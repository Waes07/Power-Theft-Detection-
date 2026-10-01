"""
ntl_common.py -- shared physics, topology, load-profile and RNG utilities.

Replaces the ad-hoc network construction that used to live inline in
phase1_sgcc_current_generator.py.

Key changes vs the original code
--------------------------------
1.  The feeder is modelled at 0.4 kV (LV), where `NAYY 4x50/150 SE` is the
    correct cable.  The original ran a 0.4 kV LV cable at 11 kV, which
    (a) exceeds the cable's insulation rating ~27x and (b) made the reported
    "line current" ~92% capacitive charging current instead of load current.
2.  Loads are aggregated service points (K households each), so the feeder
    carries a realistic 40-120 kW and technical losses land in the 3-10%
    band instead of 0.02%.
3.  An independent backward/forward-sweep (BFS) power-flow solver is provided
    alongside pandapower, so (a) the simulator runs without pandapower and
    (b) the two engines can be cross-validated against each other -- a
    reviewer-facing correctness argument the original code did not have.
4.  General radial *tree* topology (parent array), not just a 3-node chain,
    so the study can scale to a feeder with laterals.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

F_HZ = 50.0

# ---------------------------------------------------------------------------
# Cable library.  Values match pandapower's standard types so the BFS solver
# and pandapower agree.  (r/x in ohm/km, c in nF/km, max_i in kA, un in kV)
# ---------------------------------------------------------------------------
CABLES: Dict[str, Dict[str, float]] = {
    # Low-voltage aluminium cables -- these are 0.4 kV parts.
    "NAYY 4x50 SE":  dict(r=0.642, x=0.083, c=210.0, max_i=0.142, un=0.4),
    "NAYY 4x120 SE": dict(r=0.225, x=0.080, c=264.0, max_i=0.242, un=0.4),
    "NAYY 4x150 SE": dict(r=0.208, x=0.080, c=261.0, max_i=0.270, un=0.4),
    # Medium-voltage cable, for the optional 11 kV variant of the study.
    "NA2XS2Y 1x185 RM/25 12/20 kV": dict(r=0.161, x=0.117, c=273.0, max_i=0.362, un=20.0),
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class FeederConfig:
    """Everything about the physical network and the measurement chain."""

    # --- topology -----------------------------------------------------------
    n_nodes: int = 5                       # number of metered service points
    topology: str = "chain"                # "chain" | "branched"
    parent: Optional[List[int]] = None     # explicit tree; index i -> parent bus
    segment_km: float = 0.16
    cable: str = "NAYY 4x150 SE"
    vn_kv: float = 0.4
    slack_vm_pu: float = 1.02
    power_factor: float = 0.95             # lagging; original code used 1.0

    # --- load aggregation ---------------------------------------------------
    households_per_node: int = 15
    load_scale: float = 1.0                # multiply node load (>1 = industrial feeder)

    # --- model-mismatch / "inverse crime" removal ---------------------------
    # The feature extractor and the rule baseline use NOMINAL constants.  The
    # simulator perturbs the true ones, so no estimator can invert the exact
    # equations that generated the data.  Without this the study is an inverse
    # crime and every physics-informed feature looks better than it is.
    line_length_jitter: float = 0.15   # true segment length vs nominal
    pf_node_sigma: float = 0.03        # per-node power factor spread
    unmetered_load_frac: float = 0.03  # communal/street load behind no meter
    streetlight_kw: float = 1.5        # unmetered dusk-to-dawn load on the feeder

    # --- distributed generation -------------------------------------------
    pv_nodes: Optional[List[int]] = None   # which nodes carry rooftop PV
    pv_peak_frac: float = 0.0              # peak PV output / that node's mean load
    pv_metered: bool = True                # True: PV sits behind the meter (net
                                           # metering, utility sees net import).
                                           # False: generation the meter never
                                           # sees -- e.g. a tap ahead of the meter

    # --- theft / threat model ----------------------------------------------
    theft_severity: float = 0.15           # alpha: theft as a fraction of the
                                           # thief's OWN mean load (was: an
                                           # absolute 3-8 kW, i.e. 3.3x the
                                           # whole feeder's load)
    severity_jitter: float = 0.25          # +/- relative spread around alpha
    attack_mix: Tuple[float, ...] = (0.40, 0.25, 0.175, 0.175)
    #            hook, tamper, intermittent (evening), ev (overnight)
    episodes_per_node: float = 6.0         # expected episodes over the horizon
    episode_mean_days: float = 9.0
    episode_min_days: int = 3
    episode_max_days: int = 45
    max_concurrent_thieves: int = 1        # set >1 for the multi-thief study
    intermittent_hours: Tuple[int, int] = (18, 23)   # evening-peak theft
    ev_hours: Tuple[int, int] = (1, 5)               # overnight EV-charging theft

    # --- measurement chain (all absent from the original simulator) ---------
    sigma_meter: float = 0.005             # IEC Class 0.5S random error
    sigma_meter_bias: float = 0.003        # per-meter systematic offset
    sigma_substation: float = 0.002
    sigma_ct: float = 0.010                # line current sensors
    sigma_volt: float = 0.005              # smart-meter voltage channel (Class 1)
    report_voltage: bool = True            # do the meters report voltage magnitude?
    sensor_coverage: str = "all"           # all | trunk | sparse | none
    n_sensors: int = 2                     # used when sensor_coverage == "sparse"
    sensor_lines: Optional[List[int]] = None   # explicit placement, 1-indexed
    p_missing: float = 0.02                # AMI read failures

    # --- reproducibility ----------------------------------------------------
    seed: int = 20260821

    def resolved_parent(self) -> List[int]:
        """Return parent[i] for buses 1..n_nodes (bus 0 is the substation)."""
        if self.parent is not None:
            return list(self.parent)
        if self.topology == "chain":
            return list(range(0, self.n_nodes))          # 0,1,2,... daisy chain
        if self.topology == "branched":
            # A short main trunk with laterals hanging off it.
            trunk = max(2, self.n_nodes // 3)
            par = list(range(0, trunk))                  # trunk buses 1..trunk
            for i in range(trunk, self.n_nodes):
                par.append(1 + (i - trunk) % trunk)      # attach to a trunk bus
            return par
        raise ValueError(f"unknown topology {self.topology!r}")

    def to_json(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)


# ---------------------------------------------------------------------------
# Topology helpers
# ---------------------------------------------------------------------------
def subtree_members(parent: Sequence[int], n_nodes: int) -> List[List[int]]:
    """
    downstream[i] = list of load buses (1-indexed) fed through line i
    (line i is the branch whose *to* bus is i).  Used both by the physics and
    by the physics-informed residual features.
    """
    children: Dict[int, List[int]] = {b: [] for b in range(n_nodes + 1)}
    for i in range(1, n_nodes + 1):
        children[parent[i - 1]].append(i)

    downstream: List[List[int]] = [[] for _ in range(n_nodes + 1)]

    def walk(b: int) -> List[int]:
        acc = [b] if b != 0 else []
        for c in children[b]:
            acc = acc + walk(c)
        downstream[b] = acc
        return acc

    walk(0)
    return [downstream[i] for i in range(n_nodes + 1)]   # index by bus


def topo_order(parent: Sequence[int], n_nodes: int) -> List[int]:
    """Buses ordered so a parent always precedes its children."""
    order, frontier = [], [0]
    children: Dict[int, List[int]] = {b: [] for b in range(n_nodes + 1)}
    for i in range(1, n_nodes + 1):
        children[parent[i - 1]].append(i)
    while frontier:
        b = frontier.pop(0)
        order.append(b)
        frontier.extend(children[b])
    return order


# ---------------------------------------------------------------------------
# Backward / forward sweep power flow (radial networks)
# ---------------------------------------------------------------------------
class BFSSolver:
    """
    Positive-sequence backward/forward sweep for a radial feeder, using the
    same pi-model as pandapower (series r+jx, shunt charging split half at
    each end).  Reports |I| at both line terminals and returns the max, which
    is exactly what pandapower's `res_line.i_ka` reports.

    Verified against pandapower output from the original project to 4
    significant figures (see phase2 audit, `--check-solver`).
    """

    def __init__(self, cfg: FeederConfig, lengths_km=None):
        self.cfg = cfg
        cab = CABLES[cfg.cable]
        n = cfg.n_nodes
        self.n = n
        self.parent = cfg.resolved_parent()
        self.order = topo_order(self.parent, n)
        self.down = subtree_members(self.parent, n)

        L = np.asarray(lengths_km if lengths_km is not None
                       else [cfg.segment_km] * n, dtype=float)
        self.z = np.array([complex(cab["r"] * l, cab["x"] * l) for l in L])
        self.y = np.array([complex(0.0, 2 * math.pi * F_HZ * cab["c"] * 1e-9 * l) for l in L])
        self.max_i_a = cab["max_i"] * 1000.0
        self.v_ln = cfg.vn_kv * 1000.0 / math.sqrt(3.0)
        self.v0 = self.v_ln * cfg.slack_vm_pu

        # shunt admittance lumped at each bus = half of every incident line
        self.y_bus = np.zeros(n + 1, dtype=complex)
        for i in range(1, n + 1):
            self.y_bus[i] += self.y[i - 1] / 2.0
            self.y_bus[self.parent[i - 1]] += self.y[i - 1] / 2.0

    def solve(self, p_kw: np.ndarray, q_kvar: np.ndarray,
              tol: float = 1e-9, max_iter: int = 60):
        """
        p_kw / q_kvar: three-phase load at buses 1..n (kW / kvar).
        Returns dict with substation_kw, line_amps (n,), vm_pu (n+1,), loss_kw.
        """
        n = self.n
        v = np.full(n + 1, self.v0, dtype=complex)
        s_ph = (p_kw * 1000.0 + 1j * q_kvar * 1000.0) / 3.0     # per phase, VA

        i_series = np.zeros(n, dtype=complex)
        for _ in range(max_iter):
            v_prev = v.copy()

            # ---- backward sweep: accumulate currents from leaves to source
            i_inj = np.zeros(n + 1, dtype=complex)
            for b in range(1, n + 1):
                i_inj[b] = np.conj(s_ph[b - 1] / v[b]) + v[b] * self.y_bus[b]
            for b in reversed(self.order):
                if b == 0:
                    continue
                i_series[b - 1] = i_inj[b]
                i_inj[self.parent[b - 1]] += i_inj[b]

            # ---- forward sweep: update voltages from source to leaves
            v[0] = self.v0
            for b in self.order:
                if b == 0:
                    continue
                v[b] = v[self.parent[b - 1]] - self.z[b - 1] * i_series[b - 1]

            if np.max(np.abs(v - v_prev)) < tol:
                break

        # terminal currents including the local half-shunt, as pandapower does
        i_from = np.array([i_series[i] + v[self.parent[i]] * self.y[i] / 2.0
                           for i in range(n)])
        i_to = np.array([i_series[i] - v[i + 1] * self.y[i] / 2.0
                         for i in range(n)])
        line_amps = np.maximum(np.abs(i_from), np.abs(i_to))

        i_slack = np.zeros(1, dtype=complex)
        for i in range(n):
            if self.parent[i] == 0:
                i_slack = i_slack + i_from[i]
        p_slack_kw = 3.0 * np.real(v[0] * np.conj(i_slack[0])) / 1000.0

        return dict(
            substation_kw=float(p_slack_kw),
            line_amps=line_amps.astype(float),
            vm_pu=(np.abs(v) / self.v_ln).astype(float),
            loss_kw=float(p_slack_kw - p_kw.sum()),
            loading_pct=(line_amps / self.max_i_a * 100.0).astype(float),
        )


# ---------------------------------------------------------------------------
# pandapower backend (optional -- used when the package is installed)
# ---------------------------------------------------------------------------
class PandapowerSolver:
    """Thin wrapper so phase 1 can swap engines with --solver pandapower."""

    def __init__(self, cfg: FeederConfig, lengths_km=None):
        import pandapower as pp  # noqa: F401  (imported lazily and on purpose)

        self.pp = pp
        self.cfg = cfg
        self.parent = cfg.resolved_parent()
        n = cfg.n_nodes

        net = pp.create_empty_network(f_hz=F_HZ)
        buses = [pp.create_bus(net, vn_kv=cfg.vn_kv, name="Substation")]
        for i in range(1, n + 1):
            buses.append(pp.create_bus(net, vn_kv=cfg.vn_kv, name=f"Node {i}"))
        pp.create_ext_grid(net, bus=buses[0], vm_pu=cfg.slack_vm_pu)

        lens = lengths_km if lengths_km is not None else [cfg.segment_km] * n
        self.lines = []
        for i in range(1, n + 1):
            self.lines.append(
                pp.create_line(net, from_bus=buses[self.parent[i - 1]], to_bus=buses[i],
                               length_km=float(lens[i - 1]), std_type=cfg.cable,
                               name=f"Line {i}")
            )
        self.loads = [pp.create_load(net, bus=buses[i], p_mw=0.0, q_mvar=0.0)
                      for i in range(1, n + 1)]
        self.net = net
        self.max_i_a = CABLES[cfg.cable]["max_i"] * 1000.0

    def solve(self, p_kw: np.ndarray, q_kvar: np.ndarray):
        net = self.net
        for k, li in enumerate(self.loads):
            net.load.at[li, "p_mw"] = p_kw[k] / 1000.0
            net.load.at[li, "q_mvar"] = q_kvar[k] / 1000.0
        try:
            self.pp.runpp(net)
        except Exception:
            return None                      # caller records a NaN row
        return dict(
            substation_kw=float(net.res_ext_grid.p_mw.iloc[0] * 1000.0),
            line_amps=net.res_line.i_ka.values[: self.cfg.n_nodes] * 1000.0,
            vm_pu=net.res_bus.vm_pu.values.astype(float),
            loss_kw=float(net.res_ext_grid.p_mw.iloc[0] * 1000.0 - p_kw.sum()),
            loading_pct=net.res_line.loading_percent.values[: self.cfg.n_nodes],
        )


def make_solver(cfg: FeederConfig, backend: str = "auto", lengths_km=None):
    if backend in ("auto", "pandapower"):
        try:
            return PandapowerSolver(cfg, lengths_km), "pandapower"
        except Exception:
            if backend == "pandapower":
                raise
    return BFSSolver(cfg, lengths_km), "bfs"


# ---------------------------------------------------------------------------
# Load-profile loading
# ---------------------------------------------------------------------------
def load_lcl_blocks(paths: Sequence[str], cache: Optional[str] = None) -> pd.DataFrame:
    """
    Read raw Low Carbon London block files and return an hourly wide frame
    (index = timestamp, one column per LCLid, values = kWh in that hour).

    NOTE ON PROVENANCE.  These files are the *SmartMeter Energy Consumption
    Data in London Households* set (UK Power Networks / Low Carbon London,
    2011-2014, half-hourly).  They are **not** the SGCC dataset, which is
    daily consumption from ~42k Chinese customers, 2014-2016.  The original
    code named every file and variable "sgcc"; that is corrected here.

    This also fixes a data bug: the original read the first 48,000 rows of a
    block file and treated them as one household.  Each block holds 50
    households concatenated, so those 48,000 rows actually spliced several
    different households end-to-end into one fake "consumer".
    """
    if cache and os.path.exists(cache):
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    frames = []
    for p in paths:
        df = pd.read_csv(p, usecols=[0, 1, 2], header=0,
                         names=["id", "ts", "kwh"], dtype={"id": "str"})
        df["kwh"] = pd.to_numeric(df["kwh"], errors="coerce")
        df["ts"] = pd.to_datetime(df["ts"].str.strip().str.slice(0, 19), errors="coerce")
        df = df.dropna(subset=["ts"])
        df["h"] = df["ts"].dt.floor("h")
        frames.append(df.groupby(["id", "h"])["kwh"].sum().unstack(0))

    wide = pd.concat(frames, axis=1)
    # keep the densest common window and well-covered households
    dense = wide.loc[wide.notna().mean(axis=1) > 0.9]
    cov = dense.notna().mean()
    dense = dense[cov[cov >= 0.98].index].interpolate(limit_direction="both")
    start = dense.index[dense.index.hour == 0][0]
    dense = dense.loc[start:]
    dense = dense.iloc[: (len(dense) // 24) * 24]
    if cache:
        dense.round(4).to_csv(cache)
    return dense


def aggregate_to_nodes(hourly: pd.DataFrame, n_nodes: int, hh_per_node: int,
                       rng: np.random.Generator) -> Tuple[np.ndarray, List[List[str]]]:
    """
    Group real households into aggregated service points.

    Aggregation is what makes the feeder loading realistic (tens of kW rather
    than 1.7 kW) and therefore what makes technical losses non-negligible.
    Returns (n_hours, n_nodes) array in kW and the household assignment.
    """
    ids = list(hourly.columns)
    rng.shuffle(ids)
    need = n_nodes * hh_per_node
    if need > len(ids):                     # sample with replacement if short
        ids = list(rng.choice(ids, size=need, replace=True))
    groups = [ids[i * hh_per_node:(i + 1) * hh_per_node] for i in range(n_nodes)]
    mat = np.column_stack([hourly[g].sum(axis=1).values for g in groups])
    return mat.astype(float), groups


# ---------------------------------------------------------------------------
# Physics-informed features
# ---------------------------------------------------------------------------
def physics_features(df: pd.DataFrame, cfg: FeederConfig) -> pd.DataFrame:
    """
    Derive the two physics channels the models actually need, from *measured*
    quantities only (substation power, meter reads, line currents).

    Unbalance_kW
        substation power minus the sum of metered consumption.  Detects that
        theft is happening somewhere, but says nothing about where.

    Resid_j  (physics-informed localisation residual -- new in this version)
        Kirchhoff's current law applied at node j:
            I_node_j = I_line_j - sum(I_line_c for c a child of j)
        compared against the current node j's *meter* says it should be drawing.
        A hook raises I_node_j; meter tampering lowers the expectation.  Either
        way the residual rises at the offending node and nowhere else, which is
        the "topological elimination" the project claims -- expressed as an
        explicit feature rather than left for the network to rediscover.
    """
    n = cfg.n_nodes
    parent = cfg.resolved_parent()
    children = {b: [] for b in range(n + 1)}
    for i in range(1, n + 1):
        children[parent[i - 1]].append(i)

    meters = df[[f"Meter_{j+1}_kW" for j in range(n)]].interpolate(limit_direction="both")
    lines = df[[f"Line_{j+1}_A" for j in range(n)]].to_numpy()
    observed = ~np.isnan(lines).all(axis=0)     # which segments actually have a CT

    out = pd.DataFrame(index=df.index)
    out["Unbalance_kW"] = df["Substation_kW"].to_numpy() - meters.to_numpy().sum(axis=1)

    v_ll = cfg.vn_kv * 1000.0 * cfg.slack_vm_pu
    k = 1.0 / (math.sqrt(3.0) * v_ll * cfg.power_factor)
    for j in range(1, n + 1):
        # KCL at node j is only computable if this segment and every downstream
        # segment leaving it carry a current sensor.  With sparse metering the
        # residual is undefined for some nodes -- that is the realistic case,
        # and it is where a temporal model has something to contribute.
        needed = [j] + list(children[j])
        if not all(observed[b - 1] for b in needed):
            out[f"Resid_{j}"] = np.nan
            continue
        i_node = lines[:, j - 1] - sum(lines[:, c - 1] for c in children[j]) \
            if children[j] else lines[:, j - 1]
        i_expect = meters[f"Meter_{j}_kW"].to_numpy() * 1000.0 * k
        out[f"Resid_{j}"] = i_node - i_expect

    # ---- voltage-profile residual -----------------------------------------
    # Smart meters already report voltage magnitude, so this costs no new
    # hardware.  A linearised DistFlow estimate predicts each node's voltage
    # from the METERED loads; theft downstream of node t pushes extra current
    # through lines 1..t only, so the measured-minus-predicted drop grows with
    # distance up to node t and then flattens.  The position of that knee is
    # the thief -- localisation information from meters alone.
    if all(f"Meter_{j+1}_V" in df.columns for j in range(n)):
        cab = CABLES[cfg.cable]
        v_ln = cfg.vn_kv * 1000.0 / math.sqrt(3.0)
        tan_phi = math.sqrt(1.0 / cfg.power_factor ** 2 - 1.0)
        L = cfg.segment_km
        down = subtree_members(parent, n)
        p_kw = meters.to_numpy()
        drop = np.zeros((len(df), n + 1))
        for b in topo_order(parent, n):
            if b == 0:
                continue
            idx = [m - 1 for m in down[b]]                  # loads fed via line b
            p_down = p_kw[:, idx].sum(axis=1) * 1000.0      # W, three-phase
            q_down = p_down * tan_phi
            drop[:, b] = drop[:, parent[b - 1]] + \
                (cab["r"] * L * p_down + cab["x"] * L * q_down) / (3.0 * v_ln)
        for j in range(1, n + 1):
            v_pred = v_ln * cfg.slack_vm_pu - drop[:, j]
            out[f"VResid_{j}"] = v_pred - df[f"Meter_{j}_V"].to_numpy()
    return out
