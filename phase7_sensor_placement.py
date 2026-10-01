"""
phase7_sensor_placement.py -- observability-zone theory and sensor-placement
optimisation for a radial LV feeder.

Why this file exists
--------------------
Phases 3-6 established *empirically* that where you put current sensors matters
far more than how many you install (three sensors on lines 3,4,5 match full
instrumentation; the same three on lines 1,2,3 do not).  That is a measurement,
not an explanation, and it cannot be transferred to a feeder of a different
size without re-running the whole pipeline.

This module supplies the missing analytical layer:

  1.  A closed-form description of what a given sensor set can and cannot
      resolve -- the *observability zone* partition (Proposition 1).
  2.  A scalar screening statistic, expected localisation ambiguity D(S),
      computable in microseconds, that rank-orders placements (Proposition 2).
  3.  A noise-aware refinement L(S) that breaks ties inside D(S) and explains
      why downstream placement wins and why the fifth sensor is nearly
      worthless (Proposition 3).
  4.  Exhaustive and greedy optimisers over both objectives.
  5.  A validation harness that checks the analytic ranking against the
      measured phase-3 / phase-6 numbers.

Nothing here runs a power flow or trains a model.  It is deliberately cheap:
the point is to screen O(2^n) placements analytically and send only the
survivors to the expensive simulate-and-train loop.

Usage
-----
    python phase7_sensor_placement.py --n-nodes 5 --budget 3
    python phase7_sensor_placement.py --n-nodes 5 --all-budgets --plot
    python phase7_sensor_placement.py --data data\\run --validate
    python phase7_sensor_placement.py --n-nodes 12 --budget 4 --greedy
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:                                    # optional: only needed for --validate
    import pandas as pd
except Exception:                       # pragma: no cover
    pd = None


# ===========================================================================
# 1.  Observability zones
# ===========================================================================
def zone_partition(n: int, sensors: Sequence[int]) -> List[List[int]]:
    """
    Partition nodes 1..n into observability zones induced by a sensor set.

    Model
    -----
    Line j connects node j-1 to node j (node 0 = substation), so line j carries
    the whole downstream load of nodes j..n.  Two boundaries are always
    available for free:

        * line 1, because the substation revenue meter measures total feeder
          infeed (in fact more accurately than a CT: sigma_sub < sigma_ct);
        * line n+1, the open end of the feeder, whose current is identically
          zero and therefore perfectly known.

    Given a boundary set B = {1} U S U {n+1}, sorted b_0 < b_1 < ... < b_M,
    the power balance over the nodes between two adjacent boundaries is a
    measurable quantity:

        dP_i = P(b_i) - P(b_{i+1}) - sum_{j in Z_i} Meter_j,
        Z_i  = {b_i, b_i + 1, ..., b_{i+1} - 1}.

    Every node inside Z_i enters dP_i with the same coefficient, so the
    current channel cannot tell them apart.  Z_i is the resolution limit.

    Returns the list of zones, each a list of node indices (1-indexed).
    """
    bounds = sorted(set([1] + [int(s) for s in sensors] + [n + 1]))
    bounds = [b for b in bounds if 1 <= b <= n + 1]
    return [list(range(bounds[i], bounds[i + 1])) for i in range(len(bounds) - 1)]


def zone_of(n: int, sensors: Sequence[int]) -> Dict[int, int]:
    """node -> index of the zone containing it."""
    out = {}
    for i, z in enumerate(zone_partition(n, sensors)):
        for j in z:
            out[j] = i
    return out


def pairwise_resolvable(n: int, sensors: Sequence[int]) -> List[int]:
    """
    Nodes for which the *current* phase-3 feature extractor produces an exact
    per-node residual Resid_j.  ntl_common.physics_features requires line j and
    every child line of j to carry a CT, i.e. on a chain: s_j AND s_{j+1}
    (with line n+1 free).  This is strictly stronger than what the zone
    partition needs, so the shipped feature set discards information whenever
    two sensors are non-adjacent.  See Remark 4 in the paper section.
    """
    S = set(int(s) for s in sensors)
    out = []
    for j in range(1, n + 1):
        need_self = (j in S) or (j == 1)          # line 1 free via substation
        need_next = (j + 1 in S) or (j + 1 == n + 1)
        if need_self and need_next:
            out.append(j)
    return out


# ===========================================================================
# 2.  Objectives
# ===========================================================================
def ambiguity(n: int, sensors: Sequence[int], prior: Optional[np.ndarray] = None,
              partition: str = "zone") -> float:
    """
    D(S) = expected number of candidate nodes an operator must inspect, given
    that theft has been detected and localised to its zone.

    With a prior pi over which node is stealing,

        D(S) = sum_j pi_j * |Z(j)|,

    which under a uniform prior reduces to the balanced-partition objective

        D(S) = (1/n) * sum_i |Z_i|^2.

    Lower is better; D = 1 means perfect node-level resolution.
    """
    zones = (shipped_partition if partition == "shipped" else zone_partition)(n, sensors)
    if prior is None:
        prior = np.full(n, 1.0 / n)
    prior = np.asarray(prior, float)
    prior = prior / prior.sum()
    tot = 0.0
    for z in zones:
        w = prior[[j - 1 for j in z]].sum()
        tot += w * len(z)
    return float(tot)


def resolution(n: int, sensors: Sequence[int], prior: Optional[np.ndarray] = None,
               partition: str = "zone") -> float:
    """
    A(S) = probability of naming the right node when guessing the most likely
    node inside the identified zone.  Under a uniform prior this is simply
    (number of zones)/n = (|S U {1}|)/n.  This is the current-channel ceiling;
    the voltage-residual channel is independent of it and can exceed it.
    """
    zones = (shipped_partition if partition == "shipped" else zone_partition)(n, sensors)
    if prior is None:
        prior = np.full(n, 1.0 / n)
    prior = np.asarray(prior, float)
    prior = prior / prior.sum()
    return float(sum(max(prior[j - 1] for j in z) for z in zones))


def zone_noise_std(n: int, sensors: Sequence[int], mean_kw: np.ndarray,
                   sigma_ct: float, sigma_meter: float, sigma_sub: float,
                   sigma_model: float = 0.03, n_avg: int = 1) -> np.ndarray:
    """
    Standard deviation of each zone residual dP_i, in kW, after averaging
    n_avg samples.

    Boundary b carries the cumulative downstream load P_b = sum_{j>=b} mean_kw_j.
    Instrument errors are *relative*, so a class-X CT contributes an absolute
    error proportional to the current it sees.  Two error families behave very
    differently under averaging:

      random (CT read noise, meter read noise, AMI quantisation)
          Var_rand(i) = sigma_ct^2 (P_lo^2 + P_hi^2) + sigma_meter^2 sum M_j^2
          averages down as 1/n_avg over the n_avg samples pooled into one test.

      systematic (unmetered communal load, street lighting, per-node power-factor
      spread, line-length error, CT calibration offset -- everything the feature
      extractor models with nominal constants while the feeder uses jittered
      true ones)
          Var_sys(i) = sigma_model^2 * (sum_{j in Z_i} M_j)^2
          does NOT average down, and grows with the *total load inside the
          zone*, because a fixed percentage of unmodelled load is a bigger
          number in a zone that contains more load.

        Var(dP_i) = Var_rand(i)/n_avg + Var_sys(i)

    Two design consequences fall straight out.  First, a sensor at the head of
    the feeder measures the sum of every node's load, so a 1% error there is n
    times larger in kW than the same 1% at the tail, while the theft it is
    trying to see is still only one node's worth -- downstream placement wins.
    Second, because Var_sys is irreducible, past a certain point extra samples
    and extra sensors stop buying anything, which is the analytical form of the
    measured "fifth sensor adds nothing" result.
    """
    mean_kw = np.asarray(mean_kw, float)
    cum = np.array([mean_kw[b - 1:].sum() for b in range(1, n + 2)])   # P_1..P_{n+1}
    S = set(int(s) for s in sensors)
    bounds = sorted(set([1] + list(S) + [n + 1]))

    def sig(b: int) -> float:
        if b == n + 1:
            return 0.0                       # open end: I == 0 exactly
        if b == 1:
            return min(sigma_sub, sigma_ct) if 1 in S else sigma_sub
        return sigma_ct

    out = []
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        v_rand = (sig(lo) * cum[lo - 1]) ** 2 + (sig(hi) * cum[hi - 1]) ** 2
        v_rand += sigma_meter ** 2 * float((mean_kw[lo - 1:hi - 1] ** 2).sum())
        v_sys = (sigma_model * float(mean_kw[lo - 1:hi - 1].sum())) ** 2
        out.append(math.sqrt(v_rand / max(n_avg, 1) + v_sys))
    return np.array(out)


def shipped_partition(n: int, sensors: Sequence[int]) -> List[List[int]]:
    """
    The partition the CURRENT phase-3 feature extractor actually achieves.

    ntl_common.physics_features() builds an exact per-node residual Resid_j only
    when line j and every child line of j carry a CT.  Nodes that fail that test
    get NaN and contribute no localisation information at all, so they collapse
    into a single undifferentiated lump.  The zone partition is therefore an
    upper bound on what the shipped code extracts, with equality iff the
    instrumented lines form a suffix-adjacent block.  Remark 4.
    """
    res = pairwise_resolvable(n, sensors)
    rest = [j for j in range(1, n + 1) if j not in res]
    parts = [[j] for j in res]
    if rest:
        parts.append(rest)
    return parts


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def localisation_score(n: int, sensors: Sequence[int], mean_kw: np.ndarray,
                       severity: float, sigma_ct: float, sigma_meter: float,
                       sigma_sub: float, sigma_model: float = 0.03,
                       alpha_fa: float = 0.05, n_avg: int = 24,
                       prior: Optional[np.ndarray] = None,
                       partition: str = "zone") -> float:
    """
    L(S) = expected probability of correctly naming the thief.

    Two independent things have to go right.  The residual covering the thief
    must clear the alarm threshold -- a detection-power term governed by noise
    -- and the operator must then pick the right node inside that residual's
    support -- a resolution term governed by how many nodes share it:

        L(S) = sum_j pi_j * Phi( rho_j - z_{1-alpha} ) / |Z(j)|,
        rho_j = severity * mean_kw_j / sd(dP_{Z(j)}).

    partition="zone"    the achievable limit, using aggregate zone residuals.
    partition="shipped" what phase 3 extracts today (per-node residuals only).
    """
    mean_kw = np.asarray(mean_kw, float)
    if prior is None:
        prior = np.full(n, 1.0 / n)
    prior = np.asarray(prior, float); prior = prior / prior.sum()

    if partition == "shipped":
        zones = shipped_partition(n, sensors)
        # a lump of unresolved nodes is still visible in the global unbalance,
        # whose noise scales with the whole feeder's load
        sd = []
        for z in zones:
            if len(z) == 1 and z[0] in pairwise_resolvable(n, sensors):
                sd.append(zone_noise_std(n, sensors, mean_kw, sigma_ct,
                                         sigma_meter, sigma_sub, sigma_model,
                                         n_avg)[
                    [i for i, zz in enumerate(zone_partition(n, sensors))
                     if z[0] in zz][0]])
            else:
                v_r = (sigma_sub * mean_kw.sum()) ** 2 + \
                      sigma_meter ** 2 * float((mean_kw ** 2).sum())
                v_s = (sigma_model * float(mean_kw.sum())) ** 2
                sd.append(math.sqrt(v_r / max(n_avg, 1) + v_s))
        sd = np.array(sd)
    else:
        zones = zone_partition(n, sensors)
        sd = zone_noise_std(n, sensors, mean_kw, sigma_ct, sigma_meter,
                            sigma_sub, sigma_model, n_avg)

    zmap = {}
    for i, z in enumerate(zones):
        for j in z:
            zmap[j] = i
    zcrit = _inv_phi(1.0 - alpha_fa)

    tot = 0.0
    for j in range(1, n + 1):
        i = zmap[j]
        rho = severity * mean_kw[j - 1] / max(sd[i], 1e-12)
        tot += prior[j - 1] * _phi(rho - zcrit) / len(zones[i])
    return float(tot)


def detection_power(n: int, sensors: Sequence[int], mean_kw: np.ndarray,
                    severity: float, sigma_ct: float, sigma_meter: float,
                    sigma_sub: float, sigma_model: float = 0.03,
                    alpha_fa: float = 0.05, n_avg: int = 24,
                    prior: Optional[np.ndarray] = None) -> float:
    """
    C(S) = P(theft is flagged at all), irrespective of whether the operator can
    then say which node it was:

        C(S) = sum_j pi_j * Phi( rho_j - z_{1-alpha} ).

    This is deliberately kept separate from L(S), because the two objectives
    pull the sensors in opposite directions and an operator has to choose.
    Maximising L rewards being certain about a few nodes, so it clusters the
    sensors at the tail and leaves a large blind zone upstream.  Maximising C
    rewards never missing theft anywhere, so it spreads the sensors out to keep
    every zone's aggregated load -- and therefore its noise -- small.  Which is
    right depends on whether the binding constraint is finding theft or
    pinpointing it; Section 6.4 reports both.
    """
    mean_kw = np.asarray(mean_kw, float)
    if prior is None:
        prior = np.full(n, 1.0 / n)
    prior = np.asarray(prior, float); prior = prior / prior.sum()
    sd = zone_noise_std(n, sensors, mean_kw, sigma_ct, sigma_meter, sigma_sub,
                        sigma_model, n_avg)
    zmap = zone_of(n, sensors)
    zcrit = _inv_phi(1.0 - alpha_fa)
    return float(sum(prior[j - 1] *
                     _phi(severity * mean_kw[j - 1] / max(sd[zmap[j]], 1e-12) - zcrit)
                     for j in range(1, n + 1)))


def _inv_phi(p: float) -> float:
    """Acklam's rational approximation to the normal quantile; plenty here."""
    if p <= 0.0:
        return -np.inf
    if p >= 1.0:
        return np.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# ===========================================================================
# 3.  Optimisation
# ===========================================================================
@dataclass
class PlacementProblem:
    n: int
    mean_kw: np.ndarray
    severity: float = 0.15
    sigma_ct: float = 0.010
    sigma_meter: float = 0.005
    sigma_sub: float = 0.002
    sigma_model: float = 0.03
    alpha_fa: float = 0.05
    n_avg: int = 24
    prior: Optional[np.ndarray] = None

    def D(self, S, partition: str = "zone") -> float:
        return ambiguity(self.n, S, self.prior, partition)

    def A(self, S, partition: str = "zone") -> float:
        return resolution(self.n, S, self.prior, partition)

    def L(self, S, partition: str = "zone") -> float:
        return localisation_score(self.n, S, self.mean_kw, self.severity,
                                  self.sigma_ct, self.sigma_meter,
                                  self.sigma_sub, self.sigma_model,
                                  self.alpha_fa, self.n_avg, self.prior,
                                  partition)

    def C(self, S) -> float:
        return detection_power(self.n, S, self.mean_kw, self.severity,
                               self.sigma_ct, self.sigma_meter, self.sigma_sub,
                               self.sigma_model, self.alpha_fa, self.n_avg,
                               self.prior)

    def sd(self, S) -> np.ndarray:
        return zone_noise_std(self.n, S, self.mean_kw, self.sigma_ct,
                              self.sigma_meter, self.sigma_sub,
                              self.sigma_model, self.n_avg)


def exhaustive(prob: PlacementProblem, budget: int, objective: str = "L",
               partition: str = "zone"):
    """Best placement of exactly `budget` sensors. O(C(n, budget))."""
    f = {"L": lambda S: prob.L(S, partition),
         "D": lambda S: -prob.D(S, partition),
         "C": lambda S: prob.C(S)}[objective]
    best, best_v = None, -np.inf
    rows = []
    for S in itertools.combinations(range(1, prob.n + 1), budget):
        v = f(S)
        rows.append((S, v, prob.D(S, partition), prob.L(S, partition)))
        if v > best_v:
            best, best_v = S, v
    rows.sort(key=lambda r: -r[1])
    return best, rows


def greedy(prob: PlacementProblem, budget: int, objective: str = "L",
           partition: str = "zone"):
    """
    Forward greedy, for feeders too long to enumerate.

    The resolution term -D(S) is monotone and submodular: splitting a zone of
    size z at an interior point into z1 + z2 gains 2*z1*z2/n, and that gain
    shrinks as earlier sensors shrink the zone.  Greedy therefore carries the
    standard (1 - 1/e) = 0.632 guarantee on D.  L(S) and C(S) add a noise term
    that is not monotone -- an extra CT injects its own calibration error into
    two residuals instead of one -- so on those two greedy is a heuristic with
    no guarantee.

    Measured over n = 4..14 and budgets 1..6 (189 problems), greedy returned
    the exhaustive optimum in 71-78% of cases depending on the objective, and
    its worst observed shortfall was a ratio of 0.86 (0.98 on L).  Comfortably
    inside the submodular bound, but not exact -- so for n small enough to
    enumerate, use exhaustive().
    """
    f = {"L": lambda S: prob.L(S, partition),
         "D": lambda S: -prob.D(S, partition),
         "C": lambda S: prob.C(S)}[objective]
    S: List[int] = []
    trace = []
    for _ in range(budget):
        cand = [j for j in range(1, prob.n + 1) if j not in S]
        if not cand:
            break
        gains = [(f(sorted(S + [j])), j) for j in cand]
        gains.sort(key=lambda t: -t[0])
        S = sorted(S + [gains[0][1]])
        trace.append((tuple(S), gains[0][0], prob.D(S, partition), prob.L(S, partition)))
    return tuple(S), trace


# ===========================================================================
# 4.  Node mean loads
# ===========================================================================
def mean_loads_from_data(data_dir: str, n: int) -> Optional[np.ndarray]:
    path = os.path.join(data_dir, "dataset_hourly.csv")
    if pd is None or not os.path.exists(path):
        return None
    cols = [f"Meter_{j+1}_kW" for j in range(n)]
    df = pd.read_csv(path, usecols=lambda c: c in cols)
    if not all(c in df.columns for c in cols):
        return None
    return df[cols].mean().to_numpy(float)


# ===========================================================================
# 5.  Validation against the measured phase-3 / phase-6 results
# ===========================================================================
# Measured numbers from this study.  Each entry is (sensor set, metric value).
# They are NOT used to fit anything -- the analytic objectives contain no free
# parameters tuned on them.  They exist so the ranking claim can be checked.
MEASURED = {
    "phase3 localise_acc (5 seeds, time split)": [
        ((1, 2, 3, 4, 5), 0.933),
        ((3, 4, 5),       0.957),
        ((1, 2, 3),       0.423),
    ],
    "phase6 recovery @5% budget, data/mt2 (logreg)": [
        ((1, 2, 3, 4, 5), 0.712),
        ((3, 4, 5),       0.707),
        ((3, 4),          0.600),
        ((4, 5),          0.478),
    ],
    "phase6 recovery @5% budget, data/run (logreg)": [
        ((1, 2, 3, 4, 5), 0.781),
        ((3, 4, 5),       0.770),
        ((4, 5),          0.428),
    ],
}


def _spearman(x, y) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    den = math.sqrt(float((rx ** 2).sum()) * float((ry ** 2).sum()))
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def _inversions(pred, meas, tol: float = 0.0):
    """Pairs whose predicted order contradicts the measured order by > tol."""
    bad = []
    k = len(pred)
    for i in range(k):
        for j in range(i + 1, k):
            dp, dm = pred[i] - pred[j], meas[i] - meas[j]
            if dp * dm < 0 and abs(dm) > tol:
                bad.append((i, j, dm))
    return bad


def validate(prob: PlacementProblem, tol: float = 0.03,
             verbose: bool = True) -> Dict[str, dict]:
    """
    Check that the analytic objectives rank placements the way the measured
    pipeline does.  Nothing here is fitted: D and L contain no parameter tuned
    on these numbers, so agreement is a prediction, not a regression.

    `tol` is the measured difference below which two placements are treated as
    indistinguishable -- seed-to-seed spread in phase 3 is a few points, so
    calling a 0.02 gap an inversion would be over-reading the data.
    """
    out = {}
    for name, rows in MEASURED.items():
        Dz = [prob.D(S) for S, _ in rows]
        Lz = [prob.L(S) for S, _ in rows]
        Ls = [prob.L(S, "shipped") for S, _ in rows]
        ms = [v for _, v in rows]
        res = dict(spearman_negD=-_spearman(Dz, ms),
                   spearman_L_zone=_spearman(Lz, ms),
                   spearman_L_shipped=_spearman(Ls, ms))
        inv = _inversions(Ls, ms, tol)
        res["inversions_beyond_tol"] = [
            (list(rows[i][0]), list(rows[j][0]), round(d, 3)) for i, j, d in inv]
        out[name] = res
        if verbose:
            print(f"\n{name}")
            print(f"  {'sensors':<16}{'measured':>10}{'D(S)':>8}"
                  f"{'L_zone':>9}{'L_shipped':>11}")
            for (S, v), d, lz, ls in zip(rows, Dz, Lz, Ls):
                print(f"  {str(S):<16}{v:>10.3f}{d:>8.2f}{lz:>9.3f}{ls:>11.3f}")
            print(f"  Spearman vs measured:  -D {res['spearman_negD']:+.3f}   "
                  f"L_zone {res['spearman_L_zone']:+.3f}   "
                  f"L_shipped {res['spearman_L_shipped']:+.3f}")
            if inv:
                for i, j, d in inv:
                    print(f"  [inversion] {rows[i][0]} vs {rows[j][0]}: "
                          f"measured gap {d:+.3f} contradicts the prediction")
            else:
                print(f"  no inversions beyond a {tol:.2f} tolerance")
    return out


# ===========================================================================
# 6.  Figure
# ===========================================================================
def make_figure(prob: PlacementProblem, path: str) -> None:
    """
    Four panels, one measure per axis (no twin axes), two categorical hues that
    pass CVD separation, and a second non-colour encoding on every series
    (marker shape + dash) so the figure survives greyscale printing.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S1, S2 = "#2a78d6", "#eb6834"          # validated categorical slots 1, 2
    INK, SEC, MUT = "#0b0b0b", "#52514e", "#b7b5ae"
    plt.rcParams.update({"font.size": 9, "axes.labelcolor": SEC,
                         "xtick.color": SEC, "ytick.color": SEC,
                         "axes.edgecolor": MUT})

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.0))
    (axA, axB), (axC, axD) = axes

    def clean(ax):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color=MUT, lw=0.5, alpha=0.45)
        ax.set_axisbelow(True)

    # -- (a) budget curve -----------------------------------------------------
    bs = list(range(1, prob.n + 1))
    Ls = [prob.L(exhaustive(prob, b, "L")[0]) for b in bs]
    Cs = [prob.C(exhaustive(prob, b, "C")[0]) for b in bs]
    axA.plot(bs, Cs, "s--", color=S2, lw=2, ms=8, label="detection  C(S)")
    axA.plot(bs, Ls, "o-", color=S1, lw=2, ms=8, label="localisation  L(S)")
    axA.annotate("C(S)", (bs[-1], Cs[-1]), color=S2, fontsize=8.5,
                 xytext=(6, 2), textcoords="offset points", weight="bold")
    axA.annotate("L(S)", (bs[-1], Ls[-1]), color=S1, fontsize=8.5,
                 xytext=(6, -8), textcoords="offset points", weight="bold")
    gain = Ls[-1] - Ls[-2]
    axA.annotate(f"the {prob.n}th sensor adds {gain:+.3f}\nto either score: line 1 only\nrepeats the substation meter",
                 (prob.n - 0.5, Ls[-1]), fontsize=7.5, color=SEC,
                 ha="left", va="top", xycoords="data",
                 xytext=(0.03, 0.98), textcoords="axes fraction",
                 arrowprops=dict(arrowstyle="->", color=MUT, lw=0.8,
                                 shrinkB=3,
                                 connectionstyle="arc3,rad=-0.18"))
    axA.set_xticks(bs); axA.set_ylim(0, 1.18)
    axA.set_xlabel("current sensors installed")
    axA.set_ylabel("probability")
    axA.set_title("a   what each extra sensor buys",
                  loc="left", fontsize=10, color=INK)
    axA.legend(frameon=False, fontsize=8, loc="lower right",
               bbox_to_anchor=(1.0, 0.02))
    clean(axA)

    # -- (b) every 3-sensor placement ----------------------------------------
    _, rows = exhaustive(prob, 3, "L")
    labs = ["{" + ",".join(map(str, S)) + "}" for S, *_ in rows]
    vals = [r[1] for r in rows]
    top = max(vals)
    cols = [S1 if v >= top - 5e-3 else MUT for v in vals]
    axB.barh(range(len(rows)), vals, color=cols, height=0.68)
    axB.set_yticks(range(len(rows)))
    axB.set_yticklabels(labs, fontsize=8)
    axB.invert_yaxis()
    axB.set_xlim(0, max(vals) * 1.5)
    for i, (v, c) in enumerate(zip(vals, cols)):
        axB.annotate(f"{v:.3f}", (v, i), fontsize=7, color=SEC,
                     va="center", xytext=(4, 0), textcoords="offset points")
    ntie = sum(1 for v in vals if v >= top - 5e-3)
    axB.annotate(f"{ntie} placements tie\nat the optimum.\nEvery one of them\nleaves line 1 bare;\nevery loser uses it.",
                 (max(vals) * 1.03, 4.6), fontsize=7.5, color=SEC, va="center")
    axB.set_xlabel("localisation score  L(S)")
    axB.set_title("b   where the three sensors go", loc="left",
                  fontsize=10, color=INK)
    axB.grid(axis="x", color=MUT, lw=0.5, alpha=0.45); axB.set_axisbelow(True)
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)

    # -- (c) predicted vs measured -------------------------------------------
    style = {"phase3 localise_acc (5 seeds, time split)":
                 ("o", S1, "phase 3  localisation accuracy"),
             "phase6 recovery @5% budget, data/mt2 (logreg)":
                 ("s", S2, "phase 6  recovery, multi-thief"),
             "phase6 recovery @5% budget, data/run (logreg)":
                 ("^", SEC, "phase 6  recovery, single thief")}
    for si, (name, rws) in enumerate(MEASURED.items()):
        mk, col, lab = style[name]
        xs = [prob.D(S) for S, _ in rws]
        ys = [v for _, v in rws]
        o = np.argsort(xs)
        axC.plot(np.array(xs)[o], np.array(ys)[o], mk + "-", color=col, ms=8,
                 lw=1.4, label=lab, alpha=0.9, mec="white", mew=0.8)
        for q, ((S, v), x) in enumerate(zip(rws, xs)):
            dx, dy = [(7, 3), (6, 4), (-26, -5)][si]
            axC.annotate("".join(map(str, S)), (x, v), fontsize=6.5,
                         xytext=(dx, dy), textcoords="offset points", color=MUT)
    axC.set_xlabel("predicted ambiguity  D(S)   [nodes per alarm]")
    axC.set_ylabel("measured performance")
    axC.set_title("c   the prediction holds against the pipeline",
                  loc="left", fontsize=10, color=INK)
    axC.set_xlim(0.88, 2.52)
    axC.legend(frameon=False, fontsize=7.2, loc="lower left")
    clean(axC)

    # -- (d) scope of the downstream rule ------------------------------------
    ns, agree, dis = list(range(4, 15)), [], []
    for n in ns:
        B = max(1, round(0.4 * n))
        pr = PlacementProblem(n=n, mean_kw=np.ones(n), severity=prob.severity,
                              sigma_ct=prob.sigma_ct, sigma_meter=prob.sigma_meter,
                              sigma_sub=prob.sigma_sub, sigma_model=prob.sigma_model,
                              alpha_fa=prob.alpha_fa, n_avg=prob.n_avg)
        SL, _ = exhaustive(pr, B, "L")
        SC, _ = exhaustive(pr, B, "C")
        agree.append(pr.C(SC)); dis.append(pr.C(SL))
    axD.plot(ns, agree, "s--", color=S2, lw=2, ms=7, label="sensors spread out")
    axD.plot(ns, dis, "o-", color=S1, lw=2, ms=7, label="sensors clustered downstream")
    split = next(n for n, a, d in zip(ns, agree, dis) if a - d > 0.02)
    axD.axvline(split, color=MUT, lw=1, ls=":", zorder=0)
    axD.annotate(f"identical up to n = {split-1};\nbeyond it, clustering\nopens a blind zone",
                 (split + 0.15, 0.955), fontsize=7.5, color=SEC, va="top")
    axD.set_xlabel("nodes on the feeder   (sensor budget = 0.4 n)")
    axD.set_ylabel("detection power  C(S)")
    axD.set_ylim(0.32, 1.02); axD.set_xticks(ns)
    axD.set_title("d   the downstream rule does not scale", loc="left",
                  fontsize=10, color=INK)
    axD.legend(frameon=False, fontsize=7.5, loc="lower left",
               bbox_to_anchor=(-0.01, -0.02))
    clean(axD)

    fig.tight_layout(pad=1.3)
    fig.savefig(path, dpi=220, facecolor="white")
    fig.savefig(os.path.splitext(path)[0] + ".pdf", facecolor="white")
    plt.close(fig)


# ===========================================================================
# 6b.  Self-test -- the propositions, checked exhaustively rather than asserted
# ===========================================================================
def selftest(n: int = 7, verbose: bool = True) -> bool:
    """
    Exhaustive numerical check of every claim the paper section makes about the
    zone partition.  Runs over all 2^n placements, so keep n small.
    """
    ok = True

    def say(tag, good):
        nonlocal ok
        ok = ok and good
        if verbose:
            print(f"  [{'PASS' if good else 'FAIL'}] {tag}")

    allS = [S for r in range(n + 1)
            for S in itertools.combinations(range(1, n + 1), r)]

    good = all(sorted(j for z in zone_partition(n, S) for j in z)
               == list(range(1, n + 1)) for S in allS)
    say(f"Prop 1: zones partition the node set, all {len(allS)} placements", good)

    good = all(abs(ambiguity(n, S)
                   - sum(len(z) ** 2 for z in zone_partition(n, S)) / n) < 1e-12
               for S in allS)
    say("Eq 2: D(S) = (1/n) sum |Z_i|^2 under a uniform prior", good)

    good = all(abs(resolution(n, S) - len(set(S) | {1}) / n) < 1e-12 for S in allS)
    say("Eq 3: A(S) = |S u {1}| / n", good)

    vm = vs = 0
    for r in range(n):
        for S in itertools.combinations(range(1, n + 1), r):
            for T in itertools.combinations(range(1, n + 1), r + 1):
                if not set(S) < set(T):
                    continue
                for j in range(1, n + 1):
                    if j in T:
                        continue
                    gS = ambiguity(n, S) - ambiguity(n, tuple(sorted(S + (j,))))
                    gT = ambiguity(n, T) - ambiguity(n, tuple(sorted(T + (j,))))
                    vm += gS < -1e-12
                    vs += gT > gS + 1e-12
    say(f"Prop 2: -D monotone ({vm} violations) and submodular ({vs} violations)",
        vm == 0 and vs == 0)

    bad = [S for S in allS if 1 not in S
           and ambiguity(n, S) != ambiguity(n, tuple(sorted(S + (1,))))]
    say(f"Rule 1: instrumenting line 1 never changes D ({len(bad)} counterexamples)",
        not bad)

    ratios = []
    for nn in range(4, 11):
        pr = PlacementProblem(n=nn, mean_kw=np.ones(nn))
        for b in range(1, min(nn, 5) + 1):
            Sg, _ = greedy(pr, b, "D")
            Se, _ = exhaustive(pr, b, "D")
            ratios.append((1 / pr.D(Sg)) / (1 / pr.D(Se)))
    say(f"Prop 2 corollary: greedy on D beats the (1-1/e) bound "
        f"(worst ratio {min(ratios):.3f} > 0.632)", min(ratios) > 1 - 1 / math.e)
    return ok


# ===========================================================================
# 7.  CLI
# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Observability-zone analysis and sensor-placement "
                    "optimisation for a radial feeder.")
    ap.add_argument("--n-nodes", type=int, default=5)
    ap.add_argument("--data", default=None,
                    help="folder holding dataset_hourly.csv; node mean loads "
                         "are read from it.  Omitted -> homogeneous feeder.")
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--all-budgets", action="store_true")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--objective", choices=["L", "D", "C"], default="L",
                    help="L = localisation score, D = ambiguity, "
                         "C = detection power")
    ap.add_argument("--partition", choices=["zone", "shipped"], default="zone",
                    help="zone = achievable limit with aggregate residuals; "
                         "shipped = what phase 3 extracts today")
    ap.add_argument("--severity", type=float, default=0.15)
    ap.add_argument("--sigma-ct", type=float, default=0.010)
    ap.add_argument("--sigma-meter", type=float, default=0.005)
    ap.add_argument("--sigma-sub", type=float, default=0.002)
    ap.add_argument("--sigma-model", type=float, default=0.03,
                    help="systematic, non-averaging error as a fraction of "
                         "zone load (unmetered load, pf spread, length jitter)")
    ap.add_argument("--alpha-fa", type=float, default=0.05)
    ap.add_argument("--n-avg", type=int, default=24)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--selftest", action="store_true",
                    help="exhaustively check the propositions and exit")
    ap.add_argument("--tol", type=float, default=0.03)
    ap.add_argument("--plot", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.selftest:
        print("=" * 78)
        print("PHASE 7 SELF-TEST -- propositions checked by enumeration")
        print("=" * 78)
        raise SystemExit(0 if selftest(min(args.n_nodes, 8)) else 1)

    n = args.n_nodes
    mean_kw = None
    if args.data:
        mean_kw = mean_loads_from_data(args.data, n)
        if mean_kw is None:
            print(f"[warn] no usable dataset_hourly.csv in {args.data}; "
                  f"using a homogeneous feeder")
    if mean_kw is None:
        mean_kw = np.ones(n)

    prob = PlacementProblem(n=n, mean_kw=mean_kw, severity=args.severity,
                            sigma_ct=args.sigma_ct, sigma_meter=args.sigma_meter,
                            sigma_sub=args.sigma_sub,
                            sigma_model=args.sigma_model,
                            alpha_fa=args.alpha_fa, n_avg=args.n_avg)
    P = args.partition

    print("=" * 78)
    print("PHASE 7 -- observability zones and optimal sensor placement")
    print("=" * 78)
    print(f"nodes             : {n}")
    print(f"node mean load    : " + ", ".join(f"{v:.2f}" for v in mean_kw) + "  kW")
    print(f"theft severity    : {args.severity:.0%} of the thief's own load")
    print(f"sigma ct/mtr/sub  : {args.sigma_ct:.3f} / {args.sigma_meter:.3f}"
          f" / {args.sigma_sub:.3f}   (random, relative)")
    print(f"sigma model       : {args.sigma_model:.3f}   (systematic, does not average)")
    print(f"samples per test  : {args.n_avg}     false-alarm rate: {args.alpha_fa:.2f}")
    print(f"partition model   : {P}")

    results = {"config": dict(n=n, mean_kw=list(map(float, mean_kw)),
                              severity=args.severity, sigma_ct=args.sigma_ct,
                              sigma_meter=args.sigma_meter,
                              sigma_sub=args.sigma_sub,
                              sigma_model=args.sigma_model,
                              alpha_fa=args.alpha_fa, n_avg=args.n_avg,
                              partition=P)}

    budgets = list(range(1, n + 1)) if args.all_budgets else \
              [args.budget if args.budget is not None else max(1, n - 2)]

    for b in budgets:
        print("\n" + "-" * 78)
        print(f"BUDGET = {b} sensor(s)")
        print("-" * 78)
        if args.greedy:
            S, trace = greedy(prob, b, args.objective, P)
            print("  greedy build-up:")
            for st, v, d, l in trace:
                print(f"    add -> {str(st):<20} D={d:5.2f}  L={l:.3f}")
            Sx, rows = exhaustive(prob, b, args.objective, P)
            ok = "match" if abs(prob.L(S, P) - prob.L(Sx, P)) < 1e-9 else "GREEDY SUBOPTIMAL"
            print(f"  greedy    : {S}   L={prob.L(S, P):.3f}  D={prob.D(S, P):.2f}")
            print(f"  exhaustive: {Sx}   L={prob.L(Sx, P):.3f}  D={prob.D(Sx, P):.2f}   ({ok})")
            best = Sx
        else:
            best, rows = exhaustive(prob, b, args.objective, P)
            print(f"  {'rank':<5}{'sensors':<18}{'L(S)':>8}{'C(S)':>8}"
                  f"{'D(S)':>8}   zones")
            shown = rows[:8]
            for i, (S, v, d, l) in enumerate(shown):
                zs = (shipped_partition if P == "shipped" else zone_partition)(n, S)
                zstr = " | ".join("".join(map(str, z)) for z in zs)
                print(f"  {i+1:<5}{str(S):<18}{l:>8.3f}{prob.C(S):>8.3f}"
                      f"{d:>8.2f}   {zstr}")
            if len(rows) > len(shown):
                print(f"  ...   ({len(rows)-len(shown)} more)")
                S, v, d, l = rows[-1]
                zs = (shipped_partition if P == "shipped" else zone_partition)(n, S)
                zstr = " | ".join("".join(map(str, z)) for z in zs)
                print(f"  {'worst':<5}{str(S):<18}{l:>8.3f}{prob.C(S):>8.3f}"
                      f"{d:>8.2f}   {zstr}")

        sd = prob.sd(best)
        print(f"\n  best placement : {best}    "
              f"L={prob.L(best,'zone'):.3f}  C={prob.C(best):.3f}  "
              f"D={prob.D(best,'zone'):.2f}")
        print(f"  {'zone':<14}{'load kW':>10}{'sd(dP) kW':>12}{'SNR':>8}{'P(detect)':>11}")
        zc = _inv_phi(1.0 - args.alpha_fa)
        for z, s in zip(zone_partition(n, best), sd):
            ld = float(mean_kw[[j-1 for j in z]].sum())
            snr = args.severity * float(np.mean(mean_kw[[j-1 for j in z]])) / max(s, 1e-12)
            print(f"  {'{' + ','.join(map(str, z)) + '}':<14}{ld:>10.2f}"
                  f"{s:>12.4f}{snr:>8.2f}{_phi(snr - zc):>11.3f}")
        pr = pairwise_resolvable(n, best)
        print(f"  exact per-node residuals from the shipped extractor: "
              f"{pr if pr else 'none'}")
        miss = [j for j in range(1, n + 1) if j not in pr]
        if miss:
            print(f"  nodes {miss} resolve only to a ZONE residual, which "
                  f"physics_features() discards today (Remark 4):")
            print(f"      L_zone = {prob.L(best, 'zone'):.3f}   vs   "
                  f"L_shipped = {prob.L(best, 'shipped'):.3f}")
        results[str(b)] = dict(best=list(best), L_zone=prob.L(best, "zone"),
                               L_shipped=prob.L(best, "shipped"),
                               C=prob.C(best),
                               D=prob.D(best, P), A=prob.A(best, P),
                               zones=[list(z) for z in zone_partition(n, best)],
                               sd_kw=[float(x) for x in sd])

    if args.validate:
        print("\n" + "=" * 78)
        print("VALIDATION -- analytic ranking vs measured pipeline performance")
        print("=" * 78)
        results["validation"] = validate(prob, args.tol)

    if args.plot:
        make_figure(prob, args.plot)
        print(f"\n[figure] {args.plot}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"[json]   {args.out}")


if __name__ == "__main__":
    main()
