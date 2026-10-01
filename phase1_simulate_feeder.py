"""
phase1_simulate_feeder.py -- physics + threat-model simulator.

Replaces `phase1_sgcc_current_generator.py.py`.

WHAT CHANGED AND WHY
--------------------
(1) LABELS.  The old script re-drew a thief independently every hour, then
    phase 3 labelled each day from hour 0 only.  Result: 743 of 745 "normal"
    days actually contained theft.  Theft is now a *persistent episode*
    (onset day + duration in days), so a day-level label is well defined and
    onset-detection latency becomes measurable.

(2) SEVERITY.  The old script injected an absolute 3-8 kW, i.e. 3.3x the
    entire feeder's honest load, against perfect meters.  Theft is now
    `alpha x the thief's own mean load`, and alpha is a swept parameter --
    the paper reports a curve, not a single number.

(3) MEASUREMENT CHAIN.  The old script had none: `meter_p = actual_p`.  Meter
    random error, per-meter systematic bias, substation error, CT error and
    AMI read failures are all modelled now.  This is what turns
    Grid_Unbalance from a theft readout into a real signal-in-noise problem.

(4) PHYSICS.  0.4 kV feeder with a cable that is actually rated for it, loads
    aggregated to realistic magnitude, non-unity power factor.  Technical
    losses now land in the 3-10% band instead of 0.024%, and line current
    tracks load instead of cable charging current.

(5) ATTACK TYPES.  Direct hooking, meter tampering and intermittent
    (peak-hour) theft, optionally with several simultaneous thieves -- the
    "downstream bleed" case the write-up claimed but never implemented.

(6) INTEGRITY.  Everything is seeded.  A non-converged hour is written as a
    NaN row instead of `continue`, which used to silently shift every
    subsequent day boundary and mis-slice the 24-hour windows.

Usage
-----
    python phase1_simulate_feeder.py --blocks block1.csv block2.csv block3.csv \
        --severity 0.20 --out data/alpha020
"""
from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import pandas as pd

from ntl_common import (FeederConfig, aggregate_to_nodes, load_lcl_blocks,
                        make_solver, subtree_members)

ATTACKS = ["hook", "tamper", "intermittent", "ev"]


# ---------------------------------------------------------------------------
def draw_episodes(cfg: FeederConfig, n_days: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Persistent theft episodes.  Each row: node, attack, start_day, end_day,
    severity.  With max_concurrent_thieves == 1 the episodes are forced to be
    mutually non-overlapping so `Thief_ID` stays a valid single label.
    """
    rows: List[dict] = []
    for node in range(1, cfg.n_nodes + 1):
        n_ep = rng.poisson(cfg.episodes_per_node)
        for _ in range(int(n_ep)):
            dur = int(np.clip(rng.lognormal(np.log(cfg.episode_mean_days), 0.6),
                              cfg.episode_min_days, cfg.episode_max_days))
            if dur >= n_days:
                continue
            start = int(rng.integers(0, n_days - dur))
            sev = cfg.theft_severity * (1.0 + rng.uniform(-cfg.severity_jitter,
                                                          cfg.severity_jitter))
            rows.append(dict(node=node, attack=rng.choice(ATTACKS, p=cfg.attack_mix),
                             start_day=start, end_day=start + dur,
                             severity=float(max(sev, 0.01))))

    eps = pd.DataFrame(rows).sort_values("start_day").reset_index(drop=True) \
        if rows else pd.DataFrame(columns=["node", "attack", "start_day", "end_day", "severity"])

    if cfg.max_concurrent_thieves <= 1 and len(eps):
        keep, occupied = [], np.zeros(n_days, dtype=bool)
        for i, e in eps.iterrows():
            sl = slice(int(e.start_day), int(e.end_day))
            if occupied[sl].any():
                continue
            occupied[sl] = True
            keep.append(i)
        eps = eps.loc[keep].reset_index(drop=True)
    else:
        # cap simultaneous thieves at the configured maximum
        keep, count = [], np.zeros(n_days, dtype=int)
        for i, e in eps.iterrows():
            sl = slice(int(e.start_day), int(e.end_day))
            if (count[sl] >= cfg.max_concurrent_thieves).any():
                continue
            count[sl] += 1
            keep.append(i)
        eps = eps.loc[keep].reset_index(drop=True)
    return eps


# ---------------------------------------------------------------------------
def simulate(cfg: FeederConfig, hourly: pd.DataFrame, backend: str = "auto",
             max_days: int | None = None, realization: int = 0):
    """
    One independent feeder realization: its own household->node assignment and
    its own theft-episode draw.  Pooling several realizations is how the study
    gets enough days out of a fixed 485-day measurement window, and it enables
    the train-on-one-feeder / test-on-another generalisation test.
    """
    rng = np.random.default_rng(cfg.seed + 100003 * realization)
    n = cfg.n_nodes

    node_kw, groups = aggregate_to_nodes(hourly, n, cfg.households_per_node, rng)
    node_kw = node_kw * cfg.load_scale
    if max_days is not None:
        node_kw = node_kw[: max_days * 24]
    n_hours = (len(node_kw) // 24) * 24
    node_kw = node_kw[:n_hours]
    n_days = n_hours // 24
    index = hourly.index[:n_hours]

    # ---- true network parameters differ from the nominal ones the feature
    # ---- extractor assumes (removes the inverse crime)
    lengths = cfg.segment_km * (1.0 + rng.uniform(-cfg.line_length_jitter,
                                                  cfg.line_length_jitter, size=n))
    pf_node = np.clip(rng.normal(cfg.power_factor, cfg.pf_node_sigma, size=n), 0.85, 0.999)
    solver, backend_used = make_solver(cfg, backend, lengths_km=lengths)
    episodes = draw_episodes(cfg, n_days, rng)
    node_mean = node_kw.mean(axis=0)

    # per-hour true theft state ------------------------------------------------
    hook_kw = np.zeros((n_hours, n))       # unmetered load actually drawn
    tamper_f = np.ones((n_hours, n))       # meter under-reporting factor
    theft_on = np.zeros((n_hours, n), dtype=bool)
    attack_of = np.full((n_days, n), "", dtype=object)

    hours_of_day = np.tile(np.arange(24), n_days)
    lo, hi = cfg.intermittent_hours
    peak_mask = (hours_of_day >= lo) & (hours_of_day <= hi)
    duty = max(peak_mask[:24].sum(), 1) / 24.0
    elo, ehi = cfg.ev_hours                      # overnight window, e.g. 01:00-05:00
    ev_mask = (hours_of_day >= elo) & (hours_of_day <= ehi)
    ev_duty = max(ev_mask[:24].sum(), 1) / 24.0

    for _, e in episodes.iterrows():
        j = int(e.node) - 1
        h0, h1 = int(e.start_day) * 24, int(e.end_day) * 24
        sev = float(e.severity)
        attack_of[int(e.start_day):int(e.end_day), j] = e.attack
        if e.attack == "hook":
            hook_kw[h0:h1, j] += sev * node_mean[j]
        elif e.attack == "tamper":
            tamper_f[h0:h1, j] *= (1.0 - sev)
        elif e.attack in ("intermittent", "ev"):
            # A duty-cycled tap: the same daily energy is drawn, but only inside
            # a fixed window, so the hourly amplitude is 1/duty times larger.
            # "intermittent" is evening-peak theft; "ev" models an overnight
            # electric-vehicle charger running on an unmetered connection.
            win, dty = (peak_mask, duty) if e.attack == "intermittent" else (ev_mask, ev_duty)
            m = np.zeros(n_hours, dtype=bool)
            m[h0:h1] = True
            m &= win
            hook_kw[m, j] += sev * node_mean[j] / dty
            theft_on[m, j] = True                # only the hours actually stolen
            continue
        theft_on[h0:h1, j] = True

    # --- unmetered but perfectly legitimate load ------------------------------
    # Communal supplies and street lighting sit behind no customer meter, so
    # they show up in the substation-minus-meters unbalance exactly the way
    # theft does.  This is the dominant real-world confounder and the original
    # simulator had none of it.
    communal_kw = node_kw * cfg.unmetered_load_frac
    dusk = (hours_of_day >= 18) | (hours_of_day <= 5)
    street = np.zeros((n_hours, n))
    street[dusk, :] = cfg.streetlight_kw / n
    unmetered_kw = communal_kw + street

    # --- distributed generation (rooftop PV) ----------------------------------
    # A clear-sky bell between 06:00 and 18:00, scaled by a daily weather factor.
    pv_kw = np.zeros((n_hours, n))
    if cfg.pv_nodes and cfg.pv_peak_frac > 0:
        shape = np.clip(np.sin(np.pi * (hours_of_day - 6) / 12.0), 0, None)
        weather = np.repeat(rng.uniform(0.25, 1.0, size=n_days), 24)
        for j in [int(b) - 1 for b in cfg.pv_nodes]:
            pv_kw[:, j] = cfg.pv_peak_frac * node_mean[j] * shape * weather

    # --- run the power flow ---------------------------------------------------
    actual_kw = node_kw + hook_kw + unmetered_kw - pv_kw   # what the grid delivers
    # Behind-the-meter PV is netted off the reading; generation the meter never
    # sees is not -- and that mismatch is what corrupts the unbalance channel.
    metered_true_kw = (node_kw - pv_kw if cfg.pv_metered else node_kw) * tamper_f
    tan_phi = np.sqrt(1.0 / pf_node ** 2 - 1.0)         # per-node, not a constant

    sub_kw = np.full(n_hours, np.nan)
    line_a = np.full((n_hours, n), np.nan)
    bus_v = np.full((n_hours, n), np.nan)
    vmin = np.full(n_hours, np.nan)
    loss_kw = np.full(n_hours, np.nan)
    load_pct = np.full((n_hours, n), np.nan)

    for t in range(n_hours):
        p = actual_kw[t]
        res = solver.solve(p, p * tan_phi)
        if res is None:                                  # non-convergence ->
            continue                                     # leave the row as NaN
        sub_kw[t] = res["substation_kw"]
        line_a[t] = res["line_amps"]
        vmin[t] = float(np.min(res["vm_pu"]))
        bus_v[t] = res["vm_pu"][1:n + 1] * cfg.vn_kv * 1000.0 / np.sqrt(3.0)
        loss_kw[t] = res["loss_kw"]
        load_pct[t] = res["loading_pct"]

    # --- measurement chain ----------------------------------------------------
    bias = rng.normal(0.0, cfg.sigma_meter_bias, size=n)
    meter_meas = metered_true_kw * (1.0 + bias) * \
        (1.0 + rng.normal(0.0, cfg.sigma_meter, size=(n_hours, n)))
    sub_meas = sub_kw * (1.0 + rng.normal(0.0, cfg.sigma_substation, size=n_hours))
    line_meas = line_a * (1.0 + rng.normal(0.0, cfg.sigma_ct, size=(n_hours, n)))

    volt_meas = bus_v * (1.0 + rng.normal(0.0, cfg.sigma_volt, size=(n_hours, n)))

    missing = rng.random((n_hours, n)) < cfg.p_missing
    meter_meas = np.where(missing, np.nan, meter_meas)

    # ---- current-sensor coverage --------------------------------------------
    # A real LV feeder is not instrumented on every segment.  Un-instrumented
    # segments are written as NaN so downstream code cannot silently use them.
    if cfg.sensor_lines is not None:          # explicit placement wins
        sensors = np.zeros(n, dtype=bool)
        for j in cfg.sensor_lines:
            sensors[int(j) - 1] = True
    elif cfg.sensor_coverage == "all":
        sensors = np.ones(n, dtype=bool)
    elif cfg.sensor_coverage == "trunk":
        sensors = np.zeros(n, dtype=bool); sensors[0] = True
    elif cfg.sensor_coverage == "none":
        sensors = np.zeros(n, dtype=bool)
    else:
        # "sparse": a CONSECUTIVE block at the downstream end of the feeder.
        # Measured, not assumed: the head-of-feeder segment is the worst place
        # to spend a sensor.  Line 1 carries every customer, so one stolen kW
        # is a tiny relative change there, and its residual duplicates what
        # Grid_Unbalance already says.  Consecutive downstream sensors let
        # Kirchhoff's law be applied exactly at the nodes between them.
        sensors = np.zeros(n, dtype=bool)
        sensors[max(0, n - cfg.n_sensors):] = True
    line_meas[:, ~sensors] = np.nan

    # --- assemble -------------------------------------------------------------
    out = pd.DataFrame({"timestamp": index})
    out["day"] = np.repeat(np.arange(n_days), 24)
    out["hour"] = hours_of_day
    out["Substation_kW"] = sub_meas
    for j in range(n):
        out[f"Meter_{j+1}_kW"] = meter_meas[:, j]
    for j in range(n):
        out[f"Line_{j+1}_A"] = line_meas[:, j]
    if cfg.report_voltage:                       # smart meters already have this
        for j in range(n):
            out[f"Meter_{j+1}_V"] = np.where(missing[:, j], np.nan, volt_meas[:, j])
    for j in range(n):
        out[f"missing_{j+1}"] = missing[:, j].astype(int)
    # ground truth (for evaluation only -- never used as a model input)
    for j in range(n):
        out[f"true_theft_{j+1}"] = theft_on[:, j].astype(int)
    out["Substation_kW_true"] = sub_kw
    out["tech_loss_kW"] = loss_kw
    out["vm_pu_min"] = vmin
    out["max_loading_pct"] = np.nanmax(load_pct, axis=1)

    # --- daily labels ---------------------------------------------------------
    day_theft = theft_on.reshape(n_days, 24, n).any(axis=1)
    daily = pd.DataFrame({"day": np.arange(n_days)})
    for j in range(n):
        daily[f"theft_{j+1}"] = day_theft[:, j].astype(int)
        daily[f"attack_{j+1}"] = attack_of[:, j]
    n_active = day_theft.sum(axis=1)
    thief_id = np.where(n_active == 1, day_theft.argmax(axis=1) + 1, 0)
    thief_id = np.where(n_active > 1, -1, thief_id)      # -1 = multiple thieves
    daily["Thief_ID"] = thief_id
    daily["n_thieves"] = n_active
    daily["realization"] = realization

    out["realization"] = realization
    meta = dict(backend=backend_used, realization=realization,
                sensors=[int(x) for x in np.where(sensors)[0] + 1],
                mean_unmetered_kw=float(unmetered_kw.sum(axis=1).mean()),
                true_lengths_km=[float(x) for x in lengths],
                true_pf=[float(x) for x in pf_node], n_days=int(n_days), n_hours=int(n_hours),
                households=[list(g) for g in groups],
                mean_feeder_kw=float(np.nanmean(actual_kw.sum(axis=1))),
                mean_tech_loss_pct=float(100 * np.nanmean(loss_kw) /
                                         np.nanmean(actual_kw.sum(axis=1))),
                min_vm_pu=float(np.nanmin(vmin)),
                max_loading_pct=float(np.nanmax(load_pct)),
                nonconverged_hours=int(np.isnan(sub_kw).sum()))
    return out, daily, episodes, meta


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blocks", nargs="+", default=["block1.csv", "block2.csv", "block3.csv"])
    ap.add_argument("--cache", default="lcl_hourly_wide.csv",
                    help="cached hourly wide matrix (built once, reused)")
    ap.add_argument("--out", default="data/run")
    ap.add_argument("--solver", choices=["auto", "pandapower", "bfs"], default="auto")
    ap.add_argument("--severity", type=float, default=0.15)
    ap.add_argument("--nodes", type=int, default=5)
    ap.add_argument("--hh-per-node", type=int, default=15)
    ap.add_argument("--load-scale", type=float, default=1.0,
                    help=">1 scales the feeder up to industrial size")
    ap.add_argument("--topology", choices=["chain", "branched"], default="chain")
    ap.add_argument("--segment-km", type=float, default=0.16)
    ap.add_argument("--cable", default="NAYY 4x150 SE")
    ap.add_argument("--vn-kv", type=float, default=0.4)
    ap.add_argument("--max-concurrent", type=int, default=1)
    ap.add_argument("--sigma-meter", type=float, default=0.005)
    ap.add_argument("--p-missing", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--unmetered-frac", type=float, default=0.03)
    ap.add_argument("--pv-nodes", nargs="+", type=int, default=None,
                    help="nodes carrying rooftop PV, e.g. --pv-nodes 2 4 5")
    ap.add_argument("--pv-peak-frac", type=float, default=0.0)
    ap.add_argument("--pv-unmetered", action="store_true",
                    help="the meter does not see the generation (tap ahead of the meter)")
    ap.add_argument("--sensor-coverage", choices=["all", "trunk", "sparse", "none"],
                    default="all", help="which line segments carry a current sensor")
    ap.add_argument("--n-sensors", type=int, default=2)
    ap.add_argument("--sensor-lines", nargs="+", type=int, default=None,
                    help="explicit sensor placement, e.g. --sensor-lines 3 5")
    ap.add_argument("--realizations", type=int, default=8,
                    help="independent feeder realizations to pool")
    args = ap.parse_args()

    cfg = FeederConfig(n_nodes=args.nodes, topology=args.topology,
                       segment_km=args.segment_km, cable=args.cable,
                       vn_kv=args.vn_kv, households_per_node=args.hh_per_node,
                       load_scale=args.load_scale,
                       theft_severity=args.severity,
                       max_concurrent_thieves=args.max_concurrent,
                       sigma_meter=args.sigma_meter, p_missing=args.p_missing,
                       unmetered_load_frac=args.unmetered_frac,
                       pv_nodes=args.pv_nodes, pv_peak_frac=args.pv_peak_frac,
                       pv_metered=not args.pv_unmetered,
                       sensor_coverage=args.sensor_coverage, n_sensors=args.n_sensors,
                       sensor_lines=args.sensor_lines,
                       seed=args.seed)

    os.makedirs(args.out, exist_ok=True)
    print("Loading Low Carbon London half-hourly household data ...")
    hourly = load_lcl_blocks(args.blocks, cache=args.cache)
    print(f"  {hourly.shape[1]} households x {len(hourly)} hours "
          f"({hourly.index.min()} -> {hourly.index.max()})")

    print(f"Simulating {args.realizations} x {cfg.n_nodes}-node {cfg.vn_kv} kV feeder, "
          f"alpha={cfg.theft_severity} ...")
    all_h, all_d, all_e, metas = [], [], [], []
    for r in range(args.realizations):
        hourly_df, daily, episodes, meta = simulate(cfg, hourly, args.solver,
                                                    args.max_days, realization=r)
        episodes["realization"] = r
        all_h.append(hourly_df); all_d.append(daily); all_e.append(episodes)
        metas.append(meta)
        print(f"  [r{r}] load {meta['mean_feeder_kw']:6.1f} kW | "
              f"loss {meta['mean_tech_loss_pct']:5.2f}% | "
              f"Vmin {meta['min_vm_pu']:.3f} pu | "
              f"load {meta['max_loading_pct']:5.1f}% | "
              f"{len(episodes)} episodes | "
              f"{int((daily.Thief_ID != 0).sum())}/{len(daily)} theft-days")

    hourly_df = pd.concat(all_h, ignore_index=True)
    daily = pd.concat(all_d, ignore_index=True)
    episodes = pd.concat(all_e, ignore_index=True)

    hourly_df.to_csv(os.path.join(args.out, "dataset_hourly.csv"), index=False)
    daily.to_csv(os.path.join(args.out, "labels_daily.csv"), index=False)
    episodes.to_csv(os.path.join(args.out, "episodes.csv"), index=False)
    cfg.to_json(os.path.join(args.out, "config.json"))
    pd.DataFrame(metas).to_json(os.path.join(args.out, "meta.json"), indent=2)

    agg = pd.DataFrame(metas)
    print(f"\n  solver ................. {agg.backend.iloc[0]}")
    print(f"  realizations ........... {args.realizations}")
    print(f"  days (pooled) .......... {len(daily)}")
    print(f"  mean feeder load ....... {agg.mean_feeder_kw.mean():.1f} kW")
    print(f"  mean technical loss .... {agg.mean_tech_loss_pct.mean():.2f} %   (target 3-10%)")
    print(f"  min bus voltage ........ {agg.min_vm_pu.min():.4f} pu  (limit 0.90)")
    print(f"  max line loading ....... {agg.max_loading_pct.max():.1f} %   (limit 100)")
    print(f"  non-converged hours .... {int(agg.nonconverged_hours.sum())}")
    print(f"  theft episodes ......... {len(episodes)}")
    print(f"  theft-days / total ..... {int((daily.Thief_ID != 0).sum())} / {len(daily)} "
          f"({100*(daily.Thief_ID != 0).mean():.1f}%)")
    print(f"\nWritten to {args.out}/")


if __name__ == "__main__":
    main()
