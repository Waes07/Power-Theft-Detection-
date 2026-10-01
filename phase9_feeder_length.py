"""
phase9_feeder_length.py -- does the "cluster the sensors downstream" rule hold
on a longer feeder?

WHAT THIS TESTS
---------------
Phase 7 panel (d) is pure theory: it claims that clustering sensors at the tail
of the feeder is optimal for short feeders, but that from about eight nodes
onward it opens a blind zone and a spread-out placement detects far more theft.
Nothing in the study had ever simulated a feeder longer than five nodes, so that
panel was a prediction with no measurement behind it.

This stage runs the experiment, and it is deliberately built as a controlled
comparison with BOTH arms:

    n = 7   the theory says clustered and spread are EQUIVALENT here
            -> if we measure a large difference, the theory is wrong
    n = 9   the theory says spread should win clearly
            -> if we measure no difference, the theory is wrong

A control where no effect is predicted is what makes the treatment convincing.
Running only the n = 9 arm would prove much less.

KEEPING IT CHEAP
----------------
A longer feeder means more power flows, so the horizon is cut instead:
`--days 120` uses the first 120 days of the cached household data rather than
all 485.  That is enough for a placement comparison (the effect sizes in
question are 0.2-0.3) while running in minutes rather than hours.  The reduced
horizon is reported with the result, and the episode count is printed so the
sampling error is visible.

USAGE
-----
    python phase9_feeder_length.py --cache lcl_hourly_wide.csv --days 120 ^
           --nodes 7 9 --models cnn_lstm --seeds 3 --out sweep

    # cheap first look, no torch needed
    python phase9_feeder_length.py --days 90 --nodes 7 9 --models logreg

    # what the theory predicts, instantly, with no simulation
    python phase9_feeder_length.py --design --nodes 5 7 9 11 13
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import pandas as pd

from ntl_common import FeederConfig, load_lcl_blocks
from phase1_simulate_feeder import simulate
from phase3_train_and_benchmark import CHANNEL_GROUPS, run_once, split_indices
import phase7_sensor_placement as p7


# ---------------------------------------------------------------------------
def arms(n: int, budget: int):
    """
    The two placements being compared, both chosen by phase 7 itself.

    clustered : the set that maximises localisation L(S) -- on a long feeder
                this piles every sensor at the tail
    spread    : the set that maximises detection power C(S) -- this spaces them
                out so no zone aggregates too much load
    """
    pr = p7.PlacementProblem(n=n, mean_kw=np.ones(n))
    SL, _ = p7.exhaustive(pr, budget, "L")
    SC, _ = p7.exhaustive(pr, budget, "C")
    return dict(clustered=tuple(SL), spread=tuple(SC))


def empirical_prior(daily: pd.DataFrame, n: int, split: str) -> np.ndarray:
    """Measured share of theft days per node on the test split (see phase 8)."""
    tid = daily["Thief_ID"].to_numpy()
    _, _, te = split_indices(tid, daily["realization"].to_numpy(), split)
    t = tid[te]
    c = np.array([(t == j).sum() for j in range(1, n + 1)], dtype=float)
    return c / c.sum() if c.sum() else np.full(n, 1.0 / n)


def shorten(hourly: pd.DataFrame, days: int) -> pd.DataFrame:
    """First `days` whole days of the cached household frame."""
    if days is None or days * 24 >= len(hourly):
        return hourly
    return hourly.iloc[: days * 24]


# ---------------------------------------------------------------------------
def run(args) -> pd.DataFrame:
    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "feeder_length.csv")
    hourly = shorten(load_lcl_blocks(args.blocks, cache=args.cache), args.days)
    n_days_avail = len(hourly) // 24
    print(f"household frame: {len(hourly)} hours = {n_days_avail} days, "
          f"{hourly.shape[1]} households\n")
    groups = [g for g in CHANNEL_GROUPS if g in args.groups]
    rows = []

    for n in args.nodes:
        budget = args.budget if args.budget else max(1, round(0.4 * n))
        A = arms(n, budget)
        same = set(A["clustered"]) == set(A["spread"])
        print("=" * 78)
        print(f"n = {n} nodes,  budget = {budget} sensors")
        print(f"  clustered (max L) : {A['clustered']}")
        print(f"  spread    (max C) : {A['spread']}"
              f"{'   <-- SAME SET: theory predicts no difference' if same else ''}")
        print("=" * 78)

        for arm, place in A.items():
            if same and arm == "spread":
                print(f"  [{arm}] identical to clustered, not re-run")
                continue
            cfg = FeederConfig(n_nodes=n, households_per_node=args.hh_per_node,
                               theft_severity=args.severity,
                               sensor_lines=list(place), seed=args.seed)
            hs, ds = [], []
            for r in range(args.realizations):
                h, d, _, _ = simulate(cfg, hourly, args.solver, realization=r)
                hs.append(h); ds.append(d)
            df = pd.concat(hs, ignore_index=True)
            daily = pd.concat(ds, ignore_index=True)

            mean_kw = df[[f"Meter_{j+1}_kW" for j in range(n)]].mean().to_numpy(float)
            prior = empirical_prior(daily, n, args.split) if args.prior == "empirical" else None
            prob = p7.PlacementProblem(n=n, mean_kw=mean_kw, severity=cfg.theft_severity,
                                       sigma_ct=cfg.sigma_ct, sigma_meter=cfg.sigma_meter,
                                       sigma_sub=cfg.sigma_substation, prior=prior)
            zones = p7.zone_partition(n, place)
            big = max(len(z) for z in zones)
            print(f"\n  [{arm}] sensors {place}   zones "
                  f"{'|'.join(''.join(map(str, z)) for z in zones)}")
            print(f"        predicted  D={prob.D(place):.2f}  A={prob.A(place):.2f}  "
                  f"C={prob.C(place):.3f}  L={prob.L(place):.3f}   "
                  f"largest zone = {big} nodes")

            for seed in range(args.seeds):
                res, _, _ = run_once(df, daily, cfg, groups, seed, args.split,
                                     args.models, oof_folds=args.oof_folds)
                for model, m in res.items():
                    rows.append(dict(n=n, budget=budget, arm=arm,
                                     place=",".join(map(str, place)),
                                     D=prob.D(place), A=prob.A(place),
                                     C=prob.C(place), L=prob.L(place),
                                     largest_zone=big, model=model, seed=seed,
                                     localise_acc=m.get("localise_acc", np.nan),
                                     detect_f1=m.get("detect_f1", np.nan),
                                     macro_f1=m.get("macro_f1", np.nan),
                                     n_episodes=m.get("n_episodes", np.nan)))
                got = {r["model"]: (r["detect_f1"], r["localise_acc"]) for r in rows
                       if r["n"] == n and r["arm"] == arm and r["seed"] == seed}
                print("        seed %d  " % seed + "   ".join(
                    f"{k}: detect={v[0]:.3f} localise={v[1]:.3f}" for k, v in got.items()))
            pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\n[csv] {csv_path}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def report(res: pd.DataFrame, args) -> None:
    for model in sorted(res.model.unique()):
        d = res[res.model == model]
        print("\n" + "=" * 78)
        print(f"MODEL: {model}")
        print("=" * 78)
        for n in sorted(d.n.unique()):
            g = d[d.n == n]
            print(f"\n  n = {n} nodes")
            print(f"    {'arm':<11}{'sensors':<16}{'C pred':>8}{'detect_f1':>11}"
                  f"{'sd':>7}{'L pred':>8}{'localise':>10}{'sd':>7}")
            summ = {}
            for arm in ("clustered", "spread"):
                a = g[g.arm == arm]
                if a.empty:
                    continue
                summ[arm] = (a.detect_f1.mean(), a.detect_f1.std(),
                             a.localise_acc.mean(), a.localise_acc.std())
                print(f"    {arm:<11}{a.place.iloc[0]:<16}{a.C.iloc[0]:>8.3f}"
                      f"{a.detect_f1.mean():>11.3f}{a.detect_f1.std():>7.3f}"
                      f"{a.L.iloc[0]:>8.3f}{a.localise_acc.mean():>10.3f}"
                      f"{a.localise_acc.std():>7.3f}")
            if len(summ) == 2:
                dc_pred = (g[g.arm == "spread"].C.iloc[0] - g[g.arm == "clustered"].C.iloc[0])
                dc_meas = summ["spread"][0] - summ["clustered"][0]
                seed_sd = float(np.nanmean([summ["spread"][1], summ["clustered"][1]]))
                # deterministic models give seed_sd = 0; the honest floor is the
                # sampling error of a proportion estimated from n_episodes events
                neps_ = float(np.nanmean(g.n_episodes))
                p_hat = float(np.clip(np.mean([summ["spread"][0],
                                               summ["clustered"][0]]), 1e-6, 1 - 1e-6))
                binom = math.sqrt(p_hat * (1 - p_hat) / neps_) if neps_ > 0 else np.nan
                noise = float(np.nanmax([seed_sd, binom]))
                print(f"\n    detection gap (spread − clustered):  "
                      f"predicted {dc_pred:+.3f}   measured {dc_meas:+.3f}")
                print(f"    noise floor: seed s.d. {seed_sd:.3f}, "
                      f"sampling s.e. from ~{neps_:.0f} episodes {binom:.3f} "
                      f"-> using {noise:.3f}")
                agree = (np.sign(dc_pred) == np.sign(dc_meas) and abs(dc_meas) > 2 * noise)
                if abs(dc_pred) < 0.02:
                    verdict = ("CONSISTENT: no gap predicted, none measured"
                               if abs(dc_meas) <= 2 * noise
                               else "THEORY FAILS: no gap predicted, one measured")
                else:
                    verdict = ("CONSISTENT: gap predicted and measured, same sign"
                               if agree else "THEORY FAILS: predicted gap not reproduced")
                print(f"    -> {verdict}")
            else:
                print("\n    (the two arms selected the same placement — "
                      "theory predicts no difference, nothing to compare)")
            neps = g.n_episodes.mean()
            if neps < 20:
                print(f"    [warn] only ~{neps:.0f} episodes behind each number — "
                      f"raise --days or --realizations")


def design(args) -> None:
    print("=" * 78)
    print("WHAT THE THEORY PREDICTS  (no simulation)")
    print("=" * 78)
    print(f"  {'n':>3}{'B':>4}  {'clustered (max L)':<26}{'spread (max C)':<26}"
          f"{'C gap':>8}")
    for n in args.nodes:
        B = args.budget if args.budget else max(1, round(0.4 * n))
        pr = p7.PlacementProblem(n=n, mean_kw=np.ones(n))
        A = arms(n, B)
        gap = pr.C(A["spread"]) - pr.C(A["clustered"])
        flag = "  <- same set" if set(A["clustered"]) == set(A["spread"]) else ""
        print(f"  {n:>3}{B:>4}  {str(A['clustered']):<26}{str(A['spread']):<26}"
              f"{gap:>+8.3f}{flag}")
    print("\n  A gap near zero means the theory predicts the two strategies are")
    print("  equivalent at that feeder length -- a control arm, not a null result.")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", action="store_true")
    ap.add_argument("--replot", default=None)
    ap.add_argument("--nodes", nargs="+", type=int, default=[7, 9])
    ap.add_argument("--budget", type=int, default=None,
                    help="sensors per feeder; default round(0.4 n)")
    ap.add_argument("--days", type=int, default=120,
                    help="how many days of the cached household frame to use")
    ap.add_argument("--blocks", nargs="+",
                    default=["block1.csv", "block2.csv", "block3.csv"])
    ap.add_argument("--cache", default="lcl_hourly_wide.csv")
    ap.add_argument("--models", nargs="+", default=["logreg", "cnn_lstm"])
    ap.add_argument("--groups", nargs="+", default=CHANNEL_GROUPS)
    ap.add_argument("--prior", default="empirical", choices=["uniform", "empirical"])
    ap.add_argument("--realizations", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--oof-folds", type=int, default=3)
    ap.add_argument("--split", default="time", choices=["time", "realization"])
    ap.add_argument("--severity", type=float, default=0.15)
    ap.add_argument("--hh-per-node", type=int, default=15)
    ap.add_argument("--solver", default="auto",
                    choices=["auto", "pandapower", "bfs"])
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--out", default="sweep")
    args = ap.parse_args()

    if args.design:
        design(args); return
    if args.replot:
        report(pd.read_csv(args.replot), args); return
    report(run(args), args)


if __name__ == "__main__":
    main()
