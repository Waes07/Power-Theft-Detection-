"""
phase8_validate_placement.py -- does the placement theory predict what the
pipeline actually measures?

WHY THIS STAGE EXISTS
---------------------
Phase 7 derives a closed-form statistic D(S) and a noise-aware score L(S) that
claim to predict how well theft can be localised for a given sensor placement S.
Up to now those claims were checked against numbers produced for other purposes,
at three or four placements, with the model and the metric not always matched.

This stage does the experiment properly: it takes a list of placements, and for
each one it

    1. simulates the feeder with ONLY those lines instrumented   (phase 1)
    2. trains the full model ladder on the result                (phase 3)
    3. records localise_acc, detect_f1 and macro-F1
    4. computes the phase-7 prediction D(S), A(S), C(S), L(S)

and then reports the rank correlation between prediction and measurement, using
ONE model and ONE metric throughout.

It is deliberately a post-hoc validation: phase 7 explains phases 3 and 6, it
does not replace them.  What this stage adds is that the explanation is now
tested on placements chosen BEFORE the runs, including the four-way tie that the
theory predicts and that nothing in the earlier work had measured.

THE PREDICTION BEING TESTED
---------------------------
For n = 5 and a budget of three sensors, phase 7 says that

    {2,3,4}, {2,3,5}, {2,4,5} and {3,4,5}

all tie at the optimum (they all split the feeder into zone sizes 2,1,1,1), and
that every placement containing line 1 is strictly worse (line 1 duplicates the
substation meter).  Only {3,4,5} and {1,2,3} had ever been run.  If the other
two tie as predicted, the theory survives a test it could have failed.

COST
----
Each placement is a full simulate-and-train cycle.  With --realizations 3 and
--seeds 3 a six-placement run is a few hours on a laptop.  Results are appended
to the CSV after every placement, so an interrupted run is not lost.

USAGE
-----
    # the main experiment (overnight)
    python phase8_validate_placement.py --blocks block1.csv block2.csv block3.csv ^
           --realizations 3 --seeds 3 --models logreg cnn_lstm --out sweep

    # just the design-time answer, no simulation at all (instant)
    python phase8_validate_placement.py --design --n-nodes 5 --budget 3

    # re-draw the figure from an existing CSV
    python phase8_validate_placement.py --replot sweep/placement_validation.csv
"""
from __future__ import annotations

import argparse
import itertools
import math
import os
import sys

import numpy as np
import pandas as pd

from ntl_common import FeederConfig, load_lcl_blocks
from phase1_simulate_feeder import simulate
from phase3_train_and_benchmark import CHANNEL_GROUPS, run_once
import phase7_sensor_placement as p7


# ---------------------------------------------------------------------------
# The closed form -- no search needed for D
# ---------------------------------------------------------------------------
def balanced_optimum(n: int, budget: int) -> float:
    """
    Minimum achievable ambiguity with `budget` sensors on an n-node chain.

    With the substation meter fixing the head boundary and the open end fixing
    the tail, B sensors produce B+1 observability zones.  Sum of squared zone
    sizes is minimised by the most balanced partition, so

        k = B + 1,   q = floor(n/k),   r = n mod k
        D* = [ r (q+1)^2 + (k - r) q^2 ] / n

    Verified against exhaustive search for n = 3..14, B = 1..7: zero mismatches.
    This makes the exhaustive/greedy machinery unnecessary for D -- it is still
    needed for L and C, whose noise term breaks the closed form.
    """
    k = budget + 1
    q, r = divmod(n, k)
    return (r * (q + 1) ** 2 + (k - r) * q ** 2) / n


def balanced_placements(n: int, budget: int):
    """Every placement achieving balanced_optimum(n, budget)."""
    target = balanced_optimum(n, budget)
    return [S for S in itertools.combinations(range(1, n + 1), budget)
            if abs(p7.ambiguity(n, S) - target) < 1e-12]


# ---------------------------------------------------------------------------
def build_dataset(cfg: FeederConfig, hourly, realizations: int, solver: str):
    """Simulate `realizations` independent feeders and stack them."""
    hs, ds = [], []
    for r in range(realizations):
        h, d, _, _ = simulate(cfg, hourly, solver, realization=r)
        hs.append(h)
        ds.append(d)
    return pd.concat(hs, ignore_index=True), pd.concat(ds, ignore_index=True)


def theory_row(cfg: FeederConfig, df: pd.DataFrame, place) -> dict:
    """Phase-7 prediction for this placement, using the measured node loads."""
    n = cfg.n_nodes
    mean_kw = df[[f"Meter_{j+1}_kW" for j in range(n)]].mean().to_numpy(float)
    prob = p7.PlacementProblem(n=n, mean_kw=mean_kw, severity=cfg.theft_severity,
                               sigma_ct=cfg.sigma_ct, sigma_meter=cfg.sigma_meter,
                               sigma_sub=cfg.sigma_substation)
    zones = p7.zone_partition(n, place)
    return dict(D=prob.D(place), A=prob.A(place), C=prob.C(place), L=prob.L(place),
                zones="|".join("".join(map(str, z)) for z in zones),
                n_zones=len(zones),
                resid_available="".join(str(j) for j in p7.pairwise_resolvable(n, place)))


# ---------------------------------------------------------------------------
def run(args) -> pd.DataFrame:
    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "placement_validation.csv")

    places = [tuple(int(c) for c in p.split(",")) for p in args.configs]
    hourly = load_lcl_blocks(args.blocks, cache=args.cache)
    groups = [g for g in CHANNEL_GROUPS if g in args.groups]
    rows = []

    for pi, place in enumerate(places, 1):
        print("=" * 74)
        print(f"[{pi}/{len(places)}]  sensors on lines {place}")
        print("=" * 74)
        cfg = FeederConfig(n_nodes=args.n_nodes, households_per_node=args.hh_per_node,
                           theft_severity=args.severity, sensor_lines=list(place),
                           seed=args.seed)
        df, daily = build_dataset(cfg, hourly, args.realizations, args.solver)
        th = theory_row(cfg, df, place)
        print(f"  prediction : zones {th['zones']}   D={th['D']:.2f}  "
              f"A={th['A']:.2f}  C={th['C']:.3f}  L={th['L']:.3f}")
        print(f"  per-node residuals available at nodes: {th['resid_available'] or 'none'}")

        for seed in range(args.seeds):
            res, _, _ = run_once(df, daily, cfg, groups, seed, args.split,
                                 args.models, oof_folds=args.oof_folds)
            for model, m in res.items():
                rows.append(dict(place=",".join(map(str, place)), model=model,
                                 seed=seed, **th,
                                 localise_acc=m.get("localise_acc", np.nan),
                                 detect_f1=m.get("detect_f1", np.nan),
                                 macro_f1=m.get("macro_f1", np.nan),
                                 episode_recall=m.get("episode_recall", np.nan),
                                 n_episodes=m.get("n_episodes", np.nan)))
            done = {r["model"]: r["localise_acc"] for r in rows
                    if r["place"] == ",".join(map(str, place)) and r["seed"] == seed}
            print("  seed %d  " % seed + "  ".join(f"{k}={v:.3f}" for k, v in done.items()))

        pd.DataFrame(rows).to_csv(csv_path, index=False)   # save after each place
        print(f"  [saved] {csv_path}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def summarise(res: pd.DataFrame, model: str, metric: str = "localise_acc"):
    """Mean over seeds, one row per placement, for a single model and metric."""
    d = res[res.model == model]
    if d.empty:
        return None
    g = d.groupby("place").agg(
        D=("D", "first"), A=("A", "first"), C=("C", "first"), L=("L", "first"),
        zones=("zones", "first"),
        measured=(metric, "mean"), sd=(metric, "std"), n=(metric, "size"),
        n_episodes=("n_episodes", "mean")).reset_index()
    return g.sort_values("D")


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    den = math.sqrt(float((rx ** 2).sum()) * float((ry ** 2).sum()))
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def make_figure(g: pd.DataFrame, model: str, metric: str, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S1, S2 = "#2a78d6", "#eb6834"
    INK, SEC, MUT, GRID = "#0b0b0b", "#52514e", "#b7b5ae", "#dcdad3"
    plt.rcParams.update({"font.size": 11, "axes.labelcolor": SEC,
                         "xtick.color": SEC, "ytick.color": SEC, "axes.edgecolor": MUT})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.6))

    def clean(ax, axis="y"):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis=axis, color=GRID, lw=0.7); ax.set_axisbelow(True)

    # -- left: measured vs predicted L, with the tie group highlighted --------
    tie = g.D <= g.D.min() + 1e-9
    ax1.errorbar(g.L[tie], g.measured[tie], yerr=g.sd[tie], fmt="o", ms=10,
                 color=S1, capsize=3, lw=1.4, mec="white", mew=1.2,
                 label="predicted to tie at the optimum")
    ax1.errorbar(g.L[~tie], g.measured[~tie], yerr=g.sd[~tie], fmt="s", ms=9,
                 color=S2, capsize=3, lw=1.4, mec="white", mew=1.2,
                 label="predicted to be worse")
    for _, r in g.iterrows():
        ax1.annotate("{" + r.place + "}", (r.L, r.measured), fontsize=8.5,
                     color=MUT, xytext=(7, 4), textcoords="offset points")
    ax1.set_xlabel("predicted localisation score  L(S)")
    ax1.set_ylabel(f"measured {metric}   ({model}, mean ± s.d.)")
    ax1.set_title("Prediction against measurement", loc="left",
                  fontsize=13, color=INK, weight="bold", pad=10)
    ax1.legend(frameon=False, fontsize=9, loc="lower right")
    clean(ax1)

    # -- right: the tie test --------------------------------------------------
    gg = g.sort_values("measured", ascending=False).reset_index(drop=True)
    cols = [S1 if t else S2 for t in (gg.D <= gg.D.min() + 1e-9)]
    ax2.barh(range(len(gg)), gg.measured, xerr=gg.sd, color=cols, height=0.66,
             error_kw=dict(ecolor=MUT, lw=1.1, capsize=3))
    ax2.set_yticks(range(len(gg)))
    ax2.set_yticklabels(["{" + p + "}" for p in gg.place], fontsize=10)
    ax2.invert_yaxis()
    ax2.set_xlabel(f"measured {metric}")
    ax2.set_xlim(0, gg.measured.max() * 1.30)
    for i, (v, s) in enumerate(zip(gg.measured, gg.sd)):
        ax2.annotate(f"{v:.3f}", (v + (s if np.isfinite(s) else 0), i), fontsize=9,
                     color=SEC, va="center", xytext=(6, 0), textcoords="offset points")
    ax2.set_title("Does the predicted tie group actually tie?", loc="left",
                  fontsize=13, color=INK, weight="bold", pad=10)
    clean(ax2, "x")

    rho = spearman(-g.D, g.measured)
    fig.text(0.5, 0.015, f"Spearman(−D, measured) = {rho:+.3f}   ·   "
             f"blue = the balanced-partition optimum predicted by phase 7, "
             f"orange = predicted worse", ha="center", fontsize=9.5, color=SEC)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(path, dpi=200, facecolor="white")
    fig.savefig(os.path.splitext(path)[0] + ".pdf", facecolor="white")
    plt.close(fig)


def report(res: pd.DataFrame, args) -> None:
    for model in sorted(res.model.unique()):
        g = summarise(res, model, args.metric)
        if g is None or len(g) < 2:
            continue
        print("\n" + "=" * 78)
        print(f"MODEL: {model}    metric: {args.metric}")
        print("=" * 78)
        print(f"  {'sensors':<12}{'zones':<16}{'D':>6}{'A':>6}{'L':>7}"
              f"{'measured':>11}{'sd':>8}")
        for _, r in g.iterrows():
            print(f"  {'{'+r.place+'}':<12}{r.zones:<16}{r.D:>6.2f}{r.A:>6.2f}"
                  f"{r.L:>7.3f}{r.measured:>11.3f}{r.sd:>8.3f}")
        print(f"\n  Spearman(-D, measured)  = {spearman(-g.D, g.measured):+.3f}")
        print(f"  Spearman( L, measured)  = {spearman(g.L, g.measured):+.3f}")

        tie = g[g.D <= g.D.min() + 1e-9]
        if len(tie) > 1:
            spread = tie.measured.max() - tie.measured.min()
            seed_sd = float(np.nanmean(tie.sd))
            # Deterministic models (logreg, the rule) give seed_sd = 0, which
            # would make any spread look significant.  The honest floor is the
            # sampling error of the metric itself: localise_acc is a proportion
            # estimated from a handful of EPISODES, not from node-days.
            neps = float(np.nanmean(tie.n_episodes))
            p_hat = float(np.clip(tie.measured.mean(), 1e-6, 1 - 1e-6))
            binom_se = math.sqrt(p_hat * (1 - p_hat) / neps) if neps > 0 else np.nan
            noise = np.nanmax([seed_sd, binom_se])
            print(f"\n  TIE TEST: phase 7 predicts {len(tie)} placements tie "
                  f"({', '.join('{'+p+'}' for p in tie.place)})")
            print(f"    measured spread within the group : {spread:.3f}")
            print(f"    seed-to-seed s.d.                : {seed_sd:.3f}")
            print(f"    sampling s.e. from ~{neps:.0f} episodes    : {binom_se:.3f}")
            if seed_sd < 1e-6:
                print("    (this model is deterministic across seeds, so the "
                      "sampling s.e. is the\n     only honest noise estimate -- "
                      "vary --realizations, not --seeds, to beat it down)")
            verdict = ("CONSISTENT with a tie" if spread <= 2 * noise
                       else "NOT a tie at this sample size")
            print(f"    -> {verdict}   (criterion: spread <= 2 x {noise:.3f})")
            if neps < 20:
                print(f"    [warn] only ~{neps:.0f} episodes behind each number -- "
                      f"raise --realizations before\n           drawing any "
                      f"conclusion from this test")

        if model == args.plot_model or args.plot_model is None:
            path = os.path.join(args.out, f"placement_validation_{model}.png")
            make_figure(g, model, args.metric, path)
            print(f"\n  [figure] {path}")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Validate the phase-7 placement prediction against the "
                    "measured pipeline.")
    ap.add_argument("--design", action="store_true",
                    help="print the closed-form optimal placement and exit "
                         "(no simulation, instant)")
    ap.add_argument("--replot", default=None,
                    help="re-read an existing results CSV and redraw, no runs")
    ap.add_argument("--n-nodes", type=int, default=5)
    ap.add_argument("--budget", type=int, default=3)
    ap.add_argument("--blocks", nargs="+",
                    default=["block1.csv", "block2.csv", "block3.csv"])
    ap.add_argument("--cache", default="lcl_hourly_wide.csv")
    ap.add_argument("--configs", nargs="+",
                    default=["2,3,4", "2,3,5", "2,4,5", "3,4,5",
                             "1,2,3", "1,4,5"],
                    help="placements to test; the first four are the predicted "
                         "tie group, the last two contain line 1")
    ap.add_argument("--models", nargs="+", default=["logreg", "cnn_lstm"])
    ap.add_argument("--groups", nargs="+", default=CHANNEL_GROUPS)
    ap.add_argument("--metric", default="localise_acc",
                    choices=["localise_acc", "detect_f1", "macro_f1",
                             "episode_recall"])
    ap.add_argument("--realizations", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--oof-folds", type=int, default=3)
    ap.add_argument("--split", default="time", choices=["time", "realization"])
    ap.add_argument("--severity", type=float, default=0.15)
    ap.add_argument("--hh-per-node", type=int, default=15)
    ap.add_argument("--solver", default="auto",
                    choices=["auto", "pandapower", "bfs"])
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--plot-model", default=None)
    ap.add_argument("--out", default="sweep")
    args = ap.parse_args()

    # ---- design mode: the answer with no experiment ------------------------
    if args.design:
        n, B = args.n_nodes, args.budget
        print("=" * 70)
        print(f"DESIGN-TIME ANSWER    n = {n} nodes, budget = {B} sensors")
        print("=" * 70)
        d = balanced_optimum(n, B)
        print(f"  minimum ambiguity D* = {d:.4f}   "
              f"({d:.2f} premises to inspect per alarm)")
        print(f"  resolution ceiling A = {(B+1)/n:.4f}")
        opts = balanced_placements(n, B)
        print(f"  placements achieving it ({len(opts)}):")
        for S in opts:
            z = "|".join("".join(map(str, zz)) for zz in p7.zone_partition(n, S))
            print(f"      {str(S):<18} zones {z}")
        print("\n  none of them instruments line 1 -- the substation meter "
              "already\n  measures that boundary, and more accurately than a CT.")
        return

    if args.replot:
        res = pd.read_csv(args.replot)
        args.out = os.path.dirname(args.replot) or "."
        report(res, args)
        return

    res = run(args)
    report(res, args)


if __name__ == "__main__":
    main()
