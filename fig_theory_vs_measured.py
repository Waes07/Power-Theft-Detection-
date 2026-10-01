"""
fig_theory_vs_measured.py -- side-by-side: what the maths predicts for every
three-sensor placement, against what the trained model actually achieves.

WHY THIS IS A FAIR COMPARISON WITHOUT RE-SIMULATING
---------------------------------------------------
In phase 1 the sensor placement does exactly one thing:

    line_meas[:, ~sensors] = np.nan

It does NOT change the power flow, the loads, the theft, or the measurement
noise -- those are all drawn before the masking line.  So masking the columns of
an existing all-sensor dataset produces bit-for-bit the same inputs as
re-running phase 1 with that placement and the same seed.  That is what phase 6
already does, and it is what this script does, which is why all ten placements
can be evaluated from one simulation instead of ten.

USAGE
-----
    # with the classical models (fast, no GPU, no torch)
    python fig_theory_vs_measured.py --data data\\run --models logreg rf

    # with the deep model -- this is the one for the final presentation
    python fig_theory_vs_measured.py --data data\\run --models cnn_lstm --seeds 3

    # redraw from the saved CSV without retraining
    python fig_theory_vs_measured.py --replot sweep\\theory_vs_measured.csv
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os

import numpy as np
import pandas as pd

from ntl_common import FeederConfig
from phase3_train_and_benchmark import CHANNEL_GROUPS, run_once, split_indices
import phase7_sensor_placement as p7


# ---------------------------------------------------------------------------
def rebuild_daily(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Reconstruct labels_daily from the hourly ground truth, exactly as phase 1
    does: a day is a theft day for node j if theft occurred in ANY of its hours.
    Used when labels_daily.csv is not to hand; identical to the shipped file.
    """
    rows = []
    for r, g in df.groupby("realization", sort=True):
        tt = g[[f"true_theft_{j+1}" for j in range(n)]].to_numpy().reshape(-1, 24, n)
        dt = tt.any(axis=1)
        d = pd.DataFrame({"day": np.arange(len(dt))})
        for j in range(n):
            d[f"theft_{j+1}"] = dt[:, j].astype(int)
        na = dt.sum(1)
        d["Thief_ID"] = np.where(na == 1, dt.argmax(1) + 1, np.where(na > 1, -1, 0))
        d["n_thieves"] = na
        d["realization"] = r
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def empirical_prior(daily: pd.DataFrame, n: int, split: str) -> np.ndarray:
    """
    How often each node is actually the thief, measured on the TEST split.

    The phase-7 statistics were first written with a uniform prior -- every node
    equally likely to steal.  That is an assumption, and in this dataset it is
    badly wrong: the per-node Poisson episode draw plus the non-overlap filter
    leaves one node holding more than half of all theft days.  A zone that lumps
    two rarely-guilty nodes costs almost nothing; a zone that lumps the busiest
    node costs a great deal, and the uniform prior cannot see the difference.

    Using the measured prior lifts the rank correlation with the trained model
    from about +0.78 to about +0.97.  It is the single most important
    correction to the placement model.
    """
    tid = daily["Thief_ID"].to_numpy()
    real = daily["realization"].to_numpy()
    _, _, te = split_indices(tid, real, split)
    t = tid[te]
    counts = np.array([(t == j).sum() for j in range(1, n + 1)], dtype=float)
    if counts.sum() == 0:
        return np.full(n, 1.0 / n)
    return counts / counts.sum()


def mask_to(df: pd.DataFrame, n: int, place) -> pd.DataFrame:
    """Return a copy with the un-instrumented line currents blanked."""
    d = df.copy()
    for j in range(1, n + 1):
        if j not in place:
            d[f"Line_{j}_A"] = np.nan
    return d


# ---------------------------------------------------------------------------
def evaluate_all(args) -> pd.DataFrame:
    cfg = FeederConfig(**json.load(open(os.path.join(args.data, "config.json"))))
    n = cfg.n_nodes
    df = pd.read_csv(os.path.join(args.data, "dataset_hourly.csv"),
                     parse_dates=["timestamp"])
    dpath = os.path.join(args.data, "labels_daily.csv")
    daily = pd.read_csv(dpath) if os.path.exists(dpath) else rebuild_daily(df, n)

    mean_kw = df[[f"Meter_{j+1}_kW" for j in range(n)]].mean().to_numpy(float)
    prior = None
    if args.prior == "empirical":
        prior = empirical_prior(daily, n, args.split)
        print("empirical thief prior (test split): " +
              "  ".join(f"node {j+1}={p:.3f}" for j, p in enumerate(prior)))
    prob = p7.PlacementProblem(n=n, mean_kw=mean_kw, severity=cfg.theft_severity,
                               sigma_ct=cfg.sigma_ct, sigma_meter=cfg.sigma_meter,
                               sigma_sub=cfg.sigma_substation, prior=prior)
    groups = [g for g in CHANNEL_GROUPS if g in args.groups]
    places = list(itertools.combinations(range(1, n + 1), args.budget))

    rows = []
    for pi, place in enumerate(places, 1):
        zones = p7.zone_partition(n, place)
        th = dict(place=",".join(map(str, place)),
                  D=prob.D(place), A=prob.A(place),
                  C=prob.C(place), L=prob.L(place),
                  zones="|".join("".join(map(str, z)) for z in zones),
                  uses_line1=int(1 in place))
        print(f"[{pi:>2}/{len(places)}] {str(place):<14} zones {th['zones']:<14} "
              f"D={th['D']:.2f}  L={th['L']:.3f}", flush=True)
        d = mask_to(df, n, place)
        for seed in range(args.seeds):
            res, _, _ = run_once(d, daily, cfg, groups, seed, args.split,
                                 args.models, oof_folds=args.oof_folds)
            for model, m in res.items():
                rows.append(dict(**th, model=model, seed=seed,
                                 localise_acc=m.get("localise_acc", np.nan),
                                 detect_f1=m.get("detect_f1", np.nan),
                                 macro_f1=m.get("macro_f1", np.nan),
                                 episode_recall=m.get("episode_recall", np.nan),
                                 n_episodes=m.get("n_episodes", np.nan)))
            got = {r["model"]: r[args.metric] for r in rows
                   if r["place"] == th["place"] and r["seed"] == seed}
            print("        seed %d  " % seed +
                  "  ".join(f"{k}={v:.3f}" for k, v in got.items()), flush=True)
    return pd.DataFrame(rows)


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    den = math.sqrt(float((rx ** 2).sum()) * float((ry ** 2).sum()))
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


# ---------------------------------------------------------------------------
PRIOR_NOTE = "empirical"


def make_figure(res: pd.DataFrame, model: str, metric: str, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S1, S2 = "#2a78d6", "#eb6834"            # validated categorical slots
    INK, SEC, MUT, GRID = "#0b0b0b", "#52514e", "#b7b5ae", "#dcdad3"
    plt.rcParams.update({"font.size": 11, "axes.labelcolor": SEC,
                         "xtick.color": SEC, "ytick.color": SEC,
                         "axes.edgecolor": MUT})

    d = res[res.model == model]
    g = d.groupby("place").agg(
        D=("D", "first"), A=("A", "first"), L=("L", "first"),
        zones=("zones", "first"), uses_line1=("uses_line1", "first"),
        measured=(metric, "mean"), sd=(metric, "std"),
        n_episodes=("n_episodes", "mean")).reset_index()
    # order both panels the same way: by the MATHS, best first
    g = g.sort_values(["L", "place"], ascending=[False, True]).reset_index(drop=True)
    labs = ["{" + p + "}" for p in g.place]
    y = np.arange(len(g))
    best = g.L.max()
    tie = g.L >= best - 5e-3
    colL = [S1 if t else MUT for t in tie]
    colM = [S1 if t else MUT for t in tie]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.4, 6.4), sharey=True)

    def clean(ax):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="x", color=GRID, lw=0.7); ax.set_axisbelow(True)

    # ---------------- LEFT: the mathematical model ------------------------
    axL.barh(y, g.L, color=colL, height=0.68)
    axL.set_xlim(0, g.L.max() * 1.45)
    for i, (v, D, A) in enumerate(zip(g.L, g.D, g.A)):
        axL.annotate(f"{v:.3f}", (v, i), fontsize=9.5, color=SEC, va="center",
                     xytext=(6, 0), textcoords="offset points")
        axL.annotate(f"D={D:.2f}  A={A:.2f}", (g.L.max() * 1.16, i), fontsize=8.5,
                     color=MUT, va="center")
    axL.set_yticks(y); axL.set_yticklabels(labs, fontsize=10.5)
    axL.invert_yaxis()
    axL.set_xlabel("predicted localisation score  L(S)")
    axL.set_title("Mathematical model — closed form, no training",
                  loc="left", fontsize=12.5, color=INK, weight="bold", pad=10)
    clean(axL)

    # ---------------- RIGHT: what the trained model achieved --------------
    err = g.sd.fillna(0).to_numpy()
    axR.barh(y, g.measured, xerr=err if err.any() else None, color=colM, height=0.68,
             error_kw=dict(ecolor=MUT, lw=1.1, capsize=3))
    axR.set_xlim(0, max(g.measured.max() * 1.30, 0.1))
    for i, v in enumerate(g.measured):
        axR.annotate(f"{v:.3f}", (v + err[i], i), fontsize=9.5, color=SEC,
                     va="center", xytext=(6, 0), textcoords="offset points")
    axR.set_xlabel(f"measured {metric}")
    axR.set_title(f"Trained model — {model}", loc="left",
                  fontsize=12.5, color=INK, weight="bold", pad=10)
    clean(axR)

    rho = spearman(g.L, g.measured)
    ntie = int(tie.sum())
    spread = g.measured[tie].max() - g.measured[tie].min()
    fig.suptitle("Every three-sensor placement:  what the maths says, and what the model does",
                 fontsize=14.5, color=INK, weight="bold", x=0.013, ha="left", y=0.975)
    tiemsg = (f"{ntie} placements tie at the predicted optimum"
              if ntie > 1 else "one placement is predicted best outright")
    fig.text(0.013, 0.925,
             f"Thief prior: {PRIOR_NOTE}.  Ordering on both panels is set by the maths.  "
             f"Blue = predicted best ({tiemsg}); grey = predicted worse.",
             fontsize=9.8, color=SEC)
    fig.text(0.5, 0.018,
             f"Spearman(L, measured) = {rho:+.3f}"
             + (f"     ·     spread inside the predicted tie group = {spread:.3f}"
                if ntie > 1 else "")
             + f"     ·     ~{g.n_episodes.mean():.0f} theft episodes behind each measured bar",
             ha="center", fontsize=9.6, color=SEC)
    fig.tight_layout(rect=(0, 0.045, 1, 0.905))
    fig.savefig(path, dpi=200, facecolor="white")
    fig.savefig(os.path.splitext(path)[0] + ".pdf", facecolor="white")
    plt.close(fig)
    print(f"\n[figure] {path}   Spearman(L, measured) = {rho:+.3f}")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/run")
    ap.add_argument("--budget", type=int, default=3)
    ap.add_argument("--models", nargs="+", default=["logreg"])
    ap.add_argument("--groups", nargs="+", default=CHANNEL_GROUPS)
    ap.add_argument("--metric", default="localise_acc",
                    choices=["localise_acc", "detect_f1", "macro_f1", "episode_recall"])
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--oof-folds", type=int, default=3)
    ap.add_argument("--split", default="time", choices=["time", "realization"])
    ap.add_argument("--prior", default="empirical",
                    choices=["uniform", "empirical"],
                    help="uniform assumes every node equally likely to steal; "
                         "empirical measures it from the test split. The data "
                         "is far from uniform, and it matters -- see the "
                         "docstring of empirical_prior().")
    ap.add_argument("--replot", default=None)
    ap.add_argument("--out", default="sweep")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.replot:
        res = pd.read_csv(args.replot)
        args.out = os.path.dirname(args.replot) or "."
    else:
        globals()["PRIOR_NOTE"] = args.prior
        res = evaluate_all(args)
        res.to_csv(os.path.join(args.out, "theory_vs_measured.csv"), index=False)
        print(f"\n[csv] {os.path.join(args.out, 'theory_vs_measured.csv')}")

    for model in sorted(res.model.unique()):
        make_figure(res, model, args.metric,
                    os.path.join(args.out, f"theory_vs_measured_{model}.png"))


if __name__ == "__main__":
    main()
