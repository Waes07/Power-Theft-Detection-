"""
phase6_inspection_budget.py -- what the detector is worth to a utility.

Macro-F1 is the right metric for comparing models; it is the wrong metric for
deciding whether to deploy one.  A utility does not choose a decision threshold
-- it has an inspection crew, and the real question is:

    "We can inspect k service points this month.  How much theft do we recover,
     and how many of those visits are wasted?"

This file answers that.  Every node-day in the test split is ranked by the
model's score, and the ranking is walked from the top, giving:

    hit rate   of the visits made, how many found theft          (precision@k)
    recovery   of all theft present, how much was caught         (recall@k)
    lift       how much better than inspecting at random

Confidence intervals come from bootstrapping over DAYS, not node-days, so the
correlation between nodes on the same day is preserved.  That also fixes the
small-sample problem in the multi-thief metrics: with ~10 multi-thief test days
a point estimate is not defensible, an interval is.

Nothing here trains a new kind of model.  It re-reads the same scores the other
phases already produce, so it adds an operational view at no experimental cost.

    python phase6_inspection_budget.py --data data/run
    python phase6_inspection_budget.py --data data/run --model cnn_lstm \
        --configs 1,2,3,4,5 3,4,5 4,5 3,4
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from phase2_audit_dataset import load_run
from phase3_train_and_benchmark import (CHANNEL_GROUPS, apply_scaler, build_windows,
                                        fit_scaler, make_gbm, split_indices,
                                        summary_features)

try:
    from phase5_multi_thief import HAS_TORCH, predict_ml, train_multilabel
except Exception:                                            # pragma: no cover
    HAS_TORCH = False


# ---------------------------------------------------------------------------
def node_day_scores(df, daily, cfg, groups, place, model, seed, split_mode="time"):
    """
    Score every (day, service point) pair in the test split.

    Returns S (days x nodes) suspicion scores, Y the binary truth, and the day
    and realization index of each row -- the last two let the caller rebuild
    theft episodes, and the bootstrap resample whole days.
    """
    n = cfg.n_nodes
    d = df.copy()
    for j in range(1, n + 1):
        if j not in place:
            d[f"Line_{j}_A"] = np.nan

    X, y1, real, _ = build_windows(d, daily, cfg, groups)
    Y = daily[[f"theft_{j+1}" for j in range(n)]].to_numpy().astype(int)
    tr, va, te = split_indices(y1, real, split_mode)
    Xs = apply_scaler(X, fit_scaler(X, tr))

    if model in ("cnn", "lstm", "cnn_lstm"):
        if not HAS_TORCH:
            raise SystemExit("PyTorch is required for --model " + model)
        mdl = train_multilabel(Xs[tr], Y[tr], Xs[va], Y[va], Xs.shape[1], n,
                               model, seed)
        S = predict_ml(mdl, Xs[te])
    else:
        Ftr, Fte = summary_features(Xs[tr]), summary_features(Xs[te])
        S = np.zeros((len(te), n))
        for j in range(n):
            if len(np.unique(Y[tr, j])) < 2:
                S[:, j] = float(Y[tr, j].max())
                continue
            if model == "logreg":
                m = LogisticRegression(max_iter=4000, class_weight="balanced",
                                       random_state=seed)
            elif model == "rf":
                m = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                           random_state=seed, n_jobs=-1)
            elif model == "gbm":
                m = make_gbm(seed, 2)
            else:
                raise SystemExit(f"unknown model {model!r}")
            S[:, j] = m.fit(Ftr, Y[tr, j]).predict_proba(Fte)[:, 1]
    return S, Y[te], daily["day"].to_numpy()[te], daily["realization"].to_numpy()[te]


# ---------------------------------------------------------------------------
def recovery_curve(S, Y, budgets):
    """Walk the ranking from the top; report hit rate and recovery at each budget."""
    s, y = S.ravel(), Y.ravel()
    order = np.argsort(-s)
    hits = np.cumsum(y[order])
    N, total = len(s), y.sum()
    out = []
    for b in budgets:
        k = max(1, int(round(N * b)))
        caught = hits[k - 1]
        out.append(dict(budget=b, inspections=k, caught=int(caught),
                        hit_rate=caught / k,
                        recovery=caught / total if total else np.nan,
                        lift=(caught / k) / (total / N) if total else np.nan))
    return pd.DataFrame(out)


def bootstrap_curve(S, Y, budgets, n_boot=400, seed=0):
    """Resample DAYS with replacement -- keeps same-day correlation intact."""
    rng = np.random.default_rng(seed)
    rec = np.zeros((n_boot, len(budgets)))
    hit = np.zeros((n_boot, len(budgets)))
    nd = len(S)
    for b in range(n_boot):
        idx = rng.integers(0, nd, nd)
        c = recovery_curve(S[idx], Y[idx], budgets)
        rec[b] = c.recovery.to_numpy()
        hit[b] = c.hit_rate.to_numpy()
    q = lambda a, p: np.nanpercentile(a, p, axis=0)
    return dict(rec_lo=q(rec, 2.5), rec_hi=q(rec, 97.5),
                hit_lo=q(hit, 2.5), hit_hi=q(hit, 97.5))


def budget_for(S, Y, target):
    """Smallest fraction of node-days that must be inspected to recover `target`."""
    s, y = S.ravel(), Y.ravel()
    hits = np.cumsum(y[np.argsort(-s)])
    need = target * y.sum()
    k = int(np.searchsorted(hits, need) + 1)
    return k / len(s)


# ---------------------------------------------------------------------------
def episode_ids(Y, dayno, realz):
    """
    Label each (row, node) with the id of the theft episode it belongs to.

    An inspection that finds a thief ends that episode, so value accrues per
    episode caught, not per theft-day observed.  Counting days would credit the
    same discovery many times over.
    """
    days, n = Y.shape
    ids = np.full((days, n), -1, dtype=int)
    order = np.lexsort((dayno, realz))
    nxt = 0
    for j in range(n):
        prev_day, prev_r, cur = None, None, -1
        for i in order:
            if Y[i, j] == 0:
                cur = -1
                prev_day, prev_r = dayno[i], realz[i]
                continue
            contiguous = (cur >= 0 and realz[i] == prev_r and dayno[i] == prev_day + 1)
            if not contiguous:
                cur = nxt
                nxt += 1
            ids[i, j] = cur
            prev_day, prev_r = dayno[i], realz[i]
    return ids, nxt


def economics(S, Y, ids, n_ep, cfg, df, inspect_cost, tariff, field_days):
    """
    Net benefit against inspection budget, counted per episode caught.

    The value of a catch is NOT one day of stolen energy -- an inspection ends
    the theft.  The utility gains the energy that would otherwise have been lost
    over the remaining life of the connection, plus whatever is back-billed.
    `field_days` is that horizon, and it is the single assumption that drives
    the whole result, so it is printed with the answer and swept for break-even.

    Simulated episodes are deliberately short (~9 days) so that training sees
    many of them; real undetected theft runs for months, which is what
    `field_days` represents.
    """
    n = cfg.n_nodes
    node_mean = df[[f"Meter_{j+1}_kW" for j in range(n)]].mean().mean()
    kwh_day = cfg.theft_severity * node_mean * 24.0
    value_per_catch = kwh_day * tariff * field_days

    s, idsf = S.ravel(), ids.ravel()
    order = np.argsort(-s)
    seen, caught = set(), np.zeros(len(order), dtype=int)
    c = 0
    for pos, idx in enumerate(order):
        e = idsf[idx]
        if e >= 0 and e not in seen:
            seen.add(e)
            c += 1
        caught[pos] = c
    k = np.arange(1, len(order) + 1)
    net = caught * value_per_catch - k * inspect_cost
    best = int(np.argmax(net))
    breakeven = inspect_cost * (best + 1) / max(caught[best], 1) / (kwh_day * tariff)
    return dict(kwh_day=kwh_day, value_per_catch=value_per_catch,
                n_episodes=n_ep, best_budget=(best + 1) / len(order),
                best_inspections=best + 1, best_caught=int(caught[best]),
                best_net=float(net[best]),
                caught_frac=caught[best] / max(n_ep, 1),
                breakeven_days=breakeven)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/run")
    ap.add_argument("--model", default="logreg",
                    choices=["logreg", "rf", "gbm", "cnn", "lstm", "cnn_lstm"])
    ap.add_argument("--configs", nargs="+", default=["1,2,3,4,5", "3,4,5", "4,5"],
                    help="sensing configurations to compare, e.g. 3,4,5")
    ap.add_argument("--groups", nargs="+", default=CHANNEL_GROUPS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=400)
    ap.add_argument("--inspect-cost", type=float, default=60.0,
                    help="cost of sending a crew to one service point")
    ap.add_argument("--tariff", type=float, default=0.15,
                    help="value of one recovered kWh")
    ap.add_argument("--field-days", type=float, default=180.0,
                    help="how long an undetected theft would have run in the field; "
                         "a catch is credited with stopping it")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df, daily, cfg = load_run(args.data)
    out_dir = args.out or os.path.join(args.data, "results")
    os.makedirs(out_dir, exist_ok=True)
    budgets = np.concatenate([np.arange(0.002, 0.05, 0.002),
                              np.arange(0.05, 0.51, 0.01)])
    show = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30]

    print("=" * 94)
    print(f" PHASE 6 -- INSPECTION BUDGET   {args.data}   model={args.model}")
    print("=" * 94)

    rows, curves = [], {}
    for cstr in args.configs:
        place = {int(x) for x in cstr.split(",")}
        S, Y, dayno, realz = node_day_scores(df, daily, cfg, args.groups,
                                             place, args.model, args.seed)
        c = recovery_curve(S, Y, budgets)
        ci = bootstrap_curve(S, Y, budgets, args.bootstrap, args.seed)
        curves[cstr] = (c, ci)
        base = Y.sum() / Y.size

        print(f"\n  --- current sensors on lines {cstr}   "
              f"({Y.shape[0]} test days x {Y.shape[1]} points = {Y.size} node-days, "
              f"{Y.sum()} with theft) ---\n")
        print(f"  {'inspect top':<15}{'visits':>9}{'found':>8}{'hit rate':>11}"
              f"{'recovered':>25}{'lift':>8}")
        print("  " + "-" * 78)
        for b in show:
            i = int(np.argmin(np.abs(budgets - b)))
            r = c.iloc[i]
            print(f"  {f'{b:.0%} of node-days':<15}{int(r.inspections):>9}"
                  f"{int(r.caught):>8}{r.hit_rate:>11.1%}"
                  f"{r.recovery:>16.1%}  [{ci['rec_lo'][i]:.0%}-{ci['rec_hi'][i]:.0%}]"
                  f"{r.lift:>8.1f}x")
        for tgt in (0.5, 0.8, 0.9):
            print(f"  to recover {tgt:.0%} of all theft, inspect "
                  f"{budget_for(S, Y, tgt):.1%} of node-days")
        ids, n_ep = episode_ids(Y, dayno, realz)
        e = economics(S, Y, ids, n_ep, cfg, df, args.inspect_cost, args.tariff,
                      args.field_days)
        print(f"  random inspection finds theft {base:.1%} of the time")
        print(f"\n  economics  (a catch ends the theft; assumed field duration "
              f"{args.field_days:.0f} days)")
        print(f"    stolen energy {e['kwh_day']:.0f} kWh/day -> a catch is worth "
              f"{e['value_per_catch']:,.0f} at {args.tariff}/kWh")
        print(f"    net benefit peaks at {e['best_budget']:.1%} of node-days: "
              f"{e['best_inspections']} visits, {e['best_caught']}/{n_ep} episodes "
              f"caught ({e['caught_frac']:.0%}), net {e['best_net']:,.0f}")
        print(f"    break-even needs theft to run {e['breakeven_days']:.0f}+ days "
              f"in the field at that budget")

        for b in show:
            i = int(np.argmin(np.abs(budgets - b)))
            r = c.iloc[i]
            rows.append(dict(config=cstr, budget=b, inspections=int(r.inspections),
                             caught=int(r.caught), hit_rate=r.hit_rate,
                             recovery=r.recovery, recovery_lo=ci["rec_lo"][i],
                             recovery_hi=ci["rec_hi"][i], lift=r.lift))
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "inspection_budget.csv"),
                              index=False)

    # ---- figure ------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        INK, MUT, GRID, SEC = "#0b0b0b", "#7A8899", "#E4E4E0", "#52514e"
        COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
        fig, ax = plt.subplots(1, 2, figsize=(12.2, 4.9))
        for i, (cstr, (c, ci)) in enumerate(curves.items()):
            col = COL[i % len(COL)]
            lbl = f"sensors {cstr}"
            ax[0].fill_between(budgets * 100, ci["rec_lo"] * 100, ci["rec_hi"] * 100,
                               color=col, alpha=.15, lw=0)
            ax[0].plot(budgets * 100, c.recovery * 100, color=col, lw=2.2,
                       label=lbl, zorder=3)
            ax[1].plot(budgets * 100, c.hit_rate * 100, color=col, lw=2.2,
                       label=lbl, zorder=3)
        ax[0].plot([0, 50], [0, 50], color=MUT, ls=(0, (4, 3)), lw=1.3,
                   label="random inspection")
        ax[0].set_ylabel("theft recovered  (%)", fontsize=10.5, color=SEC)
        ax[0].set_title("How much theft you recover", fontsize=12.5, color=INK,
                        fontweight="bold", loc="left", pad=11)
        ax[1].set_ylabel("inspections that find theft  (%)", fontsize=10.5, color=SEC)
        ax[1].set_title("How many visits are worth making", fontsize=12.5, color=INK,
                        fontweight="bold", loc="left", pad=11)
        for a in ax:
            a.set_xlabel("share of node-days inspected  (%)", fontsize=10.5, color=SEC)
            a.grid(axis="y", color=GRID, lw=1)
            a.set_axisbelow(True)
            a.set_xlim(0, 50)
            a.set_ylim(0, 102)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                a.spines[sp].set_color(GRID)
            a.tick_params(colors=MUT, length=0, labelsize=9.5)
            a.legend(frameon=False, fontsize=9.5)
        fig.suptitle("An inspection crew's view of the detector",
                     fontsize=13.5, fontweight="bold", color=INK, x=0.008,
                     ha="left", y=1.02)
        fig.text(0.008, -0.04, f"{args.data} · model {args.model} · shaded band is a "
                 f"95% interval from {args.bootstrap} day-level bootstrap resamples",
                 fontsize=8.5, color=MUT)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(out_dir, f"inspection_budget.{ext}"), dpi=200,
                        bbox_inches="tight", facecolor="#fcfcfb")
        print(f"\n  figure -> {out_dir}/inspection_budget.png (+ .pdf)")
    except Exception as exc:                                 # pragma: no cover
        print(f"  [note] figure skipped ({exc})")

    print(f"  table  -> {out_dir}/inspection_budget.csv")


if __name__ == "__main__":
    main()
