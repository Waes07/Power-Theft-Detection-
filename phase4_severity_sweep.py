"""
phase4_severity_sweep.py -- detection performance vs theft severity.

THIS STAGE IS NEW.  The original study reported a single accuracy figure at a
single (enormous) theft magnitude.  A single number says nothing about where a
detector stops working, which is the first question any reviewer asks.  This
sweeps the severity alpha -- theft as a fraction of the thief's own load --
and produces the "macro-F1 vs alpha" curve, which is the most informative
single figure this project can put in a paper.

It also sweeps sensor coverage, because how much of the feeder is instrumented
turns out to matter more than which model you use.

Usage:
    python phase4_severity_sweep.py --alphas 0.05 0.10 0.15 0.20 0.30 \
        --coverage all sparse --seeds 3 --out sweep
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from ntl_common import FeederConfig, load_lcl_blocks
from phase1_simulate_feeder import simulate
from phase2_audit_dataset import rule_baseline
from phase3_train_and_benchmark import CHANNEL_GROUPS, run_once


def build(cfg: FeederConfig, hourly, realizations: int, solver: str):
    hs, ds = [], []
    for r in range(realizations):
        h, d, _, _ = simulate(cfg, hourly, solver, realization=r)
        hs.append(h); ds.append(d)
    return pd.concat(hs, ignore_index=True), pd.concat(ds, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blocks", nargs="+", default=["block1.csv", "block2.csv", "block3.csv"])
    ap.add_argument("--cache", default="lcl_hourly_wide.csv")
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[0.05, 0.08, 0.12, 0.16, 0.20, 0.30])
    ap.add_argument("--coverage", nargs="+", default=["all", "sparse"])
    ap.add_argument("--models", nargs="+",
                    default=["rule", "logreg", "gbm_raw", "cnn_lstm", "hybrid"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--realizations", type=int, default=8)
    ap.add_argument("--solver", choices=["auto", "pandapower", "bfs"], default="auto")
    ap.add_argument("--out", default="sweep")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    hourly = load_lcl_blocks(args.blocks, cache=args.cache)
    rows = []

    for cov in args.coverage:
        for a in args.alphas:
            cfg = FeederConfig(theft_severity=a, sensor_coverage=cov)
            df, daily = build(cfg, hourly, args.realizations, args.solver)
            per_seed = [run_once(df, daily, cfg, CHANNEL_GROUPS, s, "time", args.models)[0]
                        for s in range(args.seeds)]
            line = f"  coverage={cov:6s} alpha={a:.2f} | "
            for model in per_seed[0]:
                v = [p[model]["macro_f1"] for p in per_seed]
                e = [p[model]["episode_recall"] for p in per_seed]
                f = [p[model]["false_alarm_rate"] for p in per_seed]
                rows.append(dict(coverage=cov, alpha=a, model=model,
                                 macro_f1=np.mean(v), macro_f1_std=np.std(v),
                                 episode_recall=np.mean(e), false_alarm=np.mean(f)))
                line += f"{model} {np.mean(v):.3f}  "
            print(line, flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.out, "severity_sweep.csv"), index=False)

    # ---- the figure -------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(args.coverage), figsize=(6 * len(args.coverage), 4.4),
                                 sharey=True, squeeze=False)
        for ax, cov in zip(axes[0], args.coverage):
            sub = res[res.coverage == cov]
            for model, g in sub.groupby("model"):
                g = g.sort_values("alpha")
                ax.errorbar(g.alpha, g.macro_f1, yerr=g.macro_f1_std,
                            marker="o", capsize=3, label=model, linewidth=1.6)
            ax.set_xlabel(r"theft severity $\alpha$  (fraction of the thief's own load)")
            ax.set_title(f"current sensors: {cov}")
            ax.grid(alpha=0.3)
        axes[0][0].set_ylabel("macro-F1 (test)")
        axes[0][-1].legend(frameon=False, fontsize=9)
        fig.suptitle("Localisation performance vs theft severity", y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "severity_sweep.png"), dpi=160,
                    bbox_inches="tight")
        print(f"\nFigure written to {args.out}/severity_sweep.png")
    except Exception as exc:                          # pragma: no cover
        print(f"[note] figure skipped ({exc})")

    print(f"Table written to {args.out}/severity_sweep.csv")


if __name__ == "__main__":
    main()
