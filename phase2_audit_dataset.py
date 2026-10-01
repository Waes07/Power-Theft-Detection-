"""
phase2_audit_dataset.py -- dataset integrity and triviality audit.

THIS STAGE IS NEW.  It exists because the original study shipped a benchmark
in which a four-line hand-written rule scored 0.9977 accuracy while the
deep model scored ~0.918, and in which 743 of 745 "normal" days actually
contained theft.  Neither fact was visible from the training script.  This
audit is the check that would have caught both, and it should be run (and
reported in the paper) before any model is trained.

It reports:
  1. Structural integrity   -- day boundaries, NaN rows, label/loss consistency.
  2. Physics plausibility   -- technical loss %, bus voltages, line loading,
                               and the share of line current that is cable
                               charging rather than load.
  3. Triviality             -- best single-feature detection AUC, class overlap,
                               and the score of the hand-written physics rule.
                               If the rule already scores > 0.99 the benchmark
                               is too easy and no model result from it is
                               meaningful.
  4. Solver agreement       -- pandapower vs the independent BFS solver.

Usage:
    python phase2_audit_dataset.py --data data/alpha020 [--check-solver]
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from ntl_common import CABLES, F_HZ, FeederConfig, physics_features

WARN = "  [WARN] "
OK = "  [ ok ] "


def load_run(path: str):
    df = pd.read_csv(os.path.join(path, "dataset_hourly.csv"), parse_dates=["timestamp"])
    daily = pd.read_csv(os.path.join(path, "labels_daily.csv"))
    cfg = FeederConfig(**json.load(open(os.path.join(path, "config.json"))))
    return df, daily, cfg


# ---------------------------------------------------------------------------
def audit_structure(df: pd.DataFrame, daily: pd.DataFrame, cfg: FeederConfig) -> None:
    print("\n1. STRUCTURAL INTEGRITY")
    ok = True
    for r, g in df.groupby("realization"):
        if len(g) % 24:
            print(WARN + f"realization {r}: {len(g)} rows is not a multiple of 24"); ok = False
        if not (g["hour"].to_numpy() == np.tile(np.arange(24), len(g) // 24)).all():
            print(WARN + f"realization {r}: hour column is not a clean 0..23 cycle"); ok = False
    nan_rows = df["Substation_kW"].isna().sum()
    if nan_rows:
        print(WARN + f"{nan_rows} non-converged hours present (kept as NaN, not dropped)")

    # the check the original code failed: do hourly truth and daily labels agree?
    n = cfg.n_nodes
    mism = 0
    for r, g in df.groupby("realization"):
        d = daily[daily.realization == r]
        for j in range(1, n + 1):
            hourly_any = g[f"true_theft_{j}"].to_numpy().reshape(-1, 24).any(axis=1).astype(int)
            mism += int((hourly_any != d[f"theft_{j}"].to_numpy()).sum())
    if mism:
        print(WARN + f"{mism} day/node label mismatches between hourly truth and daily labels")
        ok = False
    else:
        print(OK + "daily labels are consistent with hourly ground truth (0 mismatches)")

    # theft-hours inside days labelled normal -- the original's fatal bug
    bad = 0
    for r, g in df.groupby("realization"):
        d = daily[daily.realization == r]
        tt = g[[f"true_theft_{j+1}" for j in range(n)]].to_numpy().reshape(-1, 24, n)
        normal_days = (d["Thief_ID"].to_numpy() == 0)
        bad += int(tt[normal_days].any(axis=(1, 2)).sum())
    frac = bad / max(1, int((daily.Thief_ID == 0).sum()))
    if frac > 0.01:
        print(WARN + f"{bad} 'normal' days contain theft hours ({100*frac:.1f}%) -- labels are corrupt")
        ok = False
    else:
        print(OK + f"'normal' days containing theft hours: {bad} ({100*frac:.2f}%)")

    print(OK + f"class balance: {int((daily.Thief_ID!=0).sum())}/{len(daily)} theft-days "
               f"({100*(daily.Thief_ID!=0).mean():.1f}%)")
    if ok:
        print(OK + "structure passed")


# ---------------------------------------------------------------------------
def audit_physics(df: pd.DataFrame, cfg: FeederConfig) -> None:
    print("\n2. PHYSICS PLAUSIBILITY")
    load = df[[f"Meter_{j+1}_kW" for j in range(cfg.n_nodes)]].sum(axis=1)
    loss_pct = 100 * df["tech_loss_kW"].mean() / load.mean()
    tag = OK if 2.0 <= loss_pct <= 12.0 else WARN
    print(tag + f"mean technical loss ....... {loss_pct:5.2f} %   (realistic band 3-10%)")

    vmin = df["vm_pu_min"].min()
    print((OK if vmin >= 0.90 else WARN) + f"minimum bus voltage ....... {vmin:.4f} pu (limit 0.90)")
    lmax = df["max_loading_pct"].max()
    print((OK if lmax <= 100 else WARN) + f"maximum line loading ...... {lmax:5.1f} %  (limit 100)")

    cab = CABLES[cfg.cable]
    print((OK if abs(cab["un"] - cfg.vn_kv) < 1e-6 else WARN) +
          f"cable {cfg.cable!r} rated {cab['un']} kV, feeder runs at {cfg.vn_kv} kV")

    # how much of the reported line current is cable charging, not load?
    v_ln = cfg.vn_kv * 1000.0 / math.sqrt(3.0) * cfg.slack_vm_pu
    y = 2 * math.pi * F_HZ * cab["c"] * 1e-9 * cfg.segment_km
    i_charge = v_ln * y            # full-segment charging current, A
    i_line1 = df["Line_1_A"].mean()
    share = 100 * i_charge / i_line1
    tag = OK if share < 20 else WARN
    print(tag + f"cable charging share of Line_1 current ... {share:5.1f} %  "
                f"({i_charge:.3f} A of {i_line1:.2f} A)")
    if share >= 20:
        print("         -> the 'I^2R signature' is mostly capacitive; fix the voltage level or cable.")


# ---------------------------------------------------------------------------
def rule_baseline(df: pd.DataFrame, daily: pd.DataFrame, cfg: FeederConfig,
                  train_frac: float = 0.7):
    """
    Hand-written physics rule -- the baseline every learned model must beat.
    Detect on daily-mean unbalance; localise by the largest KCL residual.
    Thresholds are fitted on the training portion only.
    """
    feats = physics_features(df, cfg)
    n = cfg.n_nodes
    day_key = df["realization"].astype(str) + "_" + df["day"].astype(str)
    daily_key = daily["realization"].astype(str) + "_" + daily["day"].astype(str)

    agg = feats.groupby(day_key.values).mean()
    agg = agg.loc[daily_key.values]
    y = daily["Thief_ID"].to_numpy()

    ntr = int(len(y) * train_frac)
    unb = agg["Unbalance_kW"].to_numpy()
    resid = agg[[f"Resid_{j+1}" for j in range(n)]].to_numpy()
    # Remove each residual's own baseline offset (technical loss, unmetered
    # communal load, cable-parameter error), estimated on normal TRAIN days only.
    with np.errstate(invalid="ignore"):
        base = np.nanmedian(np.where(y[:, None] == 0, resid, np.nan)[:ntr], axis=0)
    base = np.nan_to_num(base)
    resid = resid - base
    # Nodes whose KCL residual is undefined (no current sensor) cannot be
    # ranked; fall back to the most frequent thief seen in training.
    undefined = np.isnan(resid).all(axis=0)
    if undefined.all():
        counts = np.bincount(y[:ntr][y[:ntr] > 0], minlength=n + 1)[1:]
        fallback = int(counts.argmax()) + 1
    else:
        fallback = None
    resid = np.where(np.isnan(resid), -np.inf, resid)

    # threshold = the value on train days that maximises macro-F1
    cand = np.quantile(unb[:ntr], np.linspace(0.50, 0.995, 120))
    best, best_thr = -1, cand[0]
    for thr in cand:
        loc = fallback if fallback else resid.argmax(axis=1) + 1
        p = np.where(unb > thr, loc, 0)
        s = f1_score(y[:ntr], p[:ntr], average="macro", labels=list(range(n + 1)),
                     zero_division=0)
        if s > best:
            best, best_thr = s, thr
    loc = fallback if fallback else resid.argmax(axis=1) + 1
    pred = np.where(unb > best_thr, loc, 0)
    return pred, y, ntr, unb, agg


def audit_triviality(df: pd.DataFrame, daily: pd.DataFrame, cfg: FeederConfig) -> None:
    print("\n3. TRIVIALITY / SEPARABILITY")
    pred, y, ntr, unb, agg = rule_baseline(df, daily, cfg)
    te = slice(ntr, None)

    det_true = (y > 0).astype(int)
    auc = roc_auc_score(det_true[te], unb[te])
    tag = OK if auc < 0.99 else WARN
    print(tag + f"detection AUC of the single best feature (daily unbalance) ... {auc:.4f}")

    a, b = unb[y == 0], unb[y > 0]
    d = abs(np.nanmean(b) - np.nanmean(a)) / np.sqrt((np.nanvar(a) + np.nanvar(b)) / 2)
    tag = OK if d < 8 else WARN
    print(tag + f"Cohen's d between normal and theft days .................... {d:.2f}")
    print(f"         normal: {np.nanmean(a):8.3f} +/- {np.nanstd(a):.3f} kW")
    print(f"         theft : {np.nanmean(b):8.3f} +/- {np.nanstd(b):.3f} kW")

    acc = accuracy_score(y[te], pred[te])
    mf1 = f1_score(y[te], pred[te], average="macro", zero_division=0)
    tag = OK if mf1 < 0.95 else WARN
    print(tag + f"hand-written physics rule (test split) ..... acc {acc:.4f} | macro-F1 {mf1:.4f}")
    if mf1 >= 0.95:
        print("         -> benchmark is close to trivial; a deep model cannot demonstrate value here.")
        print("         -> lower --severity, raise --sigma-meter/--p-missing, or add more nodes.")
    else:
        print("         -> this is the number every learned model in phase 3 must beat.")


# ---------------------------------------------------------------------------
def check_solver(cfg: FeederConfig, n_samples: int = 200) -> None:
    print("\n4. SOLVER CROSS-VALIDATION (pandapower vs independent BFS)")
    try:
        from ntl_common import BFSSolver, PandapowerSolver
        pps = PandapowerSolver(cfg)
    except Exception as exc:
        print(WARN + f"pandapower unavailable ({type(exc).__name__}); skipping cross-check")
        return
    bfs = BFSSolver(cfg)
    rng = np.random.default_rng(0)
    tan_phi = math.sqrt(1 / cfg.power_factor ** 2 - 1)
    dev_i, dev_p = [], []
    for _ in range(n_samples):
        p = rng.uniform(1.0, 25.0, size=cfg.n_nodes)
        a, b = pps.solve(p, p * tan_phi), bfs.solve(p, p * tan_phi)
        dev_i.append(np.max(np.abs(a["line_amps"] - b["line_amps"]) / a["line_amps"]))
        dev_p.append(abs(a["substation_kw"] - b["substation_kw"]) / abs(a["substation_kw"]))
    print(OK + f"max relative deviation, line currents ... {max(dev_i):.3e}")
    print(OK + f"max relative deviation, substation P ... {max(dev_p):.3e}")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/alpha020")
    ap.add_argument("--check-solver", action="store_true")
    ap.add_argument("--use-sensor-lines", nargs="+", type=int, default=None,
                    help="audit as if only these line segments were instrumented")
    args = ap.parse_args()

    df, daily, cfg = load_run(args.data)
    if args.use_sensor_lines:
        keep = set(args.use_sensor_lines)
        for j in range(1, cfg.n_nodes + 1):
            if j not in keep:
                df[f"Line_{j}_A"] = np.nan
        cfg.sensor_lines = sorted(keep)
    print("=" * 72)
    print(f" DATASET AUDIT -- {args.data}")
    print(f" {cfg.n_nodes} nodes | {cfg.vn_kv} kV | alpha={cfg.theft_severity} | "
          f"sigma_meter={cfg.sigma_meter} | p_missing={cfg.p_missing}")
    print(f" current sensors on lines: "
          f"{cfg.sensor_lines if cfg.sensor_lines else 'all ' + str(cfg.n_nodes)}")
    print("=" * 72)

    audit_structure(df, daily, cfg)
    audit_physics(df, cfg)
    audit_triviality(df, daily, cfg)
    if args.check_solver:
        check_solver(cfg)
    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
