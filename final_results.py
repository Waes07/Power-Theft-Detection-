"""
final_results.py -- one command that produces every number the final
presentation needs, from the CURRENT pipeline code, with no XGBoost anywhere.

    python final_results.py            # full run (about 10-20 min on a laptop CPU)
    python final_results.py --quick    # 1 seed, a few minutes -- to check it runs

It re-uses the project's own code (phase3_train_and_benchmark.py and
phase5_multi_thief.py) -- the model, the windowing, the split, the scaler, the
metrics and the early stopping are the same functions phase 3 and phase 5 call.
The only thing added is logging: per-epoch loss / validation F1, the confusion
matrix of every model on every seed, and per-class precision / recall / F1.

Everything is written to  final_results/  next to this file.  Nothing else in
the project is modified.

Models (single thief, data/run, chronological 70/15/15 split, 5 seeds):
    rule      hand-written physics rule, no learning
    logreg    logistic regression on daily summary features
    rf        random forest on daily summary features
    cnn       1-D CNN only
    lstm      LSTM only
    cnn_lstm  1-D CNN -> LSTM   (the project model)

Multi-thief (data/mt2, one sigmoid per node, 3 seeds):
    cnn_lstm  and  logreg, with per-node confusion matrices.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import sys
import time
import traceback
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score,
                             precision_recall_fscore_support)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")

import phase3_train_and_benchmark as P3                      # noqa: E402
from phase2_audit_dataset import load_run                    # noqa: E402

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = P3.HAS_TORCH
except Exception:                                            # pragma: no cover
    HAS_TORCH = False

OUT = os.path.join(HERE, "final_results")
LABELS = ["Normal", "Node 1", "Node 2", "Node 3", "Node 4", "Node 5"]
METRICS = ["accuracy", "macro_f1", "detect_f1", "detect_pr_auc", "localise_acc",
           "episode_recall", "latency_days", "false_alarm_rate"]


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Same training recipe as phase3.train_nn, with a per-epoch log added.
# The random-number sequence is identical (seed, model build, shuffled
# loader); the extra validation-loss computation draws no random numbers.
# ---------------------------------------------------------------------------
def train_logged(Xtr, ytr, Xva, yva, in_ch, n_classes, mode, seed,
                 epochs=200, patience=15, lr=1e-3, batch=64):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = P3.CNNLSTM(in_ch, n_classes, mode)
    counts = np.bincount(ytr, minlength=n_classes).astype(float)
    w = counts.sum() / np.maximum(counts, 1.0)
    w = np.clip(w, 0.0, 20.0 * max(w[counts > 0].min(), 1e-9))
    w = torch.tensor(w / w.mean(), dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    Xva_t = torch.tensor(Xva, dtype=torch.float32)
    yva_t = torch.tensor(yva, dtype=torch.long)
    loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=batch, shuffle=True)

    rows = []
    best_f1, best_state, bad, best_ep = -1.0, copy.deepcopy(model.state_dict()), 0, 0
    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss.item()) * len(xb)
            n += len(xb)
        model.eval()
        with torch.no_grad():
            lv = model(Xva_t)
            vloss = float(crit(lv, yva_t).item())
            pv = lv.argmax(1).numpy()
        f1 = f1_score(yva, pv, average="macro", labels=list(range(n_classes)),
                      zero_division=0)
        rows.append(dict(epoch=ep + 1, train_loss=tot / max(n, 1),
                         val_loss=vloss, val_macro_f1=f1))
        if f1 > best_f1 + 1e-4:
            best_f1, best_state, bad, best_ep = f1, copy.deepcopy(model.state_dict()), 0, ep + 1
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, pd.DataFrame(rows), best_ep


def n_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# ---------------------------------------------------------------------------
# Single-thief benchmark
# ---------------------------------------------------------------------------
def single_thief(data_dir: str, seeds: int, models) -> dict:
    df, daily, cfg = load_run(data_dir)
    n_classes = cfg.n_nodes + 1
    X, y, real, cols = P3.build_windows(df, daily, cfg, P3.CHANNEL_GROUPS)
    keep = y >= 0
    dayno = daily["day"].to_numpy()[keep]
    X, y, real = X[keep], y[keep], real[keep]
    tr, va, te = P3.split_indices(y, real, "time")
    sc = P3.fit_scaler(X, tr)
    Xs = P3.apply_scaler(X, sc)
    Ftr, Fte = P3.summary_features(Xs[tr]), P3.summary_features(Xs[te])
    in_ch = Xs.shape[1]

    info = dict(data=data_dir, input_shape=[int(v) for v in X.shape[1:]],
                channels=cols, n_days=int(len(y)),
                split=dict(train=int(len(tr)), val=int(len(va)), test=int(len(te))),
                test_class_counts=np.bincount(y[te], minlength=n_classes).tolist(),
                train_class_counts=np.bincount(y[tr], minlength=n_classes).tolist())
    log(f"  input per day: {in_ch} channels x 24 hours;  "
        f"train/val/test days = {len(tr)}/{len(va)}/{len(te)}")

    rule_all = None
    if "rule" in models:
        pred_all, _, _, unb, _ = P3.rule_baseline(df, daily, cfg)
        rule_all = (pred_all[keep], unb[keep])

    per_seed, per_class = [], []
    for s in range(seeds):
        for name in models:
            t0 = time.time()
            try:
                proba, extra = None, {}
                if name == "rule":
                    pred, score = rule_all[0][te], rule_all[1][te]
                elif name == "logreg":
                    m = LogisticRegression(max_iter=3000, class_weight="balanced",
                                           random_state=s).fit(Ftr, y[tr])
                    proba = m.predict_proba(Fte); pred = m.predict(Fte)
                elif name == "rf":
                    m = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                               random_state=s, n_jobs=-1).fit(Ftr, y[tr])
                    proba = m.predict_proba(Fte); pred = m.predict(Fte)
                elif name in ("cnn", "lstm", "cnn_lstm"):
                    if not HAS_TORCH:
                        log(f"    [skip] {name}: PyTorch not available")
                        continue
                    mdl, curve, best_ep = train_logged(Xs[tr], y[tr], Xs[va], y[va],
                                                       in_ch, n_classes, name, s)
                    curve.to_csv(os.path.join(OUT, f"curve_{name}_seed{s}.csv"), index=False)
                    pred, proba = P3.nn_predict(mdl, Xs[te])
                    extra = dict(best_epoch=best_ep, epochs_run=int(len(curve)),
                                 params=n_params(mdl))
                else:
                    continue
                if proba is not None:
                    score = 1.0 - proba[:, 0]
                r = P3.evaluate(y[te], pred, score, n_classes)
                r.update(P3.episode_metrics(y[te], pred, dayno[te], real[te]))
                r.update(extra)
                r.update(model=name, seed=s, seconds=round(time.time() - t0, 1))
                per_seed.append(r)
                cm = confusion_matrix(y[te], pred, labels=list(range(n_classes)))
                pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(
                    os.path.join(OUT, f"confusion_{name}_seed{s}.csv"))
                p, rc, f, sup = precision_recall_fscore_support(
                    y[te], pred, labels=list(range(n_classes)), zero_division=0)
                for k in range(n_classes):
                    per_class.append(dict(model=name, seed=s, cls=LABELS[k],
                                          precision=p[k], recall=rc[k], f1=f[k],
                                          support=int(sup[k])))
                log(f"    seed {s}  {name:9s} acc {r['accuracy']:.4f}  macro-F1 "
                    f"{r['macro_f1']:.4f}  localise {r['localise_acc']:.4f}  "
                    f"false-alarm {r['false_alarm_rate']:.4f}   ({time.time()-t0:.0f}s)")
            except Exception:
                log(f"    [error] {name} seed {s}:\n{traceback.format_exc()}")

    ps = pd.DataFrame(per_seed)
    ps.to_csv(os.path.join(OUT, "per_seed.csv"), index=False)
    pd.DataFrame(per_class).to_csv(os.path.join(OUT, "per_class.csv"), index=False)
    rows = []
    for name in models:
        sub = ps[ps.model == name] if len(ps) else ps
        if not len(sub):
            continue
        row = {"model": name, "seeds": int(len(sub))}
        for m in METRICS:
            row[f"{m}_mean"] = float(np.nanmean(sub[m]))
            row[f"{m}_std"] = float(np.nanstd(sub[m]))
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "benchmark_final.csv"), index=False)
    if len(ps) and "params" in ps:
        info["params"] = {k: int(v) for k, v in
                          ps.dropna(subset=["params"]).groupby("model")["params"].first().items()}
    return info


# ---------------------------------------------------------------------------
# Multi-thief: one sigmoid per node (phase 5 recipe), per-node confusion
# ---------------------------------------------------------------------------
def multi_thief(data_dir: str, seeds: int) -> dict:
    import phase5_multi_thief as P5
    df, daily, cfg = load_run(data_dir)
    n = cfg.n_nodes
    X, y1, real, _ = P3.build_windows(df, daily, cfg, P3.CHANNEL_GROUPS)
    Y = daily[[f"theft_{j+1}" for j in range(n)]].to_numpy().astype(int)
    tr, va, te = P3.split_indices(y1, real, "time")
    Xs = P3.apply_scaler(X, P3.fit_scaler(X, tr))
    Ftr, Fte = P3.summary_features(Xs[tr]), P3.summary_features(Xs[te])
    two = int((Y[te].sum(1) >= 2).sum())
    log(f"  multi-thief test days: {len(te)}  (days with two or more thieves: {two})")

    rows = []
    for s in range(seeds):
        preds = {}
        # logistic regression, one per node (phase 5 recipe)
        Pl = np.zeros((len(te), n), dtype=int); Sl = np.zeros((len(te), n))
        for j in range(n):
            if len(np.unique(Y[tr, j])) < 2:
                Pl[:, j] = int(Y[tr, j].max()); continue
            Sl[:, j] = LogisticRegression(max_iter=4000, class_weight="balanced",
                                          random_state=s).fit(Ftr, Y[tr, j]).predict_proba(Fte)[:, 1]
            Pl[:, j] = (Sl[:, j] > 0.5).astype(int)
        preds["logreg"] = (Pl, Sl)
        if HAS_TORCH and getattr(P5, "HAS_TORCH", False):
            try:
                mdl = P5.train_multilabel(Xs[tr], Y[tr], Xs[va], Y[va], Xs.shape[1],
                                          n, "cnn_lstm", s)
                S = P5.predict_ml(mdl, Xs[te])
                preds["cnn_lstm"] = ((S > 0.5).astype(int), S)
            except Exception:
                log(f"    [error] multi-thief cnn_lstm seed {s}:\n{traceback.format_exc()}")
        for name, (Pm, Sm) in preds.items():
            r = P5.evaluate_multilabel(Y[te], Pm, Sm)
            r.update(model=name, seed=s)
            rows.append(r)
            out = {}
            for j in range(n):
                cm = confusion_matrix(Y[te, j], Pm[:, j], labels=[0, 1])
                out[f"node{j+1}"] = cm.tolist()
            cnt = confusion_matrix(Y[te].sum(1), Pm.sum(1).clip(0, 3), labels=[0, 1, 2, 3])
            out["count"] = cnt.tolist()
            with open(os.path.join(OUT, f"multi_confusion_{name}_seed{s}.json"), "w") as fh:
                json.dump(out, fh)
            log(f"    seed {s}  {name:9s} per-node F1 {r['per_node_f1']:.4f}  "
                f"exact {r['exact_match']:.4f}  two-thief days fully right "
                f"{r['exact_multi']:.3f}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "multi_thief_final.csv"), index=False)
    return dict(data=data_dir, test_days=int(len(te)), two_thief_test_days=two)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(HERE, "data", "run"))
    ap.add_argument("--multi-data", default=os.path.join(HERE, "data", "mt2"))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--multi-seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="1 seed each, to check it runs")
    ap.add_argument("--no-multi", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.multi_seeds = 1, 1
    os.makedirs(OUT, exist_ok=True)

    t0 = time.time()
    log("=" * 78)
    log(" FINAL RESULTS for the presentation  (no XGBoost anywhere)")
    log(f" PyTorch: {'yes ' + torch.__version__ if HAS_TORCH else 'NO -- neural models will be skipped'}")
    log("=" * 78)
    summary = dict(python=platform.python_version(),
                   torch=(torch.__version__ if HAS_TORCH else None),
                   started=time.strftime("%Y-%m-%d %H:%M:%S"),
                   seeds=args.seeds)
    log("\n[1/2] single-thief benchmark, " + args.data)
    summary["single"] = single_thief(args.data, args.seeds,
                                     ["rule", "logreg", "rf", "cnn", "lstm", "cnn_lstm"])
    if not args.no_multi and os.path.exists(os.path.join(args.multi_data, "labels_daily.csv")):
        log("\n[2/2] multi-thief, " + args.multi_data)
        summary["multi"] = multi_thief(args.multi_data, args.multi_seeds)
    summary["minutes"] = round((time.time() - t0) / 60, 1)
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    log(f"\nDone in {summary['minutes']} min.  Everything is in {OUT}")
    if not HAS_TORCH:
        log("WARNING: PyTorch was not found, so CNN / LSTM / CNN+LSTM were NOT run.")


if __name__ == "__main__":
    main()
