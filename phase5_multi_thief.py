"""
phase5_multi_thief.py -- simultaneous theft at several nodes.

A SEPARATE STUDY, deliberately kept out of phases 1-4.

Phases 1-4 answer "which one service point is stealing today?" using a softmax
over nodes.  That formulation cannot represent two thieves at once, and phase 3
drops such days (`keep = y >= 0`).  Rather than complicate the main pipeline,
this file re-poses the problem:

    single-label (phases 1-4)   softmax over n+1 classes   -> exactly one answer
    multi-label  (this file)    one sigmoid per node       -> any number of answers

Nothing here modifies phases 1-4.  It imports their data plumbing -- the same
windowing, the same scaler, the same chronological split -- so results are
directly comparable, and it defines its own model head, loss and metrics.

Generate the data with phase 1, then run this:

    python phase1_simulate_feeder.py --out data/mt2 --severity 0.15 \
        --max-concurrent 2 --realizations 8
    python phase5_multi_thief.py --data data/mt2 --seeds 5
    python phase5_multi_thief.py --data data/mt2 --seeds 5 --use-sensor-lines 3 4 5

Requires PyTorch only for the neural models; the classical ladder runs without it.
"""
from __future__ import annotations

import argparse
import copy
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, hamming_loss,
                             precision_score, recall_score)

from ntl_common import physics_features
from phase2_audit_dataset import load_run
from phase3_train_and_benchmark import (CHANNEL_GROUPS, apply_scaler, build_windows,
                                        fit_scaler, make_gbm, split_indices,
                                        summary_features)

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from phase3_train_and_benchmark import CNNLSTM          # architecture is reused
    HAS_TORCH = True
except Exception:                                            # pragma: no cover
    HAS_TORCH = False

ALL_MODELS = ["rule", "logreg", "rf", "gbm_raw", "cnn", "lstm", "cnn_lstm",
              "hybrid", "hybrid_plus"]


# ---------------------------------------------------------------------------
# Metrics for the multi-label setting
# ---------------------------------------------------------------------------
def evaluate_multilabel(T: np.ndarray, P: np.ndarray, S: np.ndarray) -> dict:
    """
    T, P : (days, n_nodes) binary truth and prediction.  S : per-node scores.

    Day-level accuracy is meaningless once several nodes can be guilty at once,
    so the reported quantities are the ones a utility would act on:

      per_node_f1   macro F1 across the per-node detectors
      exact_match   every node on the feeder called correctly, guilty and innocent
      hamming       fraction of individual node-days called wrongly
      detect_f1     was ANY theft on the feeder noticed at all
      recall_multi  on days with 2+ thieves, the fraction of guilty nodes found
      exact_multi   on days with 2+ thieves, every node correct
      overcall      innocent nodes wrongly accused, per day
    """
    two = T.sum(1) >= 2
    det_t, det_p = T.any(1).astype(int), P.any(1).astype(int)
    return dict(
        per_node_f1=f1_score(T, P, average="macro", zero_division=0),
        precision=precision_score(T, P, average="micro", zero_division=0),
        recall=recall_score(T, P, average="micro", zero_division=0),
        exact_match=float((P == T).all(1).mean()),
        hamming=hamming_loss(T, P),
        detect_f1=f1_score(det_t, det_p, zero_division=0),
        detect_pr_auc=(average_precision_score(det_t, 1 - np.prod(1 - S, axis=1))
                       if len(np.unique(det_t)) > 1 else float("nan")),
        recall_multi=float(P[two][T[two] == 1].mean()) if two.any() else float("nan"),
        exact_multi=float((P[two] == T[two]).all(1).mean()) if two.any() else float("nan"),
        overcall=float(((P == 1) & (T == 0)).sum(1).mean()),
        n_multi=int(two.sum()),
    )


# ---------------------------------------------------------------------------
# Neural training with a sigmoid-per-node head
# ---------------------------------------------------------------------------
if HAS_TORCH:

    def train_multilabel(Xtr, Ytr, Xva, Yva, in_ch, n_nodes, mode, seed,
                         epochs=200, patience=15, lr=1e-3, batch=64, verbose=False):
        """
        The CNN-LSTM of phase 3, unchanged, with the head reinterpreted: n_nodes
        logits passed through independent sigmoids instead of a softmax, trained
        with BCE and a per-node positive weight to handle class imbalance.
        Early stopping on validation macro-F1 across nodes, best checkpoint restored.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = CNNLSTM(in_ch, n_nodes, mode)

        pos = Ytr.sum(axis=0).astype(float)
        pw = np.clip((len(Ytr) - pos) / np.maximum(pos, 1.0), 0.0, 50.0)
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, dtype=torch.float32))
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
        Ytr_t = torch.tensor(Ytr, dtype=torch.float32)
        Xva_t = torch.tensor(Xva, dtype=torch.float32)
        loader = DataLoader(TensorDataset(Xtr_t, Ytr_t), batch_size=batch, shuffle=True)

        best, best_state, bad = -1.0, copy.deepcopy(model.state_dict()), 0
        for ep in range(epochs):
            model.train()
            for xb, yb in loader:
                opt.zero_grad()
                crit(model(xb), yb).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                pv = (torch.sigmoid(model(Xva_t)).numpy() > 0.5).astype(int)
            f1 = f1_score(Yva, pv, average="macro", zero_division=0)
            if f1 > best + 1e-4:
                best, best_state, bad = f1, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    if verbose:
                        print(f"      early stop at epoch {ep+1}, val macro-F1 {best:.4f}")
                    break
        model.load_state_dict(best_state)
        model.eval()
        return model

    def predict_ml(model, X):
        with torch.no_grad():
            return torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).numpy()

    def embed_ml(model, X):
        with torch.no_grad():
            return model(torch.tensor(X, dtype=torch.float32),
                         extract_features=True).numpy()


# ---------------------------------------------------------------------------
def run_once(df, daily, cfg, groups, seed, split_mode, models, oof_folds=3,
             verbose=False) -> dict:
    n = cfg.n_nodes
    X, y1, real, _ = build_windows(df, daily, cfg, groups)
    Y = daily[[f"theft_{j+1}" for j in range(n)]].to_numpy().astype(int)
    tr, va, te = split_indices(y1, real, split_mode)       # no day is dropped

    Xs = apply_scaler(X, fit_scaler(X, tr))
    Ftr, Fte = summary_features(Xs[tr]), summary_features(Xs[te])
    res = {}

    def record(name, P, S):
        res[name] = evaluate_multilabel(Y[te], P, S)

    def per_node(fit_predict):
        P = np.zeros((len(te), n), dtype=int)
        S = np.zeros((len(te), n))
        for j in range(n):
            if len(np.unique(Y[tr, j])) < 2:      # node never / always guilty in train
                P[:, j] = int(Y[tr, j].max())
                S[:, j] = float(Y[tr, j].max())
                continue
            S[:, j] = fit_predict(j)
            P[:, j] = (S[:, j] > 0.5).astype(int)
        return P, S

    # --- physics rule: threshold each node's own KCL residual --------------
    if "rule" in models:
        feats = physics_features(df, cfg)
        key = df["realization"].astype(str) + "_" + df["day"].astype(str)
        dkey = daily["realization"].astype(str) + "_" + daily["day"].astype(str)
        R = feats.groupby(key.values).mean().loc[dkey.values][
            [f"Resid_{j+1}" for j in range(n)]].to_numpy()
        R = np.where(np.isnan(R), -np.inf, R)
        P = np.zeros((len(te), n), dtype=int)
        S = np.zeros((len(te), n))
        for j in range(n):
            col = R[:, j]
            fin = np.isfinite(col[tr])
            if not fin.any():
                continue
            cand = np.quantile(col[tr][fin], np.linspace(.5, .995, 80))
            best, thr = -1.0, cand[0]
            for c in cand:
                f = f1_score(Y[tr, j], (col[tr] > c).astype(int), zero_division=0)
                if f > best:
                    best, thr = f, c
            P[:, j] = (col[te] > thr).astype(int)
            span = np.ptp(col[np.isfinite(col)]) or 1.0
            S[:, j] = np.clip((col[te] - thr) / span + 0.5, 0, 1)
        record("rule", P, S)

    if "logreg" in models:
        record("logreg", *per_node(lambda j: LogisticRegression(
            max_iter=4000, class_weight="balanced", random_state=seed
        ).fit(Ftr, Y[tr, j]).predict_proba(Fte)[:, 1]))
    if "rf" in models:
        record("rf", *per_node(lambda j: RandomForestClassifier(
            n_estimators=400, class_weight="balanced", random_state=seed, n_jobs=-1
        ).fit(Ftr, Y[tr, j]).predict_proba(Fte)[:, 1]))
    if "gbm_raw" in models:
        flat_tr, flat_te = Xs[tr].reshape(len(tr), -1), Xs[te].reshape(len(te), -1)
        record("gbm_raw", *per_node(lambda j: make_gbm(seed, 2)
                                    .fit(flat_tr, Y[tr, j])
                                    .predict_proba(flat_te)[:, 1]))

    # --- neural models ------------------------------------------------------
    in_ch = Xs.shape[1]
    trained = {}
    if HAS_TORCH:
        for name, mode in (("cnn", "cnn"), ("lstm", "lstm"), ("cnn_lstm", "cnn_lstm")):
            needed = name == "cnn_lstm" and any(h in models for h in ("hybrid", "hybrid_plus"))
            if name not in models and not needed:
                continue
            mdl = train_multilabel(Xs[tr], Y[tr], Xs[va], Y[va], in_ch, n, mode,
                                   seed, verbose=verbose)
            trained[name] = mdl
            if name in models:
                S = predict_ml(mdl, Xs[te])
                record(name, (S > 0.5).astype(int), S)
    elif any(m in models for m in ("cnn", "lstm", "cnn_lstm", "hybrid", "hybrid_plus")):
        print("      [skip] PyTorch not installed -- neural models omitted")

    # --- stacked variants, out-of-fold --------------------------------------
    want = [m for m in ("hybrid", "hybrid_plus") if m in models]
    if want and HAS_TORCH:
        H = trained["cnn_lstm"].head_in
        folds = np.array_split(np.arange(len(tr)), oof_folds)
        oof_e = np.zeros((len(tr), H), dtype=np.float32)
        oof_p = np.zeros((len(tr), n), dtype=np.float32)
        for k in range(oof_folds):
            hold = folds[k]
            rest = np.concatenate([folds[j] for j in range(oof_folds) if j != k])
            m_k = train_multilabel(Xs[tr[rest]], Y[tr[rest]], Xs[va], Y[va],
                                   in_ch, n, "cnn_lstm", seed + 17 * k)
            oof_e[hold] = embed_ml(m_k, Xs[tr[hold]])
            oof_p[hold] = predict_ml(m_k, Xs[tr[hold]])
        te_e = embed_ml(trained["cnn_lstm"], Xs[te])
        te_p = predict_ml(trained["cnn_lstm"], Xs[te])
        for nm, TR, TE in (("hybrid", oof_e, te_e),
                           ("hybrid_plus", np.hstack([oof_e, oof_p]),
                            np.hstack([te_e, te_p]))):
            if nm not in models:
                continue
            record(nm, *per_node(lambda j, TR=TR, TE=TE: make_gbm(seed, 2)
                                 .fit(TR, Y[tr, j]).predict_proba(TE)[:, 1]))
    return res


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/mt2")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--split", choices=["time", "realization"], default="time")
    ap.add_argument("--models", nargs="+", default=ALL_MODELS)
    ap.add_argument("--groups", nargs="+", default=CHANNEL_GROUPS)
    ap.add_argument("--oof-folds", type=int, default=3)
    ap.add_argument("--use-sensor-lines", nargs="+", type=int, default=None,
                    help="pretend only these segments carry a current sensor")
    ap.add_argument("--out", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    df, daily, cfg = load_run(args.data)
    if args.use_sensor_lines:
        keep = set(args.use_sensor_lines)
        for j in range(1, cfg.n_nodes + 1):
            if j not in keep:
                df[f"Line_{j}_A"] = np.nan
        cfg.sensor_lines = sorted(keep)
        print(f" [sensors] using only lines {sorted(keep)} of {cfg.n_nodes}")
    out_dir = args.out or os.path.join(args.data, "results")
    os.makedirs(out_dir, exist_ok=True)

    share = daily.n_thieves.value_counts().sort_index()
    print("=" * 96)
    print(f" PHASE 5 -- SIMULTANEOUS THEFT (multi-label)   {args.data}")
    print(f" torch={'yes' if HAS_TORCH else 'NO'}  seeds={args.seeds}  split={args.split}")
    print(" days by number of active thieves: " +
          "  ".join(f"{k}->{v}" for k, v in share.items()))
    print("=" * 96)

    runs = []
    for sd in range(args.seeds):
        t0 = time.time()
        r = run_once(df, daily, cfg, args.groups, sd, args.split, args.models,
                     args.oof_folds, args.verbose)
        runs.append(r)
        print(f"  seed {sd}: " + " | ".join(f"{k} {v['per_node_f1']:.3f}"
                                            for k, v in r.items())
              + f"   ({time.time()-t0:.0f}s)")

    mets = ["per_node_f1", "exact_match", "hamming", "detect_f1",
            "recall_multi", "exact_multi", "overcall"]
    rows = []
    for model in runs[0]:
        row = {"model": model}
        for m in mets:
            v = [r[model][m] for r in runs]
            row[f"{m}_mean"], row[f"{m}_std"] = np.nanmean(v), np.nanstd(v)
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out_dir, "multi_thief.csv"), index=False)

    n_multi = runs[0][list(runs[0])[0]]["n_multi"]
    print("\n" + "=" * 96)
    print(f" RESULTS  (mean +/- std over {args.seeds} seeds)")
    print(f" {n_multi} test days carried two or more simultaneous thieves")
    print("=" * 96)
    print(f"{'model':<13}{'per-node F1':>16}{'exact match':>16}{'Hamming':>10}"
          f"{'detect-F1':>11}{'2+ recall':>11}{'2+ exact':>10}{'overcall':>10}")
    for _, r in table.iterrows():
        print(f"{r.model:<13}"
              f"{r.per_node_f1_mean:>9.4f}+-{r.per_node_f1_std:<5.3f}"
              f"{r.exact_match_mean:>9.4f}+-{r.exact_match_std:<5.3f}"
              f"{r.hamming_mean:>10.4f}{r.detect_f1_mean:>11.4f}"
              f"{r.recall_multi_mean:>11.4f}{r.exact_multi_mean:>10.4f}"
              f"{r.overcall_mean:>10.3f}")
    print(f"\n  'overcall' = innocent nodes wrongly accused per day.")
    print(f"  With only {n_multi} multi-thief test days, the last two columns carry wide")
    print(f"  intervals -- raise --max-concurrent or --episodes-per-node for more of them.")
    print(f"\nWritten to {out_dir}/multi_thief.csv")


if __name__ == "__main__":
    main()
