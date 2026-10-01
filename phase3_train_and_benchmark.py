"""
phase3_train_and_benchmark.py -- windowing, baseline ladder, hybrid model.

Replaces `phase3_sgcc_hybrid_8channel.py`.

WHAT CHANGED AND WHY
--------------------
(1) LABELS.  `y = Thief_ID[::24]` took the label from hour 0 of each day while
    theft was re-drawn hourly, so 99.7% of "normal" training days actually
    contained theft and the hour-0 label matched the day's dominant thief only
    16.4% of the time.  Labels now come from `labels_daily.csv`, which phase 1
    derives from persistent episodes and phase 2 verifies.

(2) BASELINE LADDER.  The original reported one number with nothing to compare
    it to.  Eight models are now trained on identical splits:
        rule | logreg | rf | gbm_raw | cnn | lstm | cnn_lstm | hybrid
    The physics rule is first on purpose: on the original dataset it scored
    0.9977 accuracy against the hybrid's ~0.918, and nobody noticed.

(3) EARLY STOPPING.  The write-up claimed "Early Stopping (15 epochs)"; the
    code ran a fixed 50-epoch loop.  Real patience-based early stopping on
    validation macro-F1, with best-checkpoint restore, is implemented here.

(4) NO STACKING LEAK.  The original trained XGBoost on CNN embeddings of the
    same days the CNN was fitted on, so the meta-learner saw unrealistically
    clean features.  The meta-classifier is now trained on *out-of-fold*
    embeddings.

(5) SCALING.  `Grid_Unbalance` was left unscaled (multiplied by 1000) while
    every other channel was standardised, so it silently dominated. All
    channels now share one train-fitted StandardScaler.

(6) CLASS WEIGHTS + PROPER METRICS.  75/8/8/8 imbalance was unaddressed and
    reported as accuracy.  Class-weighted loss, and macro-F1 / detection
    PR-AUC / conditional localisation accuracy are reported instead.

(7) MULTI-SEED.  Every result is mean +/- std over N seeds, not one run.

(8) ABLATIONS.  `--ablate` measures what each channel group actually buys --
    the experiment that tests the project's central "line currents are the
    breakthrough" claim instead of asserting it.

Usage:
    python phase3_train_and_benchmark.py --data data/alpha020 --seeds 5
    python phase3_train_and_benchmark.py --data data/alpha020 --ablate
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
import warnings
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, confusion_matrix, f1_score)
from sklearn.preprocessing import StandardScaler

from ntl_common import FeederConfig, physics_features
from phase2_audit_dataset import load_run, rule_baseline

warnings.filterwarnings("ignore")

# --- optional heavy dependencies, with honest fallbacks --------------------
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except Exception:                                    # pragma: no cover
    HAS_TORCH = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:                                    # pragma: no cover
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_XGB = False


class SafeGBM:
    """
    Gradient boosting with the class bookkeeping made safe.

    Two robustness fixes over the original: XGBoost requires contiguous labels
    0..K-1 and will raise if a node happens to have no theft days in the
    training split, and `predict_proba` then returns fewer columns than there
    are classes -- which silently corrupts any downstream `proba[:, 0]`.  This
    wrapper label-encodes on fit and expands probabilities back into the full
    class space on predict.
    """

    @staticmethod
    def xgb_kwargs(seed: int, n_classes: int) -> dict:
        """
        XGBoost objective depends on the number of classes, and getting it wrong
        is not a soft failure: `multi:softprob` on a two-class target leaves
        `num_class` unset and XGBoost aborts with "num_class should be greater
        equal to 1".  Binary targets arise whenever a caller fits one detector
        per node, so this is selected rather than hard-coded.
        """
        kw = dict(n_estimators=300, learning_rate=0.05, max_depth=4,
                  subsample=0.9, colsample_bytree=0.9, random_state=seed,
                  tree_method="hist")
        if n_classes <= 2:
            kw.update(objective="binary:logistic", eval_metric="logloss")
        else:
            kw.update(objective="multi:softprob", eval_metric="mlogloss")
        return kw

    def __init__(self, seed: int, n_classes: int):
        self.n_classes = n_classes
        self._const = None
        if HAS_XGB:
            self.m = XGBClassifier(**self.xgb_kwargs(seed, n_classes))
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier
            self.m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                    max_depth=4, random_state=seed)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            # a node with no positive example in this split -- every estimator
            # refuses a single-class target, so answer that class with certainty
            self._const = int(self.classes_[0])
            return self
        self._const = None
        self.m.fit(X, np.searchsorted(self.classes_, y))
        return self

    def predict_proba(self, X):
        full = np.zeros((len(X), self.n_classes))
        if self._const is not None:
            full[:, self._const] = 1.0
            return full
        full[:, self.classes_] = self.m.predict_proba(X)
        return full

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


def make_gbm(seed: int, n_classes: int):
    return SafeGBM(seed, n_classes)


CHANNEL_GROUPS = ["meters", "unbalance", "currents", "residuals", "voltage"]


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def build_windows(df: pd.DataFrame, daily: pd.DataFrame, cfg: FeederConfig,
                  groups: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    -> X (n_days, C, 24), y (n_days,), realization (n_days,), channel names.

    Meter reads that the AMI lost are interpolated here (the mask is kept as
    an input channel is deliberately NOT added, so the models cannot cheat by
    reading the outage pattern).
    """
    n = cfg.n_nodes
    feats = physics_features(df, cfg)
    cols: List[str] = []
    mats: List[np.ndarray] = []

    if "meters" in groups:
        m = df[[f"Meter_{j+1}_kW" for j in range(n)]].interpolate(limit_direction="both")
        mats.append(m.to_numpy()); cols += list(m.columns)
        sub = df[["Substation_kW"]].interpolate(limit_direction="both")
        mats.append(sub.to_numpy()); cols += ["Substation_kW"]
    if "unbalance" in groups:
        mats.append(feats[["Unbalance_kW"]].to_numpy()); cols += ["Unbalance_kW"]
    if "currents" in groups:
        live = [f"Line_{j+1}_A" for j in range(n)
                if not df[f"Line_{j+1}_A"].isna().all()]
        if live:
            c = df[live]
            mats.append(c.to_numpy()); cols += list(c.columns)
    if "residuals" in groups:
        live = [f"Resid_{j+1}" for j in range(n)
                if not feats[f"Resid_{j+1}"].isna().all()]
        if live:
            r = feats[live]
            mats.append(r.to_numpy()); cols += list(r.columns)
    if "voltage" in groups and f"VResid_1" in feats.columns:
        v = feats[[f"VResid_{j+1}" for j in range(n)]]
        mats.append(v.to_numpy()); cols += list(v.columns)

    M = np.nan_to_num(np.column_stack(mats))
    n_days = len(daily)
    X = M.reshape(n_days, 24, M.shape[1]).transpose(0, 2, 1)      # (days, C, 24)
    y = daily["Thief_ID"].to_numpy()
    return X, y, daily["realization"].to_numpy(), cols


def split_indices(y: np.ndarray, realization: np.ndarray, mode: str,
                  train_frac=0.70, val_frac=0.15):
    """Chronological split inside each realization, or hold out whole feeders."""
    tr, va, te = [], [], []
    if mode == "realization":
        rs = np.unique(realization)
        n_tr, n_va = int(len(rs) * train_frac), int(len(rs) * (train_frac + val_frac))
        for i, r in enumerate(rs):
            idx = np.where(realization == r)[0]
            (tr if i < n_tr else va if i < n_va else te).extend(idx)
    else:
        for r in np.unique(realization):
            idx = np.where(realization == r)[0]
            a, b = int(len(idx) * train_frac), int(len(idx) * (train_frac + val_frac))
            tr.extend(idx[:a]); va.extend(idx[a:b]); te.extend(idx[b:])
    return np.array(tr), np.array(va), np.array(te)


def fit_scaler(X: np.ndarray, idx: np.ndarray) -> StandardScaler:
    C = X.shape[1]
    sc = StandardScaler()
    sc.fit(X[idx].transpose(0, 2, 1).reshape(-1, C))
    return sc


def apply_scaler(X: np.ndarray, sc: StandardScaler) -> np.ndarray:
    C = X.shape[1]
    flat = sc.transform(X.transpose(0, 2, 1).reshape(-1, C))
    return flat.reshape(X.shape[0], X.shape[2], C).transpose(0, 2, 1)


def summary_features(X: np.ndarray) -> np.ndarray:
    """Daily aggregate features for the classical baselines."""
    return np.concatenate([X.mean(2), X.std(2), X.min(2), X.max(2),
                           X[:, :, 17:23].mean(2)], axis=1)      # + evening-peak mean


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def episode_metrics(y_true, y_pred, day, realization) -> Dict[str, float]:
    """
    Day-level scores overstate performance because days inside one theft
    episode are highly correlated -- 1,400 theft-days can come from only ~140
    independent events.  These are the operationally meaningful numbers:

      episode_recall  fraction of true episodes flagged at the right node on
                      at least one day
      latency_days    median days from episode onset to first correct flag
      false_alarm_rate  fraction of genuinely normal days flagged as theft
    """
    order = np.lexsort((day, realization))
    yt, yp, dd, rr = y_true[order], y_pred[order], day[order], realization[order]
    eps, cur = [], None
    for i in range(len(yt)):
        contiguous = (cur is not None and yt[i] == cur[0] and rr[i] == cur[3]
                      and dd[i] == cur[2] + 1)
        if yt[i] > 0 and contiguous:
            cur = (cur[0], cur[1], dd[i], cur[3])
        elif yt[i] > 0:
            if cur is not None:
                eps.append(cur)
            cur = (yt[i], dd[i], dd[i], rr[i])
        else:
            if cur is not None:
                eps.append(cur); cur = None
    if cur is not None:
        eps.append(cur)

    hits, lat = 0, []
    for node, d0, d1, r in eps:
        m = (rr == r) & (dd >= d0) & (dd <= d1)
        correct = np.where(yp[m] == node)[0]
        if len(correct):
            hits += 1
            lat.append(int(dd[m][correct[0]] - d0))
    normal = yt == 0
    return dict(
        n_episodes=len(eps),
        episode_recall=hits / len(eps) if eps else float("nan"),
        latency_days=float(np.median(lat)) if lat else float("nan"),
        false_alarm_rate=float((yp[normal] > 0).mean()) if normal.any() else float("nan"),
    )


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, score_theft: np.ndarray,
             n_classes: int) -> Dict[str, float]:
    det_t, det_p = (y_true > 0).astype(int), (y_pred > 0).astype(int)
    detected = (det_t == 1) & (det_p == 1)
    loc = float((y_pred[detected] == y_true[detected]).mean()) if detected.any() else float("nan")
    # Score only the classes that actually occur in this test split.  Including
    # a node with zero test days would contribute a spurious F1 of 0 and cap
    # macro-F1 at (n_classes-1)/n_classes no matter how good the model is.
    present = sorted(set(y_true.tolist()))
    return dict(
        accuracy=accuracy_score(y_true, y_pred),
        macro_f1=f1_score(y_true, y_pred, average="macro",
                          labels=present, zero_division=0),
        detect_f1=f1_score(det_t, det_p, zero_division=0),
        detect_pr_auc=average_precision_score(det_t, score_theft)
        if len(np.unique(det_t)) > 1 else float("nan"),
        localise_acc=loc,
    )


# ---------------------------------------------------------------------------
# Neural models
# ---------------------------------------------------------------------------
if HAS_TORCH:

    class CNNLSTM(nn.Module):
        """CNN feature extractor -> LSTM temporal encoder -> linear head.

        `mode` selects the ablated variants used in the baseline ladder:
        "cnn" (no LSTM), "lstm" (no CNN), "cnn_lstm" (both).
        """

        def __init__(self, in_ch: int, n_classes: int, mode: str = "cnn_lstm",
                     hidden: int = 64, p_drop: float = 0.3):
            super().__init__()
            self.mode = mode
            self.hidden = hidden
            if mode in ("cnn", "cnn_lstm"):
                self.conv = nn.Sequential(
                    nn.Conv1d(in_ch, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
                    nn.MaxPool1d(2),                                   # 24 -> 12
                    nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
                    nn.MaxPool1d(2),                                   # 12 -> 6
                )
            self.drop = nn.Dropout(p_drop)
            if mode == "cnn":
                self.head_in = 64 * 6
            elif mode == "lstm":
                self.lstm = nn.LSTM(in_ch, hidden, batch_first=True)
                self.head_in = hidden
            else:
                self.lstm = nn.LSTM(64, hidden, batch_first=True)
                self.head_in = hidden
            self.fc = nn.Linear(self.head_in, n_classes)

        def embed(self, x):
            if self.mode == "lstm":
                out, (hn, _) = self.lstm(x.permute(0, 2, 1))
                return hn[-1]
            z = self.conv(x)
            if self.mode == "cnn":
                return z.flatten(1)
            out, (hn, _) = self.lstm(z.permute(0, 2, 1))
            return hn[-1]

        def forward(self, x, extract_features: bool = False):
            f = self.drop(self.embed(x))
            return f if extract_features else self.fc(f)

    def train_nn(Xtr, ytr, Xva, yva, in_ch, n_classes, mode, seed,
                 epochs=200, patience=15, lr=1e-3, batch=64, verbose=False):
        """Class-weighted training with real early stopping on val macro-F1."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = CNNLSTM(in_ch, n_classes, mode)
        counts = np.bincount(ytr, minlength=n_classes).astype(float)
        w = counts.sum() / np.maximum(counts, 1.0)
        w = np.clip(w, 0.0, 20.0 * max(w[counts > 0].min(), 1e-9))   # cap absent classes
        w = torch.tensor(w / w.mean(), dtype=torch.float32)
        crit = nn.CrossEntropyLoss(weight=w)
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
        ytr_t = torch.tensor(ytr, dtype=torch.long)
        Xva_t = torch.tensor(Xva, dtype=torch.float32)
        loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=batch, shuffle=True)

        best_f1, best_state, bad = -1.0, copy.deepcopy(model.state_dict()), 0
        for ep in range(epochs):
            model.train()
            for xb, yb in loader:
                opt.zero_grad()
                crit(model(xb), yb).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                pv = model(Xva_t).argmax(1).numpy()
            f1 = f1_score(yva, pv, average="macro", labels=list(range(n_classes)),
                          zero_division=0)
            if f1 > best_f1 + 1e-4:
                best_f1, best_state, bad = f1, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    if verbose:
                        print(f"      early stop at epoch {ep+1}, best val macro-F1 {best_f1:.4f}")
                    break
        model.load_state_dict(best_state)          # restore best, not last
        model.eval()
        return model

    def nn_predict(model, X):
        with torch.no_grad():
            logits = model(torch.tensor(X, dtype=torch.float32))
            prob = torch.softmax(logits, 1).numpy()
        return prob.argmax(1), prob

    def nn_embed(model, X):
        with torch.no_grad():
            return model(torch.tensor(X, dtype=torch.float32),
                         extract_features=True).numpy()


# ---------------------------------------------------------------------------
# One full run
# ---------------------------------------------------------------------------
def run_once(df, daily, cfg, groups, seed, split_mode, models, oof_folds=3,
             verbose=False) -> Tuple[Dict[str, Dict[str, float]], np.ndarray, np.ndarray]:
    n_classes = cfg.n_nodes + 1
    X, y, real, cols = build_windows(df, daily, cfg, groups)
    keep = y >= 0                                   # drop multi-thief days in
    dayno = daily["day"].to_numpy()[keep]
    X, y, real = X[keep], y[keep], real[keep]       # single-label mode
    tr, va, te = split_indices(y, real, split_mode)

    sc = fit_scaler(X, tr)
    Xs = apply_scaler(X, sc)
    Ftr, Fva, Fte = (summary_features(Xs[i]) for i in (tr, va, te))
    res: Dict[str, Dict[str, float]] = {}
    cms: Dict[str, np.ndarray] = {}

    def record(name, pred, proba):
        score = 1.0 - proba[:, 0] if proba is not None else np.zeros(len(pred))
        res[name] = evaluate(y[te], pred, score, n_classes)
        res[name].update(episode_metrics(y[te], pred, dayno[te], real[te]))
        cms[name] = confusion_matrix(y[te], pred, labels=list(range(n_classes)))

    # --- B0: hand-written physics rule -------------------------------------
    if "rule" in models:
        pred_all, y_all, _, unb, _ = rule_baseline(df, daily, cfg)
        pred_all, unb = pred_all[keep], unb[keep]
        res["rule"] = evaluate(y[te], pred_all[te], unb[te], n_classes)
        res["rule"].update(episode_metrics(y[te], pred_all[te], dayno[te], real[te]))
        cms["rule"] = confusion_matrix(y[te], pred_all[te], labels=list(range(n_classes)))

    # --- B1/B2: classical models on daily summary features ------------------
    if "logreg" in models:
        m = LogisticRegression(max_iter=3000, class_weight="balanced",
                               random_state=seed)
        m.fit(Ftr, y[tr]); record("logreg", m.predict(Fte), m.predict_proba(Fte))
    if "rf" in models:
        m = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                   random_state=seed, n_jobs=-1)
        m.fit(Ftr, y[tr]); record("rf", m.predict(Fte), m.predict_proba(Fte))

    # --- B3: gradient boosting on the flattened raw window ------------------
    if "gbm_raw" in models:
        m = make_gbm(seed, n_classes)
        m.fit(Xs[tr].reshape(len(tr), -1), y[tr])
        Praw = m.predict_proba(Xs[te].reshape(len(te), -1))
        record("gbm_raw", Praw.argmax(1), Praw)

    # --- B4..B6: neural models ---------------------------------------------
    in_ch = Xs.shape[1]
    trained: Dict[str, object] = {}
    if HAS_TORCH:
        for name, mode in (("cnn", "cnn"), ("lstm", "lstm"), ("cnn_lstm", "cnn_lstm")):
            needed_by_stack = name == "cnn_lstm" and (
                "hybrid" in models or "hybrid_plus" in models)
            if name not in models and not needed_by_stack:
                continue
            mdl = train_nn(Xs[tr], y[tr], Xs[va], y[va], in_ch, n_classes,
                           mode, seed, verbose=verbose)
            trained[name] = mdl
            if name in models:
                p, prob = nn_predict(mdl, Xs[te]); record(name, p, prob)
    elif any(m in models for m in ("cnn", "lstm", "cnn_lstm", "hybrid", "hybrid_plus")):
        print("      [skip] PyTorch not installed -- neural models omitted from this run")

    # --- B7/B8: stacked hybrids, with OUT-OF-FOLD features for the meta-learner
    #
    #  hybrid       meta-classifier sees the 64-D embedding only (the original design)
    #  hybrid_plus  meta-classifier also sees the CNN-LSTM's own class probabilities,
    #               so stacking can only ADD to what the network already decided
    #               instead of throwing that decision away.  Run both to show the
    #               comparison is fair.
    want = [m for m in ("hybrid", "hybrid_plus") if m in models]
    if want and HAS_TORCH:
        H = trained["cnn_lstm"].head_in
        folds = np.array_split(np.arange(len(tr)), oof_folds)
        oof_e = np.zeros((len(tr), H), dtype=np.float32)
        oof_p = np.zeros((len(tr), n_classes), dtype=np.float32)
        for k in range(oof_folds):
            hold = folds[k]
            rest = np.concatenate([folds[j] for j in range(oof_folds) if j != k])
            m_k = train_nn(Xs[tr[rest]], y[tr[rest]], Xs[va], y[va],
                           in_ch, n_classes, "cnn_lstm", seed + 17 * k)
            oof_e[hold] = nn_embed(m_k, Xs[tr[hold]])
            oof_p[hold] = nn_predict(m_k, Xs[tr[hold]])[1]
        te_e = nn_embed(trained["cnn_lstm"], Xs[te])
        te_p = nn_predict(trained["cnn_lstm"], Xs[te])[1]
        if "hybrid" in models:
            meta = make_gbm(seed, n_classes).fit(oof_e, y[tr])
            P = meta.predict_proba(te_e); record("hybrid", P.argmax(1), P)
        if "hybrid_plus" in models:
            meta = make_gbm(seed, n_classes).fit(
                np.hstack([oof_e, oof_p]), y[tr])
            P = meta.predict_proba(np.hstack([te_e, te_p]))
            record("hybrid_plus", P.argmax(1), P)

    return res, y[te], cms


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/alpha020")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--split", choices=["time", "realization"], default="time")
    ap.add_argument("--models", nargs="+",
                    default=["rule", "logreg", "rf", "gbm_raw", "cnn", "lstm",
                             "cnn_lstm", "hybrid", "hybrid_plus"])
    ap.add_argument("--groups", nargs="+", default=CHANNEL_GROUPS)
    ap.add_argument("--oof-folds", type=int, default=3)
    ap.add_argument("--ablate", action="store_true",
                    help="run the hybrid over cumulative channel-group subsets")
    ap.add_argument("--use-sensor-lines", nargs="+", type=int, default=None,
                    help="pretend only these line segments carry a current sensor. "
                         "Masks the other Line_j_A columns at load time, so a single "
                         "simulated dataset can be re-used for any sensor placement "
                         "without re-running the power flow.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    df, daily, cfg = load_run(args.data)
    if args.use_sensor_lines:
        keep_lines = set(args.use_sensor_lines)
        for j in range(1, cfg.n_nodes + 1):
            if j not in keep_lines:
                df[f"Line_{j}_A"] = np.nan
        cfg.sensor_lines = sorted(keep_lines)
        print(f" [sensors] using only lines {sorted(keep_lines)} "
              f"of {cfg.n_nodes} -- other current channels masked")
    out_dir = args.out or os.path.join(args.data, "results")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 84)
    print(f" PHASE 3 BENCHMARK -- {args.data}")
    print(f" torch={'yes' if HAS_TORCH else 'NO'}  xgboost={'yes' if HAS_XGB else 'no (sklearn HistGBM)'}"
          f"  split={args.split}  seeds={args.seeds}")
    print("=" * 84)

    if args.ablate:
        subsets = [["meters"], ["meters", "unbalance"],
                   ["meters", "unbalance", "currents"],
                   ["meters", "unbalance", "currents", "residuals"]]
        rows = []
        for gs in subsets:
            # the rule baseline is excluded: it always uses the full physics
            # feature set by construction, so it is a fixed reference line
            accs = [run_once(df, daily, cfg, gs, s, args.split,
                             ["gbm_raw", "hybrid"], args.oof_folds)[0]
                    for s in range(args.seeds)]
            for model in accs[0]:
                vals = [a[model]["macro_f1"] for a in accs]
                rows.append(dict(channels="+".join(gs), model=model,
                                 macro_f1_mean=np.mean(vals), macro_f1_std=np.std(vals)))
                print(f"  {'+'.join(gs):40s} {model:10s} macro-F1 "
                      f"{np.mean(vals):.4f} +/- {np.std(vals):.4f}")
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "ablation.csv"), index=False)
        print(f"\nWritten to {out_dir}/ablation.csv")
        return

    all_runs: List[Dict[str, Dict[str, float]]] = []
    last_cms = {}
    for s in range(args.seeds):
        t0 = time.time()
        r, y_te, cms = run_once(df, daily, cfg, args.groups, s, args.split,
                                args.models, args.oof_folds, args.verbose)
        all_runs.append(r); last_cms = cms
        print(f"  seed {s}: " + " | ".join(f"{k} {v['macro_f1']:.3f}" for k, v in r.items())
              + f"   ({time.time()-t0:.0f}s)")

    metrics = ["accuracy", "macro_f1", "detect_f1", "detect_pr_auc", "localise_acc",
               "episode_recall", "latency_days", "false_alarm_rate"]
    rows = []
    for model in all_runs[0]:
        row = {"model": model}
        for m in metrics:
            v = [r[model][m] for r in all_runs]
            row[f"{m}_mean"], row[f"{m}_std"] = np.nanmean(v), np.nanstd(v)
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out_dir, "benchmark.csv"), index=False)

    print("\n" + "=" * 84)
    print(f" RESULTS  (mean +/- std over {args.seeds} seeds, test split)")
    print("=" * 84)
    print(f"{'model':<10}{'macro-F1':>15}{'detect-F1':>15}{'det PR-AUC':>15}"
          f"{'localise':>15}{'episode rec':>15}{'latency d':>11}{'false alarm':>13}")
    for _, r in table.iterrows():
        print(f"{r.model:<10}"
              f"{r.macro_f1_mean:>8.4f}+-{r.macro_f1_std:<5.3f}"
              f"{r.detect_f1_mean:>8.4f}+-{r.detect_f1_std:<5.3f}"
              f"{r.detect_pr_auc_mean:>8.4f}+-{r.detect_pr_auc_std:<5.3f}"
              f"{r.localise_acc_mean:>8.4f}+-{r.localise_acc_std:<5.3f}"
              f"{r.episode_recall_mean:>8.4f}+-{r.episode_recall_std:<5.3f}"
              f"{r.latency_days_mean:>10.1f}"
              f"{r.false_alarm_rate_mean:>13.4f}")

    best = table.sort_values("macro_f1_mean").iloc[-1]
    rule = table[table.model == "rule"]
    if len(rule) and best.model != "rule":
        gain = best.macro_f1_mean - rule.macro_f1_mean.iloc[0]
        print(f"\n  best model: {best.model}  (+{gain:+.4f} macro-F1 over the physics rule)")
        if gain <= 0:
            print("  [WARN] no learned model beats the hand-written rule -- do not publish this as-is.")

    if last_cms:
        name = best.model
        print(f"\n Confusion matrix, {name} (last seed).  rows = actual, cols = predicted")
        labs = ["Normal"] + [f"Node {i}" for i in range(1, cfg.n_nodes + 1)]
        print(f"{'':<10}" + "".join(f"{l:>9}" for l in labs))
        for i, l in enumerate(labs):
            print(f"{l:<10}" + "".join(f"{v:>9}" for v in last_cms[name][i]))

    print(f"\nWritten to {out_dir}/benchmark.csv")


if __name__ == "__main__":
    main()
