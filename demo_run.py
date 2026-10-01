"""
demo_run.py -- a short live demonstration (about 1 minute on a laptop CPU).

    python demo_run.py              # CNN+LSTM, phase 3 and phase 5, printed in the terminal
    python demo_run.py --plot       # same, and also opens the two confusion-matrix pictures
    python demo_run.py --model logreg   # backup: no PyTorch needed, runs in seconds

It uses the project's own code -- the same data loading, windowing, split, scaler,
CNN+LSTM model and training recipe as phase3_train_and_benchmark.py and
phase5_multi_thief.py.  Only ONE training seed is run, so it is fast.

PHASE 3  one thief at a time (data/run)  -> 6 classes: Normal, Node 1..5
PHASE 5  two thieves can steal the same day (data/mt2) -> one yes/no per node
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")

import phase3_train_and_benchmark as P3          # noqa: E402
from phase2_audit_dataset import load_run         # noqa: E402

LABELS = ["Normal", "Node 1", "Node 2", "Node 3", "Node 4", "Node 5"]
LINE = "=" * 72


def banner(t):
    print("\n" + LINE + "\n " + t + "\n" + LINE, flush=True)


def show_matrix(M, labels, rows_title="truth", cols_title="model said"):
    w = max(len(x) for x in labels) + 2
    print(f"   rows = {rows_title},  columns = {cols_title}")
    print(" " * (w + 3) + "".join(f"{l:>9}" for l in labels))
    for i, l in enumerate(labels):
        cells = "".join((f"[{v:>6}]" if i == j else f"{v:>8}") + " " for j, v in enumerate(M[i]))
        print(f"   {l:<{w}}" + cells)


# ---------------------------------------------------------------------------
def phase3(model: str, seed: int):
    banner(f"PHASE 3  --  one thief at a time  ({model}, seed {seed})")
    t0 = time.time()
    df, daily, cfg = load_run(os.path.join(HERE, "data", "run"))
    X, y, real, cols = P3.build_windows(df, daily, cfg, P3.CHANNEL_GROUPS)
    keep = y >= 0
    X, y, real = X[keep], y[keep], real[keep]
    tr, va, te = P3.split_indices(y, real, "time")
    Xs = P3.apply_scaler(X, P3.fit_scaler(X, tr))
    print(f"   one day = {X.shape[1]} channels x {X.shape[2]} hours")
    print(f"   days: train {len(tr)}  |  validation {len(va)}  |  test {len(te)} (never seen)")

    if model == "cnn_lstm":
        if not P3.HAS_TORCH:
            sys.exit("   PyTorch not found -- run:  python demo_run.py --model logreg")
        from final_results import train_logged, n_params
        print("   training CNN+LSTM ...", flush=True)
        mdl, curve, best = train_logged(Xs[tr], y[tr], Xs[va], y[va], Xs.shape[1], 6,
                                        "cnn_lstm", seed)
        print(f"   {n_params(mdl):,} weights  |  best epoch {best}, stopped at epoch "
              f"{len(curve)} (early stopping)")
        pred, _ = P3.nn_predict(mdl, Xs[te])
    else:
        F = P3.summary_features
        m = LogisticRegression(max_iter=3000, class_weight="balanced",
                               random_state=seed).fit(F(Xs[tr]), y[tr])
        pred = m.predict(F(Xs[te]))

    yt = y[te]
    M = confusion_matrix(yt, pred, labels=list(range(6)))
    clean = yt == 0
    fa = float((pred[clean] != 0).mean())
    print(f"\n   ACCURACY       {accuracy_score(yt, pred):.4f}   "
          f"({int(np.trace(M))} of {len(yt)} test days right)")
    print(f"   MACRO-F1       {f1_score(yt, pred, average='macro'):.4f}")
    print(f"   FALSE ALARMS   {fa*100:.2f} % of clean days\n")
    show_matrix(M, LABELS)
    print("\n   per-class results:")
    print("   " + classification_report(yt, pred, labels=list(range(6)), target_names=LABELS,
                                        digits=3, zero_division=0).replace("\n", "\n   "))
    print(f"   time: {time.time()-t0:.0f} s")
    return M


# ---------------------------------------------------------------------------
def phase5(model: str, seed: int):
    banner(f"PHASE 5  --  two thieves can steal the same day  ({model}, seed {seed})")
    t0 = time.time()
    df, daily, cfg = load_run(os.path.join(HERE, "data", "mt2"))
    n = cfg.n_nodes
    X, y1, real, _ = P3.build_windows(df, daily, cfg, P3.CHANNEL_GROUPS)
    Y = daily[[f"theft_{j+1}" for j in range(n)]].to_numpy().astype(int)
    tr, va, te = P3.split_indices(y1, real, "time")
    Xs = P3.apply_scaler(X, P3.fit_scaler(X, tr))
    two = Y[te].sum(1) >= 2
    print(f"   test days {len(te)}, of which {int(two.sum())} have TWO thieves at once")
    print("   output: one yes/no (sigmoid) per node instead of one softmax")

    if model == "cnn_lstm":
        import phase5_multi_thief as P5
        if not getattr(P5, "HAS_TORCH", False):
            sys.exit("   PyTorch not found -- run:  python demo_run.py --model logreg")
        print("   training CNN+LSTM (multi-label) ...", flush=True)
        mdl = P5.train_multilabel(Xs[tr], Y[tr], Xs[va], Y[va], Xs.shape[1], n,
                                  "cnn_lstm", seed)
        P = (P5.predict_ml(mdl, Xs[te]) > 0.5).astype(int)
    else:
        F = P3.summary_features
        P = np.zeros((len(te), n), dtype=int)
        for j in range(n):
            P[:, j] = LogisticRegression(max_iter=4000, class_weight="balanced",
                                         random_state=seed).fit(F(Xs[tr]), Y[tr, j]).predict(F(Xs[te]))
    T = Y[te]
    print()
    mats = []
    for j in range(n):
        M = confusion_matrix(T[:, j], P[:, j], labels=[0, 1]); mats.append(M)
        acc = (M[0, 0] + M[1, 1]) / M.sum()
        f1 = f1_score(T[:, j], P[:, j], zero_division=0)
        print(f"   Node {j+1}:  truly clean -> said clean {M[0,0]:>4}  said theft {M[0,1]:>3}   |   "
              f"truly theft -> said clean {M[1,0]:>3}  said theft {M[1,1]:>4}   "
              f"acc {acc:.4f}  F1 {f1:.3f}")
    exact = float((P == T).all(1).mean())
    both = int(((P[two] == T[two]).all(1)).sum())
    print(f"\n   PER-NODE F1 (macro)            {f1_score(T, P, average='macro', zero_division=0):.4f}")
    print(f"   EVERY NODE RIGHT ON A DAY       {exact:.4f}   ({int(exact*len(te))} of {len(te)} days)")
    print(f"   TWO-THIEF DAYS FULLY RIGHT      {both} of {int(two.sum())}")
    cnt = confusion_matrix(T.sum(1), P.sum(1).clip(0, 2), labels=[0, 1, 2])
    print("\n   how many thieves were stealing (rows) vs how many the model found (columns):")
    show_matrix(cnt, ["0", "1", "2"], "thieves stealing", "thieves found")
    print(f"\n   time: {time.time()-t0:.0f} s")
    return mats, cnt


# ---------------------------------------------------------------------------
def plot(M3, mats5, cnt5, model):
    import matplotlib.pyplot as plt
    out = os.path.join(HERE, "demo_out"); os.makedirs(out, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    ax.imshow(M3 / np.maximum(M3.sum(1, keepdims=True), 1), cmap="Blues", vmin=0, vmax=1)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, M3[i, j], ha="center", va="center", fontsize=13,
                    color="white" if i == j and M3[i, j] else "black")
    ax.set_xticks(range(6)); ax.set_xticklabels(LABELS)
    ax.set_yticks(range(6)); ax.set_yticklabels(LABELS)
    ax.set_xlabel("model said"); ax.set_ylabel("truth")
    ax.set_title(f"Phase 3 -- {model}: {int(np.trace(M3))} of {int(M3.sum())} test days right")
    fig.tight_layout(); f3 = os.path.join(out, "phase3_confusion.png"); fig.savefig(f3, dpi=150)
    fig, axs = plt.subplots(1, 6, figsize=(15, 3.2))
    for j, M in enumerate(mats5):
        axs[j].imshow(M / np.maximum(M.sum(1, keepdims=True), 1), cmap="Blues", vmin=0, vmax=1)
        for a in range(2):
            for b in range(2):
                axs[j].text(b, a, M[a, b], ha="center", va="center", fontsize=12,
                            color="white" if a == b and M[a, b] else "black")
        axs[j].set_xticks([0, 1]); axs[j].set_xticklabels(["clean", "theft"])
        axs[j].set_yticks([0, 1]); axs[j].set_yticklabels(["clean", "theft"])
        axs[j].set_title(f"Node {j+1}")
    axs[5].imshow(cnt5 / np.maximum(cnt5.sum(1, keepdims=True), 1), cmap="Blues", vmin=0, vmax=1)
    for a in range(3):
        for b in range(3):
            axs[5].text(b, a, cnt5[a, b], ha="center", va="center", fontsize=12,
                        color="white" if a == b and cnt5[a, b] else "black")
    axs[5].set_title("thieves found"); axs[5].set_xticks(range(3)); axs[5].set_yticks(range(3))
    fig.suptitle(f"Phase 5 -- {model}, two thieves allowed"); fig.tight_layout()
    f5 = os.path.join(out, "phase5_confusion.png"); fig.savefig(f5, dpi=150)
    print(f"\n   pictures saved in {out}")
    for f in (f3, f5):
        try:
            os.startfile(f)                    # opens on Windows
        except Exception:
            pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["cnn_lstm", "logreg"], default="cnn_lstm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", choices=["3", "5"], default=None)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    M3 = phase3(a.model, a.seed) if a.only != "5" else None
    m5 = phase5(a.model, a.seed) if a.only != "3" else None
    if a.plot and M3 is not None and m5 is not None:
        plot(M3, m5[0], m5[1], a.model)
    print(f"\nTotal time: {time.time()-t0:.0f} s")
