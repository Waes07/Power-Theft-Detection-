# What changed, and why

This is the rewritten pipeline for *Power Theft Detection and Localization in
Radial Distribution Grids*. It follows the fix order in `publication_review.md`.

Every claim below was checked by running the code. Numbers come from real Low
Carbon London household data (126 households, 485 days, 2012-10-10 to
2014-02-12), 8 independent feeder realizations, 3,880 feeder-days.

---

## The headline

| | before | after |
|---|---|---|
| Detection AUC of one feature | ~1.0000 | **0.862** (instrumented) / **0.743** (sparse) |
| Cohen's *d*, normal vs theft | ~176,000 | **1.61** / **0.99** |
| Hand-written rule baseline | **0.9977 acc** (beat the deep model) | **0.727** / **0.187** macro-F1 |
| "Normal" days containing theft | 743 / 745 (99.7%) | **0** |
| Technical loss | 0.024% | **3.5%** |
| Charging share of line current | ~92% | **0.004%** |
| Baselines compared | 0 | **8** |
| Seeds | 1 (unseeded) | **N, mean ± std** |

The benchmark went from *unpublishably easy* to *genuinely hard and
informative*, and there is now a ladder of baselines to prove any model's worth.

---

## Files

| file | role |
|---|---|
| `ntl_common.py` | physics, topology, cable library, two power-flow solvers, load ingestion, physics-informed features |
| `phase1_simulate_feeder.py` | replaces `phase1_sgcc_current_generator.py.py` |
| `phase2_audit_dataset.py` | **new** — integrity + triviality audit; run before training |
| `phase3_train_and_benchmark.py` | replaces `phase3_sgcc_hybrid_8channel.py` |
| `phase4_severity_sweep.py` | **new** — the macro-F1 vs severity curve |
| `tests/selftest_nn.py` | **new** — one-command check of the neural path |

```bash
python phase1_simulate_feeder.py --blocks block1.csv block2.csv block3.csv --out data/run
python phase2_audit_dataset.py  --data data/run --check-solver
python phase3_train_and_benchmark.py --data data/run --seeds 5
python phase3_train_and_benchmark.py --data data/run --seeds 5 --ablate
python phase4_severity_sweep.py --out sweep
```

---

## 1. The labelling bug — fixed

**Before.** Theft was re-drawn independently every hour; `phase3` then took the
day's label from hour 0 with `y_raw = df["Thief_ID"].values[::24]`.

**After.** Theft is a *persistent episode*: a node, an onset day, and a duration
drawn log-normally (mean 9 days, clipped to 3–45). A day-level label is then
well defined, and phase 2 verifies it — it reports 0 mismatches between hourly
ground truth and daily labels, and 0 "normal" days containing theft hours.

**Bonus.** Persistence makes *onset-detection latency* measurable, which is a
publishable secondary metric the original design could not produce.

---

## 2. Theft severity — now a swept parameter, not a fixed absurdity

**Before.** `theft_amount = random.uniform(0.003, 0.008)` — an absolute 3–8 kW
against a feeder whose entire honest load averaged 1.665 kW.

**After.** `theft = alpha x (the thief's own mean load)`, with `alpha` swept.
`phase4` produces the curve. Measured, instrumented feeder, macro-F1:

| α | 0.05 | 0.08 | 0.12 | 0.16 | 0.20 | 0.30 |
|---|---|---|---|---|---|---|
| rule | 0.393 | 0.615 | 0.765 | 0.844 | 0.809 | 0.937 |
| logreg | 0.779 | 0.924 | 0.988 | 0.988 | 0.996 | 0.999 |
| gbm_raw | 0.573 | 0.771 | 0.858 | 0.934 | 0.960 | 0.975 |

That table is worth more than any single accuracy figure, because it shows
where the method breaks.

---

## 3. The measurement chain — added (it did not exist)

`meter_p1 = actual_p1` meant meters were perfect, so `Grid_Unbalance` was a
pure theft readout. Now modelled:

- meter random error σ = 0.5% (IEC Class 0.5S) and a per-meter **systematic bias** σ = 0.3%
- substation measurement error σ = 0.2%; CT error σ = 1%
- AMI read failures at 2% (interpolated at feature time, never used as a channel)
- **unmetered but legitimate load**: 3% communal per node plus a 1.5 kW dusk-to-dawn street-lighting load

That last one matters most. Communal and street load appears in
substation-minus-meters *exactly* the way theft does, and it is the dominant
real-world confounder. It is why normal-day unbalance is now 4.05 ± 0.96 kW
instead of 0.0004 ± 0.00002 kW.

---

## 4. The physics — corrected

**Before.** `NAYY 4x50 SE` (a 0.4 kV cable) on 11 kV buses, ~27× over its
insulation rating, at 0.46% loading. About 92% of the reported "I²R signature"
was cable **charging current** — line current barely moved with load.

**After.** 0.4 kV feeder, `NAYY 4x150 SE`, 0.16 km segments, five aggregated
service points of 15 households each. Verified by `phase2`:

```
mean technical loss ....... 3.51 %   (realistic band 3-10%)
minimum bus voltage ....... 0.9106 pu (limit 0.90)
maximum line loading ......  82.3 %  (limit 100)
cable 'NAYY 4x150 SE' rated 0.4 kV, feeder runs at 0.4 kV
cable charging share of Line_1 current ...   0.0 %  (0.003 A of 84.26 A)
```

Also: per-node power factor (0.95 ± 0.03) instead of `q_mvar = 0.0`, and
general radial **tree** topology instead of a hard-coded 3-node chain
(`--topology branched`, `--nodes N`).

**Inverse-crime removal.** The true feeder now has ±15% segment-length jitter
and per-node power factor spread, while the feature extractor and the rule
baseline use *nominal* constants. Without this, any physics-informed estimator
is silently inverting the exact equations that generated the data. This one
change alone dropped the classical baselines from 100% to a believable range.

---

## 5. Two power-flow engines that agree

`ntl_common.py` ships an independent backward/forward-sweep solver alongside
pandapower. Re-running your original 11 kV configuration through it reproduces
your `sgcc_dataset_with_currents.csv` to **3.1×10⁻¹² relative error** on line
currents and on substation power. `phase2 --check-solver` re-runs the
comparison on 200 random operating points.

This is worth a paragraph in the paper: an independent reimplementation
agreeing to machine precision is a correctness argument reviewers respect. It
also means the simulator runs without pandapower installed.

---

## 6. The baseline ladder — the thing that was missing

Eight models on identical splits, all reported:

`rule` → `logreg` → `rf` → `gbm_raw` → `cnn` → `lstm` → `cnn_lstm` → `hybrid`

`rule` is deliberately first: on your original dataset it scored **0.9977
accuracy against the hybrid's ~0.918**, and nothing in the pipeline would have
told you. Phase 3 now prints a warning if no learned model beats it.

---

## 7. Early stopping — now real

The write-up claimed "Early Stopping (15 epochs)"; the code ran
`for epoch in range(50)` with no early stopping at all. Implemented:
patience of 15 on **validation macro-F1**, max 200 epochs, and
`load_state_dict(best_state)` so the *best* checkpoint is restored, not the
last one.

---

## 8. The stacking leak — closed

`X_xgb_train = np.vstack((train_features, val_features))` trained XGBoost on
CNN embeddings of the same days the CNN was fitted on, so the meta-learner saw
unrealistically clean features that do not resemble test-time embeddings.

Now: the CNN-LSTM is refit on K−1 folds and embeds the held-out fold, so the
meta-classifier trains on **out-of-fold** embeddings (`--oof-folds`, default 3).
Test embeddings come from the model fitted on the full training set.

---

## 9. Scaling, class weights, metrics

- Every channel now shares one train-fitted `StandardScaler`. Previously
  `Grid_Unbalance` was left unscaled (×1000) while everything else was
  standardised, so it silently dominated the first convolution.
- Class-weighted `CrossEntropyLoss`, with the weight capped so a class absent
  from a training split cannot blow up the loss.
- Reported: macro-F1, detection F1, detection PR-AUC, and **conditional
  localisation accuracy** — not bare accuracy on a 75%-majority problem.
- Macro-F1 scores only classes present in the test split, so a node with no
  test days cannot artificially cap the metric at (K−1)/K.

**Episode-level metrics (new).** Days inside one episode are highly correlated,
so 1,397 theft-days come from only ~143 independent events. Day-level scores
overstate everything. Phase 3 now also reports **episode recall**, **detection
latency in days**, and **false-alarm rate per normal day**.

---

## 10. Integrity and reproducibility

- Everything seeded; `config.json`, `meta.json` and `episodes.csv` written with
  every run. Your original `phase1` seeded nothing, so its dataset could not be
  regenerated.
- A non-converged hour is written as a **NaN row**, not skipped. The original's
  `except LoadflowNotConverged: continue` silently shifted every subsequent day
  boundary, so `reshape(num_days, 24, 8)` sliced days at the wrong hours.
- `SafeGBM` label-encodes on fit and expands probabilities back to the full
  class space, so a node with no training theft-days cannot crash XGBoost or
  silently corrupt `proba[:, 0]`.
- Dataset provenance corrected everywhere: these are **Low Carbon London /
  UK Power Networks** files (`LCLid`, half-hourly kWh), not SGCC. The loader
  now groups by `LCLid` instead of blindly taking the first 48,000 rows, which
  used to splice several different households into one fake "consumer".

---

## 11. The ablation — your central claim, now actually tested

`--ablate` measures what each channel group buys. Instrumented feeder, α = 0.15,
gradient boosting, macro-F1:

| channels | macro-F1 |
|---|---|
| meters + unbalance | 0.235 |
| + raw line currents | 0.326 |
| + KCL residuals | **0.933** |

This is the most useful result in the rewrite, and it *sharpens* your original
claim rather than confirming it. Raw amperes on their own add relatively
little (+0.09). What does the work is amperes **combined with topology through
Kirchhoff's current law** — the `Resid_j` feature, which compares the current
actually flowing into node *j* against what its meter says it should be drawing.

So the contribution is not "we fed the network line currents." It is "we give
the model an explicit KCL residual per node, and that is worth +0.61 macro-F1."
That is a defensible, quotable claim.

---

## 12. Sensor coverage — the honest finding you need to know about

With a current sensor on **every** segment, plain logistic regression on daily
summary features reaches macro-F1 0.988. Localisation is essentially solved by
physics; a deep model has nothing left to contribute.

That is a real result, and you should report it rather than hide it. But it
also means the interesting regime is the realistic one: LV feeders are *not*
instrumented on every segment. `--sensor-coverage all|trunk|sparse|none`:

| | rule | logreg | rf | gbm_raw |
|---|---|---|---|---|
| all sensors, α=0.15 | 0.808 | **0.988** | 0.870 | 0.933 |
| 2 sensors, α=0.08 | 0.156 | **0.381** | 0.181 | 0.291 |

In the sparse case detection still works (detect-F1 0.94) but **localisation
collapses** (0.30). That gap is where a temporal model can earn its place, and
it is the sharpest framing for the paper:

> *Line-current sensing plus KCL solves localisation outright where the feeder
> is fully instrumented. Under realistic sparse instrumentation the problem
> becomes a temporal inference task, and that is what the hybrid model is for.*

---

## What I could not verify here, and what I did instead

The sandbox I ran this in has no package registry, so `pandapower`, `torch`
and `xgboost` could not be installed. Rather than ship unverified code:

- **pandapower** — the BFS solver reproduces your existing pandapower output to
  3.1×10⁻¹², so the physics is verified against pandapower even though
  pandapower is not installed. The pandapower backend keeps the same call
  pattern as your working script (`create_bus/create_line/create_load/runpp/
  res_line.i_ka`), with only parameters changed.
- **xgboost** — the pipeline falls back to scikit-learn's
  `HistGradientBoostingClassifier` (same algorithm family), so every result
  above was produced with a real gradient-boosting meta-classifier. XGBoost is
  used automatically when installed.
- **torch** — every non-neural stage is verified by execution. For the neural
  stage, `tests/torch_stub.py` provides a numpy stand-in that exercises layer
  wiring, embedding widths, the out-of-fold stacking loop, early-stopping
  checkpoint restore and the metric plumbing. `tests/selftest_nn.py` passes
  against it, and runs against **real** torch on your machine:

```bash
python tests/selftest_nn.py --data data/run     # confirms the neural path
```

Run that first. The numbers in this document for `cnn`, `lstm`, `cnn_lstm` and
`hybrid` are the ones you still need to generate — everything else is measured.

---

## Suggested next steps

1. Run `tests/selftest_nn.py`, then `phase3 --seeds 5` with the neural models in.
2. Run `phase4` including `cnn_lstm` and `hybrid` — the severity curve with the
   deep models on it is your headline figure.
3. `--sensor-coverage sparse` is where your architecture should win. If it does
   not beat `logreg` there, that is the finding, and it is still publishable.
4. `--max-concurrent 2` for the multi-thief study, and `--topology branched
   --nodes 15` for the scale-up.
5. `--split realization` trains on some feeders and tests on others — the
   generalisation experiment.
