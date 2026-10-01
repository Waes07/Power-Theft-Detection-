# Data dictionary

Every file the pipeline reads and writes, and what each row and column means.
Column counts below assume the default 5-node feeder; an N-node feeder produces
N of each per-node column.

```
block1/2/3.csv ──► lcl_hourly_wide.csv ──► phase 1 ──► dataset_hourly.csv
                        (cache)                        labels_daily.csv
                                                       episodes.csv
                                                       config.json + meta.json
                                                            │
                                                       phase 2 (audit, prints only)
                                                       phase 3 ──► benchmark.csv
                                                                   ablation.csv
                                                       phase 4 ──► severity_sweep.csv + .png
```

---

# INPUTS

## 1. `block1.csv`, `block2.csv`, `block3.csv` — raw source data

Smart-meter readings from the **Low Carbon London** trial (UK Power Networks,
2011–2014). Not SGCC. About 70 MB each; 50 households per file, stacked one
after another.

**One row = one meter reading from one household at one half-hour.**

| Column | Meaning |
|---|---|
| `LCLid` | Household identifier, e.g. `MAC000323`. **The file is sorted by this**, so 50 different homes sit end-to-end in one file. |
| `tstp` | Timestamp of the reading, half-hourly |
| `energy(kWh/hh)` | Energy consumed in that half hour, in kWh |

> This is where the original code went wrong: it read the first 48,000 rows and
> treated them as one household. Those rows span several different homes spliced
> together. The loader now groups by `LCLid`.

## 2. `lcl_hourly_wide.csv` — the cached load matrix

Built once from the three block files, then reused. **This is the only input file
you need to keep.** ~8.8 MB.

**One row = one hour. One column = one household.**

```
h                    MAC000323  MAC000386  MAC000535  ...
2012-10-10 00:00:00      0.152      0.311      0.088   ...
2012-10-10 01:00:00      0.140      0.298      0.077   ...
```

| | |
|---|---|
| Shape | 11,640 rows × 126 columns |
| Rows | 11,640 hours = **485 complete days**, 2012-10-10 → 2014-02-12 |
| Columns | 126 households that had ≥98% coverage over that window |
| Values | kWh consumed in that hour (= kW average). Range 0 – 18.0 |

Two half-hour readings are summed to make one hour. Households below 98%
coverage, and dates where fewer than 90% of households reported, are dropped —
that filtering is what reduces 150 households to 126 and the full 2011–2014 span
to 485 dense days.

---

# OUTPUTS OF PHASE 1 (the simulator)

## 3. `dataset_hourly.csv` — the main simulated dataset

**One row = one hour on one simulated feeder.** ~30 MB.

| | |
|---|---|
| Shape | 93,120 rows × 29 columns |
| Rows | 485 days × 24 hours × 8 realizations = 93,120 |

### Columns the models are allowed to use

| Column | Unit | Meaning |
|---|---|---|
| `timestamp` | — | Real calendar time from the London data |
| `day` | int | Day index 0–484 **within its realization** |
| `hour` | int | 0–23 |
| `Substation_kW` | kW | Power the substation delivered, **as measured** (0.2% error) |
| `Meter_1_kW` … `Meter_5_kW` | kW | What each node's meter reported. Carries 0.5% random error, a 0.3% fixed per-meter bias, and **`NaN` where the reading was lost** (2% of the time) |
| `Line_1_A` … `Line_5_A` | amperes | Current in each line segment, with 1% CT error. **A whole column is `NaN` if that segment has no sensor.** Line *j* carries every node from *j* downstream. |

### Columns for bookkeeping and evaluation only — never model inputs

| Column | Meaning |
|---|---|
| `missing_1` … `missing_5` | 1 if that meter reading was lost this hour |
| `true_theft_1` … `true_theft_5` | **Ground truth.** 1 if node *j* was stealing in this hour |
| `Substation_kW_true` | Substation power before measurement error |
| `tech_loss_kW` | Genuine I²R loss this hour — used to verify the physics |
| `vm_pu_min` | Lowest bus voltage, per unit. Must stay above 0.90 |
| `max_loading_pct` | Highest line loading. Must stay below 100 |
| `realization` | Which of the 8 feeder versions this row belongs to |

*(With `--report-voltage` the file also carries `Meter_1_V` … `Meter_5_V`,
the phase voltage each meter measured, with 0.5% error.)*

## 4. `labels_daily.csv` — the answer key

**One row = one day on one feeder.** This is what phase 3 trains against.

| | |
|---|---|
| Shape | 3,880 rows × 14 columns |
| Rows | 485 days × 8 realizations |

| Column | Meaning |
|---|---|
| `day` | 0–484, matches `dataset_hourly.csv` |
| `theft_1` … `theft_5` | 1 if node *j* stole at **any** hour that day |
| `attack_1` … `attack_5` | Which method: `hook`, `tamper`, `intermittent`, or blank |
| `Thief_ID` | **The training label.** 0 = normal, 1–5 = the guilty node, −1 = several thieves at once |
| `n_thieves` | How many nodes were stealing that day |
| `realization` | Which feeder version |

> `Thief_ID` is derived from the **whole day**, not from hour 0. That is the fix
> for the `[::24]` bug — and phase 2 verifies it every run.

## 5. `episodes.csv` — the theft events

**One row = one theft episode.** Used for episode-level metrics and for the paper's
description of the threat model.

| | |
|---|---|
| Shape | 143 rows × 6 columns (~17 episodes per realization) |

| Column | Meaning |
|---|---|
| `node` | Which node stole, 1–5 |
| `attack` | `hook` (unmetered load added), `tamper` (meter under-reports), `intermittent` (peak-hours only) |
| `start_day`, `end_day` | The span. Duration is log-normal, mean ~9 days, clipped to 3–45 |
| `severity` | α for this episode — theft as a fraction of that node's own mean load |
| `realization` | Which feeder version |

## 6. `config.json` — every setting used

The full `FeederConfig` for the run: nodes, voltage, cable, segment length,
households per node, theft severity, all noise levels, sensor coverage, and the
random seed. **This is what makes the run reproducible** — same config, same seed,
identical dataset.

## 7. `meta.json` — what actually happened

One record per realization, for verification rather than modelling:

| Field | Meaning |
|---|---|
| `backend` | `pandapower` or `bfs` — which solver ran |
| `sensors` | Which line segments got a current sensor |
| `households` | **The exact LCLids assigned to each node** |
| `mean_feeder_kw` | Average total load |
| `mean_tech_loss_pct` | Technical loss — should be 3–10% |
| `min_vm_pu` | Lowest voltage — should be above 0.90 |
| `max_loading_pct` | Highest loading — should be below 100 |
| `nonconverged_hours` | Power-flow failures — should be 0 |
| `true_lengths_km`, `true_pf` | The *true* line lengths and power factors, which differ from the nominal values the feature extractor assumes |

---

# OUTPUTS OF PHASE 3 (the benchmark)

## 8. `benchmark.csv` — the results table

**One row = one model.** Every metric appears twice: `_mean` and `_std` across seeds.

| Metric | What it answers |
|---|---|
| `accuracy` | Fraction of days exactly right. **Inflated — 64% of days are normal.** |
| `macro_f1` | All classes weighted equally. The honest headline number. |
| `detect_f1` | Ignoring *which* node — was theft noticed at all? |
| `detect_pr_auc` | How well suspicious days are ranked |
| `localise_acc` | Of correctly flagged theft days, how often the right node was named |
| `episode_recall` | Fraction of theft *events* caught at the right node at least once |
| `latency_days` | Median days from episode start to first correct flag |
| `false_alarm_rate` | Fraction of innocent days that accused someone |

## 9. `ablation.csv` — what each channel group buys

Written by `--ablate`. Columns: `channels`, `model`, `macro_f1_mean`, `macro_f1_std`.
One row per (feature set × model).

---

# OUTPUTS OF PHASE 4 (the sweep)

## 10. `severity_sweep.csv`

**One row = one (sensor coverage × theft severity × model).**
Columns: `coverage`, `alpha`, `model`, `macro_f1`, `macro_f1_std`,
`episode_recall`, `false_alarm`.

## 11. `severity_sweep.png`, `sensor_coverage.png` / `.pdf`

The figures. Use the **PDF** in the paper — it stays sharp at any print size.

---

# Quick reference

| File | Rows mean | Size | Keep it? |
|---|---|---|---|
| `block*.csv` | one meter reading | 200 MB | No — the cache replaces them |
| `lcl_hourly_wide.csv` | one hour, all households | 8.8 MB | **Yes** — the only input needed |
| `dataset_hourly.csv` | one hour, one feeder | 30 MB | No — regenerates in ~5 min |
| `labels_daily.csv` | one day, one feeder | 110 KB | Yes |
| `episodes.csv` | one theft event | 6 KB | Yes |
| `config.json` / `meta.json` | run settings / outcome | small | **Yes** — reproducibility |
| `benchmark.csv` | one model | small | **Yes** — your results |
