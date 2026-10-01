# NTL Detection and Localisation on a Radial LV Feeder

Rewritten pipeline. See `METHOD_CHANGES.md` for what changed from the original
three-script version and why, and `publication_review.md` for the review that
motivated it.

## Requirements

```
numpy pandas scikit-learn matplotlib
pandapower   # optional -- an equivalent BFS solver ships in ntl_common.py
torch        # required for the cnn / lstm / cnn_lstm / hybrid models
xgboost      # optional -- falls back to sklearn HistGradientBoostingClassifier
```

## Data

Put the three Low Carbon London block files (`block1.csv`, `block2.csv`,
`block3.csv`) next to the scripts. The first run builds and caches an hourly
wide matrix in `lcl_hourly_wide.csv`; later runs reuse it.

These are UK Power Networks / Low Carbon London files (`LCLid`, half-hourly
kWh), **not** SGCC. If you also want a real-data validation section, download
the actual SGCC set separately — it has genuine theft labels but no topology,
so it can test detection and not localisation.

## Pipeline

```bash
# 1. simulate
python phase1_simulate_feeder.py --blocks block1.csv block2.csv block3.csv \
    --out data/run --severity 0.15 --realizations 8

# 2. audit BEFORE training -- this is the step that catches a broken benchmark
python phase2_audit_dataset.py --data data/run --check-solver

# 3. benchmark ladder + ablation
python phase3_train_and_benchmark.py --data data/run --seeds 5
python phase3_train_and_benchmark.py --data data/run --seeds 5 --ablate

# 4. the severity curve (headline figure)
python phase4_severity_sweep.py --out sweep
```

Simultaneous theft at several nodes is a **separate study**, deliberately kept
out of phases 1-4. Phases 1-4 use a softmax over nodes, which can only name one
thief; `phase5` re-poses the problem with one sigmoid per node:

```bash
python phase1_simulate_feeder.py --out data/mt2 --severity 0.15 \
    --max-concurrent 2 --realizations 8
python phase5_multi_thief.py --data data/mt2 --seeds 5
python phase5_multi_thief.py --data data/mt2 --seeds 5 --use-sensor-lines 3 4 5
```

Check the neural path is wired correctly on your machine first:

```bash
python tests/selftest_nn.py --data data/run
```

## Useful switches

| flag | what it does |
|---|---|
| `--severity 0.05` | theft as a fraction of the thief's own load |
| `--sensor-coverage sparse --n-sensors 2` | realistic partial CT instrumentation |
| `--max-concurrent 2` | simultaneous thieves (the "downstream bleed" case) |
| `--topology branched --nodes 15` | feeder with laterals |
| `--split realization` | train on some feeders, test on others |
| `--solver bfs` | run without pandapower |

## Reading the results

`phase2` gates everything. If it reports a detection AUC above 0.99 or a rule
baseline above 0.95 macro-F1, the benchmark is too easy and no model result
from it means anything — lower `--severity`, raise `--sigma-meter` /
`--p-missing`, or thin out `--sensor-coverage`.

`phase3` prints a warning if no learned model beats the hand-written physics
rule. Take it seriously.
