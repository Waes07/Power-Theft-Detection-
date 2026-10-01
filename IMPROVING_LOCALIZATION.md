# Improving localization — measured results

Everything here was measured on your feeder, not guessed. Severity α = 0.08,
5 consumers, 8 realizations. **Random guessing gives localization 0.20.**

Detection and localization are two different jobs:

| Job | Question | Needs current sensors? |
|---|---|---|
| Detect | Is *somebody* stealing? | No — substation minus meters is enough |
| Localize | *Which house*? | Yes — this is the only job that needs them |

Detection is near-saturated in every configuration below (0.88–0.98). Everything
that follows is about localization.

---

## Lever 1 — Sensor placement. Free, and by far the biggest.

Same number of sensors. Only the position changes.

| Sensors on lines | Count | Localization | macro-F1 |
|---|---|---|---|
| 1, 2 | 2 | 0.289 | 0.402 |
| 2, 3 | 2 | 0.393 | 0.487 |
| **4, 5** | **2** | **0.527** | **0.568** |
| 1, 2, 3 | 3 | 0.414 | 0.592 |
| 2, 3, 4 | 3 | 0.789 | 0.651 |
| **3, 4, 5** | **3** | **0.929** | **0.807** |
| 2, 3, 4, 5 | 4 | 0.946 | 0.832 |
| all five | 5 | 0.978 | 0.949 |

**Three sensors on lines 3, 4, 5 score 0.929 — nearly as good as all five (0.978),
at 60% of the hardware cost.** The same three sensors at the head of the feeder
score only 0.414.

### Two rules, and why they hold

**Put sensors downstream, never at the substation.** Line 1 carries every
customer — about 84 A. Stealing 0.9 kW moves it by roughly 1%, which is inside
the sensor's own noise. Line 5 carries one customer — about 17 A — so the same
theft is a five-times larger relative change. Line 1's residual is also
redundant: `Grid_Unbalance` already measures the whole feeder.

**Keep them consecutive.** Lines 2,3,4 → 0.789. Lines 1,3,5 → 0.544. Kirchhoff's
law at a node needs the line above it *and* the line below it, so a gap breaks
the calculation for every node in that gap.

Placement is now explicit:

```bash
python phase1_simulate_feeder.py --out data\hard --severity 0.08 --sensor-lines 3 4 5
```

`--sensor-coverage sparse` also changed: it now selects a consecutive block at
the downstream end instead of forcing a sensor onto line 1.

---

## Lever 2 — Meter voltage. A device you already own.

Real smart meters report voltage magnitude, not just energy. The simulator was
not emitting it; it now does (`Meter_j_V`, 0.5% Class-1 error), and
`physics_features` derives a voltage-profile residual from it.

| Feature set | Localization |
|---|---|
| meters + unbalance only | 0.205 |
| + 2 line current sensors (KCL residual) | 0.300 |
| + meter voltage profile | 0.337 |
| voltage but **no** current sensors | 0.218 |

A modest, free gain on top of currents. On its own it is barely above chance —
so it supplements current sensing, it does not replace it.

### Why it carries any information at all

Voltage falls along the feeder. Stolen current makes it fall *faster*, but only
up to the thief — past the thief that extra current is gone. So the drop profile
bends exactly at the offending node:

```
voltage
drop   │        ____________________     flat after the thief
       │      /
       │    /                            steeper before the thief
       │  /
       └──────────────────────────────
         C1   C2   C3   C4   C5
                    ^
              the bend is the thief
```

The predicted profile comes from a linearised DistFlow estimate using the
*metered* loads; the residual is prediction minus measurement.

---

## Lever 3 — A longer observation window

The models classify one day at a time, but a theft episode lasts about nine days.
Averaging over a longer causal window cancels measurement noise.

| Window | Localization | macro-F1 | Episode recall |
|---|---|---|---|
| 1 day (current code) | 0.337 | 0.370 | 0.722 |
| 7 days | 0.324 | 0.376 | 0.500 |
| **14 days** | **0.414** | **0.428** | 0.556 |
| 21 days | 0.422 | 0.413 | 0.556 |

**Report the trade-off honestly:** longer windows localize better but detect
later. Episode recall falls from 0.72 to 0.56. You identify the thief more
accurately, but you take longer to notice the theft at all. That tension is
itself a good paragraph for the paper.

---

## All three stacked

| Configuration | Localization |
|---|---|
| 2 sensors on lines 1, 2 · 1-day window | 0.255 |
| move the sensors to lines 3, 5 | 0.488 |
| + meter voltage profile | 0.544 |
| + 14-day window | **0.613** |
| *(reference: all 5 sensors, 1-day window)* | *0.978* |

**0.255 → 0.613 with the same two sensors.** No extra hardware — only a
placement decision, a channel the meters already produce, and a wider window.

---

## What about a better model?

Least important of the four. Placement moved the number by 2.4×; architecture
changes move it by a few percent. Evidence from your own instrumented run: a
plain CNN, a CNN–LSTM, the full hybrid and a plain logistic regression were all
statistically indistinguishable.

If you still want to try, in order of expected value:

1. **Two-stage design.** Detect first, then run a separate localizer trained
   *only* on theft days. Stops the 64% majority of Normal days from dominating
   the loss.
2. **Multi-label output.** One sigmoid per node instead of one softmax over
   nodes, so several simultaneous thieves are representable
   (`--max-concurrent 2` generates that data).
3. **Graph neural network.** Message passing along the real feeder topology
   rather than leaving the network to infer the wiring from data.

Do these after the three levers above, not before.
