# START HERE — simple guide

Plain words. No jargon. Just do the steps.

---

# PART 1 — What was wrong with your old code

Five problems. All five are fixed now.

### 1. The test was too easy

You made the thief steal **3 to 8 kW**.
But your whole street only used **1.6 kW** in total.

So the thief was using **3 times more power than the whole street**.
That is like hiding an elephant in a bedroom. Anyone can see it.

I wrote a tiny program — just 4 lines, no AI at all.
It got **99.77% correct**. Your big AI model got 91.8%.

**A 4-line program beat your deep learning model.**
If a reviewer tries this, your paper is rejected immediately.

### 2. Your answer key was wrong

In your `phase1` file, you picked a new thief **every hour**.

In your `phase3` file, you wrote this line:

```python
y_raw = df["Thief_ID"].values[::24]
```

This takes the answer from **hour 0 only** and uses it for the whole day.

I counted what this did to your data:

- You had **745 days** marked "no theft"
- But **743 of them actually had theft in them**
- Only **16%** of your day-labels matched what really happened that day

So your model was learning from wrong answers.
Every accuracy number in your report is measuring the wrong thing.

### 3. Wrong dataset name

Open `block1.csv`. The first line says:

```
LCLid,tstp,energy(kWh/hh)
MAC000323,2012-03-07 00:00:00.0000000, 0.15
```

`LCLid` means **Low Carbon London**. This is UK data, from London homes, 2011–2014.

**SGCC** is a totally different dataset — from China, daily readings, 2014–2016.

Your files and your whole report say "SGCC". This is the wrong name.
A reviewer who knows this field will see it in 5 seconds. Fix it everywhere.

Also: each block file holds **50 different houses**, one after another.
Your code read the first 48,000 rows and called it "one house".
That was really several houses glued together.

### 4. Wrong cable, wrong voltage

You used cable `NAYY 4x50 SE` at **11,000 volts**.
That cable is made for **400 volts**. It is 27 times too weak.

Because of this, the "current" you measured was not caused by the load.
It was caused by the cable itself (a physics effect called charging current).

**92% of your "I²R heat signature" was not heat at all.**

Look at your own file, hours 1, 2, 3:

```
Line_1_Amps = 0.64197, 0.64209, 0.64203
```

Almost the same number, even though the load changed.
The current was not following the load. So your main idea did not work as you thought.

### 5. Your report said things your code did not do

| Your report says | Your code actually does |
|---|---|
| "Early Stopping (15 epochs)" | `for epoch in range(50)` — no early stopping at all |
| "Defeating Direct Hooking" | no such test exists |
| "Curing Downstream Bleed" | no such test exists |
| "~91.8% precision" | no random seed, so nobody can repeat it |

You must never claim something the code does not do. Reviewers check this.

---

# PART 2 — What I made for you

New folder: **`c306\ntl_v2\`**

| File | What it does |
|---|---|
| `ntl_common.py` | The shared parts: the grid, the cables, the physics |
| `phase1_simulate_feeder.py` | Makes the fake grid data (your old phase1) |
| `phase2_audit_dataset.py` | **NEW** — checks your data before you train |
| `phase3_train_and_benchmark.py` | Trains all the models (your old phase3) |
| `phase4_severity_sweep.py` | **NEW** — makes the main picture for your paper |
| `tests/selftest_nn.py` | **NEW** — checks the AI part works on your PC |
| `METHOD_CHANGES.md` | Long version, with all numbers |
| `README.md` | Short command list |

### The two most important new things

**`phase2` is a safety check.**
Run it **before** you train anything.
It tells you if your test is too easy. Your old code had no such check.
That is why nobody noticed the 4-line program was winning.

**`phase4` makes the picture.**
It shows how well the model works when theft is small, medium, and large.
One number tells a reviewer nothing. This picture tells them everything.

---

# PART 3 — What you do now

## Step 0 — Install what you need

Open Command Prompt in the `ntl_v2` folder. Type:

```
pip install numpy pandas scikit-learn matplotlib pandapower torch xgboost
```

Wait for it to finish.

## Step 1 — Nothing to do. Already done.

There is a file called `lcl_hourly_wide.csv` inside `ntl_v2`. **I already put it there.**

Check it is there:
```
dir C:\Users\FD0007NE\Desktop\c306\ntl_v2\lcl_hourly_wide.csv
```
It should be about **8.8 MB**.

**What is this file?**
Your `block1.csv`, `block2.csv`, `block3.csv` are 71 MB each and slow to read.
I already read them once and saved the useful part into this small file.

It has 126 houses and 485 days of real London electricity readings.
Every hour, for every house, how much power they used.

**You do NOT need to copy the block files into `ntl_v2`.**
The scripts see `lcl_hourly_wide.csv` and use it. The block files are never opened again.

(There is also a `_verification` folder next to `c306`. That was my working folder.
You can delete it. Nothing needs it.)

## Step 2 — Make the data

```
python phase1_simulate_feeder.py --out data\run --severity 0.15 --realizations 8
```

Takes a few minutes. You should see something like:

```
mean technical loss .... 3.25 %   (target 3-10%)
min bus voltage ........ 0.9119 pu  (limit 0.90)
max line loading ....... 82.1 %   (limit 100)
theft-days / total ..... 1397 / 3880 (36.0%)
```

**Check these three numbers:**
- loss must be between 3 and 10
- voltage must be above 0.90
- loading must be below 100

If they are, your grid is realistic. This is a real result — write it in your paper.

## Step 3 — Check the AI part works

Do this **after** Step 2. It needs the data to exist.

```
python tests\selftest_nn.py --data data\run
```

You should see:

```
[selftest] using real PyTorch
[selftest] PASSED
```

If you see PASSED, everything is connected correctly. Go to Step 4.
If you see an error, send me the error message.

## Step 4 — Check the data is not too easy

```
python phase2_audit_dataset.py --data data\run --check-solver
```

Look for lines starting with `[ ok ]` or `[WARN]`.

**The most important line is this one:**

```
hand-written physics rule (test split) ..... acc 0.7371 | macro-F1 0.6356
```

This is the score of the simple no-AI program.
**Your AI model must beat this number.** If it does not, you have no paper.

If you see `[WARN]` about the test being too easy, make theft smaller:
use `--severity 0.08` in Step 3 and run again.

## Step 5 — Train everything and compare

```
python phase3_train_and_benchmark.py --data data\run --seeds 5
```

This is slow. **Maybe 30 to 60 minutes.** Let it run.

It trains 8 different models on the same data:

| Model | What it is |
|---|---|
| `rule` | Simple program, no AI. Your model must beat this. |
| `logreg` | Simplest possible AI |
| `rf` | Random Forest |
| `gbm_raw` | XGBoost only, no deep learning |
| `cnn` | Just the CNN |
| `lstm` | Just the LSTM |
| `cnn_lstm` | CNN + LSTM together |
| `hybrid` | CNN + LSTM + XGBoost — **your model** |

At the end you get a table. **This table is Table 1 of your paper.**

Look at the bottom line. If it says:

```
[WARN] no learned model beats the hand-written rule
```

then stop and tell me. Do not submit that.

## Step 6 — Prove your main idea

```
python phase3_train_and_benchmark.py --data data\run --seeds 5 --ablate
```

This answers: **do the line currents actually help?**

I already ran this part. Here is what it says:

| What you give the model | Score |
|---|---|
| Just meters | 0.235 |
| Meters + raw line currents | 0.326 |
| Meters + currents + **my new KCL feature** | **0.933** |

**Read this carefully. It changes your main claim.**

Raw currents alone only helped a little (0.235 → 0.326).
The big jump came from the new feature I added (0.326 → 0.933).

That new feature compares:
- how much current is **really** going into a house (from the line sensors)
- how much current the **meter says** should be going in

If these two do not match, that house is stealing. This is Kirchhoff's law.

So your paper should NOT say *"we gave the AI line currents."*
Your paper SHOULD say *"we combine line currents with grid topology using Kirchhoff's current law, and this is worth +0.61 in score."*

**This is a much stronger claim. And it is true.**

## Step 7 — Make the main picture

```
python phase4_severity_sweep.py --out sweep
```

**Very slow. Maybe 2 to 4 hours.** Start it and leave your PC on.

It makes `sweep\severity_sweep.png`. I already made one for you so you can see it.

The picture shows: when theft is small, the model struggles.
When theft is big, the model wins easily.

**This is Figure 1 of your paper.** It is the most useful picture you can show.

---

# PART 4 — One thing you must know

## The two jobs are NOT the same

Your project does two different jobs. Keep them separate in your head:

| Job | Question | What it needs |
|---|---|---|
| **Detect** | Is *somebody* stealing on this street? | Substation reading minus all meter readings |
| **Localise** | *Which house* is stealing? | **Line current sensors** |

**Detecting** is easy. You just check if the substation gave out more power than
the meters recorded. You do not need current sensors for this at all.

**Localising** is the hard job, and it is the job your paper is really about.
This is the ONLY job that needs line currents.

## Proof — I tested it

Same data, same theft size (alpha = 0.12), 5 houses. I only changed how many
wires have a current sensor:

| Current sensors | Find THAT someone steals | Find WHO steals |
|---|---|---|
| 5 of 5 (every wire) | 0.99 | **1.00** |
| 2 of 5 | 0.97 | **0.33** |
| 1 of 5 (head only) | 0.95 | **0.25** |
| 0 of 5 (none) | 0.95 | **0.23** |

Read the two columns separately.

**Left column barely moves.** 0.95 to 0.99. Sensors do not help you detect.

**Right column falls off a cliff.** 1.00 down to 0.23.

With 5 houses, random guessing gives you 0.20. So **0.23 means the AI is
basically guessing.** With no current sensors, your model cannot localise. At all.

## So: do NOT remove the currents

If you remove line currents from the table:
- your model can still say "someone is stealing" — but so can a calculator
- your model can no longer say **who** — which is your entire contribution
- your paper has nothing new in it

**Removing currents does not make the test harder in a useful way.
It makes the task impossible, and deletes your paper.**

## What to do instead: use FEW sensors, not ZERO

Real electricity companies do not put a sensor on every wire. Too expensive.
They put one or two on the whole street.

Look at the table again. **2 of 5 sensors** is the interesting row:

- Localise score is 0.33
- Not solved (that would be 1.00)
- Not guessing (that would be 0.20)
- **Partly solved — some information is there, but not enough**

That gap between 0.33 and 1.00 is the space where a smart model can win.
A simple model cannot fill it. Your CNN+LSTM might.

**That is your experiment. That is where your paper lives.**

Run it like this:

```
python phase1_simulate_feeder.py --out data\hard --severity 0.08 --sensor-coverage sparse --n-sensors 2 --realizations 8
python phase3_train_and_benchmark.py --data data\hard --seeds 5
```

## The sentence to build your paper around

> When every wire has a sensor, physics alone finds the thief and AI is not needed.
> When only a few wires have sensors — which is the real situation — finding the thief
> becomes a hard time-pattern problem. That is where our hybrid model helps.

This is a much better paper than "our AI got 91.8%".

*(Note: in the table above, the `rule` model shows a strange 0.487 when sensors
are missing. Ignore it. It is not real — the rule just gives up and always
guesses the same house every time. The `logreg` row is the honest one.)*

# PART 5 — What to fix in your written report

Do these before you submit anywhere:

1. Change **SGCC** to **Low Carbon London (UK Power Networks)** everywhere. Files, text, everything.
2. Delete the claim about **"early stopping 15 epochs"** — or keep it, because the new code really does it now.
3. Delete **"Downstream Bleed"** and **"Direct Hooking"** as achievements. Run `--max-concurrent 2` first, then you can claim the multi-thief case honestly.
4. Change your main claim from "we used Amperes" to "we used Kirchhoff's law on Amperes" (see Step 6).
5. Add a related-work section. Compare with existing theft-detection papers. This is required.
6. Add the picture from Step 7 and the table from Step 5.

---

# PART 6 — If something goes wrong

**"No module named pandapower"**
You do not need it. Add `--solver bfs` to the phase1 command. It works the same.
(I checked: my solver gives the same answer as pandapower to 12 decimal places.)

**"No module named xgboost"**
The code switches to a scikit-learn version automatically. It is fine.

**"No module named torch"**
Then the CNN and LSTM models are skipped. Install torch to run them.

**Phase 3 is too slow**
Use `--seeds 2` instead of `--seeds 5` while testing. Use 5 for the final results.

**Everything scores 100%**
Your test is too easy. Lower `--severity` to 0.05 and run phase1 again.

---

# Quick summary

| Step | Command | Time |
|---|---|---|
| 2 | `python phase1_simulate_feeder.py --out data\run --severity 0.15 --realizations 8` | 5 min |
| 3 | `python tests\selftest_nn.py --data data\run` | 1 min |
| 4 | `python phase2_audit_dataset.py --data data\run` | 1 min |
| 5 | `python phase3_train_and_benchmark.py --data data\run --seeds 5` | 30–60 min |
| 6 | `python phase3_train_and_benchmark.py --data data\run --seeds 5 --ablate` | 30 min |
| 7 | `python phase4_severity_sweep.py --out sweep` | 2–4 hours |

Do them in order. Send me the output of Step 5 when it finishes.
