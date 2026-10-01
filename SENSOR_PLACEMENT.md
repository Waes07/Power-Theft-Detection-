---
title: "Observability Zones and Sensor Placement on a Radial Distribution Feeder"
subtitle: "A design heuristic for current-sensor siting in non-technical-loss localisation"
author: "Md. Nafiz Istiak — Roll 2206040, Department of Electrical and Electronic Engineering"
date: "Draft section for the NTL detection and localisation manuscript"
geometry: margin=1in
fontsize: 11pt
linestretch: 1.06
colorlinks: true
header-includes:
  - \usepackage{amsmath,amssymb,booktabs}
  - \usepackage{mathtools}
---

# 1  Why this section exists

The experimental part of this study established, across five random seeds and
three independent scoring pipelines, that **where** current sensors are placed
on a radial feeder matters far more than **how many** are installed. Three
sensors on lines 3, 4 and 5 gave a localisation accuracy of 0.957 and recovered
77.0% of stolen energy at a 5% inspection budget; full instrumentation of all
five lines gave 0.933 and 78.0%. The same three sensors moved to lines 1, 2 and
3 collapsed localisation to 0.423.

That is a measurement, not an explanation, and it has two practical
deficiencies. First, it offers a utility planner no reason to believe the result
transfers to a feeder that is not this feeder. Second, obtaining it required a
full simulate-and-train cycle for each candidate placement, which is infeasible
as a design tool: a 12-node feeder has 4 095 non-empty placements.

This section supplies the missing analytical layer. It is presented as a
**screening heuristic**, not an optimality theorem: the claim is that a
closed-form statistic computable in microseconds reproduces the ranking that the
full pipeline produces in hours, and that the mechanism behind that statistic
explains both the "downstream" and the "consecutive" rules observed
experimentally. Section 8 states plainly where the heuristic stops being
reliable.

![Observability-zone analysis of the feeder studied here. **(a)** what each additional current sensor buys, on this study's measured node loads. **(b)** every three-sensor placement ranked by the localisation score; the four that tie at the optimum are exactly the four that leave line 1 uninstrumented. **(c)** the analytic ambiguity statistic against measured pipeline performance, for all placements tested experimentally. **(d)** the scope limit: clustering sensors downstream is optimal only up to seven nodes.](sensor_placement.pdf){width=100%}


\bigskip

# 2  Measurement model and notation

The feeder is a radial chain of $n$ metered service points. Node $0$ is the
distribution substation; node $j\in\{1,\dots,n\}$ is an aggregated service point
of $K$ households. Line $j$ joins node $j-1$ to node $j$, so line $j$ carries the
entire downstream load of nodes $j,\dots,n$.

Write

- $M_j$ — the metered active power of node $j$ (kW), reported by its smart meter;
- $\mu_j = \mathbb{E}[M_j]$ — its mean load;
- $P_b = \sum_{j \ge b}\mu_j$ — the mean power flowing through line $b$;
- $\hat{P}_b$ — the power inferred from the current sensor on line $b$, if one exists;
- $S \subseteq \{1,\dots,n\}$ — the set of instrumented lines;
- $\alpha$ — theft severity, expressed as a fraction of the thief's own mean load.

Two boundaries are measured whether or not a CT is bought for them, and this
turns out to drive the entire result:

1. **Line 1.** The substation revenue meter already measures the total feeder
   infeed. Indeed it measures it *better* than a CT would
   ($\sigma_{\text{sub}} = 0.002$ versus $\sigma_{\text{ct}} = 0.010$ in this study's
   measurement chain). A current sensor bought for line 1 is therefore a
   duplicate purchase.
2. **Line $n+1$.** The open end of the feeder carries no current. $P_{n+1}\equiv 0$
   is known exactly, at zero cost and with zero error.

Define the **boundary set**

$$
B(S) \;=\; \{1\} \,\cup\, S \,\cup\, \{n+1\}, \qquad
b_0 < b_1 < \dots < b_M .
$$

\bigskip

# 3  Observability zones

## 3.1 The partition

**Proposition 1.** *Let $Z_i = \{b_i,\,b_i+1,\,\dots,\,b_{i+1}-1\}$ for
$i = 0,\dots,M-1$. Then $\{Z_i\}$ partitions $\{1,\dots,n\}$, and the current
measurements determine the location of an anomaly only up to membership of a
single $Z_i$.*

*Proof sketch.* For any two adjacent boundaries $b_i < b_{i+1}$, conservation of
power across the intervening span is a measurable quantity:

$$
\Delta P_i \;=\; \hat{P}_{b_i} \;-\; \hat{P}_{b_{i+1}} \;-\; \sum_{j\in Z_i} M_j .
\tag{1}
$$

Honest operation makes $\Delta P_i$ zero up to measurement error. A hook load at
node $j$ raises $\hat{P}_{b_i}$ without raising $M_j$; a tampered meter lowers
$M_j$ without lowering $\hat{P}_{b_i}$. Either way $\Delta P_i$ moves. Crucially,
**every node inside $Z_i$ enters equation (1) with the same coefficient**, so no
linear combination of the available current measurements separates them. No span
crossing a boundary yields an additional independent equation, because such a
span is the sum of the spans it contains. $\square$

## 3.2 In plain words

A current sensor is a *fence*. Each fence, plus the substation meter at the head
and the dead end at the tail, cuts the feeder into sections. Within a section
you can prove that power is going missing, but you cannot say from which house
on that section. The entire localisation question reduces to: **how do the
fences cut the feeder?**

This immediately explains both empirical rules without any appeal to machine
learning:

- *Why downstream.* A sensor on line 1 falls on a boundary the substation meter
  already provides. It adds no fence, so it adds no section.
- *Why consecutive.* Two sensors far apart leave a wide section between them.
  Two adjacent sensors isolate a single node between them.

## 3.3 Worked example, $n=5$

| Sensors $S$ | Boundaries $B(S)$ | Zones | $D(S)$ | $A(S)$ |
|---|---|---|---|---|
| $\{1,2,3,4,5\}$ | 1,2,3,4,5,6 | {1}{2}{3}{4}{5} | 1.00 | 1.00 |
| $\{3,4,5\}$ | 1,3,4,5,6 | {1,2}{3}{4}{5} | 1.40 | 0.80 |
| $\{3,4\}$ | 1,3,4,6 | {1,2}{3}{4,5} | 1.80 | 0.60 |
| $\{4,5\}$ | 1,4,5,6 | {1,2,3}{4}{5} | 2.20 | 0.60 |
| $\{1,2,3\}$ | 1,2,3,6 | {1}{2}{3,4,5} | 2.20 | 0.60 |

Note the last row. Adding a sensor to line 1 costs a CT and buys nothing: the
boundary set is unchanged from $\{2,3\}$ alone.

\bigskip

# 4  The screening statistic

## 4.1 Definitions

Given a prior $\pi$ over which node is stealing (uniform throughout this study),
define the **expected ambiguity**

$$
D(S) \;=\; \sum_{j=1}^{n} \pi_j \,\bigl|Z(j)\bigr|
\;\;\overset{\pi\ \text{uniform}}{=}\;\;
\frac{1}{n}\sum_{i} \bigl|Z_i\bigr|^{2},
\tag{2}
$$

read as *the number of premises an inspection crew must visit per alarm*.
$D(S)=1$ is perfect node-level resolution; $D(S)=n$ is no localisation at all.
Under a uniform prior, minimising $D$ is the classical balanced-partition
objective: **cut the feeder into equally sized sections.**

The companion quantity, the probability of naming the right node by guessing the
most likely member of the identified zone, is

$$
A(S) \;=\; \sum_i \max_{j \in Z_i} \pi_j
\;\;\overset{\pi\ \text{uniform}}{=}\;\; \frac{\bigl|S \cup \{1\}\bigr|}{n}.
\tag{3}
$$

$A(S)$ is the **current-channel resolution ceiling**. Section 5.2 shows the
trained model exceeds it, and identifies what supplies the surplus.

## 4.2 Submodularity

**Proposition 2.** *$-D$ is monotone and submodular on $2^{\{1,\dots,n\}}$.*

*Proof sketch.* Placing a sensor at an interior point of a zone of size $z$,
splitting it into $z_1 + z_2 = z$, changes the objective by $2z_1z_2/n > 0$
(monotone). If earlier sensors have already shrunk that zone, both $z_1$ and
$z_2$ are smaller and the gain is strictly smaller (diminishing returns).
$\square$

Greedy maximisation of $-D$ therefore carries the standard $(1-e^{-1})\approx0.632$
guarantee. Section 7 reports what greedy actually achieves, which is much better
than the bound but not exact.

\bigskip

# 5  Validation against the measured pipeline

Neither $D$ nor the refined score $L$ of Section 6 contains a single parameter
fitted to the results below. The agreement is a prediction, not a regression.

## 5.1 Rank agreement

| Measurement | Placements compared | Spearman $\rho$ against $-D(S)$ |
|---|---|---|
| Phase 3 localisation accuracy, 5 seeds | 3 | $+0.500$ (no inversion beyond $0.03$) |
| Phase 6 energy recovery @5% budget, multi-thief | 4 | $\mathbf{+1.000}$ |
| Phase 6 energy recovery @5% budget, single thief | 3 | $\mathbf{+1.000}$ |

Detail for the four-placement comparison, which is the most informative:

| Sensors | $D(S)$ | Measured recovery @5% |
|---|---|---|
| $\{1,2,3,4,5\}$ | 1.00 | 0.712 |
| $\{3,4,5\}$ | 1.40 | 0.707 |
| $\{3,4\}$ | 1.80 | 0.600 |
| $\{4,5\}$ | 2.20 | 0.478 |

A statistic that took microseconds to evaluate reproduces, without error, the
ordering that four full simulate-and-train runs produced.

## 5.2 The two disagreements, stated honestly

**(a) $\{3,4,5\}$ scored 0.957 against full instrumentation's 0.933.** The theory
predicts the reverse. The measured gap is 0.024, smaller than the seed-to-seed
spread in phase 3, and the two numbers came from runs with different fold
settings. We treat them as tied rather than claim the theory is refuted — but we
do not claim it is confirmed on this pair either.

**(b) The measured 0.957 exceeds the current-channel ceiling $A(\{3,4,5\})=0.80$.**
This one is real and is not noise. It says the trained model resolves node 1
from node 2 *inside a zone that the current measurements cannot split*. The only
other localisation channel in the feature set is the voltage-profile residual
$\text{VResid}_j$, computed from smart-meter voltage magnitudes against a
linearised DistFlow prediction. Theft downstream of node $t$ increases the
current in lines $1..t$ only, so the measured-minus-predicted drop grows with
distance up to node $t$ and then flattens; the position of that knee is
independent information about location.

We therefore read the excess over $A(S)$ as a **lower bound on what the voltage
channel contributes**: at least $0.957 - 0.80 = 0.157$ of localisation accuracy
comes from meter voltages rather than line currents. This is a useful number for
a utility, because smart meters already report voltage and the channel is free.

\bigskip

# 6  Noise: why the fences should sit downstream

## 6.1 The variance of a zone residual

Instrument errors are *relative*, so their absolute magnitude scales with the
quantity being measured. Two families behave differently under averaging.

$$
\operatorname{Var}(\Delta P_i)
= \underbrace{\frac{1}{N}\Bigl[\sigma_{\text{ct}}^2\bigl(P_{b_i}^2+P_{b_{i+1}}^2\bigr)
+ \sigma_{\text{m}}^2\!\!\sum_{j\in Z_i}\!\mu_j^2\Bigr]}_{\text{random: CT and meter read noise}}
\;+\;
\underbrace{\sigma_{\text{mod}}^2\Bigl(\sum_{j\in Z_i}\mu_j\Bigr)^{\!2}}_{\text{systematic: unmodelled load}}
\tag{4}
$$

where $N$ is the number of samples pooled into one test (24 hourly reads per day
here). The systematic term collects everything the feature extractor models with
nominal constants while the feeder uses jittered true ones: unmetered communal
load and street lighting (3% plus 1.5 kW in this study), per-node power-factor
spread, $\pm15\%$ line-length error, and CT calibration offset. It does **not**
average away, and it grows with the *total load inside the zone*.

The per-node signal-to-noise ratio for theft at node $j$ is then

$$
\rho_j \;=\; \frac{\alpha\,\mu_j}{\operatorname{sd}\bigl(\Delta P_{Z(j)}\bigr)} .
\tag{5}
$$

## 6.2 Consequence 1 — head sensors are noisy sensors

Evaluate (4) for this study's feeder with a sensor on every line, random terms
only, $N=1$:

| Zone | Line currents at its boundaries | $\operatorname{sd}(\Delta P)$ |
|---|---|---|
| $\{1\}$ | $P_1 = 5\mu,\; P_2 = 4\mu$ | $0.0415\,\mu$ |
| $\{2\}$ | $P_2 = 4\mu,\; P_3 = 3\mu$ | $0.0502\,\mu$ |
| $\{3\}$ | $P_3 = 3\mu,\; P_4 = 2\mu$ | $0.0364\,\mu$ |
| $\{4\}$ | $P_4 = 2\mu,\; P_5 = 1\mu$ | $0.0229\,\mu$ |
| $\{5\}$ | $P_5 = 1\mu,\; P_6 = 0$ | $0.0112\,\mu$ |

The residual at the head of the feeder is **3.7× noisier** than the residual at
the tail — and 5.7× noisier if line 1 is instrumented with a CT rather than read
from the substation meter. The reason is elementary and worth stating plainly in
the paper: *a sensor near the substation measures the sum of every node's load,
so a fixed 1% error there is a large number of kilowatts, while the theft it is
trying to see is still only one house's worth.*

This is the second, independent reason downstream placement wins, and it is
separate from the zone-counting argument of Section 3.

## 6.3 Consequence 2 — the optimum is interior

Splitting a zone at line $p$ replaces one CT-error term $\sigma_{\text{ct}}^2P_p^2$
with two, once in each daughter zone. Adding a sensor therefore **reduces
ambiguity but increases total residual noise**. "Instrument everything" is not
automatically optimal, and the measured result that the fifth sensor changes
nothing is the visible form of this trade-off.

## 6.4 A combined score

Two things must go right: the residual covering the thief must clear the alarm
threshold, and the operator must then pick the right node inside it.

$$
L(S) \;=\; \sum_{j} \pi_j\,
\underbrace{\Phi\bigl(\rho_j - z_{1-\alpha_{\mathrm{FA}}}\bigr)}_{\text{detection power}}
\cdot
\underbrace{\frac{1}{|Z(j)|}}_{\text{resolution}} ,
\qquad
C(S) \;=\; \sum_{j} \pi_j\,\Phi\bigl(\rho_j - z_{1-\alpha_{\mathrm{FA}}}\bigr) .
\tag{6}
$$

$L$ is expected localisation probability; $C$ is detection power alone,
irrespective of pinpointing. They are reported separately because, as Section 8
shows, they do not always agree.

Evaluated on this feeder's measured node loads
$\mu = (10.46,\,12.19,\,10.24,\,10.58,\,9.03)$ kW, $\alpha=0.15$,
$\alpha_{\mathrm{FA}}=0.05$, $N=24$:

| Sensors | $D$ | $L$ | $C$ | Zones |
|---|---|---|---|---|
| $\{1,2,3,4,5\}$ | 1.00 | 0.999 | 0.999 | 1 \| 2 \| 3 \| 4 \| 5 |
| $\{3,4,5\}$ | 1.40 | 0.759 | 0.918 | 12 \| 3 \| 4 \| 5 |
| $\{3,4\}$ | 1.80 | 0.519 | 0.837 | 12 \| 3 \| 45 |
| $\{4,5\}$ | 2.20 | 0.501 | 0.704 | 123 \| 4 \| 5 |
| $\{1,2,3\}$ | 2.20 | 0.501 | 0.704 | 1 \| 2 \| 345 |

\bigskip

# 7  Optimisation

For $n \lesssim 20$ the problem is enumerable: $\binom{n}{B}$ candidates, each
scored in microseconds. `phase7_sensor_placement.py --budget B` does this
exhaustively. For longer feeders, forward greedy on $-D$ inherits the
$(1-e^{-1})$ guarantee of Proposition 2; $L$ and $C$ carry no guarantee because
their noise term is not monotone.

Measured over $n=4\ldots14$ and budgets $1\ldots6$ (189 problems):

| Objective | Greedy returned the exact optimum | Worst observed ratio |
|---|---|---|
| $L$ | 49/63 (78%) | 0.980 |
| $C$ | 46/63 (73%) | 0.861 |
| $-D$ | 45/63 (71%) | 0.857 |

Comfortably inside the submodular bound, but not exact — so enumerate when you
can.

**Result for the feeder studied here.** At a budget of three sensors, four
placements tie at the optimum to within 0.002: $\{2,3,5\}$, $\{2,3,4\}$,
$\{3,4,5\}$ and $\{2,4,5\}$. Every one of them leaves line 1 uninstrumented;
every placement that uses line 1 is strictly worse. The experimental programme
tested only one member of the tied set, $\{3,4,5\}$. **The theory therefore makes
a falsifiable prediction: $\{2,3,4\}$, $\{2,3,5\}$ and $\{2,4,5\}$ should perform
indistinguishably from $\{3,4,5\}$, and all four should beat every placement
containing line 1.** Two additional phase-3 runs settle it.

\bigskip

# 8  Scope: where the downstream rule stops working

The rule extracted from this five-node feeder — *push the sensors to the tail* —
is optimal here, but it is a small-$n$ artefact. Maximising $L$ rewards being
certain about a few nodes; on a long feeder that is achieved by clustering every
sensor at the tail and abandoning the head. Maximising $C$ rewards never missing
theft anywhere, which spreads the sensors out to keep each zone's aggregated
load, and therefore its systematic noise, small.

The two objectives select the same placement for $n \le 7$ and diverge from
$n = 8$ onward. At $n=12$ with four sensors:

| Objective | Optimal placement | $L$ | $C$ | $D$ |
|---|---|---|---|---|
| maximise $L$ | $\{9,10,11,12\}$ | **0.346** | 0.436 | 5.67 |
| maximise $C$ | $\{4,7,9,11\}$ | 0.280 | **0.643** | 2.50 |

Clustering buys 24% more localisation and costs 32% of detection power, because
nodes 1–8 fall into one zone whose detection probability is 0.154. On a feeder
of this length the "downstream" rule is actively harmful if the operator's
binding constraint is finding theft rather than pinpointing it.

**This limit must be stated in the paper.** It is the difference between a
result and an overclaim, and a referee will find it if we do not.

\bigskip

# 9  A feature the present extractor discards

The current implementation (`ntl_common.physics_features`) forms an exact
per-node residual $\text{Resid}_j$ only when line $j$ *and* every child line of
$j$ carry a CT — on a chain, $s_j \wedge s_{j+1}$. Nodes failing that test are
assigned `NaN` and contribute nothing.

That test is strictly stronger than Proposition 1 requires. Equation (1) is
computable for *any* pair of adjacent boundaries, adjacent sensors or not. With
$S=\{1,2,3\}$, for instance, the aggregate residual over nodes 3–5,

$$
\Delta P \;=\; \hat{P}_3 \;-\; 0 \;-\; (M_3 + M_4 + M_5),
$$

is perfectly well defined, and the present code throws it away.

Quantitatively, on this feeder the gap between the zone limit and the shipped
feature set is $L_{\text{zone}} = 0.759$ against $L_{\text{shipped}} = 0.657$ at
$S=\{3,4,5\}$, widening for non-adjacent placements. Adding the zone residual as
an explicit feature is a few lines of code and costs no hardware.

One honest qualification: in the validation of Section 5, $D$ and $L_{\text{zone}}$
rank-correlate with measurement at $\rho = +1.000$ while $L_{\text{shipped}}$
manages only $+0.800$. The shipped-partition model is *too pessimistic*, which
implies the trained network already recovers some zone-level information from
the raw line-current features on its own. The proposed feature would make that
information explicit rather than leave it to be rediscovered — a smaller claim
than "it unlocks new information", and the one the evidence supports.

\bigskip

# 10  Design rules

For a radial LV feeder with a substation revenue meter and smart meters at every
service point:

1. **Never instrument line 1.** The substation meter already provides that
   boundary, more accurately than a CT would.
2. **Count fences, not sensors.** A placement's worth is
   $D(S)=\frac{1}{n}\sum_i|Z_i|^2$. Screen with $D$; simulate only the survivors.
3. **Prefer equal-sized zones.** Balanced partitions minimise $D$ for a fixed
   budget.
4. **Break ties downstream.** Among placements with equal $D$, the one whose
   boundaries carry the least current has the lowest residual variance.
5. **Choose your objective before you choose your sensors.** Localisation and
   detection agree only for short feeders ($n \le 7$ here). Past that, decide
   whether the crew's constraint is finding theft or pinpointing it.

\bigskip

# 11  Limitations

- The analysis is for a **chain**. A branched feeder generalises by replacing
  "adjacent boundaries" with "a cut set of the tree", but the closed forms in
  Sections 4 and 6 are written for the chain and are not proved for laterals.
- $\Phi(\cdot)$ assumes a Gaussian residual, which is reasonable after averaging
  24 hourly samples but is not exact in the tails that govern low false-alarm
  operation.
- The uniform prior over thief location is a modelling choice. A utility with
  historical NTL data should substitute its own $\pi$; both $D$ and $L$ accept it
  unchanged.
- $\sigma_{\text{mod}}$ is a lumped stand-in for several distinct model errors and
  was set to 0.03 by inspection of the simulator's configured mismatch, not
  estimated from residuals. Conclusions about *ranking* are insensitive to it;
  conclusions about *absolute* detection probability are not.
- The theory describes the current and — indirectly, via Section 5.2 — the
  voltage channel. It says nothing about the temporal structure that the
  CNN–LSTM exploits, which is why measured performance can exceed $A(S)$.

\bigskip

# 12  Reproducing the numbers

```
python phase7_sensor_placement.py --data data\run --n-nodes 5 \
       --all-budgets --validate --plot sensor_placement.png

python phase7_sensor_placement.py --data data\run --budget 3 --partition shipped
python phase7_sensor_placement.py --n-nodes 12 --budget 4 --objective C
python phase7_sensor_placement.py --n-nodes 12 --budget 4 --greedy
python phase7_sensor_placement.py --selftest --n-nodes 7
```

`--selftest` checks Propositions 1 and 2, equations (2) and (3) and design rule
1 by enumerating all $2^{n}$ placements rather than asserting them, and reports
greedy's worst observed ratio against the $(1-e^{-1})$ bound. Every table in
this section is printed by the first command. The module imports
nothing beyond NumPy (pandas only for `--data`, matplotlib only for `--plot`),
runs in under two seconds, and performs no power flow and no training.
