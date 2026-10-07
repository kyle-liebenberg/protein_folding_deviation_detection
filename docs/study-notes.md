# Study Notes (Semester Test 2)

Theory and Q&A on **our** design choices, built up phase by phase. Numbers refer to our data (training
set unless stated otherwise). Each section links back to where the decision is made.

## 0. The big picture

**Task in one sentence:** given only measurements of a candidate protein shape, predict how far off
it is from the true shape.

1. Scientists **measure** a protein's real 3D structure in the lab and keep it secret (CASP competition).
2. Prediction software produces many **candidate shapes (decoys)** for that protein.
3. **F1–F9** are computed from each decoy *alone*: surface areas, how much water-repelling
   (non-polar) surface is exposed, secondary-structure penalty, etc. No knowledge of the truth is needed.
4. Once the truth is revealed, each decoy gets its grade, the **RMSD** (Å): how far its atoms are from
   the real ones. 0 = perfect, ~2 = very close, ~15 = essentially wrong.
5. **Our MLP:** 8 features in → predicted RMSD out (**regression**). Each row is one decoy. We don't know
   which protein or team it came from.

**Why it's useful:** for a new protein, the truth is unknown, so step 4 can't happen. A model that
predicts RMSD from features alone can rank thousands of candidates and pick the likely best. This is
*model quality assessment*. **Intuition for the features:** a well-folded protein tucks its
water-repelling parts inside, while a bad decoy leaves them exposed. That's why F3 (the non-polar
fraction of the exposed area) has the strongest linear relationship with RMSD.

**What the assignment is really about:** the protein data is a vehicle. It tests (1) defensible data
preparation, (2) controlled, hypothesis-driven experiments on MLP design choices, and (3) the research
question: does **learning-rate warmup** make a network more robust to **magnitude pruning**, and is
that because warmup allows a larger learning rate?

**Contents**
- [Part A: Data preparation](#part-a-data-preparation) (decisions: [`01-data-preparation.md`](01-data-preparation.md))
  - [A1 Descriptive statistics](#a1-descriptive-statistics)
  - [A2 Histograms](#a2-histograms)
  - [A3 The target: RMSD, Å, bimodal](#a3-the-target-rmsd-å-bimodal)
  - [A4 Splitting and cross-validation](#a4-splitting-and-cross-validation)
  - [A5 Feature relevance: Pearson, Spearman, mutual information](#a5-feature-relevance-pearson-spearman-mutual-information)
  - [A6 Outliers: log1p, IQR, transform and clip](#a6-outliers-log1p-iqr-transform-and-clip)
  - [A7 Scaling: z-score vs min-max vs batch normalisation](#a7-scaling-z-score-vs-min-max-vs-batch-normalisation)
  - [A8 Test-prep questions with answers](#a8-test-prep-questions-with-answers)
- [Part B: The MLP and training](#part-b-the-mlp-and-training) (Phase 2)
- [Part C: Core investigation](#part-c-core-investigation) (Phase 3)
- [Part D: Warmup and pruning](#part-d-warmup-and-pruning) (Phase 4)

---

## Part A: Data preparation

### A1 Descriptive statistics

What each row of `df.describe()` means:

| Statistic | Meaning |
|---|---|
| **count** | Number of **non-missing** values. We have no missing values, so it equals the 44,019 deduplicated rows. |
| **mean** | Average: sum of values ÷ count. |
| **std** | Standard deviation: the typical distance of a value from the mean (see below). |
| **min / max** | Smallest / largest value. |
| **p-th percentile** | The value that p % of rows are at or below. **50 % = the median** (the middle value). Example: F7's 99th percentile is 7,244, so 99 % of rows have F7 ≤ 7,244. |
| **skew** | How lopsided the distribution is (see below). |

`9.875e+03` is scientific notation: 9.875 × 10³ = 9,875.

**Standard deviation.** How spread out the values are around the mean, in the column's own units:
1. Compute the mean x̄.
2. For each value, take its distance from the mean (xᵢ − x̄) and square it. Squaring makes every
   distance positive and weights large distances more.
3. Average the squares. This is the **variance**. pandas divides by n − 1 rather than n, which corrects
   a small bias when estimating from a sample.
4. Take the square root to return to the original units.

$$\sigma = \sqrt{\frac{\sum_i (x_i - \bar{x})^2}{n-1}}$$

Example: RMSD has mean 7.8 Å, std 6.1 Å, so a typical row is about 6 Å from the mean.

**Skew.** Asymmetry. Roughly the average of the *cubed* z-scores, mean of ((x − x̄)/σ)³. Cubing keeps
the sign, so values far above the mean push skew positive.
- ≈ 0: symmetric.
- **> 0 (right skew):** a long tail of large values, which pulls the mean above the median.
- < 0: a long tail of small values.
- Rule of thumb: |skew| > 1 is strongly skewed. F7's skew of **21** comes from a handful of enormous values.

**Reading the row as a whole:** look at min, percentiles and max together. F7's 99th percentile is 7,244
but its max is 105,948, so the top 1 % is spread out wildly. That's an outlier signal.

### A2 Histograms

- **x-axis:** the feature's value. **Bar height:** the number of rows whose value falls in that bar's
  range. Our panels use 60 equal-width bins. The y-axis numbers are hidden on purpose, so compare
  *shapes*, not exact counts.
- **Axis range vs tick labels:** the *range* is set automatically to roughly [min, max] plus ~5 %
  padding. The *labelled ticks* are "nice" round numbers that matplotlib picks inside that range.
  They are not the min and max.
- **Why the F7 panel looks empty:** the axis must stretch to 105,948 for a few rows, so the bulk of the
  data (around 4,000) is squashed into one bar. The figure shows the problem that log + clip fixes.

### A3 The target: RMSD, Å, bimodal

- **RMSD (root-mean-square deviation):** overlay a predicted protein structure on the true one in the
  best possible way, then take the root of the average squared distance between matching atoms.
  This is the same idea as RMSE, applied to atom positions. **0 = perfect, larger = worse.**
- **Å = ångström**, a standard length unit: 1 Å = 10⁻¹⁰ m = 0.1 nm. Atomic distances are a few
  ångströms (a C–C bond ≈ 1.5 Å), so structural biology, the Protein Data Bank and CASP report RMSD in
  Å. (The UCI page doesn't state the unit, but the 0–21 range matches the CASP convention.) In the
  report, write it out on first use: "RMSD, measured in ångströms (Å)".
- **Bimodal = two peaks.** A *mode* is the most common value, so two modes means two separate clusters.
  RMSD has a tall peak at **1–3 Å** (near-native predictions) and a broad hump at **~11–17 Å** (poor
  predictions), with a dip in between. *Why it matters:* a model minimising squared error is pulled
  towards the middle, between the humps, where there is little data.

### A4 Splitting and cross-validation

**"Stratified on RMSD bin"** means: group the rows by RMSD range, then split *each group* in the same
proportion. Both sides of the split get the same mix of easy and hard cases. A random split is usually
close, but stratifying guarantees it, which matters most for the rare bins.

**Our procedure, step by step:**
1. **Bin:** every row goes into one of 7 RMSD bins of 3 Å (0–3, 3–6, …, 18–21).
2. **Hold out the test set:** within *each* bin, 20 % of rows go to test and 80 % to train, with a fixed
   seed (42). Result: 8,804 test / 35,215 train, with identical bin percentages. The test set is then
   **locked** and used once, at the very end.
3. **5 folds:** cut the 35,215 training rows into 5 stratified folds of ~7,043 rows.
4. **5 rounds:** each fold takes one turn as the **validation fold** (it scores the model). The
   other 4 folds are for training.
5. **Early stopping inside each round:** 10 % of those 4 training folds is set aside to decide *when to
   stop training*. The model trains on the remaining 90 %.
6. **Average:** 5 scores → **mean ± std**. If setting A beats setting B by less than the std, the
   difference may just be noise.

**Misconception to avoid: K = 5 is *not* tied to the 80/20 split, and no fold is the test set.**
The test set is removed *first* and never takes part in cross-validation. Each fold is 20 % of the
*training* set (16 % of all data). That 1/5 = 20 % is a coincidence.

**Why K = 5:** the usual trade-off. Larger K means more training data per round and more scores to
average, but K full trainings *per configuration*. 5 and 10 are both standard. 5 keeps our many
configurations × seeds affordable, and a ~7k-row validation fold still gives a stable score.

### A5 Feature relevance: Pearson, Spearman, mutual information

All three measure how strongly a feature is related to RMSD. We also used Spearman between *pairs of
features* (the heatmap) to detect redundancy.

**Pearson r: linear relationship**, from −1 to +1:

$$r = \frac{\sum_i (x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum_i (x_i-\bar{x})^2 \; \sum_i (y_i-\bar{y})^2}}$$

The numerator is the *covariance*: do x and y tend to be above their means at the same time? Dividing
by both spreads removes the units. r = 1 means a perfect rising straight line, and r = 0 means no
*linear* trend.

**Spearman ρ: monotonic relationship.** It is Pearson computed on the **ranks** (smallest = 1, next =
2, …) instead of the raw values. It asks "when x goes up, does y consistently go up (or down)?", even
along a curve. Ranks also mean outliers can't dominate it. Example: y = x³ gives Spearman = 1 and
Pearson < 1.

**Mutual information (MI): any relationship.** How much knowing X reduces uncertainty about Y:

$$I(X;Y) = \sum_{x,y} p(x,y)\,\log\frac{p(x,y)}{p(x)\,p(y)}$$

If X and Y are independent, p(x,y) = p(x)p(y), so the log is log 1 = 0 and **MI = 0**. More dependence
gives a larger MI. It has **no upper limit of 1**, which is why we added a **random-noise column** as a
"nothing" reference (MI = 0.000). For continuous data, sklearn estimates MI with nearest neighbours.

**Why all three:** a U-shaped relationship gives Pearson ≈ 0 and Spearman ≈ 0 but a high MI. In our
data, **F7 has r ≈ −0.003 but the highest MI (0.350)**. A correlation-only filter would have discarded
our most informative feature.

### A6 Outliers: log1p, IQR, transform and clip

**`log1p(x) = log(1 + x)`** compresses long right tails. A log turns ratios into differences:
1,000 → 6.9, 10,000 → 9.2, 100,000 → 11.5, so each ×10 becomes just +2.3. Skewed features become roughly
symmetric, and extreme values move much closer to the bulk. The **+1** is there because log(0) is
undefined and F7/F8 contain zeros. log1p(0) = 0.

**IQR (interquartile range)** = Q3 − Q1 = 75th percentile − 25th percentile: the spread of the middle
50 % of the data. The **IQR rule** (Tukey) flags a value as an outlier if it is
**< Q1 − 1.5·IQR** or **> Q3 + 1.5·IQR**. This works for roughly symmetric data. On skewed data it flags
the *natural* long tail. That's why it flagged ~1,900 F8 values that are not errors.

**Transform and clip:**
- **Transform:** apply log1p to change the *shape* of a skewed feature.
- **Clip (winsorise):** set limits from the **training** data at the 0.5th and 99.5th percentiles. A
  value below the lower limit is replaced by the lower limit, and a value above the upper limit by the
  upper limit. Example: F7 = 105,948 becomes ≈ 11,068, its 99.5th percentile.
- **The row is kept.** Only the extreme *input value* is capped. It still says "very high", just not
  "50 standard deviations high".
- We clip *after* the log. Because log never changes the order of values, that is equivalent to
  clipping the raw values at their raw percentiles.

### A7 Scaling: z-score vs min-max vs batch normalisation

**Z-score (standardisation)**, per feature:

$$z = \frac{x - \mu}{\sigma}$$

μ and σ come from the **training set only**. Afterwards every feature has mean 0 and std 1, and
z = 2 means "2 standard deviations above the training mean". Validation and test data are
transformed with the *same* training μ and σ.

**Why not min-max**, x' = (x − min)/(max − min), which maps to [0, 1]?
- **Depends on two single extreme points.** One outlier stretches the range and squashes everything
  else. (Our clipping already reduces this, so in fairness min-max would be workable here. The
  next two points decide it.)
- **Zero-centring.** Z-scores are centred on 0, min-max values are all positive. If every input to a
  neuron is positive, the gradients of all of that neuron's incoming weights share the same sign, so
  the weights can only all rise or all fall together. That causes zig-zagging and slower training. Zero-centred
  inputs avoid this and also suit tanh, which is centred on 0.
- **Initialisation assumptions.** Xavier and He initialisation assume roughly zero-mean, unit-variance
  inputs.

**Why not batch normalisation?** It's a different kind of tool:
- **It's a layer inside the network, not data preprocessing.** It normalises *hidden-layer
  activations* with each mini-batch's mean and std, then applies a learnable scale and shift. You'd
  still standardise the inputs.
- **It would compromise Part 2.** Batch norm makes a layer's output insensitive to the overall scale of its weights,
  which changes how the learning rate acts and what a "small weight" means. Both are exactly what
  the warmup and magnitude-pruning experiments measure, so it would be a confounding variable.
- **Its benefit is mainly for deep networks.** Our MLPs have 1–4 hidden layers.
- **It behaves differently in training and evaluation** (batch statistics vs running averages).

### A8 Test-prep questions with answers

**1. Why remove duplicates *before* splitting? What goes wrong otherwise?**
If a row and its exact copy land on opposite sides of a split, the model is tested on a row it
memorised during training. The test score is then optimistic (**data leakage**). Removing duplicates
after splitting would be too late, because the copies are already on both sides. We removed 1,711.

**2. A feature has Pearson r ≈ 0 with the target. Is it uninformative?**
Not necessarily. Pearson only detects *linear* relationships. Check Spearman (monotonic), mutual
information (any relationship) or a non-linear model. F7 has r = −0.003, yet it has the **highest MI**
(0.350) and removing it costs 0.017 R².

**3. What is the difference between an *uninformative* and a *redundant* feature? Which did we drop,
and why?**
*Uninformative:* no relationship with the target (MI ≈ the noise reference). *Redundant:* informative,
but its information is already contained in other features. We found **no** uninformative features.
We dropped one redundant feature: **F2 = F1 × F3 exactly** (non-polar area = total area × non-polar
fraction). Adding it back to the MLP changes nothing. **F5 ≈ 138.5 × F1** *looked* redundant and was
dropped at first, but the MLP check (5 seeds) showed it helps (~0.06 Å, 5/5 seeds). The ~2.6 %
deviation, F5/F1 ≈ average exposed-residue mass, carries information, and after the log it's a simple
difference the MLP can compute. **Lesson: "redundant" depends on the model.**

**4. Why does the IQR rule flag ~1,900 F8 values, and why did we not delete them?**
F8 is right-skewed (skew 1.7). The IQR rule assumes a roughly symmetric distribution, so it flags the
natural long tail. After log1p only 4 F8 values are beyond |z| > 4, so they are normal spread, not
errors. They are valid rows with valid targets. Clipping handles the few extremes.

**5. Why clip instead of delete? Why is the target never clipped?**
Clip: (a) the rows' targets are valid information, (b) at prediction time we can't refuse to predict
for a decoy, so the model must cope with such inputs, and (c) deleting would also remove hard rows from
the test set and flatter the score. Target: a large RMSD is a real bad fold, which is exactly what we
want to predict. Clipping it would teach the model wrong answers and distort the evaluation.

**6. Why fit the scaler and clip bounds on training data only? What does this prevent?**
It prevents **data leakage**. If test data influenced μ, σ or the percentiles, information about the
test set would be built into the model, and the test score would stop being an honest estimate of
performance on unseen data. It also mirrors deployment: future data isn't available when you fit.

**7. Why standardisation rather than min-max here? Give two reasons tied to how MLPs train.**
(1) **Zero-centred** inputs avoid all of a neuron's weight gradients sharing one sign, which gives
faster and less zig-zagged training. Zero-centring also suits tanh. (2) **Xavier/He initialisation assume
unit-variance inputs**, so activations neither explode nor vanish layer by layer. Also, min-max depends
on the two most extreme values. (Bonus: standardised inputs keep sigmoid and tanh out of their flat,
saturated regions.)

**8. What does "imbalance" mean for a regression target? How did we handle it?**
Some target ranges are much rarer than others. Ours is bimodal: the 0–3 Å bin holds 34 % of rows, the
18–21 Å bin 7 % (4.9×). *Training:* stratified folds and early-stopping splits. In Phase 2 we test
inverse-bin-frequency sample weights. *Testing:* a stratified test set. *Evaluation:* per-bin RMSE and
bias alongside the overall metrics, so failures in rare ranges can't hide behind a good average.

**9. Why stratify a *regression* split, and how?**
Stratification needs categories, but the target is continuous, so we **bin it** (7 bins of 3 Å) and
stratify on the bin labels. Why: every split then has the same target distribution, including
enough of the rarer 6–12 Å decoys. Train/test bin shares match to within 0.01 percentage points.

**10. Why a separate early-stopping split inside each CV fold?**
The stopping epoch is effectively a hyperparameter. Choosing it on the validation fold we report would
use evaluation data to make a decision, and the reported score would be optimistic. The 10 %
early-stopping split keeps "choosing" and "scoring" separate.

**11. Why might our test RMSE be optimistic?**
The data has no protein IDs. Many decoys come from the *same* protein and are similar, so
near-identical decoys can sit in both train and test, and the model partly "recognises" the protein.
We can't group by protein. *Comparisons* between settings are still fair, because every setting uses
the same splits.


---

## Part B: The MLP and training

Decisions: [`02-core-investigation.md`](02-core-investigation.md#baseline-phase-2). Code:
`src/model.py`, `src/training.py`, `src/cv.py`.

### B1 One training step, walked through

`src/training.py` does exactly this for every mini-batch of 256 rows:
1. **Set the learning rate** for this step from the schedule (`src/schedules.py`). This is where
   warmup happens.
2. **Forward pass:** `predictions = model(x_batch)`. Each layer computes `activation(W·x + b)`. The
   output layer has *no* activation, because regression outputs must be free to take any value.
3. **Loss:** mean squared error between the predictions and the true (standardised) RMSD.
4. **Backward pass:** `loss.backward()` applies the chain rule from the loss back through every layer
   (**backpropagation**). This gives ∂loss/∂w for every weight.
5. **Update:** `optimizer.step()` moves every weight against its gradient. For plain SGD that is
   w ← w − η·∂loss/∂w. Momentum, RMSprop and Adam modify this step (Part C).
6. `zero_grad()` clears the gradients first, because PyTorch *adds* new gradients to old ones.

One **epoch** = one pass over all training rows (≈ 100 steps for ~25k rows at batch 256). Rows are
reshuffled every epoch so the batches differ.

### B2 Early stopping

- **What it monitors:** MSE on a separate 10 % "early-stopping" split of the fold's training data,
  measured after every epoch.
- **What it does:** remembers the weights from the best epoch so far. If 20 epochs (**patience**) pass
  without improvement, it stops and **restores the best weights**, not the last ones.
- **Why:** training loss keeps falling forever, as the network memorises the training rows, but the
  error on unseen rows stops improving and eventually rises (**overfitting**). Our learning curves show
  exactly this: train MSE keeps dropping to ~0.37, while early-stopping MSE flattens at ~0.43–0.46
  after ~100 epochs. Early stopping is a form of **regularisation**.
- **Why a separate split:** choosing the epoch is a decision. Making it on the fold we report would
  make the reported score optimistic.
- **Patience** exists because the validation curve is noisy. Stopping at the first uptick would stop
  far too early.

### B3 Why baselines (predict-the-mean, linear regression)?

A number like "RMSE 4.09 Å" means nothing on its own. Baselines give it a scale:
- **Predict the mean** (R² = 0, RMSE 6.14 Å = the target's std) is the floor. A model that can't
  beat it has learned nothing.
- **Linear regression** (RMSE 5.19 Å, R² 0.29) tests whether non-linearity is needed. The MLP's ~21 %
  lower error says yes.
- **Gradient boosting** (4.13 Å) is a strong non-neural reference. The MLP matching it says our MLP
  is not under-performing. Both stopping at R² ≈ 0.55 *seemed* to show that the features set the
  limit, but Phase 3 disproved it (a 4 × 256 MLP reaches R² 0.615). See B8 Q10.

### B4 Weight initialisation (in `src/model.py`)

Goal: keep the variance of the signals roughly constant from layer to layer, so they neither
explode nor vanish.
- **Xavier/Glorot:** Var(w) = 2 / (fan_in + fan_out). Derived for symmetric activations such as tanh
  and sigmoid.
- **He/Kaiming:** Var(w) = 2 / fan_in. Twice Xavier's (roughly), because ReLU zeroes about half of its
  inputs, halving the variance.
- **"random" (naive):** a fixed std of 0.01 regardless of layer size. In deep nets, signals shrink
  layer by layer.
- Biases start at 0. Weights must *not* all start equal (e.g. all zero), or every unit in a layer
  computes the same thing and receives the same update forever (no **symmetry breaking**).

### B5 Seeds, folds and paired comparisons

- **Fold-to-fold variation** (std 0.03–0.05 Å) comes from *which rows* are in each fold.
- **Seed-to-seed variation** (~0.015 Å) comes from the random initial weights and batch order.
- **Paired comparison:** to compare two settings, run both with the *same* seeds and folds and look
  at the per-seed *difference*. Shared noise cancels. Example: the 8-feature set beat the 7-feature
  set in 5/5 seeds, even though the gap (0.056 Å) is similar in size to the fold std.

### B6 Loss weighting for imbalance: tested, rejected

Weighting each row by the inverse frequency of its RMSD bin makes every bin contribute equally to
the loss. Result: rare-bin errors fell 0.15–0.54 Å, but the common 0–3 and 3–6 Å bins got much worse
(+1.02, +0.51 Å), and overall RMSE rose 4.09 → 4.31 Å. The bias at high RMSD barely changed
(−5.6 → −5.4 Å), so the under-prediction of bad decoys is a **feature limitation (regression to the
mean)**, not a loss-function problem.

### B7 Practical: why one CPU thread?

Measured: 0.018 s/epoch on 1 thread vs 0.065 s/epoch on 8. For tiny matrix multiplications,
coordinating threads costs more than the computation. Parallelism comes from running the 5 CV folds
in separate processes instead. A full 5-fold CV takes ~5 s.

### B8 Test-prep questions with answers

**1. Walk through one training step.** Set LR → forward pass → MSE loss → `zero_grad` → `backward`
(backprop computes all gradients) → `optimizer.step()` (update weights). See B1.

**2. What does early stopping monitor, and why?** The MSE on a held-out 10 % split after each epoch.
It keeps the best epoch's weights and stops after 20 epochs without improvement. It prevents
overfitting: the training loss keeps falling, but the error on unseen data stops improving. See B2.

**3. Why compare against a predict-the-mean baseline?** It gives the error scale (R² = 0, RMSE =
the target's std) that any model must beat. Without it, "4.09 Å" is meaningless.

**4. Why does the output layer have no activation?** It's regression: the output must be able to take
any real value. A sigmoid would cap it to (0, 1), and a ReLU would forbid negatives (and our target is
standardised, so it *is* often negative).

**5. Why He init for ReLU and Xavier for tanh?** Both keep the activation variance stable across
layers. ReLU discards about half its inputs, so He uses twice the variance (2/fan_in).

**6. Why can't all weights start at zero?** Every unit in a layer would compute the same output and get
the same gradient, so they'd stay identical forever (no symmetry breaking).

**7. The training loss keeps falling but the early-stopping loss flattens. What's happening?** The
start of overfitting: the network fits patterns specific to its training rows that don't generalise.

**8. Why is the early-stopping curve so jagged?** Mini-batch gradients are noisy estimates of the
true gradient. With a constant LR the weights keep bouncing around the minimum. LR decay shrinks the
steps so they settle.

**9. Why did loss weighting not fix the high-RMSD errors?** The model under-predicts high RMSD because
the features don't distinguish those decoys well, so it hedges towards the middle. Weighting mostly
shifted all predictions up (hurting the common low bins) instead of fixing that.

**10. Two models (MLP, boosted trees) both reach R² ≈ 0.55. What does that suggest?** It *suggests*
a ceiling set by the features, and we believed that in Phase 2. **But it was wrong:** a larger MLP
(4 × 256) reaches R² 0.615. Both baselines were limited by capacity and default settings. Lesson: two
models agreeing is weak evidence of a ceiling. Test it by scaling the model up.

---

## Part C: Core investigation

Hypotheses and results: [`02-core-investigation.md`](02-core-investigation.md). Notebook section 3.

### C1 The four optimisers

Notation: w = a weight, g = its gradient at this step, η = learning rate.

| Optimiser | Update rule (per weight) | Idea in one line |
|---|---|---|
| **SGD** | w ← w − η·g | Step downhill, using the mini-batch gradient |
| **SGD + momentum** | v ← β·v + g  then  w ← w − η·v  (β = 0.9) | Keep a running "velocity". Consistent directions build up speed, and zig-zags cancel out |
| **RMSprop** | s ← ρ·s + (1 − ρ)·g²  then  w ← w − η·g / (√s + ε) | Divide by the recent typical gradient size, so every weight takes similar-sized steps |
| **Adam** | m ← β₁m + (1 − β₁)g, s ← β₂s + (1 − β₂)g², bias-correct both, then w ← w − η·m̂ / (√ŝ + ε) | Momentum (m) **plus** RMSprop-style scaling (s) |

**Key facts:**
- **Momentum's effective step:** if the gradient stays roughly the same, v grows to g / (1 − β), so with
  β = 0.9 the step is about **10× η**. That's why momentum tolerates a smaller maximum LR than plain SGD.
- **Adaptive methods (RMSprop, Adam)** make the step size roughly independent of the gradient's scale.
  That makes them forgiving of the LR choice, and it partly hides vanishing gradients (a tiny g is
  divided by a tiny √s).
- **Bias correction in Adam:** m and s start at 0, so early on they underestimate the true averages.
  Dividing by (1 − βᵗ) fixes that for the first steps.
- **Why compare on an LR grid, not one LR:** each optimiser has its own natural LR scale (SGD ~1e-2,
  Adam ~1e-3). One fixed LR would favour whichever optimiser it happens to suit.

### C2 Depth vs width

- **Width** (units per layer): how many features a layer can compute side by side. **Depth** (number of
  layers): how many times features can be built *from* other features (composition).
- **Universal approximation:** one hidden layer can approximate any continuous function, *given enough
  units*. Depth can do the same job with far fewer units for functions with a compositional structure.
- **Costs of depth:** longer backpropagation paths, so a higher risk of vanishing or exploding gradients,
  and harder optimisation.
- **Costs of size in general:** more parameters means more capacity to memorise the training rows
  (overfitting), and slower training. Early stopping limits the overfitting.
- **Parameter count** of a dense layer = inputs × outputs + outputs (the biases). Our 2×64 net:
  (8·64 + 64) + (64·64 + 64) + (64·1 + 1) = 4,801.

### C3 Activation functions

| Activation | Formula | Derivative | Output range | Notes |
|---|---|---|---|---|
| **Sigmoid** | 1 / (1 + e⁻ˣ) | σ(x)(1 − σ(x)) ≤ **0.25** | (0, 1) | Saturates. Not zero-centred |
| **tanh** | (eˣ − e⁻ˣ) / (eˣ + e⁻ˣ) | 1 − tanh²(x) ≤ **1** | (−1, 1) | Saturates, but zero-centred |
| **ReLU** | max(0, x) | 1 if x > 0, else 0 | [0, ∞) | No saturation for x > 0. Cheap. Can "die" |
| **Leaky ReLU** | x if x > 0, else 0.01x | 1 or 0.01 | (−∞, ∞) | Never fully dead |

- **Saturation:** for large |x|, sigmoid and tanh flatten out, so their derivative ≈ 0 and the
  gradient through them almost vanishes.
- **Vanishing gradients:** backprop multiplies by one activation derivative per layer. With sigmoid,
  each factor is ≤ 0.25, so over L layers the gradient reaching the first layer can shrink by up to
  0.25ᴸ. The early layers then barely learn. ReLU's derivative is exactly 1 for active units, so it
  passes gradients through undiminished.
- **Zero-centring:** sigmoid outputs are all positive, so the next layer's inputs are all positive, and
  all of a unit's weight gradients share one sign (zig-zag updates). tanh is centred on 0, which avoids
  this.
- **Dead ReLU:** if a unit's input becomes ≤ 0 for every row (e.g. after a large update pushes its bias
  very negative), it outputs 0 everywhere, its gradient is 0, and it can never recover. Leaky ReLU keeps
  a small slope (0.01) for negative inputs, so the unit can come back.
- **Matching initialisation:** Xavier for sigmoid/tanh, He for the ReLU family (B4).

### C4 What we found (verdicts in one place)

| Hypothesis | Verdict | Key number |
|---|---|---|
| H-A1 Adam/RMSprop fastest, SGD slowest | **Partly.** Adam fastest, SGD slowest, but momentum beat RMSprop | epochs to 0.50: Adam 9.7, momentum 12.5, RMSprop 29.5, SGD 293 |
| H-A2 All optimisers end within 0.05 Å | **Contradicted** | 3.99 (Adam) … 4.39 Å (SGD) |
| H-A3 Adaptive = LR-tolerant. Momentum breaks before SGD | **Adam yes, RMSprop no (least tolerant). Momentum part yes** | RMSprop diverges at ≥ 3e-2. Momentum breaks at 1e-1, SGD doesn't |
| H-B1 Width: big gain 16→64, < 0.05 Å for 64→256 | **Partly.** Diminishing, but still −0.14 Å | −0.33 / −0.14 Å |
| H-B2 Depth: 1→2 helps, beyond 2 doesn't | **Half.** 1→2 yes, 2→4 still −0.12 Å | best: 4 × 256 = 3.81 Å |
| H-B3 Bigger nets overfit sooner, early stopping saves them | **Supported** | best epoch 310 → 54, gap 0 → 0.19 |
| H-C1 ReLU > tanh > sigmoid in speed | **Supported** | 8.6 / 21 / 340 epochs (4 × 64) |
| H-C2 Sigmoid gradients vanish with depth, moderated by Adam | **Supported** | ≈ 4× shrink per layer. First ÷ last = 5.8e-7 at depth 8 |
| H-C3 tanh ≈ ReLU ≈ Leaky, Leaky gains nothing | **Partly.** Leaky = ReLU (< 0.5 % dead), but tanh beat ReLU at depth 4 | 3.90 vs 3.97 Å |

**The big lesson:** several hypotheses failed because they rested on the "feature ceiling" from Phase
2. The baseline was actually limited by capacity. Being wrong for a clear, identifiable reason is a
legitimate and useful result.

### C5 Test-prep questions with answers

**1. Why compare optimisers on an LR grid rather than at one LR?** Each optimiser has its own natural
LR scale. At a single LR, the comparison mostly measures which optimiser that LR happens to suit. On a
grid we compare each at its *best* LR and also see how sensitive it is.

**2. Why did momentum diverge at an LR where plain SGD still trained?** With β = 0.9 the velocity
accumulates to ≈ g / (1 − β) = 10g, so the effective step is ~10× the LR. At LR 0.1 that is like plain
SGD at LR 1, which is too large.

**3. Why was RMSprop the most fragile at high LRs, when Adam was the most tolerant?** PyTorch's RMSprop
has no bias correction. Its g² average starts at 0, so the first steps are ~10× the LR (we measured
exactly 10× on step 1). Adam bias-corrects, so its first step is 1× the LR. Oversized early steps are
also the classic motivation for warmup.

**4. Why did "final quality" differ even though the network was the same?** Under a fixed epoch budget
and early stopping, a slow optimiser (plain SGD) never reaches the good region. Its best epoch was
≈ 500 = the limit. Speed turns into quality when training time is limited.

**5. Bigger networks overfit more. Why was the biggest one still the best?** Early stopping stopped it
at epoch ~54, before the overfitting hurt the validation error. Capacity helps as long as something
(here early stopping) controls the overfitting.

**6. Depth or width: which was more efficient here?** Depth. 4 × 64 (13k parameters) matched 2 × 256
(68k parameters).

**7. Explain the sigmoid gradient plot.** Backprop multiplies by each layer's activation derivative.
Sigmoid's is at most 0.25, so each layer shrinks the gradient ≈ 4×. Over 8 layers that compounds to
≈ 10⁻⁶ at the first layer: **vanishing gradients**. tanh (derivative up to 1) and ReLU (exactly 1 for
active units) keep gradients roughly constant across hidden layers.

**8. If sigmoid's gradients vanish so badly, why did it still reach 4.26 Å?** Adam divides each
weight's step by its own running gradient size, so tiny gradients still produce normal-sized steps. The
vanishing showed up as *slow* training (340 vs 8.6 epochs), not as failure. With plain SGD it would
be far worse.

**9. Why didn't Leaky ReLU beat ReLU?** Leaky ReLU fixes *dead* units, and fewer than 0.5 % of our ReLU
units died (small network, He init, standardised inputs, moderate LR). There was nothing to fix. At
high LRs, though (Adam at 1e-1: 48 % dead), it might matter.

**10. Why choose SGD + momentum for Part 2 when Adam was best?** Part 2 tests whether warmup raises
the LR a network can tolerate. Momentum has a *sharp* tolerance boundary (fine at 3e-2, broken at
1e-1), so an effect of warmup on that boundary is easy to see. Adam degrades gradually and already
self-corrects its first steps (bias correction), which would blur the effect. Momentum SGD is also
what the original papers used.

---

## Part D: Warmup and pruning

Design and results: [`03-warmup-pruning.md`](03-warmup-pruning.md). Notebook section 4.

### D1 Key concepts

- **Learning-rate warmup:** start training with a tiny LR and increase it (here linearly over the first
  5 epochs) to the target ("peak") LR, then follow the normal schedule (here cosine decay to 0).
- **Why warmup helps** (Kalra & Barkeshli): at initialisation the loss landscape can be sharp (badly
  conditioned), and a large step there overshoots and blows up. Small early steps let the network move
  to a flatter, better-conditioned region, after which a large LR is safe. **Its main benefit is
  letting you use a larger target LR.**
- **Magnitude pruning:** remove (set to 0) the weights with the smallest |w|, assuming they matter
  least. **Sparsity** = fraction of weights removed.
- **Global vs layer-wise:** global ranks all weights together (layers can end up with different
  sparsities). Layer-wise prunes the same fraction from each layer. We used global. Biases are not
  pruned.
- **One-shot vs iterative:** one-shot prunes to the target sparsity in one go. Iterative alternates
  pruning a little and retraining (as in the Lottery Ticket paper). We used one-shot.
- **Masks:** a 0/1 tensor per weight matrix. After every optimiser step during fine-tuning, the weights
  are multiplied by the mask, or the optimiser (and momentum) would move pruned weights away from 0.
- **Retraining after pruning** (Renda et al.):
  - **Fine-tuning:** continue from the final weights with a *small fixed* LR (ours: 0.001, 10 epochs).
  - **Weight rewinding:** reset the surviving weights to an earlier point in training and retrain with
    the original schedule.
  - **LR rewinding:** keep the final weights but retrain with the *original* LR schedule (large LR).
    Both rewinding methods beat fine-tuning.
- **Lottery Ticket Hypothesis** (Frankle & Carbin): a dense, randomly initialised network contains a
  sparse subnetwork (a "winning ticket", often 10–20 % of the weights) that, trained from the *same
  initial weights*, matches the full network's accuracy. For deeper networks they needed warmup (or a
  lower LR) to find such tickets.

### D2 The experiment in one paragraph

We trained the same network three ways: **A** without warmup at the largest LR that works without it
(0.005), **B** with warmup at a much larger LR (0.2) chosen so its unpruned accuracy matched A's, and
**C** with warmup at A's LR (the control). Then we pruned each at 0–98 % sparsity, fine-tuned all of
them identically, and compared the error added by pruning (Δ), over 5 seeds.

### D3 What we found

| Hypothesis | Verdict | Key number |
|---|---|---|
| H1 warmup raises the tolerable LR | **Strongly supported** | 0.005 → 0.1 (trains at 0.2). ≥ 20×. All no-warmup failures in epoch 1 |
| H2 at matched accuracy, warmup net is more pruning-robust | **Supported** | Δ smaller by 0.22–0.27 Å at ≥ 70 %, 5/5 seeds |
| H3 benefit comes from the larger LR, not warmup itself | **Supported** | C (warmup, small LR) ≈ A. B beats C by 0.26–0.32 Å |
| E1 LR rewinding shrinks the A–B gap | **Contradicted:** the gap widens (0.19 → 0.70 Å) | Large retraining LR helps recovery ([2] reproduced) |
| E2 longer warmup adds nothing | **Partly:** 1 ≈ 5 epochs, but 15 is *less* robust | Fewer epochs at the large LR |

**Answer to the research question:** warmup improves pruning robustness **indirectly**. It unlocks a
large learning rate, and the large learning rate is what makes the network robust (and, with LR
rewinding, helps it recover).

### D4 Test-prep questions with answers

**1. What is learning-rate warmup, and what is its main benefit according to Kalra & Barkeshli?** Start
with a tiny LR and ramp it up to the target over the first steps/epochs. Main benefit: the network
tolerates a *larger target LR*, because the small early steps move it to a better-conditioned (less
sharp) region before the large steps start.

**2. What did our LR sweep show, and why did all failures happen in epoch 1?** Without warmup, LR ≥ 0.01
diverged (0.005 was the max). With warmup everything up to 0.2 trained. All 17 failures were in the first
epoch: at initialisation the landscape is sharp, so full-size steps (amplified ~10× by momentum)
overshoot immediately. Warmup's small first steps avoid exactly this.

**3. Why did we need a control condition C?** A vs B differs in *two* things: warmup *and* LR. If B is
more robust, we can't tell which caused it. C has warmup but A's LR. C ≈ A while B ≫ both, so the LR is
the cause and warmup only matters as the enabler.

**4. What does "comparable unpruned accuracy" mean here, and why does it matter?** The spec requires the
unpruned networks to perform similarly, so that differences after pruning reflect *robustness*, not a
better starting point. Our pre-registered rule: B's validation RMSE within 0.05 Å of A's (3.73 vs
3.74 Å). B's test RMSE was even slightly worse than A's.

**5. Why fix the decision rules before running?** Otherwise we could (even unintentionally) pick the B
learning rate or the threshold that makes the result look best. Committing the rules first (`243937b`)
means the conditions were chosen mechanically.

**6. How do we know dead units didn't create B's robustness?** Large LRs can kill ReLU units, whose
weights could then be pruned for free. But B had the *fewest* dead units (0.2 % vs 3.6 % for A), so this
can't explain it.

**7. Why measure both before and after fine-tuning?** Before fine-tuning shows the raw damage of removing
weights. After fine-tuning shows how much the network can *recover*. The spec asks for the recovered
accuracy. B was better on both (dramatically before fine-tuning: 4.6 vs 8.0 Å at 70 %).

**8. Why do pruning masks have to be re-applied after every step?** The gradient of a pruned weight is
generally non-zero, and momentum keeps pushing too, so one optimiser step would make it non-zero again.
Multiplying by the mask after each step keeps the network truly sparse.

**9. Why did the A–B gap get *bigger* with LR rewinding?** With rewinding, each network retrains with its
own schedule: B at LR 0.2, A at 0.005. A large retraining LR recovers better (Renda et al.), so B
benefits twice: from more robust trained weights and from more effective recovery.

**10. A possible reason a large-LR network is more robust to *magnitude* pruning?** (Post-hoc only.)
B's weight mass is more concentrated in its largest weights (top 10 % hold 60.6 % of Σw² vs 56.5 % for
A), so removing the many small weights costs less. Another untested candidate: large-LR training finds
flatter minima, where zeroing weights changes the loss less.

**11. What are the main limitations?** One dataset and architecture (a tabular MLP, unlike the papers'
CNNs). One-shot rather than iterative pruning. A single fit/validation split. B's LR at the top of the
grid. The *why* only explored post-hoc.
