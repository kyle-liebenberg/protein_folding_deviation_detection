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
- Part C: Core investigation *(Phase 3)*
- Part D: Warmup and pruning *(Phase 4)*

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
  is not under-performing. Both stopping at R² ≈ 0.55 suggests the *features* set the limit.

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

**10. Two models (MLP, boosted trees) both reach R² ≈ 0.55. What does that suggest?** The limit is
the information in the features, not the model's capacity. Expect small differences between
reasonable MLP configurations.
