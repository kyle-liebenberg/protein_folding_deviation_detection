# 01: Data Preparation

> Covers spec §2.1 items 1–5. This is the source for the report's data section.
> Code: `src/data.py`, `src/preprocessing.py`, `src/evaluation.py`. Evidence: notebook section 1,
> figures in `figures/eda/`.

## Summary

| Step | Decision | Fitted on |
|---|---|---|
| Cleaning | Remove 1,711 exact duplicate rows → **44,019 rows** | — |
| Splitting | 80/20 train/test (35,215 / 8,804), **stratified on 3 Å RMSD bins**, fixed seed 42. **5-fold stratified CV** on train. 10 % of each fold's training part held out for early stopping | — |
| Feature relevance | Keep **8** features. Drop **F2** (= F1 × F3 exactly) as **redundant**. F5 looked redundant but is kept (MLP check). No feature is uninformative | train |
| Outliers | `log1p` on skewed features, then **clip** to the 0.5th–99.5th percentiles. **No rows deleted.** Target untouched | train |
| Scaling | **Standardise** features (z-score). Standardise the target for training, report metrics in Å | train |
| Imbalance | Stratified splits, **per-bin RMSE/bias** in evaluation. Loss weighting tested and rejected (Phase 2) | train |

**One rule behind every step:** after the test set is split off, every statistic (correlations, MI,
percentiles, means, stds) is computed on the **training set only**. Anything learned from the test
set would leak into the model and make the test score optimistic.

---

## The dataset

45,730 rows from the CASP protein-structure-prediction experiments. Each row is a *decoy* (a
candidate 3D structure of a protein). The target **RMSD** (Å) measures how far the decoy is from the
true structure: 0 = perfect, larger = worse. The 9 features are physicochemical descriptors:

| Feature | Meaning (UCI) |
|---|---|
| F1 | Total surface area |
| F2 | Non-polar exposed area |
| F3 | Fractional area of exposed non-polar residue |
| F4 | Fractional area of exposed non-polar part of residue |
| F5 | Molecular-mass-weighted exposed area |
| F6 | Average deviation from standard exposed area of residue |
| F7 | Euclidean distance |
| F8 | Secondary structure penalty |
| F9 | Spatial distribution constraints (N, K value) |

First look: no missing values, all values non-negative. F8 is an integer count (341 distinct values).
Six features have skew > 1, and F7 is extreme (skew 21). RMSD ranges over 0–21 Å (272 rows exactly 0).

## 0. Cleaning: duplicates

**1,711 rows (3.7 %) are exact duplicates** (features *and* target). We remove them before splitting.

*Why:* a duplicated row could land in both train and test. The model would then be scored on a row
it memorised, which is data leakage. A further 83 rows share identical features but have
*different* RMSD. These are kept: they are genuine label noise (no model can predict both values),
and they set a small floor on achievable error.

## 1. Feature relevance

Evidence on the training set (`figures/eda/feature_correlation.png`):

| Feature | Pearson r | Spearman ρ | Mutual info | ΔR² when left out* |
|---|---|---|---|---|
| F1 | −0.014 | −0.008 | 0.231 | −0.005 |
| F2 | 0.159 | 0.163 | 0.187 | **+0.001** |
| F3 | 0.375 | 0.389 | 0.176 | −0.006 |
| F4 | −0.169 | −0.144 | 0.280 | −0.055 |
| F5 | −0.013 | −0.011 | 0.228 | **−0.003** |
| F6 | −0.035 | −0.021 | 0.246 | −0.021 |
| F7 | −0.003 | −0.029 | 0.350 | −0.017 |
| F8 | −0.000 | 0.073 | 0.243 | −0.053 |
| F9 | 0.062 | 0.061 | 0.297 | −0.008 |
| *random noise* | 0.013 | 0.013 | **0.000** | — |

\*Gradient-boosted trees, 5-fold CV R² on train. All 9 features: 0.546 ± 0.009. Without F2 and F5:
0.544 ± 0.006.

**Findings**

1. **No feature is uninformative.** Every feature's mutual information (0.18–0.35) is far above the
   noise reference (0.000). Yet F1, F5, F6, F7 and F8 have almost zero linear correlation with RMSD.
   The relationships are **non-linear**, so a correlation-based filter would wrongly throw away
   useful features. This is also the argument for a non-linear model (MLP) over linear regression.
2. **F2 is an exact function of other features: F2 = F1 × F3.** The ratio F2 / (F1·F3) has a
   coefficient of variation of 0.0000. That follows from the definitions: F3 is the *fraction* of the
   exposed area that is non-polar, so non-polar exposed area = total area × fraction. Leaving F2 out
   changes R² by +0.001 (noise).
3. **F5 is almost, but not exactly, proportional to F1: F5 ≈ 138.5 × F1** (coefficient of variation of
   the ratio 2.6 %, correlation 1.00). Mass-weighted area ≈ area × average residue mass. The tree
   model barely notices its removal (−0.003 R²). **But the ratio F5/F1 itself has MI 0.089 with RMSD**
   (noise: 0.000): the small deviation describes *composition* (average mass of the exposed
   residues), not size.
4. F4 and F6 also correlate strongly with F1 (ρ ≈ 0.94–0.96) but are **not** functions of it. Removing
   them costs real accuracy (F4 costs the most of any feature), so they stay.

**Decision (revised in Phase 2): drop only F2, keep 8 inputs: F1, F3, F4, F5, F6, F7, F8, F9.**

*History of this decision (worth knowing for the test):* Phase 1 initially dropped both F2 and F5,
based on the correlations and the tree model. Phase 2 re-checked with the **MLP itself** over 5 seeds
(notebook §2.3):

| Inputs | CV RMSE (Å), mean of 5 seeds | Better than the 7-feature set in |
|---|---|---|
| 7: drop F2 and F5 | 4.153 | — |
| 8: drop F5 only | 4.160 | 3/5 seeds (noise) |
| **8: drop F2 only (chosen)** | **4.097** | **5/5 seeds** |
| 9: all | 4.115 | 4/5 seeds |

- **F2:** adding it back changes nothing meaningful (+0.018 Å, within seed noise). It is an exact
  product of other features, so it's dropped.
- **F5:** removing it costs ~0.06 Å, consistently. After `log1p`, log F5 − log F1 is *exactly* the log
  of the informative ratio. That's a simple difference an MLP's first layer computes with two weights.
  A tree splits on one feature at a time and needs many splits to approximate a difference, which is
  why the tree check missed it. **Lesson: "redundant" depends on the model. Check with the model you
  will actually use.**

*Why drop a redundant feature at all, if it doesn't hurt?* Fewer inputs means fewer first-layer
weights, and a perfectly dependent input adds no information, only another weight to fit.

*Test-ready distinction:* **uninformative** = no relationship with the target (MI ≈ noise).
**Redundant** = informative, but the information is already present in other features. The spec asks
about uninformative features. We found none, but found one truly redundant one (F2).

## 2. Outliers

| Feature | IQR-rule flags (raw) | \|z\|>4 (raw) | IQR-rule flags (after log1p) | \|z\|>4 (after log1p) |
|---|---|---|---|---|
| F1 | 885 | 39 | 29 | 0 |
| F3 | 295 | 7 | 295 | 7 |
| F4 | 924 | 52 | 9 | 3 |
| F5 | 821 | 38 | 19 | 0 |
| F6 | 978 | 25 | 14 | 0 |
| **F7** | 411 | **150** | **1,053** | **121** |
| F8 | 1,864 | 156 | 573 | 4 |
| F9 | 95 | 0 | 95 | 0 |

(F3 and F9 are not log-transformed, so their counts don't change.)

**Findings**

1. Textbook rules (IQR, z-score) assume a roughly symmetric distribution. On right-skewed, positive
   measurements like areas, they mostly flag the **natural long tail**, not errors. After `log1p`,
   the flags for F1, F4, F6 and F8 almost disappear.
2. **F7 is genuinely anomalous.** Its maximum (105,948) is ~28× its median (3,841), and it keeps
   extreme values at *both* ends even after the log. (One row has F7 = 0 → z ≈ −25. Its IQR count
   *rises* after the log because the compressed bulk makes the low tail stand out.) See
   `figures/eda/outliers_before_after_log.png`.
3. The 177 rows above F7's 99.5th percentile have **ordinary** RMSD values (median 5.95 Å vs 5.11 Å
   overall, spanning the full range), and their other features look normal. The anomaly is in F7
   alone.

**Decision: transform and clip (winsorise), don't delete.**
- `log1p` on the skewed inputs (F1, F4, F5, F6, F7, F8). We use `log1p(x) = log(1+x)` because F7 and F8
  contain zeros.
- Then **clip every feature to its training-set 0.5th–99.5th percentile.** An extreme input then
  becomes "very high" instead of "50 standard deviations high", so it cannot dominate a neuron's
  weighted sum or produce huge gradients.
- *Why not delete the rows?* (a) Their targets are valid and informative. (b) At prediction time we
  can't refuse to predict for a decoy, so the model must handle such inputs anyway. (c) Deletion
  would also remove rows from the test set, flattering the score.
- **The target is never treated as an outlier.** A large RMSD is a real bad fold, which is exactly
  what we want to predict.

## 3. Scaling

Pipeline (in `Preprocessor`, fitted on training data): **select 8 features → log1p (skewed ones) →
clip → standardise (z = (x − μ)/σ)**. After this, every training input has mean 0, std 1 and range
≈ [−3.5, 3.5] (`figures/eda/preprocessed_distributions.png`).

**Why standardisation:**
- **Conditioning of gradient descent.** Raw F5 ≈ 10⁶ and F3 ≈ 0.3. Unscaled, the weights for different
  inputs would need learning rates differing by orders of magnitude. One global learning rate only
  works if the inputs share a scale.
- **Initialisation assumptions.** Xavier/He initialisation set weight variances assuming unit-variance
  inputs, so that activations neither explode nor vanish layer by layer.
- **Saturating activations.** Sigmoid and tanh flatten out (gradient ≈ 0) for large inputs. With
  standardised inputs, pre-activations start in the responsive region. This matters for our
  activation-function axis.
- **Why not min-max?** Min-max uses the extremes. F7's maximum would squeeze 99 % of its values into
  a tiny slice of [0, 1]. Log + clip + z-score is robust to that.
- All features are continuous (F8 is a count with 341 levels and is treated as continuous), so one
  treatment fits all. No one-hot encoding is needed.

**Target:** standardised for training, so the network's output and loss are on a unit scale and
the learning rate means the same thing regardless of the target's units. Predictions are converted
back to Å before *any* metric is computed.

## 4. Splitting

- **Hold-out test set: 20 % (8,804 rows), stratified on 3 Å RMSD bins, seed 42.** Stratification
  makes the test set's target distribution match training to within 0.01 percentage points per bin.
  Without it, a random split could under-represent the rarer 6–12 Å decoys. The test set is
  **locked**: used once, for final reporting.
- **5-fold stratified cross-validation on the 35,215 training rows** for every hyperparameter
  comparison (spec requirement). Reporting mean ± std over folds shows whether a difference between
  two settings is bigger than the fold-to-fold noise.
- **Early-stopping split inside each fold:** 10 % of the fold's training part (stratified). If we
  early-stopped on the fold's evaluation part, we would choose the stopping epoch using the data we
  report, which is optimistic.
- *Why 80/20 and k = 5?* With 44k rows, 20 % (≈ 8.8k) gives a precise test estimate. Each CV
  validation fold still has ≈ 7k rows, and 5 folds keeps the cost of many experiments manageable.
- **Known limitation:** the data has no protein identifier. Many decoys come from the same protein,
  so near-identical decoys can sit on both sides of a split. We can't group by protein, so absolute
  scores may be somewhat optimistic. *Comparisons* between settings are unaffected because every
  setting uses the same splits.

## 5. Imbalance

RMSD is not just skewed, it is **bimodal**: a tall peak of near-native decoys at 1–3 Å and a broad
second hump at ~11–17 Å (`figures/eda/rmsd_distribution.png`). Per 3 Å bin (training set):

| Bin (Å) | 0–3 | 3–6 | 6–9 | 9–12 | 12–15 | 15–18 | 18–21 |
|---|---|---|---|---|---|---|---|
| Share | 34.3 % | 18.8 % | 7.5 % | 8.9 % | 12.1 % | 11.3 % | 7.0 % |

The largest bin is ~4.9× the smallest. A model minimising plain MSE is pulled towards the dense
regions.

**The problem, shown with a predict-the-mean baseline** (CV fold 1, *not* the test set): overall RMSE
6.13 Å, R² = 0.00. Per bin, the RMSE ranges from 0.9 Å (6–9, near the mean of 7.8) to 11.6 Å
(18–21), with bias +5.9 Å in the lowest bin and −11.6 Å in the highest. A single overall number hides
*where* a model fails. It also shows **regression to the mean**, which we expect weaker models to
show too.

**How it is handled:**
- **Training:** stratified CV folds and early-stopping splits, so every fold sees every RMSD range in
  the same proportion. Loss: **plain (unweighted) MSE**. Phase 2 (notebook §2.4) tested
  inverse-bin-frequency sample weights and **rejected** them. They cut the error in the five rarer
  bins by 0.15–0.54 Å but raised it in the two most common bins (0–3 Å: +1.02 Å, 3–6 Å: +0.51 Å; 53 %
  of rows), so overall RMSE got worse (4.09 → 4.31 Å). They also hardly reduced the under-prediction
  of high RMSD (bias −5.6 → −5.4 Å). That error comes from the features not separating bad decoys
  well, not from the loss ignoring them.
- **Testing:** the test set is stratified on the same bins.
- **Evaluation:** overall RMSE / MAE / R² **plus per-bin RMSE and bias** (`per_bin_metrics`), so
  rare-bin performance is always visible.

*Why not log-transform the target?* That is the usual fix for a *right-skewed* target with a long
tail. Ours is bounded (0–21 Å) and bimodal. A log would stretch the 0–3 Å region and compress the
second mode, making the high-RMSD decoys look more similar than they are.

---

## 🎓 Test-prep questions

Answers: [`study-notes.md` § A8](study-notes.md#a8-test-prep-questions-with-answers).

1. Why remove duplicates *before* splitting? What goes wrong otherwise?
2. A feature has Pearson r ≈ 0 with the target. Is it uninformative? (No. Check MI or a non-linear
   model. F7 has r ≈ 0 but the highest MI.)
3. What's the difference between an *uninformative* and a *redundant* feature? Which did we drop, and
   why?
4. Why does the IQR rule flag ~1,900 F8 values as outliers, and why did we not delete them?
5. Why clip instead of delete? Why is the target never clipped?
6. Why fit the scaler/clip bounds on training data only? Name the problem this prevents.
7. Why standardisation rather than min-max for this data? Give two reasons tied to how MLPs train.
8. What does "imbalance" mean for a regression target? How did we handle it in training, testing and
   evaluation?
9. Why stratify a *regression* split, and how? (Bin the target.)
10. Why a separate early-stopping split inside each CV fold?
11. Why might our test RMSE be optimistic? (Decoys of the same protein across splits.)
