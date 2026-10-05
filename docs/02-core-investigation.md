# 02: Core Investigation

> Covers spec §2.2: an MLP baseline, then controlled experiments on **optimisers**, **architecture**
> and **activation functions**, evaluated with k-fold CV on the training set.
> Code: `src/model.py`, `src/schedules.py`, `src/training.py`, `src/cv.py`. Evidence: notebook
> sections 2–3, figures in `figures/phase2/` (and `figures/phase3/` later), numbers in `results/`.

## Experimental protocol (applies to every experiment)

1. **Hypothesis first.** Each axis's hypothesis is written in this file *before* its experiment runs.
2. **One variable at a time.** Everything not under study stays at the [baseline](#baseline-configuration).
3. **Same data for everyone.** 5-fold stratified CV on the 35,215 training rows, with the same folds for
   every configuration. Inside each fold, 10 % of the fold's training part is used for early stopping.
   The preprocessor is fitted on the remaining 90 % only.
4. **Metrics in Å.** Validation RMSE (headline), MAE and R², reported as mean ± std over folds.
   Per-bin errors where relevant.
5. **Noise awareness.** Seed-to-seed variation of the baseline is ~0.015 Å RMSE, and fold-to-fold std
   is 0.03–0.05 Å. Differences smaller than this are not treated as real. Where differences are
   small, we repeat over seeds and compare *paired* (same seed, same folds).
6. **The test set stays locked** until final reporting.

---

## Baseline (Phase 2)

### Baseline configuration

| Setting | Value | Why |
|---|---|---|
| Inputs | 8 features (F2 dropped) | [01](01-data-preparation.md#1-feature-relevance) + check below |
| Architecture | 2 hidden layers × 64 units (4,801 parameters) | a common, modest starting point |
| Activation / init | ReLU, He init | the standard pairing (He compensates for ReLU zeroing half its inputs) |
| Optimiser | Adam, LR 1e-3, batch 256, constant LR | widely used defaults |
| Loss | unweighted MSE on the standardised target | check below |
| Stopping | early stopping on the 10 % split, patience 20, max 500 epochs. The best epoch's weights are kept | prevents overfitting without a fixed epoch count |

### Hypothesis

The MLP will clearly beat linear regression, because Phase 1 showed the feature–RMSD relationships
are mostly non-linear (high MI, near-zero Pearson r). It should land close to gradient-boosted trees,
a strong reference model for tabular data.

### Results (5-fold CV, training set)

| Model | RMSE (Å) | MAE (Å) | R² |
|---|---|---|---|
| Predict the mean | 6.14 ± 0.01 | 5.52 | 0.00 |
| Linear regression | 5.19 ± 0.05 | 4.25 | 0.29 |
| Gradient boosting (reference) | 4.13 ± 0.04 | 3.12 | 0.55 |
| **MLP 2×64 (3 seeds)** | **4.08–4.10 ± 0.03–0.05** | **3.00–3.03** | **0.555** |

Learning curves: `figures/phase2/baseline_learning_curves.png`.

### Interpretation: hypothesis supported

- The MLP has ~21 % lower RMSE than linear regression, which confirms the non-linearity. It matches
  (slightly beats) gradient boosting.
- **Learning curves:** training loss keeps falling, while the early-stopping loss flattens after ~100
  epochs, and the gap widens. That's the onset of **overfitting**. Early stopping kept epochs 117–196.
  The early-stopping curve is jagged because the LR is constant: mini-batch noise keeps the weights
  jittering around a minimum. LR decay would let them settle (relevant to the warmup study).
- **A ceiling:** an MLP and boosted trees both stop at R² ≈ 0.55. The limit is probably the
  *information in the features*, not the model. Phase 3 should therefore expect **small differences
  in final RMSE** between reasonable configurations. The more revealing differences may be in
  **convergence speed and stability**.

### Two decisions re-checked with the MLP

**Feature set (5 seeds, paired):** dropping only F2 is best (4.097 Å), beating the 7-feature set in
5/5 seeds. F5 had been dropped in Phase 1 but carries a small amount of information that the MLP uses.
Full story in [01 §1](01-data-preparation.md#1-feature-relevance).

**Loss weighting (3 seeds):** inverse-bin-frequency weights *reduce* rare-bin error (0.15–0.54 Å) but
*raise* error in the two most common bins (+1.02 and +0.51 Å), and overall RMSE worsens 4.09 → 4.31 Å.
The high-RMSD bias barely changes (−5.6 → −5.4 Å), so the under-prediction of bad decoys comes from
the features, not the loss. **Rejected:** we keep plain MSE and handle imbalance through stratification
and per-bin reporting. Details: [01 §5](01-data-preparation.md#5-imbalance).

---

## Axis A: Optimisers *(Phase 3, to come)*

## Axis B: Architecture *(Phase 3, to come)*

## Axis C: Activation functions *(Phase 3, to come)*
