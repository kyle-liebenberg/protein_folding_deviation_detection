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
- **~~A ceiling~~ (corrected after Phase 3):** an MLP and boosted trees both stopped at R² ≈ 0.55,
  which suggested the features set a ceiling. **Phase 3 disproved this:** a 4 × 256 MLP reaches
  R² ≈ 0.615 (3.81 Å). Both baselines were limited by capacity and default settings. Two models
  agreeing is weak evidence of a ceiling. (The Phase 3 hypotheses were written while we still believed
  in the ceiling, which is why several of them under-estimate the effect of capacity.)

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

## Phase 3: shared measurements

> **Status:** hypotheses approved and committed on 2026-10-06 (commit `90ef669`) **before** any Phase 3
> experiment ran. The hypothesis text below is unchanged from that commit. Results and verdicts are
> added under each axis.

Every axis reports the same four things, so the axes can be compared:

| Measure | Definition | Answers |
|---|---|---|
| **Final quality** | CV RMSE (Å) of the early-stopped model, mean ± std over 5 folds, averaged over **3 seeds** | "How good does it get?" |
| **Convergence speed** | Epochs until the early-stopping MSE first drops below **0.50** (standardised units: predict-the-mean = 1.0, baseline ends at ≈ 0.43) | "How fast does it get good?" |
| **Stability** | Folds that diverged, plus the spread (std) across folds and seeds | "How reliable is it?" |
| **Learning curves** | Early-stopping MSE per epoch, averaged over folds | Shows *how* the above differ |

**What counts as a real difference:** larger than the baseline's noise (seed ~0.015 Å, fold std
0.03–0.05 Å) and in the same direction for all 3 seeds.

**Expectation that frames all three axes:** Phase 2 showed an MLP and boosted trees both stop at
R² ≈ 0.55, so the features probably set the ceiling. We therefore expect **small differences in final
RMSE** between sensible configurations, and **larger differences in speed and stability**. Several
hypotheses below are stated that way on purpose.

---

## Axis A: Optimisers

**Design.** SGD, SGD + momentum (β = 0.9), RMSprop and Adam, each on the **same learning-rate grid**
{1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1} × 3 seeds. Everything else is the baseline (2×64 ReLU,
He init, batch 256, constant LR, early stopping). Why one shared grid: each optimiser has a different
natural LR scale, so comparing them at one fixed LR would be unfair. The grid finds each optimiser's
best LR *and* shows how sensitive it is to the choice. This also feeds Part 2, which is about LR
tolerance.

**Hypotheses.**
- **H-A1 (speed):** at their best LRs, **Adam and RMSprop reach MSE < 0.50 in the fewest epochs,
  plain SGD in the most, with momentum in between.**
  *Why:* momentum accumulates a velocity along directions the gradient keeps pointing in, which speeds
  up progress and damps zig-zagging. Adam and RMSprop also divide each weight's step by a running
  average of its gradient size, so weights with small gradients still move.
- **H-A2 (final quality):** at their best LRs, **all four optimisers reach a final CV RMSE within
  ~0.05 Å of each other.**
  *Why:* if the features cap performance (R² ≈ 0.55), the optimiser decides how quickly we reach the
  ceiling, not how high it is.
- **H-A3 (LR sensitivity and stability):** **Adam and RMSprop give near-best RMSE over a wider range of
  LRs than SGD does. SGD with momentum diverges at a lower LR than plain SGD.**
  *Why:* adaptive methods normalise the step size, which makes them forgiving. With β = 0.9,
  momentum's effective step is about η / (1 − β) = 10η, so it should blow up roughly one grid step
  earlier than plain SGD.

**Evidence that would contradict them:** SGD converging as fast as Adam (H-A1). A gap > 0.05 Å that is
consistent across seeds (H-A2). Adam failing over as wide an LR range as SGD (H-A3).

### Results (notebook §3.1, `results/phase3_optimisers.csv`, `figures/phase3/optimisers_*.png`)

Each optimiser at its own best LR (3 seeds × 5 folds). Speed = epochs until early-stopping MSE < 0.50.

| Optimiser | Best LR | CV RMSE (Å) | Epochs to 0.50 | Grid LRs within 0.10 Å of the overall best |
|---|---|---|---|---|
| **Adam** | 1e-2 | **3.99** ± 0.05 | **9.7** | 3 |
| SGD + momentum | 3e-2 | 4.03 ± 0.04 | 12.5 | 2 |
| RMSprop | 3e-3 | 4.21 ± 0.08 (seed std 0.09) | 29.5 | 0 |
| SGD | 3e-3 | 4.39 ± 0.06 | 293 (only 6/15 folds reached it) | 0 |

Failures at high LR: RMSprop diverged in 11/15 folds at 3e-2 and 15/15 at 1e-1. Momentum broke at 1e-1
(RMSE 5.78 Å, 2/15 diverged). Plain SGD still trained at 1e-1 (4.44 Å). Adam degraded without
diverging (4.21 Å at 3e-2, 4.75 Å at 1e-1).

### Verdicts

- **H-A1 (speed): partly supported.** Adam fastest and SGD slowest, as predicted. But **momentum beat
  RMSprop**: momentum alone gives a ~20× speed-up over plain SGD. Adam's speed seems to come largely
  from the momentum it contains, not only from its adaptive scaling.
- **H-A2 (final quality within 0.05 Å): contradicted.** The spread is 0.40 Å. Only Adam and momentum
  are close. Plain SGD at small LRs hit the 500-epoch limit still improving, and at larger LRs early
  stopping ended it after 40–85 epochs at a worse RMSE. Under a fixed budget and stopping rule,
  **speed becomes final quality**. The hypothesis also relied on the feature "ceiling", which Axis B
  disproved.
- **H-A3 (LR tolerance): supported for Adam, contradicted for RMSprop. The momentum part is supported.**
  Adam is the most tolerant (3 good LRs), but RMSprop is the **least** tolerant. Momentum fails at
  1e-1 while plain SGD survives, consistent with its ~10× effective step.

**Why RMSprop is fragile (verified with a separate measurement):** PyTorch's RMSprop has **no bias
correction**. Its running average of g² starts at 0, so its first update is **10× the LR** (≈ 4× after
5 steps), while Adam's bias-corrected first update is exactly 1× the LR. Oversized steps at the very
start of training is precisely the problem **warmup** is meant to solve.

**Side effect of large LRs: dead units.** Adam's fraction of dead ReLU units rises from 0.3 % (LR 1e-3)
to 10 % (1e-2), 32 % (3e-2) and 48 % (1e-1). Big steps knock units into the never-active region. This
is relevant to Part 2, because dead units are "free" sparsity.

---

## Axis B: Architecture (depth vs width)

**Design.** Depth {1, 2, 3, 4} hidden layers × width {16, 64, 256} units per layer, giving 12
architectures from 161 to ~200k parameters, × 3 seeds. Baseline optimiser (Adam 1e-3), ReLU + He init,
early stopping. Also recorded: **parameter count**, **best epoch**, and the **train–validation gap**
at the best epoch (an overfitting indicator).

**Hypotheses.**
- **H-B1 (width, diminishing returns):** **going from 16 to 64 units clearly lowers RMSE. Going from
  64 to 256 helps much less (< 0.05 Å).**
  *Why:* 16 units may be too few to model the non-linear relationships, but with only 8 inputs and a
  feature-limited ceiling, extra capacity beyond ~64 has little left to capture.
- **H-B2 (depth):** **1 hidden layer is worse than 2 at the same width, but going beyond 2 layers
  gives little further gain (< 0.05 Å).**
  *Why:* a second layer lets the network build features out of features (composition), which is more
  efficient than one wide layer. Beyond that, the same ceiling applies, and deeper nets are harder to
  optimise.
- **H-B3 (overfitting vs early stopping):** **larger networks overfit sooner (earlier best epoch,
  larger train–validation gap), but early stopping prevents them from ending up *worse* than the
  medium networks.**
  *Why:* more parameters means more capacity to memorise the training rows. Early stopping cuts
  training off before that memorisation hurts the validation error.

**Evidence that would contradict them:** 256-wide nets clearly beating 64-wide ones (H-B1). One layer
matching two (H-B2). The largest nets ending worse than medium ones despite early stopping (H-B3).

*Caveat to state in the report:* all architectures share Adam at LR 1e-3. A different LR might suit
some sizes better. We control for this by keeping the LR fixed, and note it as a limitation.

### Results (notebook §3.2, `results/phase3_architecture.csv`, `figures/phase3/architecture.png`)

CV RMSE (Å), mean over 3 seeds (seed std ≤ 0.03 everywhere):

| depth \ width | 16 | 64 | 256 |
|---|---|---|---|
| 1 | 4.63 | 4.39 | 4.23 |
| 2 | 4.42 | **4.09** (baseline) | 3.95 |
| 3 | 4.33 | 4.00 | 3.83 |
| 4 | 4.30 | 3.97 | **3.81** (R² 0.615) |

Best epoch falls from ~310 (1 × 16) to ~54 (4 × 256). The train–validation gap at the kept epoch rises
from ≈ 0 to 0.19. Every paired comparison below was consistent in 3/3 seeds.

### Verdicts

- **H-B1 (width, diminishing returns): partly supported.** 16 → 64: −0.33 Å, 64 → 256: −0.14 Å (depth
  2). The returns diminish, but the second step is ~3× the 0.05 Å we predicted.
- **H-B2 (depth): first half supported, second half contradicted.** 1 → 2 layers: −0.30 Å. But 2 → 3
  (−0.09 Å) and 2 → 4 (−0.12 Å) still help clearly.
- **H-B3 (overfitting vs early stopping): supported.** Bigger nets overfit sooner and more (earlier
  best epoch, larger gap), yet early stopping lets the biggest net finish best.

**Extra observations:** depth is more parameter-efficient than width here: 4 × 64 (13k parameters)
≈ 2 × 256 (68k parameters), within 0.014 Å. Bigger nets also *learn faster* (4 × 256 reaches MSE < 0.50
in 5 epochs vs 21 for 2 × 64). **Our mistake:** H-B1/H-B2 assumed the Phase 2 "ceiling", which was
really the baseline's capacity.

---

## Axis C: Activation functions

**Design.** Sigmoid, tanh, ReLU and Leaky ReLU (slope 0.01), each with its matching initialisation
(Xavier for sigmoid/tanh, He for the ReLU family). Run at the baseline depth (**2 × 64**) and a
deeper net (**4 × 64**), × 3 seeds, with the baseline optimiser (Adam 1e-3). Two extra diagnostics:
- **Gradient flow at initialisation:** on one batch, the average gradient size of the *first* layer's
  weights divided by the *last* layer's. A ratio far below 1 means gradients shrink as they travel
  backwards, i.e. **vanishing gradients**.
- **Dead units (ReLU family):** the fraction of hidden units that output 0 for *every* validation row
  after training.

**Hypotheses.**
- **H-C1 (speed):** **the ReLU family reaches MSE < 0.50 fastest, tanh next, and sigmoid slowest.**
  *Why:* sigmoid's derivative is at most 0.25 and its outputs are all positive (not zero-centred),
  which slows learning. tanh is zero-centred with derivative up to 1. ReLU's derivative is exactly 1
  for active units.
- **H-C2 (depth makes sigmoid worse: vanishing gradients):** **sigmoid's gradient ratio (first ÷ last
  layer) is far smaller than the others' and shrinks further from 2 to 4 layers. Its RMSE gap to the
  other activations is larger at depth 4 than at depth 2.**
  *Why:* backpropagation multiplies by the activation's derivative once per layer. For sigmoid that
  factor is ≤ 0.25, so the gradient can shrink by up to 4× per layer.
  *Expected moderation:* Adam rescales each weight's update by its own gradient history, which partly
  undoes small gradients. So we expect the effect to be clearly visible in the gradient diagnostic,
  but **smaller in final RMSE** than it would be with plain SGD.
- **H-C3 (final quality, ReLU vs Leaky ReLU):** **tanh, ReLU and Leaky ReLU end within ~0.05 Å of each
  other, and Leaky ReLU gives no real advantage over ReLU, because few ReLU units die in a small,
  well-initialised network with standardised inputs** (dead fraction < 10 %).
  *Why:* Leaky ReLU fixes "dead" units, which stop learning once they output 0 for every input. That
  only matters if many units die.

**Evidence that would contradict them:** sigmoid training as fast as ReLU (H-C1). No depth-dependent
shrinking of sigmoid's gradients (H-C2). Many dead ReLU units and Leaky ReLU clearly winning (H-C3).

### Results (notebook §3.3, `results/phase3_activations.csv`, `figures/phase3/activations_*.png`)

| Activation | RMSE 2×64 | RMSE 4×64 | Epochs to 0.50 (2×64 / 4×64) | Dead units |
|---|---|---|---|---|
| sigmoid | 4.34 | 4.26 | 423 / 340 (2×64: only 11/15 folds reached it) | — |
| tanh | 4.06 | **3.90** | 64 / 21 | — |
| ReLU | 4.09 | 3.97 | 21 / 8.6 | 0.3 % / 0.4 % |
| Leaky ReLU | 4.10 | 3.96 | 21 / 8.5 | 0.2 % / 0.0 % |

**Gradient flow at initialisation** (mean |gradient| of the first layer ÷ the output layer, 10 inits):

| | depth 2 | depth 4 | depth 8 |
|---|---|---|---|
| sigmoid | 2.7e-3 | 1.7e-4 | **5.8e-7** |
| tanh | 0.31 | 0.27 | 0.33 |
| ReLU | 0.095 | 0.091 | 0.080 |

With 8 sigmoid layers the gradient shrinks ≈ 4× per layer going backwards (0.56 at the output layer
→ 3 × 10⁻⁷ at the first layer). ReLU and tanh hidden layers all receive similar gradients (flat).

### Verdicts

- **H-C1 (speed: ReLU family > tanh > sigmoid): supported.**
- **H-C2 (vanishing gradients): supported, with the predicted moderation by Adam.** The ≈ 4× shrink per
  layer is exactly what sigmoid's maximum derivative of 0.25 predicts, and it compounds with depth,
  while tanh and ReLU stay flat. In RMSE, though, sigmoid's gap to ReLU grows only slightly (0.25 →
  0.30 Å), and sigmoid itself *improves* from depth 2 to 4. Adam rescales each weight's step by its
  own gradient history, so tiny gradients still give normal-sized updates. The damage shows up in
  speed, not final quality.
- **H-C3 (tanh ≈ ReLU ≈ Leaky ReLU, Leaky gains nothing): partly supported.** Leaky ReLU = ReLU (mixed
  across seeds), and < 0.5 % of ReLU units die, so the reason we gave holds. But **tanh beat ReLU at
  depth 4** (3.90 vs 3.97 Å, 3/3 seeds), beyond our threshold, while training ~2.5× slower. A plausible,
  untested reason: tanh nets are smooth, whereas ReLU nets are piecewise-linear.

---

## Final evaluation on the locked test set

Run once, after all CV comparisons. Same protocol as a CV fold (90 % fit, 10 % early stopping) on the
whole training set, 5 seeds. Only CV-validated configurations were evaluated. Untested combinations of
separately-best settings (Adam at 1e-2 with 4 × 256, or tanh) were deliberately not tried, because that
would mean selecting on the test set. Notebook §3.5, `results/final_test*.csv`.

| Model | CV RMSE (Å) | **Test RMSE (Å)** | Test R² |
|---|---|---|---|
| Predict the mean | 6.14 | 6.14 | 0.00 |
| Linear regression | 5.19 | 5.19 | 0.29 |
| Baseline 2 × 64 | 4.09 | 4.05 ± 0.04 | 0.566 |
| **Best: 4 × 256, ReLU, Adam 1e-3** | 3.81 | **3.78 ± 0.03** | **0.621** |

Test and CV agree, so comparing many configurations with CV did not overfit the validation folds.

**Per-bin test errors (the imbalance check), best model:** RMSE 2.75 Å in the common 0–3 Å bin, rising
to 6.56 Å in the rarest, worst 18–21 Å bin. Bias +1.3 Å (0–3 Å) to −4.8 Å (18–21 Å) shows regression to
the mean. The larger network helps mostly at the extremes (0–3 Å: 3.08 → 2.75 Å; 18–21 Å: 7.09 → 6.56 Å;
18–21 Å bias −5.7 → −4.8 Å).

## Choice of base configuration for Part 2 (warmup vs pruning)

| Choice | Value | Evidence |
|---|---|---|
| Architecture | **4 × 256** (~200k weights) | best RMSE, and plenty of redundant capacity to prune. ~10 s per fold |
| Activation / init | **ReLU, He** | fastest. The standard in the pruning papers we follow. tanh's 0.07 Å edge isn't worth a slower, less comparable setup |
| Optimiser | **SGD + momentum (β = 0.9)** | has a **sharp LR-tolerance boundary** (fine at 3e-2, broken at 1e-1), which is exactly what the warmup hypothesis concerns. Used by Frankle & Carbin and Renda et al. Adam degrades gradually and self-corrects its early steps, which would blur the effect |

The LR boundary was measured on 2 × 64. Part 2's first experiment re-measures it on 4 × 256, with and
without warmup.
