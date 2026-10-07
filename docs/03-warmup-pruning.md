# 03: Research Investigation: Warmup and Pruning Robustness

> Covers spec §2.3. **Status:** hypotheses, design and decision rules approved and committed on
> 2026-10-06 (commit `243937b`) **before** the experiments ran. Sections 1–6 are unchanged from that
> commit. Results are in §7. Code: `src/warmup_pruning.py`, `src/pruning.py`. Notebook section 4.
> Figures in `figures/phase4/`. Raw numbers in `results/phase4_*.csv`.

## 1. Background (from the three references)

> Summaries are based on the papers' abstracts and the spec. 🧑 Check the details against the PDFs,
> especially Appendix I.5 of [1] (warmup iterations), which the spec singles out.

- **[1] Frankle & Carbin (2019), Lottery Ticket Hypothesis.** Dense networks contain sparse subnetworks
  ("winning tickets", often 10–20 % of the weights) that train to comparable accuracy in isolation.
  Found by iterative magnitude pruning. For deeper networks (VGG/ResNet on CIFAR-10), the spec notes
  that warmup and a constant LR gave *similar unpruned accuracy*, but **warmup gave much higher accuracy
  as the network was pruned further**.
- **[2] Renda, Frankle & Carbin (2020), Rewinding vs fine-tuning.** Three ways to retrain after pruning:
  **fine-tuning** (continue from the final weights with a *small fixed LR*), **weight rewinding** (reset
  the surviving weights to an earlier point and retrain with the original schedule), and **LR
  rewinding** (keep the final weights but retrain with the *original LR schedule*). Both rewinding
  methods clearly beat fine-tuning. So *larger* retraining LRs help recovery.
- **[3] Kalra & Barkeshli (2024), Why warmup?** Warmup's main benefit is letting the network **tolerate
  a larger target LR**, by moving it into better-conditioned (less sharp) regions of the loss landscape
  before the LR becomes large. They also propose a non-zero initialisation of Adam's variance estimate
  as a substitute for warmup. That is the same root cause as the RMSprop fragility we measured in Phase 3
  (variance estimate starting at 0 → first update 10× the LR).

**The hypothesised chain (spec §2.3.1):** warmup → a larger LR becomes trainable [3] → training at a
larger LR yields a network that recovers better from pruning [2] → warmup *indirectly* improves
pruning robustness. The spec stresses this causal chain is **unproven**. Our job is to test it.

## 2. Setup (fixed for every run)

| Item | Value | Why |
|---|---|---|
| Network | 4 × 256, ReLU, He init (~200k weights) | Best architecture from Phase 3. Lots of redundancy to prune |
| Optimiser | SGD + momentum 0.9, batch 256 | Sharp LR-tolerance boundary (Phase 3). Used in [1, 2] |
| Training budget | **Fixed 100 epochs, no early stopping** (final weights are kept) | Equal budgets make the conditions comparable. The pilot showed 100 > 60 epochs |
| LR schedule | Cosine decay to 0. Warmup conditions add a **linear warmup of 5 epochs** (5 % of training) first | The *only* schedule difference between conditions is the warmup |
| Data | Training set split once (stratified) into **fit (90 %)** and **validation (10 %)**. **Test set** (8,804 rows) used only for the reported pruning results | All *choices* (LRs, "comparable") are made on validation. The test set is touched only at the end |
| Seeds | **5** per condition (spec: ≥ 3) | Seed changes initialisation and batch order. Mean ± std reported |

**Calibration pilot (run before these hypotheses were written, 1 seed, no warmup, no pruning):**
4 × 256 momentum SGD **without warmup diverged at epoch 1 for LR 0.03 and 0.1**, but trained at 0.01
(val RMSE 3.80 Å at 100 epochs, 3.93 Å at 60). This was used to choose the LR grid and the epoch budget.
It already hints that the no-warmup limit lies between 0.01 and 0.03 for this network.

## 3. Experiments, in order

### Step 1: LR-tolerance sweep (tests H1)
Train with and without warmup at LR ∈ {0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2} × 3 seeds. Record the
final validation RMSE and whether training diverged.
- **Max stable LR** (per schedule) = the largest grid LR at which **all 3 seeds** finish without
  diverging and the mean val RMSE is within **0.10 Å** of that schedule's best.

### Step 2: Choose the three conditions (rules fixed now, applied to the Step 1 results)
- **A: No warmup at η₀.** η₀ = the no-warmup LR with the lowest val RMSE.
- **B: Warmup at η₁ > η₀.** η₁ = the **largest** grid LR above η₀ at which warmup training is stable
  (all seeds) **and** its val RMSE is within **0.05 Å** of condition A's ("comparable unpruned accuracy",
  spec step 1). If no such LR exists, warmup did not enable a larger comparable LR. We report that
  (it bears directly on the mechanism) and still run A vs C.
- **C: Warmup at η₀** (control). Same LR as A, with warmup. It separates "warmup itself" from "the
  larger LR that warmup allows".

### Step 3: Train, prune, fine-tune (tests H2 and H3)
For each condition × 5 seeds: train (100 epochs), then for each sparsity
**s ∈ {0, 30, 50, 70, 90, 95, 98} %**:
1. **Prune:** one-shot **global magnitude pruning**. Rank all weights of all weight matrices together
   (biases are not pruned) by |w|, and set the smallest s % to zero.
2. **Measure** test RMSE immediately after pruning ("before fine-tuning").
3. **Fine-tune:** identical for every condition. SGD + momentum 0.9 at a **small constant LR of 0.001
   for 10 epochs**. Pruned weights are held at zero after every step (mask).
4. **Measure** test RMSE after fine-tuning.

**Robustness** at sparsity s, per seed: **Δ(s) = RMSE_after_FT(s) − RMSE(0 %)**, i.e. how much error
pruning adds once the network has had the chance to recover. Smaller Δ = more robust.

Also recorded per condition: the fraction of **dead ReLU units** before pruning. Phase 3 showed large
LRs kill units, and a dead unit's weights can be pruned "for free", so this is a possible confounder
(see §5).

## 4. Hypotheses

- **H1 (warmup raises LR tolerance):** the max stable LR **with** warmup is **at least one grid step
  higher** than without warmup.
  *Contradicted if:* the two max stable LRs are equal, or no-warmup's is higher.
- **H2 (warmup improves pruning robustness):** at comparable unpruned RMSE, **B (warmup, larger LR)
  has a smaller Δ(s) than A (no warmup) at high sparsity** (s ≥ 70 %). We count it as a real effect
  if B's Δ is smaller at **at least 2 of the 4 sparsities ≥ 70 %**, in **≥ 4 of 5 seeds** (paired), by
  more than the seed-to-seed std.
  *Contradicted if:* A and B have Δ within noise, or A is more robust.
- **H3 (the mechanism is the larger LR, not warmup itself):** the control **C (warmup at η₀) behaves
  like A, not like B**. B's Δ is smaller than C's at high sparsity (same criterion as H2), while C vs A
  shows no consistent difference.
  *Contradicted if:* C is as robust as B (then warmup helps even without a larger LR, so the "LR
  tolerance" route is not needed), or C is *more* robust than A.
- **Exploratory, no hypothesis:** the RMSE *before* fine-tuning vs sparsity, and dead-unit fractions.

**Honest prior:** our network has ~200k weights for 8 inputs, so it is heavily over-parameterised.
Pruning may barely hurt *any* condition until very high sparsity, which is why 90/95/98 % are
included. A null result (no difference) would be a legitimate finding: tabular regression with a
small MLP is very different from the deep image classifiers in [1, 2].

## 5. Threats to validity (to discuss in the report)

- **Comparability:** B trains at a larger LR, so it may also reach a different unpruned RMSE. The 0.05 Å
  rule limits this, but cannot remove every difference.
- **Dead units as hidden sparsity:** if B's larger LR kills more units, part of its "robustness" could
  just be that magnitude pruning removes weights that were already useless. Dead-unit fractions are
  reported to judge this.
- **Fine-tuning LR:** a single fixed fine-tuning recipe is required for fairness, but its LR (0.001) is a
  choice. [2] suggests a larger retraining LR (LR rewinding) recovers more (optional extension E1).
- **One-shot vs iterative pruning:** [1] prunes iteratively. One-shot is simpler and the spec allows it,
  but the effect may be weaker.
- **Single fit/validation split** (no k-fold here). Seeds capture initialisation noise, not
  data-split noise.

## 6. Optional extensions (only if time allows, after the main results)

- **E1: LR rewinding instead of fine-tuning** [2]: retrain each pruned net for 10 epochs with its own
  original schedule shape (warmup if it had one, cosine to 0 from its own peak LR). Prediction: better
  recovery for all conditions, and a smaller A–B gap if the benefit really comes from the LR used *after*
  pruning.
- **E2: Warmup duration:** condition B with warmup of 1, 5 and 15 epochs. Prediction: once warmup is
  long enough to make η₁ stable, longer warmup adds no further robustness (a threshold effect).

## 7. Results

### Step 1: LR tolerance (H1). **Strongly supported**

| Peak LR | 0.005 | 0.01 | 0.02 | 0.03 | 0.05 | 0.1 | 0.2 |
|---|---|---|---|---|---|---|---|
| No warmup: val RMSE (diverged seeds) | 3.73 (0/3) | 3.80 (**2/3**) | — (3/3) | — (3/3) | — (3/3) | — (3/3) | — (3/3) |
| 5-epoch warmup: val RMSE (diverged seeds) | 3.68 (0/3) | **3.63** (0/3) | 3.63 (0/3) | 3.66 (0/3) | 3.71 (0/3) | 3.71 (0/3) | 3.73 (0/3) |

- Max stable LR: **0.005 without warmup, 0.1 with warmup** (0.2 also trains, but is 0.005 Å outside
  the 0.10 Å tolerance). The tolerable LR is **≥ 20× higher**, 4 grid steps.
- **All 17 no-warmup divergences happen in the first epoch.** The first large steps destroy the network
  before it can reach a region where large steps are safe. This is the mechanism described in [3].
- Warmup also gave better unpruned models (best 3.63 vs 3.73 Å).

**Conditions selected by the pre-registered rules:** A = no warmup, η₀ = 0.005. **B = warmup, η₁ = 0.2**
(the largest warmup LR within 0.05 Å of A: 3.730 vs 3.732 Å. Warmup LRs 0.01–0.03 were ~0.1 Å *better*
than A, so not "comparable"). C = warmup at 0.005.

### Step 3: Pruning robustness (H2, H3). **Both supported**

Test RMSE (Å), mean of 5 seeds. Unpruned: A 3.81, B 3.84, C 3.78 (validation: 3.74, 3.73, 3.68).

| Sparsity | 30 % | 50 % | 70 % | 90 % | 95 % | 98 % |
|---|---|---|---|---|---|---|
| **Before fine-tuning:** A / B / C | 4.35 / 3.89 / 4.60 | 5.69 / 4.08 / 6.15 | 7.97 / **4.61** / 7.39 | 8.59 / **5.86** / 7.82 | 7.51 / 6.40 / 7.15 | 6.74 / 6.45 / 6.69 |
| **Δ after fine-tuning:** A / B / C | 0.03 / 0.00 / 0.03 | 0.08 / 0.02 / 0.11 | 0.31 / **0.09** / 0.35 | 0.71 / **0.44** / 0.74 | 0.95 / **0.69** / 1.01 | 1.34 / **1.09** / 1.40 |

(Δ = test RMSE after fine-tuning − unpruned test RMSE. Plot: `figures/phase4/pruning_vs_sparsity.png`.)

**Paired comparisons at s ≥ 70 % (pre-registered criterion: smaller Δ in ≥ 4/5 seeds, by more than the
std of the paired differences, at ≥ 2 of the 4 sparsities):**

| Comparison | Mean Δ difference | Seeds | Meets criterion |
|---|---|---|---|
| B vs A | −0.22 to −0.27 Å | 5/5 at every sparsity | **Yes, 4/4 sparsities** → **H2 supported** |
| B vs C | −0.26 to −0.32 Å | 5/5 at every sparsity | **Yes, 4/4** |
| C vs A | +0.03 to +0.06 Å (C slightly *worse*) | 0–2/5 better | **No** → with B vs C, **H3 supported** |

**Confounders:**
- **Dead units:** A 3.6 % ± 3.9, B 0.2 %, C 0.0 %. B has the *fewest*, so "free" sparsity does not explain
  its robustness.
- **Fine-tuning LR:** 0.001 is 20 % of A's/C's training LR but 0.5 % of B's. By [2], that favours A and C.
  B still wins.
- **Reference point:** measuring Δ against the *fine-tuned* dense network gives the same ranking (B lower by
  0.19–0.24 Å).
- **Comparability:** B's unpruned test RMSE is slightly *worse* than A's (3.84 vs 3.81), so B's advantage
  is not from a better starting point.

**Before fine-tuning** (exploratory): A and C collapse to errors *worse than predicting the mean*
(6.14 Å) from 70 % sparsity. B stays far better (4.61 Å at 70 %). Their curves come back down at 98 %
because a nearly empty network outputs almost a constant.

### Post-hoc exploration (added after seeing the results, so not a hypothesis test)

Weight magnitudes of one network per condition (seed 0, `figures/phase4/weight_magnitudes.png`): A and
C are virtually identical (share of Σw² in the largest 10 % of weights: 56.5 % vs 56.4 %), matching their
identical pruning behaviour. B's distribution is shifted to larger weights (median |w| 0.084 vs 0.062),
with more of its weight mass in its largest weights (top 10 %: 60.6 %; top 30 %: 85.6 % vs 83.3 %). A
network that relies more on a few large weights loses less when magnitude pruning removes the many
small ones. Untested alternative or complement: large-LR training ends in flatter minima.

### Extensions

**E1: LR rewinding** (Δ measured against each network's own retrained dense version, because rewinding
also improves the unpruned nets: A 3.81 → 3.79, B 3.84 → **3.72**, C 3.78 → 3.75):

| Sparsity | 70 % | 90 % | 95 % | 98 % |
|---|---|---|---|---|
| Fine-tuning Δ: A / B | 0.28 / 0.09 | 0.68 / 0.44 | 0.92 / 0.69 | 1.31 / 1.09 |
| **LR rewinding Δ: A / B** | 0.22 / **0.02** | 0.58 / **0.03** | 0.81 / **0.15** | 1.14 / **0.45** |

Recovery improves for all conditions, as predicted, but the **A–B gap widens** (0.19 → 0.70 Å, 5/5
seeds). We predicted it would shrink, so that part is **contradicted**. With rewinding each network
retrains at its *own* peak LR (B: 0.2, A: 0.005), which reproduces [2]'s finding that a large retraining
LR recovers better. The large LR that warmup unlocks therefore helps **twice**: it makes the trained
network more robust, and it makes recovery more effective.

**E2: warmup length at LR 0.2** (5 seeds each):

| Warmup | Unpruned test RMSE | Δ at 70 / 90 / 95 / 98 % |
|---|---|---|
| 1 epoch | 3.87 | 0.06 / 0.31 / 0.56 / 1.00 |
| 5 epochs | 3.84 | 0.09 / 0.44 / 0.69 / 1.09 |
| 15 epochs | **3.79** | 0.15 / 0.57 / 0.80 / 1.13 |

Even **1 epoch** of warmup is enough to make LR 0.2 trainable. 1 vs 5 epochs: no consistent difference
(1–2/5 seeds), as the threshold prediction said. But **15 epochs is less robust than 5** (4–5/5 seeds,
0.04–0.12 Å), while giving the best unpruned RMSE. So the prediction is **partly supported**. A longer
warmup leaves fewer epochs at the large LR, which fits the "robustness comes from the large LR" picture
(untested explanation).

### Conclusion (spec step 4)

**We observed the effect.** At matched unpruned accuracy, the warmup-trained network lost 0.22–0.27 Å
less after pruning and fine-tuning (≥ 70 % sparsity, 5/5 seeds), and far less before fine-tuning.

**The results support the proposed mechanism, on both links:** (1) warmup raises the tolerable LR ≥ 20×
(without it the network diverges in epoch 1). (2) The robustness comes from the large LR, not warmup
itself: warmup at the small LR (C) gave **no** benefit. Warmup's effect on pruning robustness is
therefore **indirect**: it matters because it unlocks a large learning rate. E1 shows the same large LR
also helps recovery.

**Limitations:** a single tabular dataset and MLP (the original papers used CNNs on images). One-shot
pruning, not iterative. A single fit/validation split. B's LR was set by our comparability rule at the
top of the grid. The *reason* large-LR networks are more robust (weight concentration? flatter minima?)
is only explored post-hoc.
