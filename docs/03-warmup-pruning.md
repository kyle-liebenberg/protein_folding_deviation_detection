# 03: Research Investigation: Warmup and Pruning Robustness

> Covers spec §2.3. **Status: hypotheses and design drafted on 2026-10-06, awaiting your approval.
> Not yet run** (apart from the calibration pilot described below). Once approved, commit this file
> *before* the experiments run.

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

*(to be added after the experiments run)*
