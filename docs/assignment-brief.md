# Assignment Brief (condensed from `COS711-A1.pdf`)

> The PDF is the source of truth. This page is a shorter version for quick reference.
> Section numbers (§) refer to the PDF.

**Course:** COS711, Assignment 1: *Neural Network Fundamentals: Learning Rate Warmup and Pruning Robustness*
**Due:** 9 October 2026, uploaded to ClickUP
**Graded?** No, but **you must submit it to be admitted to Semester Test 2**. The test asks
questions about the concepts, experiments and design choices in this assignment.

---

## Our dataset: Protein Tertiary Structure (UCI id 265)

- **Source:** CASP protein-folding competitions.
- **Task:** regression. Predict the **RMSD** (root-mean-square deviation, in Ångström) of a candidate
  protein fold from 9 physicochemical descriptors (`F1`–`F9`).
- **Size:** 45,730 rows × 9 features, the largest dataset on the list (the spec calls it "well suited to
  the Part 2 warmup/pruning experiments").
- **Loading:**
  ```python
  from ucimlrepo import fetch_ucirepo
  protein_structure = fetch_ucirepo(id=265)
  X, y = protein_structure.data.features, protein_structure.data.targets
  ```

## Part 0: Data preparation (§2.1, compulsory, must appear in the report)

| # | Topic | What we must do |
|---|-------|-----------------|
| 1 | Feature relevance | Find uninformative features and justify removing them (or keeping them all) |
| 2 | Outliers | Develop and justify a strategy to identify and handle outliers |
| 3 | Scaling | Normalise/standardise, and justify the choice given the feature types |
| 4 | Splitting | Define and justify the train/validation/test split strategy |
| 5 | Imbalance | The target distribution may be skewed. Explain how this is handled during training, testing and evaluation |

## Part 1: Core investigation (§2.2)

- Train an **MLP baseline**.
- Run controlled experiments on **at least two** of these axes:
  - Initialisation: random vs Xavier/He
  - Activation: sigmoid vs tanh vs ReLU (and variants)
  - Architecture: depth vs width, number of hidden units
  - Optimisers: SGD vs momentum vs Adam/RMSprop (convergence speed and stability)
  - Regularisation: weight decay, dropout, early stopping (overfitting behaviour)
- **k-fold cross-validation on the training set is required** to evaluate hyperparameter effects.
- **State a hypothesis before each experiment**, then say whether the results supported or
  contradicted it. Interpretation matters more than tables of numbers.

## Part 2: Research investigation, warmup vs pruning robustness (§2.3)

**Hypothesis to test:** learning-rate warmup lets a network tolerate a *larger* learning rate [3], and
larger learning rates help a network *recover from pruning* [2]. So warmup may *indirectly* improve
pruning robustness. The spec says this causal story is unproven and is ours to test.

Required steps (each experiment uses **≥ 3 random seeds**, reporting **mean ± std**):

1. Train two networks, **with** and **without** warmup (no-warmup = constant LR or a standard decay).
   Choose configurations so the **unpruned** models have **comparable** performance.
2. Apply **magnitude-based pruning** to both at one or more sparsities (e.g. 30 %, 50 %, 70 %), using the
   same procedure and sparsities for both.
3. **Fine-tune** the pruned networks with the **same** fine-tuning procedure, then compare recovered
   performance.
4. Report and interpret: did we see the effect? Do the results support or contradict the
   "LR tolerance → pruning robustness" mechanism, and why?

Optional extensions (§2.3.3): pruning *timing* relative to warmup; warmup *duration*; distillation from
a warmup-trained teacher; lightweight NAS under different schedules.

## Notes (§3)

- Python, submitted as a **Jupyter notebook**. ML frameworks are allowed. Building parts from first
  principles is encouraged.
- The report must cover **all data preparation steps**, give a **hypothesis for every experiment**, and
  include **plots of accuracy vs sparsity for both conditions** (for regression, "accuracy" means our
  chosen metric, e.g. RMSE or R²).

## Deliverables (§4)

1. A **technical report, 2–4 pages** (excluding code) with learning curves, comparison tables/plots,
   and a written interpretation of every experiment.
2. A **Jupyter notebook** that reproduces the reported results.
3. Both in **one zip**, uploaded to ClickUP (only the last upload counts).

## The spirit of it (§5)

> "Depth of understanding on fewer experiments is preferred over shallow coverage of many."

Show that we can design a fair experiment, state a hypothesis, interpret evidence honestly (including
when it contradicts the hypothesis), and connect the findings back to theory from lectures.

## References

1. Frankle & Carbin (2019): *The Lottery Ticket Hypothesis*, ICLR. **Appendix I.5 covers warmup.**
2. Renda, Frankle & Carbin (2020): *Comparing Rewinding and Fine-tuning in Neural Network Pruning*, ICLR.
3. Kalra & Barkeshli (2024): *Why Warmup the Learning Rate? Underlying Mechanisms and Improvements*, NeurIPS.
