# Protein Folding Deviation Detection (COS711 Assignment 1)

An MLP that predicts the **structural deviation (RMSD, Å)** of candidate protein folds from 9
physicochemical descriptors (UCI *Physicochemical Properties of Protein Tertiary Structure*, CASP data).
We use it to study two things:

1. **Core investigation:** how MLP design choices (optimiser, architecture, …) affect a regression
   model, evaluated with k-fold cross-validation.
2. **Research investigation:** whether **learning-rate warmup** makes a network more **robust to
   magnitude pruning**, and whether that happens because warmup allows a larger learning rate.

> **Status:** Phases 0–2 done (setup, data preparation, MLP + training + baseline). See [`plan.md`](plan.md) for the task list and progress.

## Documentation map

| Doc | Purpose |
|-----|---------|
| [`plan.md`](plan.md) | TODO list, decisions needed, decision log |
| [`docs/assignment-brief.md`](docs/assignment-brief.md) | Condensed version of the assignment spec |
| [`docs/01-data-preparation.md`](docs/01-data-preparation.md) | Data prep decisions and justifications |
| [`docs/02-core-investigation.md`](docs/02-core-investigation.md) | Baseline, plus hypotheses, results and interpretation per axis *(axes: Phase 3)* |
| `docs/03-warmup-pruning.md` | Research hypotheses, design, results *(Phase 4)* |
| [`docs/study-notes.md`](docs/study-notes.md) | Theory and Q&A for Semester Test 2 (built up each phase) |

## Project structure

```
├── README.md
├── plan.md                  # task list
├── requirements.txt         # pinned dependencies
├── pytest.ini               # lets tests import `src`
├── COS711-A1.pdf            # assignment spec (source of truth)
├── docs/                    # written explanations, which feed the report
├── src/                     # all reusable code, one concept per file
│   ├── utils.py             # seeding, device selection
│   ├── data.py              # loading, cleaning, splitting (train/test, k-fold)
│   ├── preprocessing.py     # log → clip → standardise, fitted on train only
│   ├── model.py             # the MLP
│   ├── schedules.py         # learning-rate schedules (incl. warmup)
│   ├── training.py          # training loop, early stopping
│   ├── evaluation.py        # RMSE / MAE / R², per-bin metrics
│   ├── plotting.py          # shared figure style and colours
│   ├── cv.py                # k-fold cross-validation runner
│   └── pruning.py           # magnitude pruning and masks
├── notebooks/
│   └── A1_protein_mlp.ipynb # the submitted notebook: narrative + experiments
├── tests/                   # small unit tests for the code above
├── results/                 # experiment outputs (CSV/JSON)
├── figures/                 # saved plots used in the report
├── report/                  # the 2–4 page technical report
└── data/                    # cached raw data (git-ignored)
```

**Design principle:** the notebook tells the story, and `src/` holds the mechanics. Each `src/` file is
short and does one thing, so it can be read top to bottom before the test.

## Setup

Requires Python 3.14 (tested).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest                    # run the unit tests. The first run also downloads the dataset
```

The dataset (~3.5 MB) is downloaded from UCI the first time `src.data.load_raw()` is called, and
cached in `data/raw/protein.csv`. The spec's `ucimlrepo` snippet does not work for this dataset, so we
use UCI's direct download link instead.
