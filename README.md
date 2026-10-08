# COS711 Assignment 1: MLP on Protein Tertiary Structure

An MLP is trained to predict the RMSD (in Å) of candidate protein structures from the CASP experiments,
using physicochemical features (UCI dataset 265). The project has two parts:

1. **Core investigation:** how the optimiser, architecture and activation function affect the MLP,
   compared with 5-fold cross-validation.
2. **Research investigation:** whether learning-rate warmup makes the network more robust to magnitude
   pruning, and whether this happens because warmup allows a larger learning rate.

The findings are in `report/main.pdf`. The notebook reproduces every result in the report.

## Contents

```
├── README.md
├── requirements.txt
├── report/
│   └── main.pdf               # the report
├── notebooks/
│   └── A1_protein_mlp.ipynb   # all experiments and results
├── src/                       # code used by the notebook
│   ├── data.py                # loading, cleaning and splitting
│   ├── preprocessing.py       # log transform, clipping and scaling
│   ├── model.py               # the MLP
│   ├── schedules.py           # learning-rate schedules, including warmup
│   ├── training.py            # training loop with early stopping
│   ├── evaluation.py          # RMSE, MAE, R² and per-bin metrics
│   ├── cv.py                  # 5-fold cross-validation
│   ├── experiments.py         # experiment grids for the core investigation
│   ├── diagnostics.py         # gradient size, dead units, convergence speed
│   ├── pruning.py             # global magnitude pruning
│   ├── warmup_pruning.py      # the warmup and pruning experiments
│   ├── final_evaluation.py    # final evaluation on the test set
│   ├── plotting.py            # figure style
│   └── utils.py               # random seeds
├── results/                   # saved experiment results, reused by the notebook
└── data/raw/protein.csv       # the dataset
```

## Running the code

Requires Python 3.14 (other recent versions should also work).

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/A1_protein_mlp.ipynb
```

Then choose **Kernel → Restart & Run All**.

- The notebook reuses the saved results in `results/`, so a full run takes about 3 minutes.
- To retrain every model from scratch, set `RERUN = True` at the start of section 3 (about 15 minutes on
  a 12-core machine).
- If `data/raw/protein.csv` is missing, the dataset is downloaded from UCI on the first run.
