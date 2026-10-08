"""Regression metrics, all in Å (the units of RMSD).

- RMSE: root mean squared error. Penalises large mistakes heavily. The headline metric.
- MAE:  mean absolute error. The "typical" error size.
- R²:   fraction of the target's variance explained (1 = perfect, 0 = no better than the mean).

`per_bin_metrics` handles the imbalanced target: the overall RMSE is dominated by the common
low-RMSD rows, so errors are also reported separately for each RMSD bin.
"""

import numpy as np
import pandas as pd

from src.data import RMSD_BIN_LABELS, rmsd_bin


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    residual_ss = np.sum((y_true - y_pred) ** 2)
    total_ss = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - residual_ss / total_ss)


def regression_metrics(y_true, y_pred) -> dict:
    return {"rmse": rmse(y_true, y_pred), "mae": mae(y_true, y_pred), "r2": r2(y_true, y_pred)}


def per_bin_metrics(y_true, y_pred) -> pd.DataFrame:
    """RMSE, MAE, mean error (bias) and row count for each RMSD bin of the TRUE value.

    A positive bias means the model over-predicts in that bin. Models typically over-predict the low
    bins and under-predict the high bins (regression towards the mean).
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    bins = rmsd_bin(y_true)
    rows = []
    for b, label in enumerate(RMSD_BIN_LABELS):
        in_bin = bins == b
        if not in_bin.any():  # can happen on small subsets. Report the bin as empty
            rows.append({"bin": label, "n": 0, "rmse": np.nan, "mae": np.nan, "bias": np.nan})
            continue
        rows.append({
            "bin": label,
            "n": int(in_bin.sum()),
            "rmse": rmse(y_true[in_bin], y_pred[in_bin]),
            "mae": mae(y_true[in_bin], y_pred[in_bin]),
            "bias": float(np.mean(y_pred[in_bin] - y_true[in_bin])),
        })
    return pd.DataFrame(rows).set_index("bin")
