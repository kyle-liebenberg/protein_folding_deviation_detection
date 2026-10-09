"""The one-time evaluation on the locked TEST set (Part 1's final models).

Same protocol as a CV fold, but on the whole training set:
    training set -> 90 % fit (weights are learned) + 10 % early stopping
    test set     -> scored once, in Å, overall and per RMSD bin (the imbalance check)
Only configurations already validated by cross-validation are evaluated here. Nothing is chosen
using these numbers.
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.data import REPO_ROOT, TARGET, split_train_val
from src.evaluation import per_bin_metrics, regression_metrics
from src.preprocessing import Preprocessor
from src.training import TrainConfig, predict, train

RESULTS_DIR = REPO_ROOT / "results"


def _run(label: str, config: TrainConfig, train_df: pd.DataFrame, test_df: pd.DataFrame):
    fit_idx, stop_idx = split_train_val(np.arange(len(train_df)), train_df[TARGET].to_numpy())
    fit, stop = train_df.iloc[fit_idx], train_df.iloc[stop_idx]
    prep = Preprocessor().fit(fit)
    result = train(config, prep.transform_X(fit), prep.transform_y(fit), prep.transform_X(stop), prep.transform_y(stop))

    y_true = test_df[TARGET].to_numpy()
    y_pred = prep.inverse_transform_y(predict(result.model, prep.transform_X(test_df)))
    metrics = {"label": label, "seed": config.seed, **regression_metrics(y_true, y_pred), "best_epoch": result.best_epoch}
    bins = per_bin_metrics(y_true, y_pred).assign(label=label, seed=config.seed)
    return metrics, bins


def evaluate_on_test(configs: dict[str, TrainConfig], train_df: pd.DataFrame, test_df: pd.DataFrame,
                     seeds=range(5), rerun: bool = False, n_jobs: int = -1):
    """Returns (metrics: one row per configuration × seed, per_bin: per-bin errors per configuration × seed)."""
    metrics_path, bins_path = RESULTS_DIR / "final_test.csv", RESULTS_DIR / "final_test_per_bin.csv"
    if metrics_path.exists() and bins_path.exists() and not rerun:
        return pd.read_csv(metrics_path), pd.read_csv(bins_path)

    jobs = [delayed(_run)(label, TrainConfig(**{**config.__dict__, "seed": seed}), train_df, test_df)
            for label, config in configs.items() for seed in seeds]
    outputs = Parallel(n_jobs=n_jobs)(jobs)
    metrics = pd.DataFrame([m for m, _ in outputs])
    per_bin = pd.concat([b.reset_index() for _, b in outputs], ignore_index=True)
    metrics.to_csv(metrics_path, index=False)
    per_bin.to_csv(bins_path, index=False)
    return metrics, per_bin
