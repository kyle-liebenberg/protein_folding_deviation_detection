"""k-fold cross-validation on the training set (spec requirement for every hyperparameter comparison).

For each of the k folds:
    fold's training part  ->  90 % "fit" (weights are learned here) + 10 % "stop" (early stopping)
    fold's validation part -> scored, in Å (this is the reported number)
The preprocessor is fitted on the "fit" rows only, so nothing about the scored rows leaks in.

Every configuration sees exactly the same folds (fixed SPLIT_SEED), so differences between
configurations come from the configuration, not from luck of the split.
"""

from dataclasses import dataclass, field

import pandas as pd
from joblib import Parallel, delayed

from src.diagnostics import dead_unit_fraction, epochs_to_threshold
from src.data import INPUT_FEATURES, TARGET, inverse_bin_frequency_weights, split_train_val, stratified_kfold
from src.evaluation import regression_metrics
from src.preprocessing import Preprocessor
from src.training import TrainConfig, predict, train

METRICS = ["rmse", "mae", "r2"]


@dataclass
class CVResult:
    folds: pd.DataFrame                                 # one row per fold: metrics and training stats
    histories: list = field(default_factory=list)      # per-fold learning curves (MLP only)
    predictions: pd.DataFrame | None = None             # every training row's prediction from the fold
                                                        # where it was the validation row ("out-of-fold")

    def summary(self) -> pd.Series:
        """mean and std of each metric across folds, e.g. rmse_mean, rmse_std, ..."""
        out = {}
        for m in METRICS:
            out[f"{m}_mean"] = self.folds[m].mean()
            out[f"{m}_std"] = self.folds[m].std()
        for extra in ["best_epoch", "epochs_to_threshold", "gap_at_best", "dead_fraction", "seconds"]:
            if extra in self.folds:
                out[f"{extra}_mean"] = self.folds[extra].mean()
        if "epochs_to_threshold" in self.folds:
            out["folds_reaching_threshold"] = int(self.folds["epochs_to_threshold"].notna().sum())
        if "diverged" in self.folds:
            out["diverged_folds"] = int(self.folds["diverged"].sum())
        return pd.Series(out)


def fold_splits(train_df: pd.DataFrame, k: int = 5):
    """[(fit_idx, stop_idx, val_idx), ...] for the k folds."""
    y = train_df[TARGET].to_numpy()
    splits = []
    for train_idx, val_idx in stratified_kfold(y, k):
        fit_idx, stop_idx = split_train_val(train_idx, y)
        splits.append((fit_idx, stop_idx, val_idx))
    return splits


def cross_validate(config: TrainConfig, train_df: pd.DataFrame, k: int = 5,
                   features=INPUT_FEATURES, n_jobs: int = 1) -> CVResult:
    """Train and score one MLP configuration on each of the k folds.

    n_jobs > 1 runs folds in parallel processes (each on one CPU thread).
    """
    return cross_validate_many([config], train_df, k, features, n_jobs)[0]


def cross_validate_many(configs: list[TrainConfig], train_df: pd.DataFrame, k: int = 5,
                        features=INPUT_FEATURES, n_jobs: int = -1) -> list[CVResult]:
    """cross_validate for a list of configurations, with ALL (configuration, fold) runs in one parallel
    batch, so a whole experiment grid keeps every CPU core busy. Returns one CVResult per config."""
    splits = fold_splits(train_df, k)
    jobs = [delayed(_run_mlp_fold)(config, train_df, fold, fold_split, features)
            for config in configs for fold, fold_split in enumerate(splits)]
    outputs = Parallel(n_jobs=n_jobs)(jobs)
    results = []
    for i in range(len(configs)):
        mine = outputs[i * k:(i + 1) * k]
        results.append(CVResult(folds=pd.DataFrame([row for row, _, _ in mine]),
                                histories=[history for _, history, _ in mine],
                                predictions=pd.concat([preds for _, _, preds in mine]).sort_index()))
    return results


def cross_validate_sklearn(make_model, train_df: pd.DataFrame, k: int = 5, features=INPUT_FEATURES) -> CVResult:
    """Same folds and preprocessing, but for a scikit-learn regressor (the simple baselines)."""
    rows, preds = [], []
    for fold, (fit_idx, stop_idx, val_idx) in enumerate(fold_splits(train_df, k)):
        prep, (X_fit, y_fit), _, (X_val, y_val_angstrom) = _prepare(train_df, fit_idx, stop_idx, val_idx, features)
        model = make_model().fit(X_fit, y_fit.ravel())
        y_pred = prep.inverse_transform_y(model.predict(X_val))
        rows.append({"fold": fold, **regression_metrics(y_val_angstrom, y_pred)})
        preds.append(_predictions_frame(val_idx, fold, y_val_angstrom, y_pred))
    return CVResult(folds=pd.DataFrame(rows), predictions=pd.concat(preds).sort_index())


def _prepare(df, fit_idx, stop_idx, val_idx, features):
    """Fit the preprocessor on the fit rows, then transform all three parts."""
    fit, stop, val = df.iloc[fit_idx], df.iloc[stop_idx], df.iloc[val_idx]
    prep = Preprocessor(features).fit(fit)
    return (prep,
            (prep.transform_X(fit), prep.transform_y(fit)),
            (prep.transform_X(stop), prep.transform_y(stop)),
            (prep.transform_X(val), val[TARGET].to_numpy()))   # validation target stays in Å


def _predictions_frame(val_idx, fold, y_true, y_pred) -> pd.DataFrame:
    return pd.DataFrame({"fold": fold, "y_true": y_true, "y_pred": y_pred}, index=val_idx)


def _run_mlp_fold(config, df, fold, splits, features):
    fit_idx, stop_idx, val_idx = splits
    prep, (X_fit, y_fit), (X_stop, y_stop), (X_val, y_val_angstrom) = _prepare(df, fit_idx, stop_idx, val_idx, features)
    weights = inverse_bin_frequency_weights(df.iloc[fit_idx][TARGET]) if config.weighted_loss else None

    result = train(config, X_fit, y_fit, X_stop, y_stop, sample_weights=weights)
    y_pred = prep.inverse_transform_y(predict(result.model, X_val))

    best = result.best_epoch - 1
    row = {"fold": fold, **regression_metrics(y_val_angstrom, y_pred), "best_epoch": result.best_epoch,
           "epochs_run": result.epochs_run, "diverged": result.diverged, "seconds": result.seconds,
           # convergence speed: first epoch with early-stopping MSE below the threshold
           "epochs_to_threshold": epochs_to_threshold(result.history["val_loss"]),
           # overfitting indicator: validation minus training loss at the kept epoch
           "gap_at_best": result.history["val_loss"][best] - result.history["train_loss"][best] if best >= 0 else float("nan"),
           "dead_fraction": dead_unit_fraction(result.model, X_val) if config.activation in ("relu", "leaky_relu") else float("nan")}
    return row, result.history, _predictions_frame(val_idx, fold, y_val_angstrom, y_pred)
