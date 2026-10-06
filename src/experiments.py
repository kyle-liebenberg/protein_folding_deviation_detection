"""Running an experiment grid (many configurations × seeds) with results cached in results/.

    table, curves = run_grid("optimisers", {"adam lr=1e-3": TrainConfig(...), ...})

- `table`:  one row per (configuration, seed), with the CV summary (mean/std over folds) and the
            configuration's settings, so it can be grouped by optimiser, lr, depth, ...
- `curves`: the early-stopping MSE per epoch, averaged over folds and seeds, per configuration.

Results are saved to results/phase3_<name>.csv and results/phase3_<name>_curves.csv. Later calls load
them instead of retraining (pass rerun=True to force a fresh run).
"""

from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from src.cv import cross_validate_many
from src.data import INPUT_FEATURES, REPO_ROOT
from src.training import TrainConfig

RESULTS_DIR = REPO_ROOT / "results"
SEEDS = (0, 1, 2)


def run_grid(name: str, configs: dict[str, TrainConfig], train_df: pd.DataFrame, seeds=SEEDS,
             features=INPUT_FEATURES, rerun: bool = False, n_jobs: int = -1):
    table_path = RESULTS_DIR / f"phase3_{name}.csv"
    curves_path = RESULTS_DIR / f"phase3_{name}_curves.csv"
    if table_path.exists() and curves_path.exists() and not rerun:
        return pd.read_csv(table_path), pd.read_csv(curves_path)

    jobs = [(label, seed, replace(config, seed=seed)) for label, config in configs.items() for seed in seeds]
    results = cross_validate_many([config for _, _, config in jobs], train_df, features=features, n_jobs=n_jobs)

    rows, curves = [], []
    for (label, seed, config), result in zip(jobs, results):
        settings = asdict(config)
        settings["hidden_sizes"] = "x".join(map(str, config.hidden_sizes))  # e.g. "64x64"
        settings["depth"], settings["width"] = len(config.hidden_sizes), max(config.hidden_sizes, default=0)
        rows.append({"label": label, **settings, **result.summary().to_dict()})
        for fold, history in enumerate(result.histories):
            curves += [{"label": label, "seed": seed, "fold": fold, "epoch": e + 1, "val_loss": v, "train_loss": t}
                       for e, (v, t) in enumerate(zip(history["val_loss"], history["train_loss"]))]

    table = pd.DataFrame(rows)
    curves = (pd.DataFrame(curves).groupby(["label", "epoch"], as_index=False)[["val_loss", "train_loss"]].mean())
    RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(table_path, index=False)
    curves.to_csv(curves_path, index=False)
    return table, curves


def aggregate(table: pd.DataFrame, by) -> pd.DataFrame:
    """Average the per-seed rows over seeds.

    rmse = mean over seeds of the CV RMSE. rmse_seed_std = how much it moves between seeds.
    rmse_fold_std = the typical fold-to-fold std within one seed.
    """
    grouped = table.groupby(by, sort=False)
    out = pd.DataFrame({
        "rmse": grouped["rmse_mean"].mean(),
        "rmse_seed_std": grouped["rmse_mean"].std(),
        "rmse_fold_std": grouped["rmse_std"].mean(),
        "r2": grouped["r2_mean"].mean(),
        "epochs_to_0.50": grouped["epochs_to_threshold_mean"].mean(),
        "reached_0.50": grouped["folds_reaching_threshold"].sum().astype(str) + "/" + (grouped.size() * 5).astype(str),
        "best_epoch": grouped["best_epoch_mean"].mean(),
        "gap_at_best": grouped["gap_at_best_mean"].mean(),
        "diverged_folds": grouped["diverged_folds"].sum(),
    })
    if table["dead_fraction_mean"].notna().any():
        out["dead_fraction"] = grouped["dead_fraction_mean"].mean()
    return out


def seeds_agree(table: pd.DataFrame, by: str, a, b) -> str:
    """Paired comparison of RMSE: in how many seeds is configuration `a` better than `b`?"""
    wide = table.pivot_table(index="seed", columns=by, values="rmse_mean")
    diff = wide[a] - wide[b]
    return f"{a} vs {b}: mean diff {diff.mean():+.3f} Å, {a} better in {int((diff < 0).sum())}/{len(diff)} seeds"
