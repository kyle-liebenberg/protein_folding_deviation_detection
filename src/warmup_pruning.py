"""Part 2 (spec §2.3): does learning-rate warmup make the network more robust to pruning?

The decision rules below were fixed before any of these experiments ran (notebook section 4):
    Step 1  lr_sweep()          train with/without warmup over an LR grid       -> tests H1
    Step 2  choose_conditions() apply the pre-registered rules -> A, B, C
    Step 3  prune_and_finetune() train each condition, prune at each sparsity,
                                 fine-tune identically, measure test RMSE       -> tests H2, H3
Optional extensions, same machinery:
    retrain="lr_rewind"   retrain pruned nets with their own schedule instead of fine-tuning
    warmup_epochs=...     condition B with a shorter or longer warmup
Results are cached in results/phase4_*.csv (pass rerun=True to retrain).
"""

import copy
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.data import REPO_ROOT, TARGET, split_train_val
from src.diagnostics import dead_unit_fraction
from src.evaluation import rmse
from src.preprocessing import Preprocessor
from src.pruning import apply_masks, global_magnitude_masks, weight_sparsity
from src.training import TrainConfig, predict, train

RESULTS_DIR = REPO_ROOT / "results"

# Fixed setup
BASE = TrainConfig(hidden_sizes=(256, 256, 256, 256), activation="relu", optimizer="momentum",
                   batch_size=256, max_epochs=100, decay="cosine", patience=None)  # fixed budget, final weights
WARMUP_EPOCHS = 5
FINE_TUNE = TrainConfig(hidden_sizes=BASE.hidden_sizes, optimizer="momentum", lr=0.001,
                        max_epochs=10, decay="constant", patience=None)              # identical for every condition

SWEEP_LRS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2]
SWEEP_SEEDS = [0, 1, 2]
SEEDS = [0, 1, 2, 3, 4]
SPARSITIES = [0.0, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98]
STABLE_TOLERANCE = 0.10      # Å: "within 0.10 Å of that schedule's best" -> counts as stable
COMPARABLE_TOLERANCE = 0.05  # Å: B must be within this of A's unpruned validation RMSE


@dataclass
class Data:
    """Preprocessed arrays. The preprocessor is fitted on the fit rows only."""
    prep: Preprocessor
    X_fit: np.ndarray
    y_fit: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray     # standardised (for training curves)
    y_val_angstrom: np.ndarray
    X_test: np.ndarray
    y_test_angstrom: np.ndarray


def prepare_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Data:
    fit_idx, val_idx = split_train_val(np.arange(len(train_df)), train_df[TARGET].to_numpy())
    fit, val = train_df.iloc[fit_idx], train_df.iloc[val_idx]
    prep = Preprocessor().fit(fit)
    return Data(prep, prep.transform_X(fit), prep.transform_y(fit), prep.transform_X(val), prep.transform_y(val),
                val[TARGET].to_numpy(), prep.transform_X(test_df), test_df[TARGET].to_numpy())


def schedule(warmup: bool, lr: float, seed: int, warmup_epochs: float = WARMUP_EPOCHS) -> TrainConfig:
    return replace(BASE, lr=lr, seed=seed, warmup_epochs=warmup_epochs if warmup else 0.0)


def retrain_config(original: TrainConfig, retrain: str) -> TrainConfig:
    """How a pruned network is retrained (always 10 epochs, the same for every condition).

    "finetune":  small constant LR 0.001 (the main experiment, as in [2]'s fine-tuning)
    "lr_rewind": the network's OWN schedule shape compressed into 10 epochs: same peak LR, warmup for
                 the same 5 % of the time if it had warmup, cosine to 0 (as in [2]'s LR rewinding)
    """
    if retrain == "finetune":
        return replace(FINE_TUNE, seed=original.seed)
    if retrain == "lr_rewind":
        fraction = original.warmup_epochs / original.max_epochs
        return replace(original, max_epochs=FINE_TUNE.max_epochs, warmup_epochs=fraction * FINE_TUNE.max_epochs)
    raise ValueError(retrain)


def _rmse_angstrom(model, data: Data, X, y_angstrom) -> float:
    return rmse(y_angstrom, data.prep.inverse_transform_y(predict(model, X)))


# ---------------------------------------------------------------- Step 1: LR-tolerance sweep
def _sweep_run(data: Data, warmup: bool, lr: float, seed: int) -> dict:
    result = train(schedule(warmup, lr, seed), data.X_fit, data.y_fit, data.X_val, data.y_val)
    val_rmse = float("nan") if result.diverged else _rmse_angstrom(result.model, data, data.X_val, data.y_val_angstrom)
    return {"warmup": warmup, "lr": lr, "seed": seed, "diverged": result.diverged,
            "diverged_at_epoch": result.epochs_run if result.diverged else np.nan, "val_rmse": val_rmse}


def lr_sweep(data: Data, rerun: bool = False, n_jobs: int = -1) -> pd.DataFrame:
    path = RESULTS_DIR / "phase4_lr_sweep.csv"
    if path.exists() and not rerun:
        return pd.read_csv(path)
    jobs = [delayed(_sweep_run)(data, w, lr, s) for w in [False, True] for lr in SWEEP_LRS for s in SWEEP_SEEDS]
    sweep = pd.DataFrame(Parallel(n_jobs=n_jobs)(jobs))
    sweep.to_csv(path, index=False)
    return sweep


def summarise_sweep(sweep: pd.DataFrame) -> pd.DataFrame:
    """Per (warmup, lr): mean/std val RMSE over seeds, and how many seeds diverged."""
    g = sweep.groupby(["warmup", "lr"])
    return pd.DataFrame({"val_rmse": g["val_rmse"].mean(), "val_rmse_std": g["val_rmse"].std(),
                         "diverged_seeds": g["diverged"].sum().astype(int), "seeds": g.size()})


def max_stable_lr(summary: pd.DataFrame, warmup: bool) -> float:
    """Largest LR where no seed diverged and val RMSE is within STABLE_TOLERANCE of this schedule's best."""
    s = summary.loc[warmup]
    ok = (s["diverged_seeds"] == 0) & (s["val_rmse"] <= s["val_rmse"].min() + STABLE_TOLERANCE)
    return float(s.index[ok].max())


# ---------------------------------------------------------------- Step 2: pre-registered condition rules
def choose_conditions(summary: pd.DataFrame) -> dict:
    """A: no warmup at its best LR (η0). C: warmup at η0.
    B: warmup at the LARGEST LR above η0 that is stable (all seeds) and within COMPARABLE_TOLERANCE of A.
    Returns {name: (warmup, lr)}. B is missing if no LR satisfies the rule (reported, not patched)."""
    no_wu, wu = summary.loc[False], summary.loc[True]
    eta0 = float(no_wu["val_rmse"].idxmin())
    a_rmse = no_wu.loc[eta0, "val_rmse"]
    candidates = wu[(wu.index > eta0) & (wu["diverged_seeds"] == 0)
                    & ((wu["val_rmse"] - a_rmse).abs() <= COMPARABLE_TOLERANCE)]
    conditions = {"A: no warmup, η0": (False, eta0)}
    if len(candidates):
        conditions["B: warmup, η1 > η0"] = (True, float(candidates.index.max()))
    conditions["C: warmup, η0 (control)"] = (True, eta0)
    return conditions


# ---------------------------------------------------------------- Step 3: train, prune, fine-tune
def _condition_run(data: Data, name: str, warmup: bool, lr: float, seed: int,
                   warmup_epochs: float = WARMUP_EPOCHS, retrain: str = "finetune") -> list[dict]:
    config = schedule(warmup, lr, seed, warmup_epochs)
    trained = train(config, data.X_fit, data.y_fit, data.X_val, data.y_val).model
    dense_test = _rmse_angstrom(trained, data, data.X_test, data.y_test_angstrom)
    common = {"condition": name, "warmup": warmup, "lr": lr, "seed": seed,
              "warmup_epochs": config.warmup_epochs, "retrain": retrain,
              "dense_val_rmse": _rmse_angstrom(trained, data, data.X_val, data.y_val_angstrom),
              "dense_test_rmse": dense_test,
              "dead_fraction": dead_unit_fraction(trained, data.X_fit)}

    rows = []
    for sparsity in SPARSITIES:
        model = copy.deepcopy(trained)              # every sparsity starts from the same trained network
        masks = global_magnitude_masks(model, sparsity)
        apply_masks(model, masks)
        before = _rmse_angstrom(model, data, data.X_test, data.y_test_angstrom)

        train(retrain_config(config, retrain), data.X_fit, data.y_fit, data.X_val, data.y_val,
              model=model, after_step=lambda m: apply_masks(m, masks))
        after = _rmse_angstrom(model, data, data.X_test, data.y_test_angstrom)

        rows.append({**common, "sparsity": sparsity, "actual_sparsity": weight_sparsity(model),
                     "test_rmse_before_ft": before, "test_rmse_after_ft": after,
                     "delta": after - dense_test})   # robustness: error added by pruning, after recovery
    return rows


def prune_and_finetune(data: Data, conditions: dict, rerun: bool = False, n_jobs: int = -1,
                       retrain: str = "finetune", cache_name: str = "phase4_pruning") -> pd.DataFrame:
    """conditions: {name: (warmup, lr)} or {name: (warmup, lr, warmup_epochs)}."""
    path = RESULTS_DIR / f"{cache_name}.csv"
    if path.exists() and not rerun:
        return pd.read_csv(path)
    jobs = [delayed(_condition_run)(data, name, spec[0], spec[1], s, *spec[2:], retrain=retrain)
            for name, spec in conditions.items() for s in SEEDS]
    results = pd.DataFrame([row for rows in Parallel(n_jobs=n_jobs)(jobs) for row in rows])
    results.to_csv(path, index=False)
    return results


def paired_check(results: pd.DataFrame, better: str, worse: str, sparsities=(0.7, 0.9, 0.95, 0.98)) -> pd.DataFrame:
    """Per sparsity: mean Δ difference (better − worse) and in how many seeds `better` has the smaller Δ.
    The H2/H3 criterion: smaller in ≥ 4/5 seeds at ≥ 2 of the 4 high sparsities, beyond the seed std."""
    rows = []
    for s in sparsities:
        d = results[results["sparsity"] == s].pivot_table(index="seed", columns="condition", values="delta")
        diff = d[better] - d[worse]
        rows.append({"sparsity": s, "mean_delta_diff": diff.mean(), "seed_std_of_diff": diff.std(),
                     "seeds_better": f"{int((diff < 0).sum())}/{len(diff)}",
                     "counts": bool((diff < 0).sum() >= 4 and -diff.mean() > diff.std())})
    return pd.DataFrame(rows).set_index("sparsity")
