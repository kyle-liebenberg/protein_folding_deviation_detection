"""Loading, cleaning and splitting the Protein Tertiary Structure dataset (UCI id 265).

Pipeline (each step justified in docs/01-data-preparation.md):
    load_raw()  ->  clean()  ->  split_train_test()  ->  stratified_kfold() / split_train_val()
Feature transforms (log, clipping, scaling) live in src/preprocessing.py.

Note: the `ucimlrepo` snippet in the spec does NOT work for this dataset. UCI reports id 265 as
"not available for import". So, as the spec allows, we download the CSV from the dataset page's
direct link instead.
"""

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

DOWNLOAD_URL = (
    "https://archive.ics.uci.edu/static/public/265/"
    "physicochemical+properties+of+protein+tertiary+structure.zip"
)
CSV_NAME_IN_ZIP = "CASP.csv"

TARGET = "RMSD"
FEATURES = [f"F{i}" for i in range(1, 10)]  # F1 ... F9, as in the raw file

# F2 is dropped as REDUNDANT, not uninformative: F2 = F1 × F3 exactly
# (non-polar exposed area = total area × non-polar fraction). The MLP gains nothing from it.
# F5 ≈ 138.5 × F1 looked redundant too, but the small deviation (F5/F1 ≈ average mass of the exposed
# residues) carries information, and the MLP is consistently better with F5. So F5 is kept (docs/01).
DROPPED_FEATURES = ["F2"]
INPUT_FEATURES = [f for f in FEATURES if f not in DROPPED_FEATURES]  # what the MLP sees

# RMSD bins (Å) used to stratify the splits and to report per-bin errors (the "imbalance" handling).
# Equal 3 Å widths are easy to interpret. Counts range from ~3.1k to ~15k rows per bin.
RMSD_BIN_EDGES = [0, 3, 6, 9, 12, 15, 18, 21]
RMSD_BIN_LABELS = ["0-3", "3-6", "6-9", "9-12", "12-15", "15-18", "18-21"]

SPLIT_SEED = 42  # fixed: the test set must be the same in every experiment
TEST_FRACTION = 0.2

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = REPO_ROOT / "data" / "raw" / "protein.csv"


def load_raw(cache_path: Path = RAW_CSV) -> pd.DataFrame:
    """Return the full dataset as one DataFrame: columns F1..F9, then RMSD.

    The first call downloads the zip from UCI and saves the CSV. Later calls read the cached CSV,
    so they work offline and always see exactly the same data.
    """
    if cache_path.exists():
        return pd.read_csv(cache_path)

    with urllib.request.urlopen(DOWNLOAD_URL) as response:
        zip_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        df = pd.read_csv(archive.open(CSV_NAME_IN_ZIP))

    df = df[FEATURES + [TARGET]]  # the CSV has RMSD first. We put features first, target last

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows (1,711 of 45,730).

    Why: if a row and its copy end up on opposite sides of a split, the model is tested on a row it
    has already seen. That leaks information and makes the scores look better than they are.
    """
    return df.drop_duplicates().reset_index(drop=True)


def rmsd_bin(y) -> np.ndarray:
    """Map RMSD values (Å) to bin indices 0..6, using RMSD_BIN_EDGES."""
    # digitize on the inner edges: <3 -> 0, [3,6) -> 1, ..., >=18 -> 6
    return np.digitize(np.asarray(y), RMSD_BIN_EDGES[1:-1])


def inverse_bin_frequency_weights(y) -> np.ndarray:
    """Per-row loss weights: rows from rare RMSD bins count more. Normalised to average 1.

    A row in a bin holding 7 % of the data gets ~4.9x the weight of a row in the 34 % bin, so every
    bin contributes equally to the total loss.
    """
    bins = rmsd_bin(y)
    counts = np.bincount(bins, minlength=len(RMSD_BIN_LABELS))
    weights = 1.0 / counts[bins]
    return (weights / weights.mean()).astype(np.float32)


def split_train_test(df: pd.DataFrame, test_fraction: float = TEST_FRACTION, seed: int = SPLIT_SEED):
    """Hold out a test set, stratified on RMSD bin so both sets share the same target distribution.

    The test set is only used for final reporting. All tuning uses the training set (k-fold CV).
    """
    train_df, test_df = train_test_split(
        df, test_size=test_fraction, random_state=seed, stratify=rmsd_bin(df[TARGET])
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def stratified_kfold(y, k: int = 5, seed: int = SPLIT_SEED):
    """Yield (train_idx, val_idx) for k folds of the training set, stratified on RMSD bin."""
    folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    yield from folds.split(np.zeros(len(y)), rmsd_bin(y))


def split_train_val(indices, y, val_fraction: float = 0.1, seed: int = SPLIT_SEED):
    """Split `indices` into (fit_idx, early_stop_idx), stratified on RMSD bin.

    Used inside each CV fold: the early-stopping set must be separate from the fold's evaluation set.
    Otherwise we would pick the stopping epoch using the very data we report on.
    """
    indices = np.asarray(indices)
    return train_test_split(
        indices, test_size=val_fraction, random_state=seed, stratify=rmsd_bin(np.asarray(y)[indices])
    )
