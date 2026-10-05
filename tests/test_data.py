import numpy as np

from src.data import (
    FEATURES, INPUT_FEATURES, TARGET, clean, load_raw, rmsd_bin, split_train_test, split_train_val,
    stratified_kfold,
)


def test_raw_data_has_expected_shape_and_columns():
    # The first run downloads from UCI and later runs read the cached CSV.
    df = load_raw()
    assert df.shape == (45730, 10)  # 9 features + 1 target
    assert list(df.columns) == FEATURES + [TARGET]


def test_clean_removes_exact_duplicates():
    df = clean(load_raw())
    assert len(df) == 45730 - 1711
    assert not df.duplicated().any()


def test_only_f2_is_dropped_from_model_inputs():
    assert INPUT_FEATURES == ["F1", "F3", "F4", "F5", "F6", "F7", "F8", "F9"]


def test_f2_is_exactly_f1_times_f3():
    """The justification for dropping F2: it can be computed exactly from F1 and F3."""
    df = clean(load_raw())
    assert np.allclose(df["F2"], df["F1"] * df["F3"], rtol=1e-3)


def test_rmsd_bins_cover_the_full_range():
    assert list(rmsd_bin([0.0, 2.99, 3.0, 17.9, 18.0, 20.999])) == [0, 0, 1, 5, 6, 6]


def test_train_test_split_is_disjoint_stratified_and_repeatable():
    df = clean(load_raw())
    train, test = split_train_test(df)
    assert len(train) + len(test) == len(df)
    assert round(len(test) / len(df), 3) == 0.2

    # Stratification: every bin's share differs by < 0.5 percentage points between the two sets.
    share = lambda d: np.bincount(rmsd_bin(d[TARGET]), minlength=7) / len(d)
    assert np.abs(share(train) - share(test)).max() < 0.005

    # Repeatable: the same seed gives the same test set.
    _, test_again = split_train_test(df)
    assert test.equals(test_again)


def test_kfold_folds_partition_the_training_set():
    df = clean(load_raw())
    train, _ = split_train_test(df)
    val_indices = [val for _, val in stratified_kfold(train[TARGET], k=5)]
    all_val = np.concatenate(val_indices)
    assert len(val_indices) == 5
    assert sorted(all_val) == list(range(len(train)))  # every row is validated exactly once


def test_early_stopping_split_stays_inside_the_fold():
    y = np.linspace(0, 20.9, 1000)
    fold_train = np.arange(0, 800)
    fit_idx, stop_idx = split_train_val(fold_train, y, val_fraction=0.1)
    assert set(fit_idx) | set(stop_idx) == set(fold_train)
    assert not set(fit_idx) & set(stop_idx)
    assert len(stop_idx) == 80
