import numpy as np

from src.data import INPUT_FEATURES, clean, load_raw, split_train_test
from src.preprocessing import Preprocessor


def _fitted():
    train, test = split_train_test(clean(load_raw()))
    return Preprocessor().fit(train), train, test


def test_training_features_are_standardised():
    prep, train, _ = _fitted()
    X = prep.transform_X(train)
    assert X.shape == (len(train), len(INPUT_FEATURES))
    assert np.allclose(X.mean(axis=0), 0, atol=1e-4)
    assert np.allclose(X.std(axis=0), 1, atol=1e-3)


def test_test_set_uses_training_statistics_only():
    """No leakage: transforming the test set must use the bounds/means learned on train."""
    prep, train, test = _fitted()
    # An absurd test value is clipped to the TRAINING upper bound and not stretched further.
    extreme = test.head(1).copy()
    extreme["F7"] = 1e9
    x_extreme = prep.transform_X(extreme)[0, INPUT_FEATURES.index("F7")]
    x_train_max = prep.transform_X(train)[:, INPUT_FEATURES.index("F7")].max()
    assert np.isclose(x_extreme, x_train_max)

    # Refitting on train gives identical parameters, so the test set never influenced them.
    again = Preprocessor().fit(train)
    assert again.x_mean.equals(prep.x_mean) and again.clip_high.equals(prep.clip_high)


def test_target_round_trip_returns_angstroms():
    prep, train, _ = _fitted()
    y_scaled = prep.transform_y(train)
    assert y_scaled.shape == (len(train), 1)
    assert np.allclose(prep.inverse_transform_y(y_scaled), train["RMSD"].to_numpy(), atol=1e-4)
