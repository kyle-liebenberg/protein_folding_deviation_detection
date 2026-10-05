import numpy as np
import pytest
import torch

from src.cv import cross_validate
from src.data import clean, load_raw, split_train_test
from src.training import TrainConfig, train


def _toy_data(n, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 7)).astype(np.float32)
    y = (np.sin(X[:, :1]) + X[:, 1:2] ** 2).astype(np.float32)
    return X, y


def test_can_overfit_a_tiny_batch():
    """Sanity check: if the model, loss and update are wired correctly, 32 rows can be memorised."""
    X, y = _toy_data(32)
    config = TrainConfig(hidden_sizes=(64, 64), lr=1e-2, batch_size=32, max_epochs=1000, patience=None)
    result = train(config, X, y, X, y)
    assert result.history["train_loss"][-1] < 1e-3


def test_same_seed_same_result_different_seed_different_result():
    X, y = _toy_data(200)
    config = TrainConfig(max_epochs=5, patience=None)
    a = train(config, X, y, X, y).history["train_loss"]
    b = train(config, X, y, X, y).history["train_loss"]
    c = train(TrainConfig(max_epochs=5, patience=None, seed=1), X, y, X, y).history["train_loss"]
    assert a == b and a != c


def test_early_stopping_keeps_the_best_epoch():
    X, y = _toy_data(300)
    X_stop, y_stop = _toy_data(100, seed=1)
    result = train(TrainConfig(max_epochs=300, patience=5), X, y, X_stop, y_stop)
    val = result.history["val_loss"]
    assert result.epochs_run < 300                                   # it did stop early
    assert result.epochs_run == result.best_epoch + 5                # ... exactly `patience` epochs after the best
    assert min(val) == pytest.approx(val[result.best_epoch - 1])


def test_warmup_lr_is_recorded():
    X, y = _toy_data(256)
    config = TrainConfig(lr=0.01, batch_size=32, max_epochs=4, warmup_epochs=2, patience=None)
    lrs = train(config, X, y, X, y).history["lr"]                  # LR at the end of each epoch
    assert lrs[0] == pytest.approx(0.005) and lrs[1:] == pytest.approx([0.01] * 3)


def test_huge_learning_rate_is_flagged_as_diverged():
    X, y = _toy_data(256)
    result = train(TrainConfig(optimizer="sgd", lr=50.0, max_epochs=20, patience=None), X, y, X, y)
    assert result.diverged


def test_after_step_hook_runs_every_step():
    X, y = _toy_data(100)
    calls = []
    train(TrainConfig(batch_size=10, max_epochs=2, patience=None), X, y, X, y, after_step=lambda m: calls.append(1))
    assert len(calls) == 20


def test_cross_validation_returns_one_row_per_fold():
    train_df, _ = split_train_test(clean(load_raw()))
    result = cross_validate(TrainConfig(max_epochs=2, patience=None), train_df.sample(2000, random_state=0), k=3)
    assert len(result.folds) == 3 and len(result.histories) == 3
    assert set(result.summary().index) >= {"rmse_mean", "rmse_std", "r2_mean"}
