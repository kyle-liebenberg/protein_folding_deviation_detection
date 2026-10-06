import math

import numpy as np
import torch

from src.cv import cross_validate_many
from src.data import clean, load_raw, split_train_test
from src.diagnostics import dead_unit_fraction, epochs_to_threshold, layer_gradient_sizes
from src.model import MLP
from src.training import TrainConfig


def test_epochs_to_threshold():
    assert epochs_to_threshold([0.9, 0.7, 0.49, 0.45]) == 3.0
    assert math.isnan(epochs_to_threshold([0.9, 0.8]))


def test_one_gradient_size_per_layer():
    torch.manual_seed(0)
    sizes = layer_gradient_sizes(MLP(8, (16, 16, 16)), torch.randn(64, 8), torch.randn(64, 1))
    assert len(sizes) == 4 and all(s > 0 for s in sizes)


def test_sigmoid_gradients_shrink_towards_the_input_more_than_relu():
    """The vanishing-gradient effect behind hypothesis H-C2, averaged over seeds."""
    X, y = torch.randn(256, 8), torch.randn(256, 1)

    def first_over_last(activation):
        ratios = []
        for seed in range(5):
            torch.manual_seed(seed)
            sizes = layer_gradient_sizes(MLP(8, (64,) * 6, activation), X, y)
            ratios.append(sizes[0] / sizes[-1])
        return np.mean(ratios)

    assert first_over_last("sigmoid") < first_over_last("relu") / 10


def test_dead_unit_fraction_counts_units_that_never_fire():
    model = MLP(2, (4,), "relu")
    with torch.no_grad():
        layer = model.linear_layers()[0]
        layer.weight.zero_()
        layer.bias.copy_(torch.tensor([1.0, 1.0, -1.0, -1.0]))  # two units always on, two always off
    assert dead_unit_fraction(model, torch.randn(100, 2)) == 0.5


def test_cross_validate_many_returns_one_result_per_config():
    train_df, _ = split_train_test(clean(load_raw()))
    small = train_df.sample(2000, random_state=0)
    configs = [TrainConfig(max_epochs=2, patience=None), TrainConfig(max_epochs=2, patience=None, activation="tanh")]
    results = cross_validate_many(configs, small, k=3, n_jobs=2)
    assert len(results) == 2 and all(len(r.folds) == 3 for r in results)
    assert results[0].folds["dead_fraction"].notna().all()     # measured for ReLU
    assert results[1].folds["dead_fraction"].isna().all()      # not applicable to tanh
