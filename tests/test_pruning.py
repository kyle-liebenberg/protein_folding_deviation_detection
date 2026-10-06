import numpy as np
import pytest
import torch

from src.model import MLP
from src.pruning import apply_masks, global_magnitude_masks, weight_sparsity
from src.training import TrainConfig, train


@pytest.mark.parametrize("sparsity", [0.0, 0.3, 0.5, 0.9, 0.98])
def test_exact_target_sparsity(sparsity):
    torch.manual_seed(0)
    model = MLP(8, (64, 64))
    apply_masks(model, global_magnitude_masks(model, sparsity))
    assert weight_sparsity(model) == pytest.approx(sparsity, abs=1e-3)


def test_pruning_removes_the_globally_smallest_weights():
    torch.manual_seed(0)
    model = MLP(8, (32, 32))
    before = torch.cat([l.weight.detach().abs().flatten() for l in model.linear_layers()])
    masks = global_magnitude_masks(model, 0.5)
    kept = torch.cat([m.flatten() for m in masks]).bool()
    assert before[kept].min() >= before[~kept].max()   # every kept weight is at least as large as every pruned one


def test_biases_are_not_pruned():
    torch.manual_seed(0)
    model = MLP(8, (16,))
    with torch.no_grad():
        for layer in model.linear_layers():
            layer.bias.fill_(1e-6)                        # tiny biases: would be pruned first if included
    apply_masks(model, global_magnitude_masks(model, 0.9))
    assert all((layer.bias == 1e-6).all() for layer in model.linear_layers())


def test_pruned_weights_stay_zero_during_fine_tuning():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(256, 8)).astype(np.float32)
    y = rng.normal(size=(256, 1)).astype(np.float32)
    torch.manual_seed(0)
    model = MLP(8, (32, 32))
    masks = global_magnitude_masks(model, 0.7)
    apply_masks(model, masks)

    config = TrainConfig(optimizer="momentum", lr=0.01, max_epochs=5, patience=None)
    train(config, X, y, X, y, model=model, after_step=lambda m: apply_masks(m, masks))

    assert weight_sparsity(model) == pytest.approx(0.7, abs=1e-3)
    for layer, mask in zip(model.linear_layers(), masks):
        assert (layer.weight[mask == 0] == 0).all()
        assert (layer.weight[mask == 1] != 0).any()      # the surviving weights did train
