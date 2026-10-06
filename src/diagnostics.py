"""Looking inside a network: gradient flow, dead units, and convergence speed.

- layer_gradient_sizes: how big the gradients are in each layer (vanishing gradients, Axis C)
- dead_unit_fraction:   how many ReLU-family units never activate (dead ReLUs, Axis C)
- epochs_to_threshold:  how quickly training gets good (convergence speed, all axes)
"""

import numpy as np
import torch

from src.model import MLP

# Convergence-speed threshold, in standardised MSE: predicting the mean gives 1.0,
# and the baseline MLP ends at about 0.43.
SPEED_THRESHOLD = 0.50


def layer_gradient_sizes(model: MLP, X, y) -> list[float]:
    """Mean |∂loss/∂w| of each layer's weights, input layer first, for one forward/backward pass.

    Backpropagation multiplies by each activation's derivative on the way back. If that derivative is
    small (sigmoid: at most 0.25), gradients shrink layer by layer, so early layers barely learn.
    """
    model.zero_grad()
    loss = torch.mean((model(torch.as_tensor(X)) - torch.as_tensor(y)) ** 2)
    loss.backward()
    return [layer.weight.grad.abs().mean().item() for layer in model.linear_layers()]


@torch.no_grad()
def dead_unit_fraction(model: MLP, X) -> float:
    """Fraction of hidden units whose input is ≤ 0 for EVERY row of X.

    For ReLU such a unit always outputs 0 and gets zero gradient, so it can never recover ("dead").
    Leaky ReLU units in that state still pass a small gradient (slope 0.01), so they are counted the
    same way to compare the two, but they are not truly stuck.
    """
    h = torch.as_tensor(X)
    dead, total = 0, 0
    layers = list(model.net)
    for i, module in enumerate(layers[:-1]):       # the last module is the output layer
        h = module(h)
        if isinstance(module, torch.nn.Linear):    # h is now a layer's pre-activation
            never_positive = (h <= 0).all(dim=0)
            dead += int(never_positive.sum())
            total += h.shape[1]
    return dead / total if total else 0.0


def epochs_to_threshold(val_losses, threshold: float = SPEED_THRESHOLD) -> float:
    """First epoch (1-based) whose early-stopping MSE is below `threshold`. NaN if never reached."""
    below = np.flatnonzero(np.asarray(val_losses) < threshold)
    return float(below[0] + 1) if len(below) else float("nan")
