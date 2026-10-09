"""The multilayer perceptron (MLP).

    input (8) -> [Linear -> activation] x len(hidden_sizes) -> Linear -> output (1)

The output layer has NO activation: this is regression, so the output must be free to take any
real value (the standardised RMSD).
"""

import torch
from torch import nn

ACTIVATIONS = {
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,  # ReLU with a small slope (0.01) for negative inputs: no "dead" units
    "gelu": nn.GELU,
}

# Which weight initialisation suits which activation (used when init="auto"):
#   Xavier/Glorot keeps the variance of activations constant for symmetric activations (sigmoid, tanh).
#   He/Kaiming doubles that variance to compensate for ReLU zeroing half of its inputs.
AUTO_INIT = {"sigmoid": "xavier", "tanh": "xavier", "relu": "he", "leaky_relu": "he", "gelu": "he"}


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_sizes=(64, 64), activation: str = "relu", init: str = "auto"):
        super().__init__()
        self.activation_name = activation
        layers = []
        prev = in_dim
        for width in hidden_sizes:
            layers += [nn.Linear(prev, width), ACTIVATIONS[activation]()]
            prev = width
        layers.append(nn.Linear(prev, 1))  # output layer: one number, no activation
        self.net = nn.Sequential(*layers)

        scheme = AUTO_INIT[activation] if init == "auto" else init
        self.reset_parameters(scheme)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def linear_layers(self) -> list[nn.Linear]:
        """All weight layers, in order. Used by pruning."""
        return [m for m in self.net if isinstance(m, nn.Linear)]

    def reset_parameters(self, scheme: str) -> None:
        """Initialise every weight matrix with `scheme`. Biases start at zero.

        - "xavier": uniform with variance 2 / (fan_in + fan_out)
        - "he":     normal  with variance 2 / fan_in
        - "random": normal  with a fixed std of 0.01, ignoring layer size (the naive baseline)
        """
        for layer in self.linear_layers():
            if scheme == "xavier":
                nn.init.xavier_uniform_(layer.weight)
            elif scheme == "he":
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            elif scheme == "random":
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)
            else:
                raise ValueError(f"unknown init scheme: {scheme}")
            nn.init.zeros_(layer.bias)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
