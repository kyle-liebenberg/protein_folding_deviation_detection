"""Training an MLP: an explicit mini-batch gradient-descent loop with early stopping.

One optimiser step, in order:
    1. set this step's learning rate (from src/schedules.py: this is where warmup happens)
    2. forward pass:  predictions = model(x_batch)
    3. loss:          mean squared error (optionally weighted per sample)
    4. backward pass: loss.backward() computes dLoss/dWeight for every weight (backpropagation)
    5. update:        optimizer.step() moves every weight against its gradient
    6. after_step hook (used by pruning to keep pruned weights at zero)
"""

import copy
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from src.model import MLP
from src.schedules import learning_rate
from src.utils import set_seed

# A standardised target has variance 1, so predicting the mean gives MSE ≈ 1.
# A loss above this threshold (or NaN/inf) means training has blown up.
DIVERGENCE_LOSS = 100.0


@dataclass
class TrainConfig:
    """Everything that defines one training run. Experiments change one field at a time."""
    hidden_sizes: tuple = (64, 64)
    activation: str = "relu"
    init: str = "auto"            # "auto" picks Xavier or He to match the activation (see model.py)
    optimizer: str = "adam"       # "sgd", "momentum", "rmsprop", "adam"
    lr: float = 1e-3              # peak learning rate
    batch_size: int = 256
    max_epochs: int = 500           # upper limit. Early stopping usually ends training well before this
    warmup_epochs: float = 0.0    # length of linear warmup, in epochs (0 = no warmup)
    decay: str = "constant"       # LR after warmup: "constant" or "cosine"
    patience: int | None = 20     # early stopping: stop after this many epochs without improvement.
                                  # None = fixed budget: train all max_epochs and keep the FINAL weights
    weighted_loss: bool = False   # weight samples by inverse RMSD-bin frequency (imbalance handling)
    seed: int = 0                 # controls weight initialisation and mini-batch order


@dataclass
class TrainResult:
    model: MLP
    history: dict = field(default_factory=lambda: {"train_loss": [], "val_loss": [], "lr": []})
    best_epoch: int = 0           # epoch whose weights were kept (lowest validation loss)
    epochs_run: int = 0
    diverged: bool = False
    seconds: float = 0.0


def make_optimizer(name: str, params, lr: float) -> torch.optim.Optimizer:
    """The four optimisers of the core investigation (the LR is overwritten every step anyway)."""
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if name == "momentum":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    raise ValueError(f"unknown optimizer: {name}")


def train(config: TrainConfig, X_fit, y_fit, X_stop, y_stop, sample_weights=None,
          model: MLP | None = None, after_step=None) -> TrainResult:
    """Train on (X_fit, y_fit), early-stopping on (X_stop, y_stop). Returns the best model found.

    Pass `model` to continue training an existing network (fine-tuning after pruning). Otherwise a new
    MLP is built from `config`. All arrays must already be preprocessed (standardised).
    """
    start = time.perf_counter()
    torch.set_num_threads(1)  # small networks train ~3.6x faster on one thread
    set_seed(config.seed)
    if model is None:
        model = MLP(X_fit.shape[1], config.hidden_sizes, config.activation, config.init)

    X_fit, y_fit = torch.as_tensor(X_fit), torch.as_tensor(y_fit)
    X_stop, y_stop = torch.as_tensor(X_stop), torch.as_tensor(y_stop)
    w_fit = None if sample_weights is None else torch.as_tensor(sample_weights, dtype=torch.float32).reshape(-1, 1)

    optimizer = make_optimizer(config.optimizer, model.parameters(), config.lr)
    shuffler = torch.Generator().manual_seed(config.seed)  # mini-batch order, reproducible

    n = len(X_fit)
    steps_per_epoch = math.ceil(n / config.batch_size)
    total_steps = config.max_epochs * steps_per_epoch
    warmup_steps = round(config.warmup_epochs * steps_per_epoch)

    result = TrainResult(model=model)
    best_val, best_state, epochs_without_improvement = math.inf, None, 0
    step = 0

    for epoch in range(config.max_epochs):
        model.train()
        order = torch.randperm(n, generator=shuffler)  # reshuffle every epoch
        epoch_loss = 0.0

        for i in range(0, n, config.batch_size):
            batch = order[i:i + config.batch_size]

            lr = learning_rate(step, config.lr, total_steps, warmup_steps, config.decay)   # 1
            for group in optimizer.param_groups:
                group["lr"] = lr

            predictions = model(X_fit[batch])                                             # 2
            squared_errors = (predictions - y_fit[batch]) ** 2                            # 3
            loss = squared_errors.mean() if w_fit is None else (w_fit[batch] * squared_errors).mean()

            optimizer.zero_grad()                                                         # 4
            loss.backward()
            optimizer.step()                                                              # 5
            if after_step is not None:                                                    # 6
                after_step(model)

            step += 1
            epoch_loss += loss.item() * len(batch)

        train_loss = epoch_loss / n  # average loss over the epoch's mini-batches
        val_loss = evaluate_loss(model, X_stop, y_stop)
        result.history["train_loss"].append(train_loss)
        result.history["val_loss"].append(val_loss)
        result.history["lr"].append(lr)
        result.epochs_run = epoch + 1

        if not (math.isfinite(train_loss) and train_loss < DIVERGENCE_LOSS):
            result.diverged = True
            break

        # Early stopping: remember the best weights. Stop if no improvement for `patience` epochs.
        if val_loss < best_val:
            best_val, best_state, result.best_epoch = val_loss, copy.deepcopy(model.state_dict()), epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if config.patience is not None and epochs_without_improvement >= config.patience:
                break

    if config.patience is not None and best_state is not None:
        model.load_state_dict(best_state)  # early stopping keeps the best epoch, not the last one
    result.seconds = time.perf_counter() - start
    return result


@torch.no_grad()
def evaluate_loss(model: MLP, X, y) -> float:
    """Unweighted MSE on (X, y), in standardised units."""
    model.eval()
    return torch.mean((model(torch.as_tensor(X)) - torch.as_tensor(y)) ** 2).item()


@torch.no_grad()
def predict(model: MLP, X) -> np.ndarray:
    """Model outputs (standardised) as a NumPy array of shape (n, 1)."""
    model.eval()
    return model(torch.as_tensor(X)).numpy()
