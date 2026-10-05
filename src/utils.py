"""Reproducibility helpers: seeding every random number generator, and choosing a device."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch so a run can be repeated exactly.

    Every experiment calls this first. The warmup/pruning study (Part 2) repeats each run over several
    seeds, and the seed is the *only* thing that changes between those repeats.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def use_single_cpu_thread() -> None:
    """Run PyTorch on one CPU thread.

    Measured on our MLPs: 0.018 s/epoch with 1 thread vs 0.065 s/epoch with 8. For tiny matrix
    multiplications, coordinating threads costs more than it saves. We get parallelism by running
    several CV folds at once in separate processes instead (see src/cv.py).
    """
    torch.set_num_threads(1)


def get_device(prefer_gpu: bool = False) -> torch.device:
    """Return the device to train on. Defaults to CPU.

    Why CPU by default: our MLPs are tiny (8 inputs, a few thousand weights). For a network this
    small, the overhead of sending each mini-batch to the Apple GPU (MPS) costs more than the GPU
    saves, and CPU results are bit-for-bit reproducible. Pass `prefer_gpu=True` to use MPS/CUDA
    when it is available.
    """
    if prefer_gpu:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")
