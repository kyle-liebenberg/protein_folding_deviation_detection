"""Global magnitude pruning, implemented with binary masks.

    masks = global_magnitude_masks(model, sparsity=0.9)   # keep the largest 10 % of all weights
    apply_masks(model, masks)                             # set the other 90 % to zero

"Global": all weight matrices are ranked TOGETHER by |w|, so layers can end up with different
sparsities (a layer full of small weights loses more). Biases are never pruned: they are few, and
each one shifts a whole unit.

During fine-tuning the masks are re-applied after every optimiser step (training.train's `after_step`
hook). The optimiser would otherwise move pruned weights away from zero again.
"""

import torch

from src.model import MLP


def global_magnitude_masks(model: MLP, sparsity: float) -> list[torch.Tensor]:
    """One 0/1 mask per weight matrix. Exactly round(sparsity × total) weights get mask 0."""
    weights = [layer.weight.detach() for layer in model.linear_layers()]
    magnitudes = torch.cat([w.abs().flatten() for w in weights])
    n_prune = round(sparsity * magnitudes.numel())

    keep = torch.ones_like(magnitudes)
    if n_prune > 0:
        smallest = torch.argsort(magnitudes)[:n_prune]  # indices of the n_prune smallest |w|
        keep[smallest] = 0.0                            # (argsort breaks ties, so the count is exact)

    # cut the flat mask back into one mask per layer
    masks, start = [], 0
    for w in weights:
        masks.append(keep[start:start + w.numel()].reshape(w.shape))
        start += w.numel()
    return masks


@torch.no_grad()
def apply_masks(model: MLP, masks: list[torch.Tensor]) -> None:
    """Zero every masked weight (in place)."""
    for layer, mask in zip(model.linear_layers(), masks):
        layer.weight.mul_(mask)


@torch.no_grad()
def weight_sparsity(model: MLP) -> float:
    """Fraction of weight-matrix entries that are exactly zero."""
    weights = [layer.weight for layer in model.linear_layers()]
    zeros = sum(int((w == 0).sum()) for w in weights)
    return zeros / sum(w.numel() for w in weights)
