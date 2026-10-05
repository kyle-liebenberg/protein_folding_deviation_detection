"""Learning-rate schedules, written as a plain function of the training step.

Every schedule is "optional linear warmup, then a decay rule":

    warmup_steps = 0,  decay = "constant"  ->  constant LR (no warmup)
    warmup_steps = 0,  decay = "cosine"    ->  cosine decay from the start
    warmup_steps > 0,  decay = "constant"  ->  linear warmup, then constant
    warmup_steps > 0,  decay = "cosine"    ->  linear warmup, then cosine decay

    LR
    peak ┤      ╭──────────────  (constant)
         │     ╱ ╲
         │    ╱    ╲             (cosine)
         │   ╱       ╲
       0 ┼──╱──────────╲──→ step
           |warmup|
"""

import math


def learning_rate(step: int, peak_lr: float, total_steps: int,
                  warmup_steps: int = 0, decay: str = "constant") -> float:
    """The learning rate to use at optimiser step `step` (counting from 0)."""
    # Warmup: rise linearly from peak_lr / warmup_steps to peak_lr.
    # (step + 1) so the very first step does not have LR = 0, which would waste the step.
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps

    if decay == "constant":
        return peak_lr

    if decay == "cosine":
        # progress goes 0 -> 1 over the steps after warmup. The LR follows half a cosine wave, peak -> 0.
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return peak_lr * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    raise ValueError(f"unknown decay: {decay}")
