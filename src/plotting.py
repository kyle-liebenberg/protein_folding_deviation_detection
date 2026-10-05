"""One consistent look for every figure in the notebook and report.

Colour rules: categorical colours are used in this fixed order (slot 1 first) and follow the
*thing* being plotted (e.g. "warmup" is always the same colour). Text is never coloured.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src.data import REPO_ROOT

# Categorical palette (colour-blind-safe order). Use in this order, never cycled.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948",
)
CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]

# Ink and surfaces
TEXT_PRIMARY, TEXT_SECONDARY, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
NEUTRAL_MID = "#f0efec"

# Diverging map for correlations: blue (-1) -> neutral grey (0) -> red (+1)
DIVERGING = LinearSegmentedColormap.from_list("blue_grey_red", [BLUE, NEUTRAL_MID, RED])

FIGURES_DIR = REPO_ROOT / "figures"


def apply_style() -> None:
    """Thin marks, recessive grid and axes, no chart junk."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9,
        "text.color": TEXT_PRIMARY, "axes.labelcolor": TEXT_SECONDARY, "axes.titlecolor": TEXT_PRIMARY,
        "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY,
        "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.prop_cycle": plt.cycler(color=CATEGORICAL),
        "lines.linewidth": 2, "lines.markersize": 5,
        "legend.frameon": False, "legend.fontsize": 8,
    })


def save(fig, name: str, subdir: str = "") -> Path:
    """Save a figure as figures/<subdir>/<name>.png and return the path."""
    out_dir = FIGURES_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path)
    return path
