"""Shared Matplotlib styling (neutral chart chrome + colour-blind-safe series colours)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend: figures are written to PNG files

SURFACE = "#fcfcfb"
INK = "#0b0b0b"        # primary ink: ground truth / titles
INK_2 = "#52514e"      # secondary ink: axis labels, legend text
MUTED = "#898781"      # tick labels
GRID = "#e1e0d9"       # hairline grid
AXIS = "#c3c2b7"       # baseline / axis line
BLUE = "#2a78d6"       # categorical slot 1
ORANGE = "#eb6834"     # categorical slot 2

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Segoe UI", "Helvetica Neue", "DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
})


def style_axes(ax) -> None:
    """Recessive chrome: horizontal hairline grid only, no top/right spines."""
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=0, labelsize=9)
