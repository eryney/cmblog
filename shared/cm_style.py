"""Core Memory visual style for matplotlib plots."""

import matplotlib.pyplot as plt
import matplotlib as mpl

COLORS = {
    "purple":      "#bc36ff",
    "purple_dark": "#A33ACB",
    "purple_deep": "#b41cff",
    "bg":          "#ffffff",
    "bg_subtle":   "#f0f0f0",
    "text":        "#363737",
    "text_muted":  "#757575",
    "border":      "#dddddd",
}

# Ordered palette for multi-series plots
PALETTE = [
    "#bc36ff",  # brand purple
    "#A33ACB",  # dark purple
    "#515151",  # dark gray
    "#929292",  # mid gray
    "#b6b6b6",  # light gray
]


def apply():
    """Apply Core Memory style to all subsequent matplotlib figures."""
    mpl.rcParams.update({
        "figure.facecolor":  COLORS["bg"],
        "axes.facecolor":    COLORS["bg"],
        "axes.edgecolor":    COLORS["border"],
        "axes.labelcolor":   COLORS["text"],
        "axes.titlecolor":   COLORS["text"],
        "axes.prop_cycle":   mpl.cycler(color=PALETTE),
        "axes.grid":         True,
        "grid.color":        COLORS["bg_subtle"],
        "grid.linewidth":    1.0,
        "xtick.color":       COLORS["text_muted"],
        "ytick.color":       COLORS["text_muted"],
        "text.color":        COLORS["text"],
        "font.family":       "sans-serif",
        "figure.dpi":        150,
        "savefig.dpi":       150,
        "savefig.bbox":      "tight",
        "savefig.facecolor": COLORS["bg"],
    })
