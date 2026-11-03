"""
Shared style module for the paper1/v2 publication figures (F1-F10).

Implements the dataviz skill's reference palette (light + dark surfaces)
plus mark specs, typography, and helper functions for publication-grade
matplotlib output. All F1-F10 figures import from this module so the
palette, ordering, and marks are consistent across the paper.

Reference: C:\\Users\\siri\\AppData\\Local\\Temp\\claude\\bundled-skills\\
2.1.241\\271f8a2ff527d30c9223470c17483c93\\dataviz\\references\\palette.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# -----------------------------------------------------------------------------
# 1. Palette — dataviz reference default (light + dark), brand-neutral.
# -----------------------------------------------------------------------------


# Categorical slots (1..N). Use as `CATEGORICAL[i-1]` for the i-th series.
# Order is the CVD-safety mechanism — never re-order on a per-chart basis.
CATEGORICAL_LIGHT = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]
CATEGORICAL_DARK = [
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#008300",
    "#9085e9",
    "#e66767",
]


# Sequential blue ramp (light surface)
SEQUENTIAL_BLUE = {
    100: "#cde2fb", 150: "#b7d3f6", 200: "#9ec5f4", 250: "#86b6ef",
    300: "#6da7ec", 350: "#5598e7", 400: "#3987e5", 450: "#2a78d6",
    500: "#256abf", 550: "#1c5cab", 600: "#184f95", 650: "#104281",
    700: "#0d366b",
}


# Diverging blue<->red, midpoint neutral
DIVERGING = ["#2a78d6", "#86b6ef", "#f0efec", "#e87ba4", "#e34948"]


# Status palette
STATUS = {"good": "#0ca30c", "warning": "#fab219",
          "serious": "#ec835a", "critical": "#d03b3b"}


# Surfaces and ink (light mode)
SURFACE_LIGHT = "#fcfcfb"
SURFACE_DARK = "#1a1a19"
INK_PRIMARY_LIGHT = "#0b0b0b"
INK_SECONDARY_LIGHT = "#52514e"
INK_MUTED = "#898781"
GRID_HAIRLINE_LIGHT = "#e1e0d9"
AXIS_BASELINE_LIGHT = "#c3c2b7"
DELTA_GOOD = "#006300"


# Model-specific slot assignments (fixed across all figures in the paper)
# TabPFN-v2 is the protagonist of the paper: slot 1 (blue).
# Baselines get slots 2-5. Tuned GBDTs are deliberately visually grouped
# but distinguished by their assigned slot.
MODEL_SLOT = {
    "TabPFN": 1,    # blue   — protagonist
    "XGBoost": 2,   # orange — most common tuned GBDT comparator
    "LightGBM": 3,  # aqua
    "CatBoost": 4,  # yellow
    "MLP": 5,       # magenta
    "SAP-RPT-1": 7, # violet — Stage-I comparator
}


def model_color(model: str, mode: str = "light") -> str:
    """Stable per-model color from MODEL_SLOT."""
    slot = MODEL_SLOT.get(model, 6)
    palette = CATEGORICAL_LIGHT if mode == "light" else CATEGORICAL_DARK
    return palette[slot - 1]


# -----------------------------------------------------------------------------
# 2. Typography — system sans, tabular figures where alignment matters.
# -----------------------------------------------------------------------------

# Matplotlib falls back to DejaVu Sans when system-ui is unavailable, so we
# declare both: the browser/Word stack as the brand intent, and DejaVu Sans
# as the always-available render-time fallback.
FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", "DejaVu Sans", sans-serif'

# -----------------------------------------------------------------------------
# 3. Mark specs — per dataviz/marks-and-anatomy.md
#    Thin marks, 4px rounded data-ends, 2px lines, ≥8px markers.
# -----------------------------------------------------------------------------

LINE_WIDTH = 2.0          # 2px lines
MARKER_SIZE = 8           # ≥8px
MARKER_EDGE_WIDTH = 1.0
CAPSIZE = 3               # error-bar caps (pt)
GRID_LINE_WIDTH = 0.5     # hairline grid
AXIS_LINE_WIDTH = 1.0
ELBOW_WIDTH = 0.6         # error-bar line width


# -----------------------------------------------------------------------------
# 4. Default rcParams — applied by apply_style().
# -----------------------------------------------------------------------------


def apply_style(mode: str = "light") -> None:
    """Apply the dataviz defaults as a matplotlib rcParams dict.

    Mode 'light' is the publication default (white-ish surface).
    Mode 'dark' is included for the artifact viewer; we don't ship dark
    figures in the PDF submission, but the helper exists so the same
    module can be reused for the artifact build.
    """
    bg = SURFACE_LIGHT if mode == "light" else SURFACE_DARK
    ink = INK_PRIMARY_LIGHT if mode == "light" else "#ffffff"
    ink2 = INK_SECONDARY_LIGHT if mode == "light" else "#c3c2b7"
    grid = GRID_HAIRLINE_LIGHT if mode == "light" else "#2c2c2a"

    mpl.rcParams.update({
        # Typography
        "font.family": FONT_FAMILY,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10,
        "axes.labelweight": "regular",
        "axes.labelcolor": ink2,
        "axes.titlecolor": ink,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.color": ink2,
        "ytick.color": ink2,
        "legend.fontsize": 9,
        "legend.title_fontsize": 9,
        "legend.frameon": False,
        # Mark specs
        "lines.linewidth": LINE_WIDTH,
        "lines.markersize": MARKER_SIZE,
        "lines.markeredgewidth": MARKER_EDGE_WIDTH,
        "errorbar.capsize": CAPSIZE,
        # Axes chrome — recessive grid/axes per dataviz
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": ink2,
        "axes.linewidth": AXIS_LINE_WIDTH,
        "axes.facecolor": bg,
        "figure.facecolor": bg,
        "figure.edgecolor": bg,
        "savefig.facecolor": bg,
        "savefig.edgecolor": bg,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "axes.grid": True,
        "grid.color": grid,
        "grid.linewidth": GRID_LINE_WIDTH,
        "grid.alpha": 1.0,
        "axes.axisbelow": True,
    })


# -----------------------------------------------------------------------------
# 5. Save helpers — single entry point for all F1-F10 figures.
# -----------------------------------------------------------------------------


PAPER_FIG_DIR = Path(__file__).resolve().parents[1] / "paper1" / "v2" / "figures"


def save(fig: mpl.figure.Figure, name: str, dpi: int = 200) -> Path:
    """Save figure to paper1/v2/figures/<name>.png at publication DPI.

    PNG only (no PDF) per user request. dpi=200 is a reasonable publication
    threshold — print-quality without bloating the manuscript bundle.
    """
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = PAPER_FIG_DIR / f"{name}.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


# -----------------------------------------------------------------------------
# 6. Convenience — sequential colormap for magnitude jobs (heatmaps etc).
# -----------------------------------------------------------------------------


def sequential_cmap(name: str = "blue_ramp") -> LinearSegmentedColormap:
    """A monotone blue sequential colormap from light to dark.

    Used by the heatmap figures (F1, paired_significance) where the
    magnitude is the message.
    """
    if name == "blue_ramp":
        stops = [SEQUENTIAL_BLUE[100], SEQUENTIAL_BLUE[200], SEQUENTIAL_BLUE[300],
                 SEQUENTIAL_BLUE[400], SEQUENTIAL_BLUE[500], SEQUENTIAL_BLUE[600],
                 SEQUENTIAL_BLUE[700]]
    else:
        raise ValueError(f"unknown cmap: {name}")
    return LinearSegmentedColormap.from_list(name, stops)


def diverging_cmap(name: str = "blue_red_div") -> LinearSegmentedColormap:
    """A symmetric blue<->red diverging colormap with neutral midpoint."""
    return LinearSegmentedColormap.from_list(name, DIVERGING)


# -----------------------------------------------------------------------------
# 7. Legend helpers — keep the legend box off the plot when possible.
# -----------------------------------------------------------------------------


def legend_outside(ax: mpl.axes.Axes, **kwargs) -> mpl.legend.Legend:
    """Place a legend outside the right spine."""
    return ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), **kwargs)


# -----------------------------------------------------------------------------
# 8. Standard ordering — must match paper1/v2/tables/T*.md for cross-reference.
# -----------------------------------------------------------------------------

N_ORDER = ["25", "50", "100", "200", "300", "500", "750", "1000", "full"]
N_TO_POS = {n: i for i, n in enumerate(N_ORDER)}

MODEL_ORDER = ["TabPFN", "XGBoost", "LightGBM", "CatBoost", "MLP"]
STAGE_I_MODEL_ORDER = ["TabPFN", "SAP-RPT-1", "XGBoost", "LightGBM", "CatBoost", "MLP"]
