"""
Centralised matplotlib style for professional-grade figures.

IEEE publishes in B&W but accepts colour for the digital version. The figures
in this project use a clean light theme that survives both greyscale printing
and colour display, with a deliberate accent palette that matches the model
identities across all charts.

All plot functions in `tabular_gauge/viz/` import the constants from this module so a
single change here propagates everywhere.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
# Deliberately chosen as a perceptually-uniform Catppuccin-inspired set. Each
# model has a fixed colour so the reader recognises it across figures.
MODEL_COLORS = {
    "TabPFN":    "#5b8dee",   # blue (foundation model accent)
    "SAP-RPT-1": "#a78bfa",   # violet (foundation model accent)
    "XGBoost":   "#10b981",   # green
    "LightGBM":  "#f59e0b",   # orange
    "CatBoost":  "#f43f5e",   # crimson
    "MLP":       "#06b6d4",   # cyan
}

# Fallback palette for models not in MODEL_COLORS
GENERIC_PALETTE = ["#6366f1", "#8b5cf6", "#ec4899", "#f43f5e", "#f97316", "#eab308", "#22c55e", "#06b6d4"]

# Light-theme palette (IEEE-friendly)
LIGHT_BG       = "#ffffff"
LIGHT_SURFACE  = "#f8fafc"
LIGHT_GRID     = "#e2e8f0"
TEXT_COLOR     = "#0f172a"
MUTED_TEXT     = "#64748b"
ACCENT         = "#5b8dee"
ACCENT2        = "#a78bfa"
ACCENT3        = "#10b981"
WARN           = "#f59e0b"
ERROR          = "#ef4444"

# ---------------------------------------------------------------------------
# rcParams
# ---------------------------------------------------------------------------
# DPI: 300 — standard for high-quality research publications.
DEFAULT_DPI = 300

def apply_style(theme: str = "light") -> None:
    """
    Apply a coherent style to all matplotlib output.
    """
    if theme == "light":
        bg = LIGHT_BG
        fg = TEXT_COLOR
        grid = LIGHT_GRID
    else:
        bg = "#0d0f14"
        fg = "#e2e8f0"
        grid = "#1f2937"

    rc = {
        "figure.facecolor":   bg,
        "axes.facecolor":     bg,
        "savefig.facecolor":  bg,
        "axes.edgecolor":     fg,
        "axes.labelcolor":    fg,
        "axes.titlecolor":    fg,
        "xtick.color":        fg,
        "ytick.color":        fg,
        "text.color":         fg,
        "axes.grid":          True,
        "grid.color":         grid,
        "grid.alpha":         0.5,
        "grid.linewidth":     0.6,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     1.0,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "axes.labelsize":     10,
        "axes.titlesize":     11,
        "legend.fontsize":    9,
        "legend.frameon":     False,
        "font.family":        "DejaVu Sans",
        "font.size":          10,
        "pdf.fonttype":       42,    # embed TrueType (not Type 3) for IEEE
        "ps.fonttype":        42,
        "savefig.dpi":        DEFAULT_DPI,
        "savefig.bbox":       "tight",
        "figure.dpi":         100,
    }
    mpl.rcParams.update(rc)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def color_for_model(model_name: str, index: int = 0) -> str:
    """Return the canonical colour for a model name (with cyclic fallback)."""
    if model_name in MODEL_COLORS:
        return MODEL_COLORS[model_name]
    return GENERIC_PALETTE[index % len(GENERIC_PALETTE)]

def palette(models: list) -> list:
    """Return the colour for each model in the given order."""
    return [color_for_model(m, i) for i, m in enumerate(models)]

def save_figure(fig, save_path: str, **kwargs) -> None:
    """
    Save with consistent metadata. Pass-through to plt.savefig with defaults
    for IEEE quality.
    """
    extra = kwargs.copy()
    extra.setdefault("dpi", DEFAULT_DPI)
    extra.setdefault("bbox_inches", "tight")
    extra.setdefault("metadata", {
        "Title": "Tabular-Gauge Benchmark",
        "Author": "Tabular-Gauge-Benchmark",
        "Subject": "Tabular foundation model comparison",
    })
    fig.savefig(save_path, **extra)
    plt.close(fig)
    print(f"  Saved figure -> {save_path}")

# Apply the default style on import so any module that uses matplotlib picks
# it up automatically.
apply_style("light")
