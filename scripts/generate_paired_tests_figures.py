"""
Generate publication-ready figures for the paired-tests appendix.

Inputs : experiments/results/aggregated/paired_stats.csv  (80 rows:
         5 overall + 75 per-dataset)
Outputs: reports/figures/paired_forest_plot.png
         reports/figures/paired_significance_heatmap.png
         reports/figures/cliffs_delta_heatmap.png

Conventions:
    All figures follow the shared light theme in tabular_gauge/viz/style.py so they
    are visually consistent with the rest of the report.

Run:
    python scripts/generate_paired_tests_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tabular_gauge.viz.style import (
    MODEL_COLORS,
    LIGHT_BG,
    LIGHT_SURFACE,
    MUTED_TEXT,
    TEXT_COLOR,
    ACCENT,
    ACCENT3,
    ERROR,
    palette,
    save_figure,
)


CSV_PATH = ROOT / "experiments" / "results" / "aggregated" / "paired_stats.csv"
FIG_DIR  = ROOT / "reports" / "figures"

# Order in which to render baselines (foundation models first, then GBMs,
# then MLP — matches the canonical model ranking used elsewhere in the
# report).
MODEL_ORDER = ["SAP-RPT-1", "LightGBM", "XGBoost", "CatBoost", "MLP"]


def load() -> pd.DataFrame:
    if not CSV_PATH.exists():
        sys.exit(f"Missing input: {CSV_PATH}. "
                 "Run: python scripts/run_paired_stats_on_benchmark.py")
    return pd.read_csv(CSV_PATH)


# --------------------------------------------------------------------------- #
# Figure A — Forest plot of per-dataset mean difference with 95% paired-bootstrap CI
# --------------------------------------------------------------------------- #
def plot_forest(df: pd.DataFrame, save_path: Path) -> None:
    """
    Per (dataset, model) row: dot = mean paired diff (TabPFN − other),
    whiskers = 95% percentile CI. Sorted by mean diff (largest TabPFN win at
    the top). One panel per model, x-axis fixed to [-1, +1].
    """
    per_ds = df[df["scope"] == "per_dataset"].copy()
    # Order the rows within each model panel by diff_mean descending
    per_ds = per_ds.sort_values("diff_mean", ascending=False)

    n_models = len(MODEL_ORDER)
    fig, axes = plt.subplots(
        n_models, 1,
        figsize=(11, 2.6 * n_models),
        sharex=True,
        facecolor=LIGHT_BG,
    )
    if n_models == 1:
        axes = [axes]

    for ax, model in zip(axes, MODEL_ORDER):
        sub = per_ds[per_ds["model"] == model]
        # Sort within this panel too
        sub = sub.sort_values("diff_mean", ascending=True)

        # Use integer y-positions because matplotlib's categorical
        # tick formatter only accepts integers on the y-axis.
        datasets = sub["dataset"].tolist()
        y_positions = list(range(len(datasets)))

        # Draw CI lines (whiskers) and mean point per dataset
        for y_pos, (_, r) in zip(y_positions, sub.iterrows()):
            color = ACCENT3 if r["diff_mean"] >= 0 else ERROR
            ax.hlines(
                y=y_pos, xmin=r["ci_low"], xmax=r["ci_high"],
                color=color, linewidth=1.4, alpha=0.9,
            )
            ax.plot(
                r["diff_mean"], y_pos,
                marker="o", color=color, markersize=6,
                markeredgecolor="white", markeredgewidth=0.6,
            )

        ax.axvline(0, color=MUTED_TEXT, linewidth=0.9, linestyle="--", alpha=0.6)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(datasets)
        ax.set_title(
            f"TabPFN vs {model}  (n = 5 seed pairs / dataset)",
            loc="left", fontsize=11, color=TEXT_COLOR,
        )
        ax.tick_params(axis="y", labelsize=8, colors=TEXT_COLOR)
        ax.tick_params(axis="x", labelsize=9, colors=TEXT_COLOR)
        ax.grid(True, axis="x", alpha=0.4)
        ax.set_xlim(-0.45, 0.45)

    axes[-1].set_xlabel(
        "Paired mean difference (TabPFN − baseline) with 95% bootstrap CI",
        fontsize=10, color=TEXT_COLOR,
    )
    fig.suptitle(
        "Per-dataset paired differences — TabPFN vs each baseline",
        fontsize=12, color=TEXT_COLOR, y=1.0,
    )
    plt.tight_layout()
    save_figure(fig, str(save_path))


# --------------------------------------------------------------------------- #
# Figure B — Holm-adjusted significance heatmap (dataset × model)
# --------------------------------------------------------------------------- #
def plot_significance_heatmap(df: pd.DataFrame, save_path: Path) -> None:
    """
    Dataset × model grid; cells show ✓ where TabPFN wins (Holm p < 0.05)
    or — otherwise. Rows are sorted by mean TabPFN diff (desc) so the
    most-favourable datasets land at the top.
    """
    per_ds = df[df["scope"] == "per_dataset"].copy()

    # pivot to (dataset × model); True where Holm p < 0.05 AND diff > 0
    pivot = (
        per_ds
        .pivot(index="dataset", columns="model", values="significant")
    )
    # Reorder columns to MODEL_ORDER
    for m in MODEL_ORDER:
        if m not in pivot.columns:
            pivot[m] = False
    pivot = pivot[MODEL_ORDER]

    # Sort rows by mean diff descending
    mean_diff = (
        per_ds
        .groupby("dataset")["diff_mean"]
        .mean()
        .reindex(pivot.index)
    )
    pivot = pivot.loc[mean_diff.sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(9, 7), facecolor=LIGHT_BG)
    ax.set_facecolor(LIGHT_BG)

    # Render as a 0/1 matrix
    matrix = pivot.astype(int).values

    import matplotlib.colors as mcolors
    cmap = mcolors.ListedColormap([LIGHT_SURFACE, ACCENT])
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right",
                       fontsize=10, color=TEXT_COLOR)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9, color=TEXT_COLOR)

    # Annotate each cell with ✓ or —
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            sig = pivot.iloc[i, j]
            sym = "✓" if bool(sig) else "—"
            color = "white" if bool(sig) else MUTED_TEXT
            ax.text(j, i, sym, ha="center", va="center",
                    color=color, fontsize=12, fontweight="bold")

    ax.set_title(
        "Per-dataset Holm-adjusted significance (TabPFN vs baseline)\n"
        "Pairing: 5 seeds; Family-wise correction: Holm-Bonferroni (α = 0.05)",
        fontsize=11, color=TEXT_COLOR,
    )
    plt.tight_layout()
    save_figure(fig, str(save_path))


# --------------------------------------------------------------------------- #
# Figure C — Cliff's δ heatmap (dataset × model)
# --------------------------------------------------------------------------- #
def plot_cliffs_delta_heatmap(df: pd.DataFrame, save_path: Path) -> None:
    """
    Dataset × model grid; cells show Cliff's δ. Diverging colour map centred
    at 0. Negative (red) = baseline beats TabPFN; positive (blue) =
    TabPFN wins.
    """
    per_ds = df[df["scope"] == "per_dataset"].copy()
    pivot = (
        per_ds
        .pivot(index="dataset", columns="model", values="cliffs_delta")
    )
    for m in MODEL_ORDER:
        if m not in pivot.columns:
            pivot[m] = np.nan
    pivot = pivot[MODEL_ORDER]
    # Sort rows by overall δ (mean across columns) so the most-TabPFN-favourable
    # datasets land at the top
    mean_delta = pivot.mean(axis=1).sort_values(ascending=False)
    pivot = pivot.loc[mean_delta.index]

    fig, ax = plt.subplots(figsize=(9, 7), facecolor=LIGHT_BG)
    ax.set_facecolor(LIGHT_BG)

    import seaborn as sns
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    im = ax.imshow(
        pivot.values, cmap=cmap, aspect="auto",
        vmin=-1.0, vmax=1.0,
    )

    # Annotate cells with formatted δ values
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iloc[i, j]
            if np.isnan(v):
                continue
            # Choose text colour based on background luminance for contrast
            color = "white" if abs(v) > 0.5 else TEXT_COLOR
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    color=color, fontsize=9, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Cliff's δ  (positive ⇒ TabPFN wins)", fontsize=9,
                   color=TEXT_COLOR)
    cbar.ax.tick_params(labelsize=8, colors=TEXT_COLOR)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right",
                       fontsize=10, color=TEXT_COLOR)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9, color=TEXT_COLOR)

    ax.set_title(
        "Per-dataset Cliff's δ effect size (TabPFN vs baseline)\n"
        "|δ|<0.147 negligible · <0.33 small · <0.474 medium · ≥0.474 large",
        fontsize=11, color=TEXT_COLOR,
    )
    plt.tight_layout()
    save_figure(fig, str(save_path))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    print("=" * 70)
    print("PAIRED-TESTS FIGURES")
    print("=" * 70)
    df = load()
    print(f"Loaded {CSV_PATH.relative_to(ROOT)}: {len(df)} rows")

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Forest plot ...")
    plot_forest(df, FIG_DIR / "paired_forest_plot.png")

    print("\n[2/3] Holm significance heatmap ...")
    plot_significance_heatmap(df, FIG_DIR / "paired_significance_heatmap.png")

    print("\n[3/3] Cliff's-delta heatmap ...")
    plot_cliffs_delta_heatmap(df, FIG_DIR / "cliffs_delta_heatmap.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
