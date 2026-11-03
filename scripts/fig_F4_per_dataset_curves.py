"""
F4 — Per-Dataset AUC learning curves (§5.3-5.4)

8 small multiples (one per dataset), 5 model curves per panel.
Identical axes across panels so the reader can compare directly.

Purpose: the headline F3 figure averages across datasets and hides
heterogeneity. F4 makes the dataset-dependent crossover behaviour
directly visible.

Source CSV: experiments/results/few_shot/multiseed_few_shot_results.csv.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_figures_style import (
    MODEL_ORDER, N_ORDER, apply_style, model_color, save,
)


def load() -> pd.DataFrame:
    return pd.read_csv("experiments/results/few_shot/multiseed_few_shot_results.csv")


def per_dataset_per_n_mean(df: pd.DataFrame, dataset: str, model: str) -> pd.Series:
    """Mean ROC-AUC per N for one (dataset, model) — averaged over 5 seeds."""
    cell = df[(df["dataset"] == dataset) & (df["model"] == model)]
    return (
        cell.groupby("train_size")["roc_auc"]
        .mean()
        .reindex(N_ORDER)
    )


def main() -> Path:
    apply_style(mode="light")
    df = load()

    datasets = sorted(df["dataset"].unique().tolist())
    n_ds = len(datasets)

    # 8 datasets in a 4x2 grid is the most readable layout.
    n_cols = 4
    n_rows = (n_ds + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13.5, 7.0), sharey=True)
    axes_flat = axes.flatten()

    # Common x-numeric mapping for log spacing.
    n_numeric = {"25": 25, "50": 50, "100": 100, "200": 200, "300": 300,
                 "500": 500, "750": 750, "1000": 1000, "full": 1500}
    x_numeric = np.array([n_numeric[n] for n in N_ORDER])

    for i, ds in enumerate(datasets):
        ax = axes_flat[i]
        for model in MODEL_ORDER:
            ys = per_dataset_per_n_mean(df, ds, model).values
            ax.plot(
                x_numeric, ys, marker="o", color=model_color(model),
                label=model, linewidth=1.6, markersize=5,
                markeredgecolor="white", markeredgewidth=0.8,
            )

        ax.set_xscale("log")
        ax.set_xticks([25, 100, 500, 1500])
        ax.set_xticklabels(["25", "100", "500", "full"], fontsize=8)
        ax.set_title(ds, loc="left", fontsize=10, fontweight="semibold")

        # One legend per panel? No — one legend for all panels at the bottom.
        # Tick formatting consistent.
        ax.tick_params(axis="both", which="both", length=3)

    # Hide any unused panel (if n_ds < n_rows*n_cols).
    for j in range(n_ds, len(axes_flat)):
        axes_flat[j].axis("off")

    # Common y-label and y-limits across panels.
    for ax in axes_flat[:n_cols]:
        ax.set_xlabel("")
    for ax in axes_flat:
        ax.set_ylim(0.55, 1.02)
    fig.supylabel("Mean ROC-AUC\n(over 5 seeds)", fontsize=10)

    # Shared legend at the bottom — one entry per model, color-coded.
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=5, fontsize=9, frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return save(fig, "F4_per_dataset_learning_curves")


if __name__ == "__main__":
    out = main()
    print(f"wrote {out}")
