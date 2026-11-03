"""
F1 — Stage-I Overall Model Comparison (§4)

Multi-panel figure showing the 6 Stage-I models across the headline metrics:
ROC-AUC, PR-AUC, log-loss (lower is better), and balanced accuracy.
Purpose: establish the Stage-I model landscape and visually justify why
TabPFN-v2 is the TFM selected for Stage II.

Source CSV: experiments/results/aggregated/results.csv (Stage I, 75 cells =
15 datasets x 5 seeds). Aggregated per-model mean +/- SE across the 75 cells.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_figures_style import (
    STAGE_I_MODEL_ORDER, apply_style, model_color, save,
)


def load_stage1_results() -> pd.DataFrame:
    """Load Stage-I per-cell results."""
    path = Path("experiments/results/aggregated/results.csv")
    df = pd.read_csv(path)
    return df


def aggregate(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Aggregate to mean +/- SE per model, ordered to match STAGE_I_MODEL_ORDER."""
    grouped = (
        df.groupby("model")[metric]
        .agg(["mean", "sem", "count"])
        .reindex(STAGE_I_MODEL_ORDER)
    )
    return grouped


def main() -> Path:
    apply_style(mode="light")

    df = load_stage1_results()
    metrics = [
        ("roc_auc", "ROC-AUC", "(higher is better)"),
        ("pr_auc", "PR-AUC", "(higher is better)"),
        ("log_loss_val", "Log-loss", "(lower is better)"),
        ("balanced_accuracy", "Balanced accuracy", "(higher is better)"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.4), sharey=False)

    for ax, (metric, title, annotation) in zip(axes, metrics):
        agg = aggregate(df, metric)
        xs = np.arange(len(agg))
        means = agg["mean"].values
        sems = agg["sem"].values
        colors = [model_color(m) for m in agg.index]

        ax.errorbar(
            xs, means, yerr=sems,
            fmt="none", ecolor="#52514e", elinewidth=0.6, capsize=3,
            zorder=2,
        )
        ax.scatter(
            xs, means, s=70, c=colors,
            edgecolors="white", linewidths=1.2, zorder=3,
        )

        # Bold TabPFN marker to make the protagonist visually obvious.
        tab_idx = STAGE_I_MODEL_ORDER.index("TabPFN")
        ax.scatter(
            [xs[tab_idx]], [means[tab_idx]],
            s=160, marker="*" if metric != "log_loss_val" else "*",
            facecolors=model_color("TabPFN"),
            edgecolors="#0b0b0b", linewidths=1.0, zorder=4,
        )

        # Compact y-range
        lo, hi = np.nanmin(means - sems), np.nanmax(means + sems)
        pad = (hi - lo) * 0.30
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(-0.6, len(agg) - 0.4)

        ax.set_xticks(xs)
        ax.set_xticklabels(agg.index, rotation=30, ha="right")
        ax.set_title(f"{title}\n{annotation}", loc="left", fontsize=10)
        ax.grid(True, axis="y")

    fig.tight_layout()

    return save(fig, "F1_stage1_overall_model_comparison")


if __name__ == "__main__":
    out = main()
    print(f"wrote {out}")
