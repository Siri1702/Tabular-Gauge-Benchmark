"""
F5 — Crossover N* by dataset (§5.4)

Forest-plot style figure showing the estimated crossover N* — the
training-set size at which the best tuned GBDT matches TabPFN-v2 — for
each of the 8 Stage-II datasets. 95% bootstrap CI on N* is shown as a
horizontal line. Datasets where TabPFN-v2 leads across the entire
measured N grid are right-censored at the dataset's full training size
(arrow head pointing right).

Source CSV: experiments/results/few_shot/per_dataset_crossover_with_ci.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_figures_style import (
    STATUS, apply_style, model_color, save,
)


def load() -> pd.DataFrame:
    return pd.read_csv("experiments/results/few_shot/per_dataset_crossover_with_ci.csv")


def main() -> Path:
    apply_style(mode="light")
    df = load()

    # Sort by lower CI bound (most-likely-crossover on left, censored on right).
    df = df.sort_values("crossover_ci_lo", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9.5, 4.6))

    y_pos = np.arange(len(df))

    # Full-train anchor (right margin): we use the largest n_train in the
    # 1800-cell grid as the "full" x position.
    full_anchor = 1500

    # Determine "censored" datasets: those where the CSV's crossover_n_point
    # is empty (NaN) — meaning no crossover point was observed in the
    # measured grid. Per CSV inspection, four datasets are censored:
    # titanic, kc1, ozone-level, spambase.
    censored = df["crossover_n_point"].isna()

    n_lo = df["crossover_ci_lo"].astype(float).values
    n_hi = df["crossover_ci_hi"].astype(float).values
    n_pt = df["crossover_n_point"].astype(float).values

    # Plot CI lines (uncensored vs censored differ in style).
    for i, (lo, hi, pt) in enumerate(zip(n_lo, n_hi, n_pt)):
        if censored.iloc[i]:
            # Right-censored: line stops at full_anchor with an arrow.
            ax.errorbar(
                pt, i, xerr=[[0], [0]], fmt="o",
                color=STATUS["serious"], markersize=8,
                markeredgecolor="#0b0b0b", markeredgewidth=0.8, zorder=3,
            )
            # arrow line from ci_lo to full_anchor
            ax.annotate(
                "", xy=(full_anchor, i), xytext=(max(lo, 25), i),
                arrowprops=dict(arrowstyle="-|>", color=STATUS["serious"],
                                lw=1.4, mutation_scale=12),
            )
        else:
            err_lo = max(pt - lo, 1)
            err_hi = max(hi - pt, 1)
            ax.errorbar(
                pt, i, xerr=[[err_lo], [err_hi]], fmt="o",
                color=model_color("TabPFN"), ecolor=model_color("TabPFN"),
                markersize=8, markeredgecolor="#0b0b0b", markeredgewidth=0.8,
                elinewidth=1.4, capsize=3, zorder=3,
            )

    # X-axis: log, with tick marks at the N grid + full anchor.
    ax.set_xscale("log")
    ticks = [25, 50, 100, 200, 500, 1000, full_anchor]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["25", "50", "100", "200", "500", "1000", "full"],
                       fontsize=9)
    ax.set_xlim(20, full_anchor * 1.58)

    # Add a vertical reference line at N=500 — a plausible practitioner
    # threshold — to make the cost-of-being-wrong point.
    ax.axvline(500, color="#898781", linestyle=":", linewidth=1.0,
               alpha=0.7, zorder=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["dataset"].values, fontsize=9)
    ax.set_xlabel("Crossover N* (training-set size) — 95% bootstrap CI",
                  fontsize=10)
    ax.grid(True, axis="x", which="both", linewidth=0.5)

    # Manual legend: censored vs uncensored.
    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], color=model_color("TabPFN"), marker="o", linestyle="",
               markersize=8, markeredgecolor="#0b0b0b", markeredgewidth=0.8,
               label="Crossover observed within grid"),
        Line2D([0], [0], color=STATUS["serious"], marker=">",
               markersize=9, markeredgecolor="#0b0b0b", markeredgewidth=0.8,
               label="Crossover right-censored (≥ full)"),
    ]
    ax.legend(handles=legend_items, loc="lower right", frameon=True,
              framealpha=0.95, fontsize=9)

    fig.tight_layout()
    return save(fig, "F5_crossover_n_star")


if __name__ == "__main__":
    out = main()
    print(f"wrote {out}")
