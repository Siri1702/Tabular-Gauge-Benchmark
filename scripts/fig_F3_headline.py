"""
F3 — AUC vs training-set size (§5.2)

The primary Stage-II figure: mean ROC-AUC vs N for each of the 5 Stage-II
models, aggregated across 8 datasets × 5 seeds. TabPFN-v2 should sit on
top across the entire grid; the four tuned GBDTs/MLP should converge
toward TabPFN-v2 at full data.

x-axis = N (training-set size), log scale to expose the low-N regime.
y-axis = ROC-AUC.
Error band = bootstrap 95% CI on the per-N mean (8 datasets × 5 seeds = 40
observations per (model, N)).

Source CSV: experiments/results/few_shot/multiseed_few_shot_results.csv.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_figures_style import (
    MODEL_ORDER, N_ORDER, N_TO_POS, apply_style, model_color, save,
)


def load() -> pd.DataFrame:
    df = pd.read_csv("experiments/results/few_shot/multiseed_few_shot_results.csv")
    # n_train is the actual number of training rows; for "full" this equals
    # the full training-set size. We use train_size (the N bin label) for
    # x-axis bucketing.
    return df


def per_n_mean_ci(df: pd.DataFrame, model: str, n_pairs: int = 400):
    """Bootstrap mean and 95% CI of ROC-AUC per N for one model.

    Each N bin has up to 8 datasets × 5 seeds = 40 observations.
    Bootstrap resamples at the (dataset, seed) row level — faster than the
    dataset-level hierarchical scheme and produces visually equivalent
    bands at the figure scale.
    """
    out = []
    rng = np.random.default_rng(42)
    for n in N_ORDER:
        cell = df[(df["model"] == model) & (df["train_size"] == n)]
        if cell.empty:
            out.append((n, np.nan, np.nan, np.nan))
            continue
        values = cell["roc_auc"].values
        m = float(np.nanmean(values))
        if len(values) < 2:
            out.append((n, m, np.nan, np.nan))
            continue
        idx = np.arange(len(values))
        sample_idx = rng.choice(idx, size=(n_pairs, len(values)), replace=True)
        sample = values[sample_idx].mean(axis=1)
        lo = float(np.nanpercentile(sample, 2.5))
        hi = float(np.nanpercentile(sample, 97.5))
        out.append((n, m, lo, hi))
    return pd.DataFrame(out, columns=["train_size", "mean", "ci_lo", "ci_hi"])


def main() -> Path:
    apply_style(mode="light")
    df = load()

    fig, ax = plt.subplots(figsize=(9.2, 4.6))

    # x positions on log axis — use rank-position so labels stay sensible.
    x_pos = np.array([N_TO_POS[n] for n in N_ORDER])
    # Use the actual n_train mid-bin (a numeric N) for the axis ticks so the
    # log spacing reflects real training-set size.
    n_numeric = {"25": 25, "50": 50, "100": 100, "200": 200, "300": 300,
                 "500": 500, "750": 750, "1000": 1000, "full": 1500}
    x_numeric = np.array([n_numeric[n] for n in N_ORDER])

    for model in MODEL_ORDER:
        stats = per_n_mean_ci(df, model)
        means = stats["mean"].values
        los = stats["ci_lo"].values
        his = stats["ci_hi"].values
        valid = ~np.isnan(means)
        color = model_color(model)
        ax.plot(
            x_numeric[valid], means[valid],
            marker="o", color=color, label=model,
            linewidth=2.0, markersize=7,
            markeredgecolor="white", markeredgewidth=1.0,
            zorder=3,
        )
        ax.fill_between(
            x_numeric[valid], los[valid], his[valid],
            color=color, alpha=0.15, linewidth=0,
        )

    ax.set_xscale("log")
    ax.set_xticks(x_numeric)
    ax.set_xticklabels(
        [n if n != "full" else "full" for n in N_ORDER],
        fontsize=9,
    )
    ax.minorticks_off()

    # y-axis: 0.7 .. 1.0 because TabPFN typically sits at 0.85+.
    ymin = 0.74
    ymax = 1.005
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(np.arange(0.75, 1.01, 0.05))

    ax.set_xlabel("Training-set size N (log scale; 'full' = dataset's full train split)",
                  fontsize=10)
    ax.set_ylabel("Mean ROC-AUC\n(8 datasets × 5 seeds)",
                  fontsize=10, linespacing=1.25)
    ax.grid(True, which="major", linewidth=0.5)
    ax.legend(loc="lower right", ncol=2, fontsize=9, frameon=False)

    # Annotation pointing at the right end of the TabPFN-v2 line — the
    # full-data convergence point. Reader's eye lands there because the
    # AUC gap closes visibly from N=25 to N=full.
    ax.annotate(
        "Crossover is\ndataset-dependent\n— see companion figures",
        xy=(1000, 0.93), xytext=(140, 0.79),
        fontsize=8.5, color="#52514e",
        arrowprops=dict(arrowstyle="->", color="#52514e", lw=0.6),
    )

    fig.tight_layout()
    return save(fig, "F3_headline_learning_curve")


if __name__ == "__main__":
    out = main()
    print(f"wrote {out}")
