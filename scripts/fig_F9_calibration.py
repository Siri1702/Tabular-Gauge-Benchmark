"""
F9 — Calibration error vs N (§5.8)

Median ECE-10 (Expected Calibration Error, 10 bins) per (model, N)
across 8 datasets × 5 seeds. ECE-10 measures reliability of predicted
probabilities: lower is better. The paper §5.8 message is that all
five models improve their calibration as N grows, and TabPFN-v2 is
competitive raw — not the best, not the worst.

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


def per_n_median(df: pd.DataFrame, model: str) -> pd.Series:
    cell = df[df["model"] == model]
    return (
        cell.groupby("train_size")["ece_10"]
        .median()
        .reindex(N_ORDER)
    )


def main() -> Path:
    apply_style(mode="light")
    df = load()

    fig, ax = plt.subplots(figsize=(9.2, 4.6))

    n_numeric = {"25": 25, "50": 50, "100": 100, "200": 200, "300": 300,
                 "500": 500, "750": 750, "1000": 1000, "full": 1500}
    x_numeric = np.array([n_numeric[n] for n in N_ORDER])

    for model in MODEL_ORDER:
        ys = per_n_median(df, model).values
        ax.plot(
            x_numeric, ys, marker="o", color=model_color(model),
            label=model, linewidth=2.0, markersize=7,
            markeredgecolor="white", markeredgewidth=1.0,
        )

    ax.set_xscale("log")
    ax.set_xticks(x_numeric)
    ax.set_xticklabels([n if n != "full" else "full" for n in N_ORDER], fontsize=9)
    ax.minorticks_off()

    # Linear y-axis here (ECE-10 is naturally bounded and small).
    ax.set_ylim(0.0, 0.20)

    ax.set_xlabel("Training-set size N (log scale; 'full' = dataset's full train split)",
                  fontsize=10)
    ax.set_ylabel("Median ECE-10\n(lower is better)",
                  fontsize=10, linespacing=1.25)
    ax.grid(True, which="both", linewidth=0.5)
    ax.legend(loc="upper right", ncol=2, fontsize=9, frameon=False)

    fig.tight_layout()
    return save(fig, "F9_calibration_ece10_vs_n")


if __name__ == "__main__":
    out = main()
    print(f"wrote {out}")
