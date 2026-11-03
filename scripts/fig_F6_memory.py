"""
F6 — Peak memory vs N (§5.5)

Median peak_memory_mb per (model, N), aggregated across 8 datasets × 5
seeds. y-axis on log scale because TabPFN-v2 memory grows with N while
the tuned GBDTs/MLP are essentially flat — a log y-axis lets both
regimes fit on one canvas without one regime flattening the other.

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
        cell.groupby("train_size")["peak_memory_mb"]
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
    ax.set_yscale("log")

    ax.set_xlabel("Training-set size N (log scale; 'full' = dataset's full train split)",
                  fontsize=10)
    ax.set_ylabel("Median peak\nmemory (MB)",
                  fontsize=10, linespacing=1.25)
    ax.grid(True, which="both", linewidth=0.5)
    ax.legend(loc="lower right", ncol=2, fontsize=9, frameon=False)

    # Annotation: TabPFN-v2 at the right end of the N axis, with the
    # numerical range so the reader can read off "≈5× to 25× the cost of
    # a tuned GBDT at the deployment N."
    tab_y_end = per_n_median(df, "TabPFN").iloc[-1]
    base_y_end = min(per_n_median(df, m).iloc[-1] for m in MODEL_ORDER if m != "TabPFN")
    ratio = tab_y_end / max(base_y_end, 1.0)
    ax.annotate(
        f"At N=full:\nTabPFN-v2 ≈ {ratio:.0f}× the\nsmallest baseline",
        xy=(1500, tab_y_end), xytext=(60, tab_y_end * 0.45),
        fontsize=8.5, color="#52514e",
        arrowprops=dict(arrowstyle="->", color="#52514e", lw=0.6),
    )

    fig.tight_layout()
    return save(fig, "F6_peak_memory_vs_n")


if __name__ == "__main__":
    out = main()
    print(f"wrote {out}")
