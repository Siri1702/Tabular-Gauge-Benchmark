"""
F7 — Inference time vs N (§5.6)

Per-row inference time (ms/row) per (model, N), aggregated across 8
datasets × 5 seeds. y-axis on log scale: TabPFN-v2 grows with N
(~7× from N=25 to N=full); tuned GBDTs/MLP are essentially constant
at sub-millisecond per-row.

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
        cell.groupby("train_size")["ms_per_test_row"]
        .median()
        .reindex(N_ORDER)
    )


def main() -> Path:
    apply_style(mode="light")
    df = load()

    # Wider canvas so the legend sits outside the data area on the right
    # (TabPFN-v2's curve owns the upper-right corner; placing the legend
    # over the lower-left cluster collides with XGBoost/LightGBM).
    fig, ax = plt.subplots(figsize=(10.0, 4.6))

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
    ax.set_ylabel("Median per-row\ninference time (ms / row)",
                  fontsize=10, linespacing=1.25)
    ax.grid(True, which="both", linewidth=0.5)

    # Legend outside the right spine — keeps the data area uncluttered.
    from paper_figures_style import legend_outside
    legend_outside(ax, ncol=1, fontsize=9)

    # Annotation: ratio of TabPFN-v2's ms/row at N=full vs N=25.
    tab_y_start = per_n_median(df, "TabPFN").iloc[0]
    tab_y_end = per_n_median(df, "TabPFN").iloc[-1]
    ratio = tab_y_end / max(tab_y_start, 1e-9)
    ax.annotate(
        f"TabPFN-v2 grows\n≈{ratio:.0f}× from N=25 to N=full",
        xy=(1500, tab_y_end), xytext=(55, tab_y_end * 0.18),
        fontsize=8.5, color="#52514e",
        arrowprops=dict(arrowstyle="->", color="#52514e", lw=0.6),
    )

    fig.tight_layout()
    return save(fig, "F7_inference_time_vs_n")


if __name__ == "__main__":
    out = main()
    print(f"wrote {out}")
