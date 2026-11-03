"""
F8 — AUC × inference-cost Pareto frontier (§5.7)

Scatter: x = mean ms/row, y = mean ROC-AUC for each (model × N) cell,
aggregated across 8 datasets × 5 seeds.

Pareto-optimal points (no other point dominates them on both axes) are
highlighted with a star; non-Pareto points are faded. The Pareto
frontier is then drawn as a step line connecting the optimal points.

Purpose: synthesize the AUC-vs-cost trade-off into a single picture
without collapsing it into one metric.

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


def per_cell_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ROC-AUC and ms/row per (model, train_size)."""
    g = (
        df.groupby(["model", "train_size"])
        [["roc_auc", "ms_per_test_row"]]
        .mean()
        .reset_index()
    )
    # Re-order rows by N for cleaner output tables.
    g["n_rank"] = g["train_size"].map({n: i for i, n in enumerate(N_ORDER)})
    g = g.sort_values(["model", "n_rank"]).reset_index(drop=True)
    return g


def pareto_mask(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Return a boolean mask of Pareto-optimal points (high y, low x)."""
    n = len(xs)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            # j dominates i if j.x <= i.x AND j.y >= i.y with at least one strict.
            if xs[j] <= xs[i] and ys[j] >= ys[i] and (
                xs[j] < xs[i] or ys[j] > ys[i]
            ):
                is_pareto[i] = False
                break
    return is_pareto


def main() -> Path:
    apply_style(mode="light")
    df = load()
    cells = per_cell_mean(df)

    fig, ax = plt.subplots(figsize=(9.2, 5.6))

    # Plot every cell as a faint point first (one colour per model), so the
    # reader sees the whole grid. Pareto points then sit on top in bold.
    n_label = {"25": "N=25", "50": "N=50", "100": "N=100", "200": "N=200",
               "300": "N=300", "500": "N=500", "750": "N=750",
               "1000": "N=1000", "full": "N=full"}

    # All cells, faded.
    for _, row in cells.iterrows():
        ax.scatter(
            row["ms_per_test_row"], row["roc_auc"],
            s=22, color=model_color(row["model"]),
            alpha=0.30, edgecolors="none", zorder=2,
        )

    # Pareto mask over the whole (cells) table.
    xs = cells["ms_per_test_row"].values
    ys = cells["roc_auc"].values
    p_mask = pareto_mask(xs, ys)

    # Highlight Pareto points with a star and label the N value.
    # The four Pareto-optimal baseline points cluster tightly on the
    # left (X ≈ 10^-2 ms/row). Use a leader-line strategy: place the
    # labels at four distinct X positions further to the right, with
    # short arrows back to the actual stars.
    pareto_indices = list(np.where(p_mask)[0])
    sorted_pareto = sorted(pareto_indices, key=lambda i: cells.iloc[i]["ms_per_test_row"])

    # Four label "lanes" — distinct X positions just above each star.
    lane_xs = [0.025, 0.040, 0.060, 0.085]  # ms/row positions on log axis

    for rank, i in enumerate(sorted_pareto):
        row = cells.iloc[i]
        ax.scatter(
            row["ms_per_test_row"], row["roc_auc"],
            s=180, marker="*",
            color=model_color(row["model"]),
            edgecolors="#0b0b0b", linewidths=0.8, zorder=4,
        )
        # Skip label-lane for the right-most star (TabPFN at full) — it
        # already sits on the right with a clean label.
        if rank == len(sorted_pareto) - 1 and row["model"] == "TabPFN":
            ax.annotate(
                n_label[row["train_size"]] + f"\n({row['model']})",
                xy=(row["ms_per_test_row"], row["roc_auc"]),
                xytext=(row["ms_per_test_row"] * 0.55, row["roc_auc"] + 0.012),
                fontsize=7.5, color="#52514e",
                ha="left", va="center",
            )
            continue
        if rank < len(lane_xs):
            lx = lane_xs[rank]
        else:
            lx = lane_xs[-1] * 1.5
        # Stagger Y to avoid label-label collision.
        y_off = 0.018 if rank % 2 == 0 else -0.018
        ly = row["roc_auc"] + y_off
        ax.annotate(
            n_label[row["train_size"]] + f"\n({row['model']})",
            xy=(row["ms_per_test_row"], row["roc_auc"]),
            xytext=(lx, ly),
            fontsize=7.5, color="#52514e",
            ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color="#898781",
                            lw=0.5, alpha=0.7),
        )

    # Draw the Pareto frontier as a step line through the optimal points.
    pareto_pts = cells[p_mask].sort_values("ms_per_test_row")
    ax.step(
        pareto_pts["ms_per_test_row"], pareto_pts["roc_auc"],
        where="post", color="#0b0b0b", linewidth=1.0, alpha=0.55,
        linestyle="--", zorder=3,
    )

    ax.set_xscale("log")
    ax.set_xlabel("Mean inference time (ms/row) — log scale", fontsize=10)
    ax.set_ylabel("Mean ROC-AUC\n(8 datasets × 5 seeds)", fontsize=10, linespacing=1.25)
    ax.grid(True, which="both", linewidth=0.5)

    # Custom legend keyed on model colour.
    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], marker="o", color=model_color(m), linestyle="",
               markersize=8, label=m)
        for m in MODEL_ORDER
    ]
    legend_items.append(
        Line2D([0], [0], marker="*", color="#0b0b0b", linestyle="",
               markersize=14, markeredgewidth=0.8, label="Pareto-optimal")
    )
    ax.legend(handles=legend_items, loc="lower right", fontsize=9,
              frameon=True, framealpha=0.95)

    fig.tight_layout()
    return save(fig, "F8_pareto_auc_vs_inference")


if __name__ == "__main__":
    out = main()
    print(f"wrote {out}")
