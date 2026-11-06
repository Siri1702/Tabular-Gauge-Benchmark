"""
Generate the main leaderboard visualisation:
- Heatmap of model x dataset AUC scores
- Summary bar chart of average ranks
- Win/loss/tie matrix

All figures use the shared light theme from :mod:`src.viz.style` so they
survive IEEE's print path and look consistent on screen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .style import (
    MUTED_TEXT,
    TEXT_COLOR,
    LIGHT_BG,
    LIGHT_SURFACE,
    palette,
    save_figure,
)
from ..evaluation.analysis import get_model_order, PREFERRED_ORDER

def plot_auc_heatmap(
    results_df: pd.DataFrame,
    metric: str = "roc_auc",
    save_path: str = "reports/figures/auc_heatmap.png",
    figsize: tuple = (12, 7),
):
    """
    Heatmap of mean AUC per (dataset, model) pair.
    """
    if results_df.empty:
        print("  Skipping auc_heatmap: results_df is empty")
        return None

    pivot = (
        results_df
        .groupby(["dataset", "model"])[metric]
        .mean()
        .unstack("model")
    )

    # Dynamic ordering based on results
    model_order = get_model_order(results_df, preferred_order=PREFERRED_ORDER)
    for m in model_order:
        if m not in pivot.columns:
            pivot[m] = np.nan
    pivot = pivot[model_order]

    # Sort datasets by TabPFN performance (descending) if it exists
    if "TabPFN" in pivot.columns:
        pivot = pivot.sort_values("TabPFN", ascending=False)

    if pivot.empty:
        print("  Skipping auc_heatmap: pivot is empty")
        return None

    fig, ax = plt.subplots(figsize=figsize, facecolor=LIGHT_BG)
    ax.set_facecolor(LIGHT_BG)

    cmap = sns.color_palette("viridis", as_cmap=True)

    # Fixed scale for direct comparison
    im = ax.imshow(
        pivot.values,
        cmap=cmap,
        aspect="auto",
        vmin=0.5, vmax=1.0,
    )

    # Annotate cells
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.isna(val):
                continue
            text_color = "white" if val < 0.78 else TEXT_COLOR
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=8, color=text_color, fontweight="600")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, color=TEXT_COLOR, fontsize=10, fontweight="600")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, color=MUTED_TEXT, fontsize=9)

    # Highlight TabPFN column
    if "TabPFN" in pivot.columns:
        tabpfn_col = list(pivot.columns).index("TabPFN")
        ax.axvline(tabpfn_col - 0.5, color="#5b8dee", lw=1.5, alpha=0.9)
        ax.axvline(tabpfn_col + 0.5, color="#5b8dee", lw=1.5, alpha=0.9)

    ax.set_title(
        "Model x Dataset ROC-AUC Heatmap",
        color=TEXT_COLOR, fontsize=12, pad=16, fontweight="600",
    )
    ax.set_xlabel("Model (sorted left to right by preference)", color=MUTED_TEXT, fontsize=10)
    ax.set_ylabel("Dataset (sorted top to bottom by TabPFN AUC)", color=MUTED_TEXT, fontsize=10)

    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.ax.yaxis.set_tick_params(color=MUTED_TEXT)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=MUTED_TEXT, fontsize=8)
    cb.set_label("ROC-AUC", color=MUTED_TEXT, fontsize=9)

    save_figure(fig, save_path)
    return fig


def plot_average_ranks(
    rank_df: pd.DataFrame,
    save_path: str = "reports/figures/average_ranks.png",
):
    """
    Horizontal bar chart of average ranks.
    """
    if rank_df.empty:
        print("  Skipping average_ranks: rank_df is empty")
        return None

    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=LIGHT_BG)
    ax.set_facecolor(LIGHT_BG)

    models = rank_df["model"].tolist()
    ranks  = rank_df["average_rank"].tolist()
    colors = palette(models)

    bars = ax.barh(models, ranks, color=colors, height=0.55, alpha=0.95)

    for bar, rank in zip(bars, ranks):
        ax.text(rank + 0.04, bar.get_y() + bar.get_height() / 2,
                f"{rank:.2f}", va="center", color=TEXT_COLOR, fontsize=10)

    ax.set_xlabel("Average Rank (1 = best across all datasets)",
                  color=MUTED_TEXT, fontsize=10)
    ax.set_title("Average Rank Across All Datasets",
                 color=TEXT_COLOR, fontsize=12, fontweight="600")
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(MUTED_TEXT)
    ax.invert_yaxis()

    if ranks:
        ax.set_xlim(0, max(ranks) * 1.18)

    save_figure(fig, save_path)
    return fig


def plot_timing_comparison(
    results_df: pd.DataFrame,
    save_path: str = "reports/figures/timing_comparison.png",
):
    """
    Two-panel timing chart.
    """
    if results_df.empty:
        print("  Skipping timing_comparison: results_df is empty")
        return None

    model_order = get_model_order(results_df, preferred_order=PREFERRED_ORDER)
    timing = (
        results_df
        .groupby("model")[["tuning_time_sec", "fit_time_sec", "predict_time_sec"]]
        .mean()
        .reindex(model_order)
        .dropna(how="all")
    )

    if timing.empty:
        print("  Skipping timing_comparison: no timing data found")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), facecolor=LIGHT_BG)

    # Left: stacked bar
    ax = axes[0]
    ax.set_facecolor(LIGHT_BG)
    bottom = np.zeros(len(timing))
    colors_map = {
        "tuning_time_sec":   "#cbd5e1",
        "fit_time_sec":      "#a78bfa",
        "predict_time_sec":  "#5b8dee",
    }
    labels_map = {
        "tuning_time_sec":   "Hyperparameter tuning",
        "fit_time_sec":      "Model fit",
        "predict_time_sec":  "Inference",
    }
    for col in ["tuning_time_sec", "fit_time_sec", "predict_time_sec"]:
        if col not in timing.columns:
            continue
        vals = timing[col].fillna(0).values
        ax.bar(timing.index, vals, bottom=bottom,
               label=labels_map[col], color=colors_map[col], alpha=0.95,
               edgecolor=LIGHT_BG, linewidth=0.5)
        bottom += vals

    for i, (idx, row) in enumerate(timing.iterrows()):
        total = row.fillna(0).sum()
        ax.text(i, total + 2, f"{total:.1f}s", ha="center", va="bottom",
                color=TEXT_COLOR, fontsize=10, fontweight="bold")

    ax.set_title("Total Wall-Clock Time per Model\n(tuning + fit + inference, mean across all (dataset, seed))",
                 color=TEXT_COLOR, fontsize=11, fontweight="600")
    ax.set_ylabel("Seconds", color=MUTED_TEXT)
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(MUTED_TEXT)

    max_total = timing.fillna(0).sum(axis=1).max()
    if max_total > 0:
        ax.set_ylim(0, max_total * 1.18)
    ax.legend(facecolor=LIGHT_SURFACE, labelcolor=TEXT_COLOR, fontsize=8, loc='upper left')

    # Right: throughput
    ax2 = axes[1]
    ax2.set_facecolor(LIGHT_BG)
    if "throughput_rows_per_sec" in results_df.columns:
        throughput = (
            results_df.groupby("model")["throughput_rows_per_sec"].mean()
            .reindex(model_order).dropna().sort_values(ascending=True)
        )
        if not throughput.empty:
            colors_t = palette(throughput.index.tolist())
            bars = ax2.bar(range(len(throughput)), throughput.values,
                           color=colors_t, alpha=0.95, edgecolor=LIGHT_BG, linewidth=0.5)
            ax2.set_xticks(range(len(throughput)))
            ax2.set_xticklabels(throughput.index, rotation=30, ha='right', color=MUTED_TEXT)
            ax2.set_yscale('log')
            ax2.set_title("Inference Throughput (rows / second, log scale)",
                          color=TEXT_COLOR, fontsize=11, fontweight="600")
            ax2.set_ylabel("Rows/sec (log scale)", color=MUTED_TEXT)
            ax2.tick_params(colors=MUTED_TEXT)
            for spine in ax2.spines.values():
                spine.set_color(MUTED_TEXT)
            for bar, val in zip(bars, throughput.values):
                ax2.text(bar.get_x() + bar.get_width() / 2, val * 1.15,
                         f"{val:.0f}", ha="center", va="bottom", color=TEXT_COLOR, fontsize=8)
        else:
            ax2.text(0.5, 0.5, "No throughput data", transform=ax2.transAxes, ha="center", va="center", color=MUTED_TEXT)
    else:
        ax2.text(0.5, 0.5, "Throughput not tracked", transform=ax2.transAxes, ha="center", va="center", color=MUTED_TEXT)

    save_figure(fig, save_path)
    return fig


def plot_memory_comparison(
    results_df: pd.DataFrame,
    save_path: str = "reports/figures/memory_comparison.png",
):
    """
    Two-panel memory chart.
    """
    if results_df.empty or "peak_memory_mb" not in results_df.columns:
        print("  No memory data available, skipping memory visualization")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=LIGHT_BG)

    # Use dynamic model ordering
    model_order = get_model_order(results_df, preferred_order=PREFERRED_ORDER)

    # Left: peak memory usage
    ax = axes[0]
    ax.set_facecolor(LIGHT_BG)
    memory = (
        results_df.groupby("model")["peak_memory_mb"]
        .mean()
        .reindex(model_order)
        .dropna()
        .sort_values(ascending=True)
    )
    colors = palette(memory.index.tolist())
    bars = ax.barh(memory.index, memory.values, color=colors, alpha=0.95,
                   edgecolor=LIGHT_BG, linewidth=0.5)
    for bar, val in zip(bars, memory.values):
        # Place text in the center of the bar for high visibility
        ax.text(val / 2,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.0f} MB", va="center", ha="center", color="#1e293b", fontsize=10, fontweight="600")
    ax.set_xlabel("Peak Memory Usage (MB)", color=MUTED_TEXT, fontsize=10)
    ax.set_title("Peak Memory Usage", color=TEXT_COLOR, fontsize=12, fontweight="600")
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(MUTED_TEXT)

    # Right: memory delta
    ax2 = axes[1]
    ax2.set_facecolor(LIGHT_BG)
    if "memory_delta_mb" in results_df.columns:
        memory_delta = (
            results_df.groupby("model")["memory_delta_mb"].mean()
            .reindex(model_order)
            .dropna()
            .sort_values(ascending=True)
        )
        colors_d = palette(memory_delta.index.tolist())
        bars2 = ax2.barh(memory_delta.index, memory_delta.values,
                         color=colors_d, alpha=0.95,
                         edgecolor=LIGHT_BG, linewidth=0.5)
        for bar, val in zip(bars2, memory_delta.values):
            # Place text to the right of the bar to avoid overlap
            ax2.text(val + (max(memory_delta.values) * 0.02 if not memory_delta.empty else 2),
                     bar.get_y() + bar.get_height() / 2,
                     f"{val:.0f} MB", va="center", ha="left", color=TEXT_COLOR, fontsize=10, fontweight="600")
        ax2.set_xlabel("Memory Delta (MB)", color=MUTED_TEXT, fontsize=10)
        ax2.set_title("Memory Increase During Fit", color=TEXT_COLOR,
                      fontsize=12, fontweight="600")
    else:
        ax2.text(0.5, 0.5, "Memory delta not tracked", transform=ax2.transAxes,
                 ha="center", va="center", color=MUTED_TEXT, fontsize=12)

    ax2.tick_params(colors=TEXT_COLOR)
    for spine in ax2.spines.values():
        spine.set_color(MUTED_TEXT)

    save_figure(fig, save_path)
    return fig
