"""
Visualisations for data-efficiency and scaling analysis:
- Learning curves (Performance vs. Training Size)
- Efficiency scalability (Time/Memory vs. Training Size)
- Stability heatmap (Variance across seeds)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .style import (
    MODEL_COLORS,
    MUTED_TEXT,
    TEXT_COLOR,
    LIGHT_BG,
    LIGHT_SURFACE,
    palette,
    save_figure,
)

def plot_learning_curves(
    df: pd.DataFrame,
    metric: str = "roc_auc",
    save_path: str = "reports/few_shot/learning_curves.png",
    figsize: tuple = (10, 6),
):
    """
    Plot mean metric vs training size with 95% confidence intervals.
    Ensures clear line trends for publication-level clarity.
    """
    if df.empty:
        return None

    # 1. Data Cleaning: Ensure train_size is handled consistently
    df = df.copy()

    # Convert train_size to a consistent format: numeric values or 'full'
    def normalize_size(s):
        if s == "full" or s is None:
            return "full"
        try:
            # Handle cases where it might be a string like "25"
            return int(float(s))
        except (ValueError, TypeError):
            return s

    df["train_size_norm"] = df["train_size"].apply(normalize_size)

    # 2. Define a logical order for sizes
    raw_sizes = df["train_size_norm"].unique()
    numeric_parts = sorted([s for s in raw_sizes if isinstance(s, (int, float))])
    has_full = "full" in raw_sizes
    ordered_sizes = numeric_parts + (["full"] if has_full else [])

    # Create a mapping for categorical positioning
    size_map = {s: i for i, s in enumerate(ordered_sizes)}
    df["size_idx"] = df["train_size_norm"].map(size_map)

    # Calculate mean and std across seeds
    stats = df.groupby(["model", "size_idx", "train_size_norm"])[metric].agg(["mean", "std", "count"]).reset_index()

    # Sort models by the performance at the largest size
    max_idx = stats["size_idx"].max()
    sorted_models = stats[stats["size_idx"] == max_idx].sort_values("mean", ascending=False)["model"].tolist()

    fig, ax = plt.subplots(figsize=figsize, facecolor=LIGHT_BG)
    ax.set_facecolor(LIGHT_BG)

    # Distinct markers for grayscale legibility
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]

    for i, model in enumerate(sorted_models):
        m_data = stats[stats["model"] == model].sort_values("size_idx")
        if m_data.empty: continue

        color = MODEL_COLORS.get(model, MUTED_TEXT)
        marker = markers[i % len(markers)]

        # Plot mean line - ensure zorder is high so lines are on top
        ax.plot(m_data["size_idx"], m_data["mean"], marker=marker, markersize=7,
                linewidth=2.5, color=color, label=model, alpha=1.0, zorder=3)

        # Plot confidence interval (approx 95% as 1.96 * std / sqrt(n))
        if m_data["count"].iloc[0] > 1:
            ci = 1.96 * (m_data["std"] / np.sqrt(m_data["count"]))
            ax.fill_between(m_data["size_idx"],
                            m_data["mean"] - ci,
                            m_data["mean"] + ci,
                            color=color, alpha=0.15, zorder=2)

    # Set X-axis to be categorical
    ax.set_xticks(range(len(ordered_sizes)))
    ax.set_xticklabels(ordered_sizes, color=TEXT_COLOR, fontsize=11, fontweight="500")

    ax.set_xlabel("Training Sample Size", color=MUTED_TEXT, fontsize=12, fontweight="600")
    ax.set_ylabel(f"Mean {metric.upper()}", color=MUTED_TEXT, fontsize=12, fontweight="600")
    ax.set_title(f"Learning Curves: {metric.upper()} vs. Training Size",
                 color=TEXT_COLOR, fontsize=15, fontweight="700", pad=20)

    ax.grid(True, axis="y", alpha=0.4, color=MUTED_TEXT, linestyle="--")
    ax.tick_params(colors=MUTED_TEXT, labelsize=11)
    for spine in ax.spines.values():
        spine.set_color(MUTED_TEXT)

    ax.legend(facecolor=LIGHT_SURFACE, labelcolor=TEXT_COLOR, fontsize=11, loc="best", frameon=True)

    plt.tight_layout()
    save_figure(fig, save_path)
    return fig

def plot_efficiency_scaling(
    df: pd.DataFrame,
    metric: str = "fit_time_sec",
    save_path: str = "reports/few_shot/efficiency_scaling.png",
    figsize: tuple = (10, 6),
):
    """
    Plot efficiency (time/memory) vs training size using categorical X-axis.
    Ensures clear line trends for publication-level clarity.
    """
    if df.empty or metric not in df.columns:
        return None

    # 1. Data Cleaning: Ensure train_size is handled consistently
    df = df.copy()
    def normalize_size(s):
        if s == "full" or s is None:
            return "full"
        try:
            return int(float(s))
        except (ValueError, TypeError):
            return s

    df["train_size_norm"] = df["train_size"].apply(normalize_size)

    # 2. Define a logical order for sizes
    raw_sizes = df["train_size_norm"].unique()
    numeric_parts = sorted([s for s in raw_sizes if isinstance(s, (int, float))])
    has_full = "full" in raw_sizes
    ordered_sizes = numeric_parts + (["full"] if has_full else [])
    size_map = {s: i for i, s in enumerate(ordered_sizes)}
    df["size_idx"] = df["train_size_norm"].map(size_map)

    stats = df.groupby(["model", "size_idx"])[metric].mean().reset_index()

    fig, ax = plt.subplots(figsize=figsize, facecolor=LIGHT_BG)
    ax.set_facecolor(LIGHT_BG)

    # Distinct markers for grayscale legibility
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]
    for i, model in enumerate(stats["model"].unique()):
        m_data = stats[stats["model"] == model].sort_values("size_idx")
        color = MODEL_COLORS.get(model, MUTED_TEXT)
        marker = markers[i % len(markers)]
        ax.plot(m_data["size_idx"], m_data[metric], marker=marker, markersize=7,
                linewidth=2.5, color=color, label=model, alpha=1.0, zorder=3)

    ax.set_xticks(range(len(ordered_sizes)))
    ax.set_xticklabels(ordered_sizes, color=TEXT_COLOR, fontsize=11, fontweight="500")
    ax.set_xlabel("Training Sample Size", color=MUTED_TEXT, fontsize=12, fontweight="600")

    label = "Fit Time (s)" if "time" in metric else "Peak Memory (MB)"
    ax.set_ylabel(label, color=MUTED_TEXT, fontsize=12, fontweight="600")
    ax.set_title(f"Computational Scalability: {label} vs. Size",
                 color=TEXT_COLOR, fontsize=15, fontweight="700", pad=20)

    ax.grid(True, axis="y", alpha=0.4, color=MUTED_TEXT, linestyle="--")
    ax.tick_params(colors=MUTED_TEXT, labelsize=11)
    for spine in ax.spines.values():
        spine.set_color(MUTED_TEXT)

    ax.legend(facecolor=LIGHT_SURFACE, labelcolor=TEXT_COLOR, fontsize=11, loc="best", frameon=True)

    plt.tight_layout()
    save_figure(fig, save_path)
    return fig

def plot_stability_heatmap(
    df: pd.DataFrame,
    metric: str = "roc_auc",
    save_path: str = "reports/few_shot/stability_heatmap.png",
    figsize: tuple = (12, 6),
):
    """
    Heatmap of standard deviation (instability) across seeds.
    """
    if df.empty:
        return None

    df = df.copy()

    # Logical sort for the X-axis (the sizes)
    raw_sizes = df["train_size"].unique()
    numeric_parts = sorted([s for s in raw_sizes if isinstance(s, (int, float))])
    has_full = "full" in raw_sizes
    ordered_sizes = [str(s) for s in numeric_parts] + ([ "full"] if has_full else [])

    df["train_size_str"] = df["train_size"].astype(str)
    stability = df.groupby(["model", "train_size_str"])[metric].std().unstack("train_size_str")

    # Reindex to ensure logical order on X-axis
    existing_sizes = [s for s in ordered_sizes if s in stability.columns]
    stability = stability[existing_sizes]

    if stability.empty or stability.isna().all().all():
        print("  Skipping stability_heatmap: No valid stability data found")
        return None

    fig, ax = plt.subplots(figsize=figsize, facecolor=LIGHT_BG)
    ax.set_facecolor(LIGHT_BG)

    sns.heatmap(stability, annot=True, fmt=".4f", cmap="YlOrRd",
                ax=ax, cbar_kws={'label': 'Std Dev'},
                annot_kws={"fontsize": 10, "color": TEXT_COLOR, "fontweight": "500"})

    ax.set_title(f"Performance Instability (Std Dev of {metric.upper()})",
                 color=TEXT_COLOR, fontsize=14, fontweight="700", pad=20)
    ax.set_xlabel("Training Size", color=MUTED_TEXT, fontsize=11, fontweight="500")
    ax.set_ylabel("Model", color=MUTED_TEXT, fontsize=11, fontweight="500")

    ax.tick_params(colors=TEXT_COLOR, labelsize=10)

    plt.tight_layout()
    save_figure(fig, save_path)
    return fig
