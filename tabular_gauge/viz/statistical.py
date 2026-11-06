"""
Statistical visualisation for benchmark results:
- Critical Difference (CD) diagram (Demšar 2006 layout)
- Bootstrap 95% confidence intervals
- Pairwise Wilcoxon significance heatmap
- Win / loss / tie matrix
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from .style import (
    MODEL_COLORS,
    MUTED_TEXT,
    TEXT_COLOR,
    LIGHT_BG,
    LIGHT_SURFACE,
    color_for_model,
    palette,
    save_figure,
)


def plot_critical_difference_diagram(
    rank_df: pd.DataFrame,
    cd: float,
    save_path: str = "reports/figures/cd_diagram.png",
    alpha: float = 0.05,
):
    """
    Standard Demšar (2006) Critical Difference diagram.

    Layout (canonical):
        - rank axis at y=1 with integer ticks 1..k
        - model markers and labels at y_models=1.30 (above the axis)
        - clique lines drawn just above the model markers (y_clique=1.55)
        - CD bar drawn BELOW the axis (y_cd=0.45) with vertical end-caps
    """
    models = rank_df["model"].tolist()
    avg_ranks = rank_df["average_rank"].tolist()
    n_models = len(models)

    fig, ax = plt.subplots(figsize=(12, 3.4), facecolor=LIGHT_BG)
    ax.set_facecolor(LIGHT_BG)

    # Rank-axis range
    x_left = 0.5
    x_right = n_models + 0.5
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(0, 2)

    # Rank axis line
    ax.axhline(y=1.0, color=MUTED_TEXT, linewidth=1.5, zorder=1)

    # Tick marks and labels on the axis
    for r in range(1, n_models + 1):
        ax.plot([r, r], [0.95, 1.05], color=MUTED_TEXT, linewidth=1.0, zorder=1)
        ax.text(r, 0.85, str(r), ha="center", va="top", color=MUTED_TEXT, fontsize=10)

    ax.text((x_left + x_right) / 2.0, 0.72, "Average Rank",
            ha="center", color=MUTED_TEXT, fontsize=11)

    # ---- Model markers + labels (above the axis) ----
    y_models = 1.30
    for model, rank in zip(models, avg_ranks):
        color = MODEL_COLORS.get(model, MUTED_TEXT)
        ax.scatter(rank, y_models, s=170, color=color, zorder=5,
                   edgecolor=LIGHT_BG, linewidth=1.5)
        # Label just above the marker
        ax.text(rank, y_models + 0.07, model, ha="center", va="bottom",
                color=color, fontsize=10, fontweight="600")

    # ---- Clique lines: connect models that are within the CD of each other ----
    y_clique = 1.62
    groups: list[list[int]] = []
    used = set()
    sorted_idx = sorted(range(n_models), key=lambda i: avg_ranks[i])
    for i in sorted_idx:
        if i in used:
            continue
        clique = [i]
        for j in sorted_idx:
            if j == i or j in used:
                continue
            if abs(avg_ranks[i] - avg_ranks[j]) <= cd:
                clique.append(j)
        for j in clique:
            used.add(j)
        groups.append(clique)

    for clique in groups:
        if len(clique) <= 1:
            continue
        clique_ranks = [avg_ranks[i] for i in clique]
        ax.plot([min(clique_ranks), max(clique_ranks)], [y_clique, y_clique],
                color="#0f172a", linewidth=1.5, zorder=3, alpha=0.85)
        # End-caps on the clique bar
        for r in (min(clique_ranks), max(clique_ranks)):
            ax.plot([r, r], [y_clique - 0.025, y_clique + 0.025],
                    color="#0f172a", linewidth=1.5, zorder=3, alpha=0.85)

    # ---- CD bar (below the axis) ----
    y_cd = 0.45
    cd_start = 1.0
    cd_end = 1.0 + cd
    ax.plot([cd_start, cd_end], [y_cd, y_cd], color="#5b8dee", linewidth=3, zorder=4)
    # End-caps
    for r in (cd_start, cd_end):
        ax.plot([r, r], [y_cd - 0.06, y_cd + 0.06], color="#5b8dee",
                linewidth=2, zorder=4)
    ax.text((cd_start + cd_end) / 2, y_cd - 0.12, f"CD = {cd:.2f}",
            ha="center", va="top", color="#5b8dee", fontsize=11, fontweight="600")

    # Annotation in the bottom-right
    ax.text(0.985, 0.05,
            f"alpha = {alpha}\nModels linked by a horizontal bar above the axis\n"
            "are not significantly different (Nemenyi post-hoc).",
            transform=ax.transAxes, ha="right", va="bottom",
            color=MUTED_TEXT, fontsize=9, style="italic")

    ax.set_title("Critical Difference Diagram (Demšar 2006 layout)",
                 color=TEXT_COLOR, fontsize=12, fontweight="600", pad=10)
    ax.axis("off")

    save_figure(fig, save_path)
    return fig


def plot_confidence_intervals(
    ci_df: pd.DataFrame,
    metric: str = "ROC-AUC",
    save_path: str = "reports/figures/confidence_intervals.png",
):
    """
    Horizontal point-and-whisker plot of per-model 95% bootstrap CIs.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=LIGHT_BG)
    ax.set_facecolor(LIGHT_BG)

    models = ci_df["model"].tolist()
    means = ci_df["mean"].tolist()
    lo = ci_df["ci_lower"].tolist()
    hi = ci_df["ci_upper"].tolist()

    y_pos = np.arange(len(models))
    for i, (model, mean, low, high) in enumerate(zip(models, means, lo, hi)):
        color = MODEL_COLORS.get(model, MUTED_TEXT)
        ax.errorbar(
            mean, i,
            xerr=[[mean - low], [high - mean]],
            fmt="o", color=color, capsize=5, capthick=2,
            markersize=9, linewidth=2,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, color=TEXT_COLOR, fontsize=10)
    ax.set_xlabel(f"{metric} (95% bootstrap CI, n=1000 resamples)",
                  color=MUTED_TEXT, fontsize=10)
    ax.set_title("Per-Model Performance with 95% Confidence Intervals",
                 color=TEXT_COLOR, fontsize=12, fontweight="600")
    ax.tick_params(colors=MUTED_TEXT)
    for spine in ax.spines.values():
        spine.set_color(MUTED_TEXT)
    ax.invert_yaxis()
    ax.xaxis.grid(True, alpha=0.4, color=MUTED_TEXT)
    ax.set_axisbelow(True)

    save_figure(fig, save_path)
    return fig


def plot_statistical_significance_heatmap(
    wilcoxon_df: pd.DataFrame,
    save_path: str = "reports/figures/significance_heatmap.png",
    metric: str = "ROC-AUC",
):
    """
    Square heatmap of pairwise Wilcoxon signed-rank p-values with significance
    markers in each cell. Cells are coloured by p-value so the strength of
    results are visible at a glance.
    """
    all_models = sorted(set(wilcoxon_df["model_a"]).union(wilcoxon_df["model_b"]))
    p_matrix = pd.DataFrame(np.nan, index=all_models, columns=all_models)
    sig_matrix = pd.DataFrame(False, index=all_models, columns=all_models)

    for _, row in wilcoxon_df.iterrows():
        p_matrix.loc[row["model_a"], row["model_b"]] = row["p_value"]
        p_matrix.loc[row["model_b"], row["model_a"]] = row["p_value"]
        sig_matrix.loc[row["model_a"], row["model_b"]] = bool(row["significant"])
        sig_matrix.loc[row["model_b"], row["model_a"]] = bool(row["significant"])

    fig, ax = plt.subplots(figsize=(8, 6.4), facecolor=LIGHT_BG)
    ax.set_facecolor(LIGHT_BG)

    cmap = sns.color_palette("RdYlGn_r", as_cmap=True)
    im = ax.imshow(p_matrix.values, cmap=cmap, vmin=0.0, vmax=0.10,
                   aspect="auto")

    ax.set_xticks(range(len(all_models)))
    ax.set_xticklabels(all_models, color=TEXT_COLOR, fontsize=10, rotation=30, ha="right")
    ax.set_yticks(range(len(all_models)))
    ax.set_yticklabels(all_models, color=TEXT_COLOR, fontsize=10)

    # Cell annotations: p-value + significance marker
    for i in range(len(all_models)):
        for j in range(len(all_models)):
            if i == j:
                ax.text(j, i, "-", ha="center", va="center",
                        color=TEXT_COLOR, fontsize=11)
                continue
            p = p_matrix.iloc[i, j]
            if pd.isna(p):
                continue
            sig_marker = "Y" if sig_matrix.iloc[i, j] else "n"
            txt = f"{p:.3f}\n{sig_marker}"
            text_color = "white" if p < 0.04 else TEXT_COLOR
            ax.text(j, i, txt, ha="center", va="center",
                    color=text_color, fontsize=9, fontweight="600")

    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cb.set_label("Wilcoxon p-value", color=MUTED_TEXT, fontsize=9)
    cb.ax.yaxis.set_tick_params(color=MUTED_TEXT)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=MUTED_TEXT, fontsize=8)

    ax.set_title(
        f"Pairwise Wilcoxon Signed-Rank ({metric})\n"
        "Cell shows p-value. Y = significant at alpha=0.05, n = not significant.",
        color=TEXT_COLOR, fontsize=11, fontweight="600", pad=12,
    )

    legend_elements = [
        mpatches.Patch(facecolor=sns.color_palette("RdYlGn_r", 4)[0], label="p < 0.025"),
        mpatches.Patch(facecolor=sns.color_palette("RdYlGn_r", 4)[2], label="p ~ 0.05"),
        mpatches.Patch(facecolor=sns.color_palette("RdYlGn_r", 4)[3], label="p > 0.10"),
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              facecolor=LIGHT_SURFACE, labelcolor=TEXT_COLOR, fontsize=8)

    save_figure(fig, save_path)
    return fig


def plot_win_loss_matrix(
    results_df: pd.DataFrame,
    metric: str = "roc_auc",
    save_path: str = "reports/figures/win_loss_matrix.png",
):
    """
    3-panel win / loss / tie matrix across the datasets.

    Each cell (i, j) counts how many datasets model i beat / lost to / tied
    model j when comparing the mean metric across seeds.
    """
    pivot = (
        results_df.groupby(["dataset", "model"])[metric].mean().unstack("model")
    )

    # Dynamic ordering based on results
    from ..evaluation.analysis import get_model_order, PREFERRED_ORDER
    model_order = get_model_order(results_df, preferred_order=PREFERRED_ORDER)
    for m in model_order:
        if m not in pivot.columns:
            pivot[m] = np.nan
    pivot = pivot[model_order]
    models = pivot.columns.tolist()

    wins   = pd.DataFrame(0, index=models, columns=models, dtype=int)
    losses = pd.DataFrame(0, index=models, columns=models, dtype=int)
    ties   = pd.DataFrame(0, index=models, columns=models, dtype=int)
    for _, row in pivot.iterrows():
        for m1 in models:
            for m2 in models:
                if m1 == m2:
                    continue
                v1, v2 = row[m1], row[m2]
                if pd.isna(v1) or pd.isna(v2):
                    continue
                if v1 > v2:
                    wins.loc[m1, m2] += 1
                elif v1 < v2:
                    losses.loc[m1, m2] += 1
                else:
                    ties.loc[m1, m2] += 1

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0), facecolor=LIGHT_BG)

    panels = [
        (wins,   "Wins",   "Greens"),
        (losses, "Losses", "Reds"),
        (ties,   "Ties",   "Blues"),
    ]
    for ax, (matrix, title, cmap) in zip(axes, panels):
        ax.set_facecolor(LIGHT_BG)
        sns.heatmap(
            matrix, annot=True, fmt="d", cmap=cmap, ax=ax,
            cbar=False, linewidths=0.5, linecolor=LIGHT_BG,
            xticklabels=models, yticklabels=models,
            annot_kws={"fontsize": 10, "fontweight": "600", "color": TEXT_COLOR},
        )
        ax.set_title(title, color=TEXT_COLOR, fontsize=12, fontweight="600")
        ax.tick_params(colors=MUTED_TEXT)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right",
                 color=TEXT_COLOR, fontsize=10)
        plt.setp(ax.yaxis.get_majorticklabels(), color=TEXT_COLOR, fontsize=10)

    plt.suptitle(f"Win / Loss / Tie Matrix ({metric})",
                 color=TEXT_COLOR, fontsize=13, fontweight="600", y=1.02)

    save_figure(fig, save_path)
    return fig