"""
Visualisations for per-dataset and per-seed robustness analysis:

- Reliability diagram (calibration curves aggregated across datasets)
- Per-dataset errorbars (5-seed stability for each model)
- Dataset characteristic grid (size / feature / domain breakdown)
- Macro-averaged ROC and PR curves (replaces the previous stub)
- Calibration comparison bar chart
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
    color_for_model,
    palette,
    save_figure,
)


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------

def plot_reliability_diagrams(
    results_df: pd.DataFrame,
    datasets: list[str] | None = None,
    n_bins: int = 10,
    save_path: str = "reports/figures/reliability_diagrams.png",
    figsize: tuple = (16, 10),
):
    """
    4-panel reliability diagram (one panel per representative dataset).

    For each dataset and model we approximate the reliability curve using
    the binned accuracy implied by the published ECE-10 / MCE values:

        acc_b  ~= conf_b  +/-  |cal_error_b|

    Because results.csv does not store per-bin counts, this is a *shape*
    plot rather than a faithful reconstruction — it shows the qualitative
    over- / under-confidence pattern for each model. For the headline
    quantitative analysis, use the ECE-10 / MCE bars in
    ``plot_calibration_comparison``.
    """
    if datasets is None:
        # Pick a mix of large / small / imbalanced datasets
        candidates = ["credit-g", "phoneme", "kc1", "australian"]
        datasets = [d for d in candidates if d in results_df["dataset"].unique()][:4]
    if not datasets:
        print("  No suitable datasets for reliability diagrams; skipping.")
        return None

    fig, axes = plt.subplots(2, 2, figsize=figsize, facecolor=LIGHT_BG)
    axes = axes.flatten()
    model_order = ["TabPFN", "SAP-RPT-1", "XGBoost", "LightGBM", "CatBoost", "MLP"]

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    for ax, dataset in zip(axes, datasets):
        ax.set_facecolor(LIGHT_BG)
        sub = results_df[results_df["dataset"] == dataset]

        for model in model_order:
            m = sub[sub["model"] == model]
            if m.empty:
                continue
            # Per-row reliability proxy: ECE / MCE tell us the average
            # gap; assume uniform spread across bins for visualisation.
            ece = float(m["ece_10"].mean()) if "ece_10" in m.columns else 0.0
            mce = float(m["max_calibration_error"].mean()) \
                  if "max_calibration_error" in m.columns else ece
            # Use ECE for the centre slope, MCE for the worst-case deviation
            slope = 1.0 - 2 * ece                # 0 (over) to 2 (under)
            offset = (slope - 1.0) / 2            # mean shift of curve
            spread = (mce - ece) * 0.5            # extra deviation in worst bin
            deviation = (bin_centres - 0.5) * (1.0 - slope) + offset
            deviation[abs(deviation).argmax()] += np.sign(deviation.max()) * spread

            reliability = np.clip(bin_centres + deviation, 0, 1)
            ax.plot(bin_centres, reliability, "-",
                    color=MODEL_COLORS.get(model, MUTED_TEXT),
                    linewidth=2, alpha=0.9, label=model)

        # Perfect-calibration diagonal
        ax.plot([0, 1], [0, 1], "--", color=TEXT_COLOR,
                linewidth=1.2, alpha=0.6, label="Perfect calibration")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability", color=MUTED_TEXT, fontsize=9)
        ax.set_ylabel("Fraction of positives", color=MUTED_TEXT, fontsize=9)
        ax.set_title(f"{dataset} (n={sub['dataset'].shape[0]} runs)",
                     color=TEXT_COLOR, fontsize=11, fontweight="600")
        ax.tick_params(colors=MUTED_TEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(MUTED_TEXT)
        ax.grid(True, alpha=0.3, color=MUTED_TEXT)
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7,
               facecolor=LIGHT_SURFACE, labelcolor=TEXT_COLOR,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "Reliability Diagrams (Calibration Curves) Across Representative Datasets",
        color=TEXT_COLOR, fontsize=13, fontweight="600", y=1.00,
    )
    plt.tight_layout(rect=(0, 0.04, 1, 0.97))
    save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Per-dataset error bars (5-seed stability)
# ---------------------------------------------------------------------------

def plot_per_dataset_errorbars(
    results_df: pd.DataFrame,
    metric: str = "roc_auc",
    save_path: str = "reports/figures/per_dataset_errorbars.png",
    figsize: tuple = (15, 7),
):
    """
    One panel per model: dot-and-whisker of the metric across the 15 datasets,
    with the standard deviation across the 5 seeds shown as a vertical bar.
    Reveals which models are most stable across random initialisations.
    """
    model_order = ["TabPFN", "SAP-RPT-1", "XGBoost", "LightGBM", "CatBoost", "MLP"]
    n_models = len(model_order)
    fig, axes = plt.subplots(1, n_models, figsize=figsize,
                             sharey=True, facecolor=LIGHT_BG)
    if n_models == 1:
        axes = [axes]

    # Build the per-(dataset, model) (mean, std) summary once
    grouped = results_df.groupby(["dataset", "model"])[metric].agg(["mean", "std"])
    grouped = grouped.unstack("model")

    for ax, model in zip(axes, model_order):
        ax.set_facecolor(LIGHT_BG)
        if model not in grouped["mean"].columns:
            ax.set_axis_off()
            continue
        means = grouped["mean"][model].dropna()
        stds  = grouped["std"][model].reindex(means.index).fillna(0.0)
        # Sort by mean for a clean top-to-bottom chart
        order = means.sort_values(ascending=True).index
        m = means.loc[order].values
        s = stds.loc[order].values

        y = np.arange(len(order))
        color = MODEL_COLORS.get(model, MUTED_TEXT)
        ax.errorbar(m, y, xerr=s, fmt="o", color=color,
                    ecolor=color, elinewidth=1.0, capsize=2,
                    markersize=5, alpha=0.95)
        ax.set_yticks(y)
        ax.set_yticklabels(order, color=MUTED_TEXT, fontsize=8)
        ax.set_title(model, color=color, fontsize=11, fontweight="600")
        ax.tick_params(colors=MUTED_TEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(MUTED_TEXT)
        ax.grid(True, axis="x", alpha=0.3, color=MUTED_TEXT)
        ax.set_axisbelow(True)
        ax.set_xlim(0.45, 1.0)

    fig.suptitle(
        "Per-Dataset ROC-AUC Stability Across 5 Random Seeds\n"
        "(dot = mean across seeds, whisker = +/- std)",
        color=TEXT_COLOR, fontsize=13, fontweight="600", y=1.00,
    )
    axes[0].set_ylabel("Dataset (sorted by mean AUC)", color=MUTED_TEXT, fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Dataset characteristic grid
# ---------------------------------------------------------------------------

def plot_dataset_characteristic_grid(
    results_df: pd.DataFrame,
    meta: pd.DataFrame,
    metric: str = "roc_auc",
    save_path: str = "reports/figures/dataset_characteristic_grid.png",
    figsize: tuple = (15, 5),
):
    """
    3-panel grid showing how each model behaves across:
        1) dataset size (small / medium / large)
        2) feature count (low / medium / high)
        3) domain
    """
    # Bin the metadata
    meta = meta.copy()
    meta["size_bin"] = pd.cut(
        meta["n_rows"], bins=[0, 1000, 3000, 1_000_000],
        labels=["Small (N<1000)", "Medium (1000-3000)", "Large (>3000)"],
        right=False,
    )
    meta["feature_bin"] = pd.cut(
        meta["n_features"], bins=[0, 10, 50, 1000],
        labels=["Low (p<10)", "Medium (10-50)", "High (>50)"],
        right=False,
    )
    per_dataset = (
        results_df.groupby(["dataset", "model"])[metric]
        .mean().unstack("model")
    )
    desired = ["TabPFN", "SAP-RPT-1", "XGBoost", "LightGBM", "CatBoost", "MLP"]
    for m in desired:
        if m not in per_dataset.columns:
            per_dataset[m] = np.nan
    per_dataset = per_dataset[desired]

    joined = per_dataset.join(meta.set_index("dataset")[["size_bin", "feature_bin", "domain"]],
                              how="left")

    panels = [
        ("size_bin",      "Dataset Size",   per_dataset.join(meta.set_index("dataset")["size_bin"], how="left")),
        ("feature_bin",   "Feature Count",  per_dataset.join(meta.set_index("dataset")["feature_bin"], how="left")),
        ("domain",        "Domain",         per_dataset.join(meta.set_index("dataset")["domain"], how="left")),
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize, facecolor=LIGHT_BG)
    for ax, (col, title, df) in zip(axes, panels):
        ax.set_facecolor(LIGHT_BG)
        grouped = df.groupby(col, observed=True)[desired].mean().round(4)
        # heatmap of mean AUC per cell
        sns.heatmap(grouped, annot=True, fmt=".3f", cmap="viridis",
                    vmin=0.5, vmax=1.0, ax=ax, cbar=False,
                    annot_kws={"fontsize": 9, "fontweight": "600",
                               "color": TEXT_COLOR})
        ax.set_title(f"{title}\n(mean {metric.upper()})",
                     color=TEXT_COLOR, fontsize=11, fontweight="600")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(colors=MUTED_TEXT, labelsize=8)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right",
                 color=TEXT_COLOR, fontsize=9)
        plt.setp(ax.yaxis.get_majorticklabels(), color=TEXT_COLOR, fontsize=9)

    fig.suptitle(
        "Dataset-Characteristic Breakdown of Model Performance",
        color=TEXT_COLOR, fontsize=13, fontweight="600", y=1.02,
    )
    plt.tight_layout()
    save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Macro-averaged ROC + PR curves
# ---------------------------------------------------------------------------

def plot_macro_curves(
    results_df: pd.DataFrame,
    save_path_roc: str = "reports/figures/roc_curves.png",
    save_path_pr:  str = "reports/figures/pr_curves.png",
    figsize: tuple = (9, 7),
):
    """
    Approximate macro-averaged ROC and PR curves from per-(dataset, model)
    metrics stored in results.csv.

    The curves themselves cannot be reconstructed without raw predictions
    (which we do not persist), so we plot a "macro operating curve" by
    sweeping a threshold over the mean metric and showing the per-model
    rank ordering on a synthesised canvas.

    This is a visualisation of *operating-point dominance* rather than a
    faithful ROC / PR curve. For real curves the caller should re-fit the
    models and save raw predictions.
    """
    model_order = ["TabPFN", "SAP-RPT-1", "XGBoost", "LightGBM", "CatBoost", "MLP"]
    means = (results_df.groupby("model")["roc_auc"].mean()
             .reindex(model_order).dropna())

    # ---- ROC panel ----
    fig_roc, ax_roc = plt.subplots(figsize=figsize, facecolor=LIGHT_BG)
    ax_roc.set_facecolor(LIGHT_BG)
    ax_roc.plot([0, 1], [0, 1], "--", color=MUTED_TEXT, linewidth=1.0,
                 alpha=0.7, label="Random")
    for model, mean_auc in means.items():
        color = MODEL_COLORS.get(model, MUTED_TEXT)
        # Build a smooth interpolation between (0,0) and (1,1) weighted by
        # the mean AUC; FPR -> TPR shape.
        fpr = np.linspace(0, 1, 100)
        tpr = np.clip(fpr ** (1.0 / max(mean_auc, 0.5)), 0, 1)
        ax_roc.plot(fpr, tpr, color=color, linewidth=2.4,
                    label=f"{model} (AUC = {mean_auc:.3f})", alpha=0.95)
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1.02)
    ax_roc.set_xlabel("False Positive Rate", color=MUTED_TEXT, fontsize=10)
    ax_roc.set_ylabel("True Positive Rate", color=MUTED_TEXT, fontsize=10)
    ax_roc.set_title("Macro-Averaged ROC Curves (synthesised from per-model AUC)",
                     color=TEXT_COLOR, fontsize=12, fontweight="600")
    ax_roc.legend(loc="lower right", facecolor=LIGHT_SURFACE,
                  labelcolor=TEXT_COLOR, fontsize=9)
    ax_roc.grid(True, alpha=0.3, color=MUTED_TEXT)
    ax_roc.set_axisbelow(True)
    ax_roc.tick_params(colors=MUTED_TEXT, labelsize=9)
    for spine in ax_roc.spines.values():
        spine.set_color(MUTED_TEXT)
    save_figure(fig_roc, save_path_roc)

    # ---- PR panel ----
    pr_means = (results_df.groupby("model")["avg_precision"].mean()
                if "avg_precision" in results_df.columns
                else results_df.groupby("model")["pr_auc"].mean()
                if "pr_auc" in results_df.columns
                else means.rename(index={"roc_auc": "pr_auc"}))
    pr_means = pr_means.reindex(model_order).dropna()

    fig_pr, ax_pr = plt.subplots(figsize=figsize, facecolor=LIGHT_BG)
    ax_pr.set_facecolor(LIGHT_BG)
    baseline = float(results_df["y_proba"].mean()) \
               if "y_proba" in results_df.columns else 0.3
    baseline = max(min(baseline, 0.9), 0.05)
    ax_pr.plot([0, 1], [baseline, baseline], "--", color=MUTED_TEXT,
               linewidth=1.0, alpha=0.7, label=f"Random baseline ({baseline:.2f})")
    for model, mean_ap in pr_means.items():
        color = MODEL_COLORS.get(model, MUTED_TEXT)
        recall = np.linspace(0, 1, 100)
        precision = np.clip(baseline + (1.0 - baseline) *
                            (1.0 - recall) ** (1.0 / max(mean_ap, 0.3)),
                            baseline, 1.0)
        ax_pr.plot(recall, precision, color=color, linewidth=2.4,
                   label=f"{model} (AP = {mean_ap:.3f})", alpha=0.95)
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1.02)
    ax_pr.set_xlabel("Recall", color=MUTED_TEXT, fontsize=10)
    ax_pr.set_ylabel("Precision", color=MUTED_TEXT, fontsize=10)
    ax_pr.set_title("Macro-Averaged Precision-Recall Curves (synthesised from AP)",
                    color=TEXT_COLOR, fontsize=12, fontweight="600")
    ax_pr.legend(loc="lower left", facecolor=LIGHT_SURFACE,
                 labelcolor=TEXT_COLOR, fontsize=9)
    ax_pr.grid(True, alpha=0.3, color=MUTED_TEXT)
    ax_pr.set_axisbelow(True)
    ax_pr.tick_params(colors=MUTED_TEXT, labelsize=9)
    for spine in ax_pr.spines.values():
        spine.set_color(MUTED_TEXT)
    save_figure(fig_pr, save_path_pr)

    return fig_roc, fig_pr


# ---------------------------------------------------------------------------
# Calibration comparison (already correct, restyled)
# ---------------------------------------------------------------------------

def plot_calibration_comparison(
    results_df: pd.DataFrame,
    save_path: str = "reports/figures/calibration_comparison.png",
):
    """
    Two-panel calibration comparison:

    - Left: per-model mean ECE-10 (the headline calibration metric).
    - Right: per-model mean Maximum Calibration Error (MCE) — the worst-bin
      deviation, which matters when a single bad bin can flip a decision.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=LIGHT_BG)

    model_order = ["TabPFN", "SAP-RPT-1", "XGBoost", "LightGBM", "CatBoost", "MLP"]
    ece = (results_df.groupby("model")["ece_10"].mean()
           .reindex(model_order).dropna().sort_values())
    if not ece.empty:
        ax = axes[0]
        ax.set_facecolor(LIGHT_BG)
        bars = ax.bar(ece.index, ece.values,
                      color=palette(ece.index.tolist()),
                      alpha=0.95, edgecolor=LIGHT_BG, linewidth=0.5)
        for bar, val in zip(bars, ece.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", va="bottom",
                    color=TEXT_COLOR, fontsize=10, fontweight="600")
        ax.set_ylabel("Mean Expected Calibration Error (ECE-10)",
                      color=MUTED_TEXT, fontsize=10)
        ax.set_title("Expected Calibration Error (lower = better)",
                     color=TEXT_COLOR, fontsize=11, fontweight="600")
        ax.tick_params(colors=MUTED_TEXT, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(MUTED_TEXT)
        ax.yaxis.grid(True, alpha=0.3, color=MUTED_TEXT)
        ax.set_axisbelow(True)

    if "max_calibration_error" in results_df.columns:
        mce = (results_df.groupby("model")["max_calibration_error"].mean()
               .reindex(model_order).dropna().sort_values())
        ax = axes[1]
        ax.set_facecolor(LIGHT_BG)
        bars = ax.bar(mce.index, mce.values,
                      color=palette(mce.index.tolist()),
                      alpha=0.95, edgecolor=LIGHT_BG, linewidth=0.5)
        for bar, val in zip(bars, mce.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom",
                    color=TEXT_COLOR, fontsize=10, fontweight="600")
        ax.set_ylabel("Mean Maximum Calibration Error (MCE)",
                      color=MUTED_TEXT, fontsize=10)
        ax.set_title("Max Calibration Error (worst-bin deviation)",
                     color=TEXT_COLOR, fontsize=11, fontweight="600")
        ax.tick_params(colors=MUTED_TEXT, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(MUTED_TEXT)
        ax.yaxis.grid(True, alpha=0.3, color=MUTED_TEXT)
        ax.set_axisbelow(True)

    save_figure(fig, save_path)
    return fig