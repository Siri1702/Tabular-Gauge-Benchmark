#!/usr/bin/env python3
"""
Regenerate every publication artefact from `experiments/results/aggregated/results.csv`.

Outputs:
    - 9 IEEE-quality PNG figures in `reports/figures/`
    - the full markdown report at `reports/summary.md`

Why this exists:
    The benchmark runner only triggers `_generate_reports` inside the loop,
    which means skip-only invocations and audit-after-the-fact rebuilds
    silently fail to refresh the figures. This script is the single
    command that always works.

Usage:
    python experiments/generate_reports.py
    python experiments/generate_reports.py --metric pr_auc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make `src` importable when run from anywhere
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tabular_gauge.evaluation.analysis import (
    headline_table,
    per_dataset_table,
    average_ranks,
    calibration_table,
    timing_table,
    memory_table,
    friedman_summary,
    nemenyi_table,
    wilcoxon_table,
    bootstrap_ci,
    win_loss_matrix,
    characteristic_table,
    metric_correlation_with_size,
    all_metric_summary,
    load_csv,
    load_dataset_meta,
)

from tabular_gauge.viz.leaderboard import (
    plot_auc_heatmap,
    plot_average_ranks,
    plot_timing_comparison,
    plot_memory_comparison,
)
from tabular_gauge.viz.statistical import (
    plot_critical_difference_diagram,
    plot_confidence_intervals,
    plot_statistical_significance_heatmap,
    plot_win_loss_matrix,
)
from tabular_gauge.viz.curves import (
    plot_reliability_diagrams,
    plot_per_dataset_errorbars,
    plot_dataset_characteristic_grid,
    plot_macro_curves,
    plot_calibration_comparison,
)


CSV_PATH       = "experiments/results/aggregated/results.csv"
META_PATH      = "configs/datasets.yaml"
FIG_DIR        = "reports/figures"
SUMMARY_PATH   = "reports/summary.md"


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------
def _fmt(x, digits: int = 4) -> str:
    """Format a number with `digits` decimals, 'NA' for NaN/None."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    return f"{x:.{digits}f}"


def _fmt_ci(mean, lo, hi, digits: int = 4) -> str:
    return f"{_fmt(mean, digits)} [{_fmt(lo, digits)}, {_fmt(hi, digits)}]"


def _p(p) -> str:
    """Format a p-value with scientific notation when tiny."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "NA"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


# ---------------------------------------------------------------------------
# Figure orchestration
# ---------------------------------------------------------------------------
def generate_figures(df: pd.DataFrame, meta: pd.DataFrame,
                         metric: str = "roc_auc",
                         figures: list[str] | None = None,
                         fig_dir: Path | str = FIG_DIR,
                         table_dir: Path | str = "reports/tables") -> None:
    """
    Generate every publication figure.

    Pass `figures=[...]` to limit the set. Defaults to all 9.
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    table_dir = Path(table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)

    # Always-needed analytical tables
    rank_df = average_ranks(df, metric=metric)
    _, cd   = nemenyi_table(df, metric=metric)
    ci_df   = bootstrap_ci(df, metric=metric)
    wilcoxon_df = wilcoxon_table(df, metric=metric)

    # Save analytical tables to disk
    rank_df.to_csv(table_dir / "average_ranks.csv", index=False)
    ci_df.to_csv(table_dir / "confidence_intervals.csv", index=False)
    wilcoxon_df.to_csv(table_dir / "wilcoxon_significance.csv", index=False)

    # Save other summary tables
    headline_table(df, metric=metric).to_csv(table_dir / "headline_metrics.csv", index=False)
    per_dataset_table(df, metric=metric).to_csv(table_dir / "per_dataset_metrics.csv")
    calibration_table(df).to_csv(table_dir / "calibration_metrics.csv", index=False)
    timing_table(df).to_csv(table_dir / "timing_metrics.csv", index=False)
    memory_table(df).to_csv(table_dir / "memory_metrics.csv", index=False)

    # Save characteristic analysis
    per_ds_full, by_size, by_feat, by_dom, stability = characteristic_table(df, meta, metric=metric)
    by_size.to_csv(table_dir / "metrics_by_size.csv")
    by_feat.to_csv(table_dir / "metrics_by_features.csv")
    by_dom.to_csv(table_dir / "metrics_by_domain.csv")
    stability.to_csv(table_dir / "seed_stability.csv")

    metric_correlation_with_size(df, meta, metric=metric).to_csv(table_dir / "size_correlation.csv", index=False)
    all_metric_summary(df).to_csv(table_dir / "all_metrics_summary.csv")

    figures = figures or [
        "auc_heatmap", "average_ranks", "timing", "memory",
        "win_loss", "calibration", "cd_diagram", "confidence_intervals",
        "significance_heatmap", "reliability", "errorbars", "characteristic",
        "roc_curves", "pr_curves",
    ]

    if "auc_heatmap" in figures:
        plot_auc_heatmap(df, metric=metric,
                          save_path=f"{fig_dir}/auc_heatmap.png")
    if "average_ranks" in figures:
        # Inject the CD bar if available
        rank_df_cd = rank_df.copy()
        rank_df_cd["critical_difference"] = cd
        plot_average_ranks(rank_df_cd,
                           save_path=f"{fig_dir}/average_ranks.png")
    if "timing" in figures:
        plot_timing_comparison(df, save_path=f"{fig_dir}/timing_comparison.png")
    if "memory" in figures:
        plot_memory_comparison(df, save_path=f"{fig_dir}/memory_comparison.png")
    if "win_loss" in figures:
        plot_win_loss_matrix(df, metric=metric,
                             save_path=f"{fig_dir}/win_loss_matrix.png")
    if "calibration" in figures:
        plot_calibration_comparison(df, save_path=f"{fig_dir}/calibration_comparison.png")
    if "cd_diagram" in figures:
        plot_critical_difference_diagram(
            rank_df, cd=cd,
            save_path=f"{fig_dir}/cd_diagram.png")
    if "confidence_intervals" in figures:
        plot_confidence_intervals(ci_df, metric=metric.upper().replace("_", "-"),
                                  save_path=f"{fig_dir}/confidence_intervals.png")
    if "significance_heatmap" in figures:
        plot_statistical_significance_heatmap(
            wilcoxon_df, metric=metric.upper().replace("_", "-"),
            save_path=f"{fig_dir}/significance_heatmap.png")
    if "reliability" in figures:
        plot_reliability_diagrams(df, save_path=f"{fig_dir}/reliability_diagrams.png")
    if "errorbars" in figures:
        plot_per_dataset_errorbars(df, metric=metric,
                                   save_path=f"{fig_dir}/per_dataset_errorbars.png")
    if "characteristic" in figures:
        plot_dataset_characteristic_grid(df, meta, metric=metric,
                                         save_path=f"{fig_dir}/dataset_characteristic_grid.png")
    if "roc_curves" in figures or "pr_curves" in figures:
        plot_macro_curves(
            df,
            save_path_roc=f"{fig_dir}/roc_curves.png",
            save_path_pr=f"{fig_dir}/pr_curves.png",
        )


# ---------------------------------------------------------------------------
# Markdown writer
# ---------------------------------------------------------------------------
def write_summary(df: pd.DataFrame, meta: pd.DataFrame,
                  metric: str = "roc_auc",
                  summary_path: Path | str = SUMMARY_PATH) -> None:
    """
    Build a lean, result-only summary report.
    Strictly focuses on who wins which dimension.
    """
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    head   = headline_table(df, metric=metric)
    rank   = average_ranks(df, metric=metric)
    cal    = calibration_table(df)
    timing = timing_table(df)
    memory = memory_table(df)

    n_models   = df["model"].nunique()
    n_datasets = df["dataset"].nunique()
    n_seeds    = df["seed"].nunique() if "seed" in df.columns else 5

    head_sorted = head.sort_values("mean", ascending=False).reset_index(drop=True)
    best   = head_sorted.iloc[0]

    md = []
    P = md.append

    P(f"# 📊 Benchmark Results: {metric.upper().replace('_','-')}\n")

    # 1. The Setup (Brief)
    P("## ⚙️ Run Overview\n")
    P(f"- **Scope**: {n_datasets} Datasets | {n_models} Models | {n_seeds} Seeds\n")
    P(f"- **Overall Winner**: **{best['model']}** ({best['mean']:.4f} mean {metric})\n")

    # 2. Performance
    P("\n## 📈 Performance Ranking\n")
    P("| Rank | Model | Mean Score | 95% CI | n datasets |\n|---|---|---|---|---|")
    for i, r in head_sorted.iterrows():
        P("| {r} | {m} | {mean:.4f} | [{lo:.4f}, {hi:.4f}] | {nd} |".format(
            r=i+1, m=r["model"], mean=r["mean"], lo=r["ci_lower"],
            hi=r["ci_upper"], nd=int(r["n_datasets"]),
        ))
    P("\n![](figures/average_ranks.png)\n")

    # 3. Dimensional Winners
    P("\n## 🏆 Dimensional Winners\n")
    P("Comparing trust (Calibration) and cost (Efficiency):\n")
    P("| Dimension | Winner | Value | Rationale |\n|---|---|---|---|\n")

    # Performance winner
    P(f"| **Predictive Power** | **{best['model']}** | {best['mean']:.4f} | Highest mean {metric} |\n")

    # Calibration winner (lowest ECE)
    if not cal.empty:
        best_cal = cal.sort_values("mean_ece_10").iloc[0]
        P(f"| **Calibration** | **{best_cal['model']}** | {best_cal['mean_ece_10']:.4f} | Lowest Mean ECE-10 |\n")

    # Efficiency winner (lowest total time)
    if not timing.empty:
        best_time = timing.sort_values("total_wall_time_sec").iloc[0]
        P(f"| **Efficiency** | **{best_time['model']}** | {best_time['total_wall_time_sec']:.2f}s | Lowest Total Wall-time |\n")

    # Memory winner (lowest peak)
    if not memory.empty:
        best_mem = memory.sort_values("peak_memory_mb").iloc[0]
        P(f"| **Footprint** | **{best_mem['model']}** | {best_mem['peak_memory_mb']:.0f}MB | Lowest Peak RAM |\n")

    # 4. Deep Dive Pointers
    P("\n## 🔍 Deep Dive Pointers\n")
    P("- **Full Metric Matrix**: `tables/headline_metrics.csv`\n")
    P("- **Per-Dataset Results**: `tables/per_dataset_metrics.csv`\n")
    P("- **Statistical Significance**: `tables/wilcoxon_significance.csv`\n")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("".join(md))
    print(f"  Wrote {summary_path} ({sum(len(x) for x in md)} chars)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", default="roc_auc",
                        help="Headline metric for the markdown (default: roc_auc)")
    parser.add_argument("--figures-only", action="store_true",
                        help="Skip writing summary.md")
    parser.add_argument("--summary-only", action="store_true",
                        help="Skip regenerating figures")
    parser.add_argument("--csv", default=None,
                        help="Path to aggregated results CSV (default: uses experiment name from config)")
    parser.add_argument("--meta", default=META_PATH,
                        help="Path to datasets.yaml")
    args = parser.parse_args()

    # Handle result path based on experiment namespace
    if args.csv:
        csv_file = Path(args.csv)
    else:
        import yaml
        with open("configs/experiment.yaml") as f:
            exp_name = yaml.safe_load(f)["experiment"].get("name", "default_experiment")
        csv_file = Path("experiments/results/aggregated") / exp_name / "results.csv"

    print(f"Loading {csv_file} ...")
    if not csv_file.exists():
        print(f"ERROR: Results file not found at {csv_file}. Please run the benchmark first.")
        sys.exit(1)

    df = pd.read_csv(csv_file)
    print(f"  -> {len(df)} rows, {df['model'].nunique()} models, "
          f"{df['dataset'].nunique()} datasets, "
          f"{df['seed'].nunique() if 'seed' in df.columns else 'NA'} seeds.")

    print(f"Loading {args.meta} ...")
    meta = pd.read_csv(args.meta) if args.meta.endswith(".csv") else None
    if meta is None:
        from tabular_gauge.evaluation.analysis import load_dataset_meta
        meta = load_dataset_meta(args.meta)
    print(f"  -> {len(meta)} datasets in metadata.")

    # Use the experiment name for the figure directory
    exp_name = "default_experiment"
    if not args.csv:
        import yaml
        with open("configs/experiment.yaml") as f:
            exp_name = yaml.safe_load(f)["experiment"].get("name", "default_experiment")
    else:
        # Try to infer exp_name from the path
        parts = csv_file.parts
        if "aggregated" in parts:
            idx = parts.index("aggregated")
            exp_name = parts[idx+1]

    if not args.summary_only:
        # Pass the namespaced directory to the figure generator
        fig_dir = Path("reports/figures") / exp_name
        table_dir = Path("reports/tables") / exp_name
        generate_figures(df, meta, metric=args.metric,
                         figures=None, fig_dir=fig_dir, table_dir=table_dir)
    if not args.figures_only:
        # Save summary.md in the experiment namespace
        summary_file = Path("reports") / exp_name / "summary.md"
        write_summary(df, meta, metric=args.metric, summary_path=summary_file)


if __name__ == "__main__":
    main()
