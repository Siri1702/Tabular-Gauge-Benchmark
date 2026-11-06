"""
Orchestrator for generating few-shot learning reports.
Transforms raw few-shot CSV results into publication-ready figures and summaries.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

# Make `src` importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tabular_gauge.evaluation.analysis import (
    average_ranks,
    nemenyi_table,
    wilcoxon_table,
    bootstrap_ci,
)
from tabular_gauge.viz.few_shot import (
    plot_learning_curves,
    plot_efficiency_scaling,
    plot_stability_heatmap,
)

FIG_DIR_BASE = "reports/few_shot/figures"
TABLE_DIR_BASE = "reports/few_shot/tables"
SUMMARY_DIR_BASE = "reports/few_shot"

def generate_few_shot_reports(csv_path: Path | str, metric: str = "roc_auc"):
    """
    Generate the complete few-shot analysis suite for a given results CSV.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"ERROR: CSV file not found at {csv_path}")
        return

    # Infer experiment name from path.
    # Results are saved as: experiments/results/few_shot/[exp_name]/all_few_shot_results.csv
    parts = csv_path.parts
    if "few_shot" in parts:
        idx = parts.index("few_shot")
        if len(parts) > idx + 1:
            exp_name = parts[idx + 1]
        else:
            exp_name = "default_few_shot"
    else:
        exp_name = "default_few_shot"

    # Define namespaced directories
    fig_dir = Path(FIG_DIR_BASE) / exp_name
    table_dir = Path(TABLE_DIR_BASE) / exp_name
    summary_dir = Path(SUMMARY_DIR_BASE) / exp_name

    for d in [fig_dir, table_dir, summary_dir]:
        d.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f"Generating few-shot report for {exp_name} using metric: {metric}...")

    # ------------------------------------------------------------------
    # 1. Visuals: Learning Curves & Scalability
    # ------------------------------------------------------------------
    print("  -> Generating learning curves...")
    plot_learning_curves(df, metric=metric, save_path=fig_dir / f"learning_curves_{metric}.png")

    if "avg_precision" in df.columns or "pr_auc" in df.columns:
        pr_metric = "avg_precision" if "avg_precision" in df.columns else "pr_auc"
        plot_learning_curves(df, metric=pr_metric, save_path=fig_dir / f"learning_curves_{pr_metric}.png")

    print("  -> Generating efficiency scaling plots...")
    if "fit_time_sec" in df.columns:
        plot_efficiency_scaling(df, metric="fit_time_sec", save_path=fig_dir / "time_scaling.png")
    if "peak_memory_mb" in df.columns:
        plot_efficiency_scaling(df, metric="peak_memory_mb", save_path=fig_dir / "memory_scaling.png")

    print("  -> Generating stability heatmap...")
    plot_stability_heatmap(df, metric=metric, save_path=fig_dir / "stability_heatmap.png")

    # ------------------------------------------------------------------
    # 2. Analytical Tables (Size-Specific)
    # ------------------------------------------------------------------
    print("  -> Generating analytical tables...")

    # Get unique training sizes
    sizes = sorted(df["train_size"].unique(), key=lambda x: (x is None, x if isinstance(x, (int, float)) else 99999))

    # a) Global Summary Table (Mean across sizes and seeds)
    global_summary = df.groupby("model")[metric].mean().sort_values(ascending=False).reset_index()
    global_summary.to_csv(table_dir / "global_few_shot_summary.csv", index=False)

    # b) Size-Specific Analysis
    # We create a table: Rows=Size, Cols=Model, Value=Mean Metric
    size_pivot = df.pivot_table(index="train_size", columns="model", values=metric, aggfunc="mean")
    # Sort index to be logical (50, 100... Full)
    size_pivot = size_pivot.reindex(sizes)
    size_pivot.to_csv(table_dir / "performance_by_size.csv")

    # c) Stability Table (Std Dev across seeds)
    stability_pivot = df.pivot_table(index="train_size", columns="model", values=metric, aggfunc="std")
    stability_pivot = stability_pivot.reindex(sizes)
    stability_pivot.to_csv(table_dir / "stability_by_size.csv")

    # ------------------------------------------------------------------
    # 3. Summary Markdown
    # ------------------------------------------------------------------
    print("  -> Writing summary.md...")
    summary_path = summary_dir / "summary.md"

    # Calculate the "Winner" at the smallest size vs the largest size
    min_size = sizes[0]
    max_size = sizes[-1]

    best_small = df[df["train_size"] == min_size].groupby("model")[metric].mean().idxmax()
    best_large = df[df["train_size"] == max_size].groupby("model")[metric].mean().idxmax()

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"# Few-Shot Learning Analysis: {exp_name}\n\n")
        f.write(f"This report analyzes model performance as the training set size increases from {min_size} to {max_size}.\n\n")

        f.write("## 📈 Key Findings\n")
        f.write(f"- **Smallest Regime ({min_size} samples)**: The top performer is **{best_small}**.\n")
        f.write(f"- **Largest Regime ({max_size})**: The top performer is **{best_large}**.\n")

        if best_small != best_large:
            f.write(f"- **Trend**: There is a shift in leadership from {best_small} to {best_large} as more data is provided.\n")
        else:
            f.write(f"- **Trend**: {best_small} maintains dominance across all sampled training sizes.\n")

        f.write("\n## 🖼️ Visualizations\n")
        f.write(f"- **Learning Curves**: `figures/learning_curves_{metric}.png`\n")
        f.write(f"- **Efficiency Scaling**: `figures/time_scaling.png` and `figures/memory_scaling.png`\n")
        f.write(f"- **Stability Heatmap**: `figures/stability_heatmap.png`\n")

        f.write("\n## 📊 Data Tables\n")
        f.write(f"- Detailed performance by size: `tables/performance_by_size.csv`\n")
        f.write(f"- Seed-stability analysis: `tables/stability_by_size.csv`\n")

    print(f"\n✅ Few-Shot Report Complete! Saved to {summary_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate reports for few-shot results")
    parser.add_argument("--csv", required=True, help="Path to the few-shot aggregated results CSV")
    parser.add_argument("--metric", default="roc_auc", help="Headline metric (default: roc_auc)")
    args = parser.parse_args()

    generate_few_shot_reports(csv_path=args.csv, metric=args.metric)
