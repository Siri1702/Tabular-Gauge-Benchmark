"""
LaTeX Table Generator for Publication-Ready Results.

This module generates LaTeX tables from benchmark results for professional reporting.
Supports:
- Main results table (ROC-AUC by dataset)
- Timing comparison table
- Statistical significance table
- Supplementary tables
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def format_number(x: float, decimals: int = 4) -> str:
    """Format number for LaTeX table."""
    if pd.isna(x):
        return "-"
    return f"{x:.{decimals}f}"


def format_number_with_std(mean: float, std: float, decimals: int = 4) -> str:
    """Format mean ± std for LaTeX."""
    if pd.isna(mean) or pd.isna(std):
        return "-"
    return f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"


def generate_main_results_table(
    results_df: pd.DataFrame,
    metric: str = "roc_auc",
    caption: str = "ROC-AUC comparison across datasets",
    label: str = "tab:main_results",
) -> str:
    """
    Generate main results table: model × dataset.

    Returns LaTeX code.
    """
    # Pivot to get models as columns, datasets as rows
    pivot = results_df.pivot_table(
        index="dataset",
        columns="model",
        values=metric,
        aggfunc="mean",
    )

    # Reorder columns by mean performance
    col_order = pivot.mean().sort_values(ascending=False).index.tolist()
    pivot = pivot[col_order]

    # Find best per row
    best_per_row = pivot.idxmax(axis=1)

    # Build LaTeX table
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{tabular}{l" + "r" * len(pivot.columns) + "}")
    lines.append("\\toprule")

    # Header
    header = "Dataset & " + " & ".join(pivot.columns) + " \\\\\_midrule"
    lines.append(header)

    # Rows
    for dataset in pivot.index:
        row_data = []
        for model in pivot.columns:
            val = pivot.loc[dataset, model]
            formatted = format_number(val)

            # Bold the best
            if dataset in best_per_row and best_per_row[dataset] == model:
                formatted = f"\\textbf{{{formatted}}}"

            row_data.append(formatted)

        row_str = f"{dataset.replace('-', ' ')} & " + " & ".join(row_data) + " \\\\"
        lines.append(row_str)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def generate_summary_table(
    results_df: pd.DataFrame,
    metrics: List[str] = None,
    caption: str = "Summary of model performance",
    label: str = "tab:summary",
) -> str:
    """
    Generate summary table with mean ± std for multiple metrics.
    """
    if metrics is None:
        metrics = ["roc_auc", "avg_precision", "f1_macro", "ece_10", "brier_score"]

    # Compute summary
    summary = results_df.groupby("model")[metrics].agg(["mean", "std"])

    # Flatten column names
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]

    # Sort by ROC-AUC
    if "roc_auc_mean" in summary.columns:
        summary = summary.sort_values("roc_auc_mean", ascending=False)

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{tabular}{l" + "r" * len(metrics) + "}")
    lines.append("\\toprule")
    lines.append("Model & " + " & ".join([m.replace("_", " ").title() for m in metrics]) + " \\\\_midrule")

    for model in summary.index:
        row = []
        for metric in metrics:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"

            if mean_col in summary.columns:
                mean_val = summary.loc[model, mean_col]
                std_val = summary.loc[model, std_col]
                formatted = format_number_with_std(mean_val, std_val)
                row.append(formatted)
            else:
                row.append("-")

        lines.append(f"{model} & " + " & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def generate_timing_table(
    results_df: pd.DataFrame,
    caption: str = "Timing comparison (seconds)",
    label: "tab:timing",
) -> str:
    """Generate timing comparison table."""
    timing = results_df.groupby("model").agg({
        "tuning_time_sec": "mean",
        "fit_time_sec": "mean",
        "predict_time_sec": "mean",
        "total_wall_time_sec": "mean",
    })

    # Sort by total time
    timing = timing.sort_values("total_wall_time_sec")

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{tabular}{lrrrr}")
    lines.append("\\toprule")
    lines.append("Model & Tuning & Fit & Predict & Total\\\\\\_midrule")

    for model in timing.index:
        row = timing.loc[model]
        lines.append(
            f"{model} & "
            f"{row['tuning_time_sec']:.1f} & "
            f"{row['fit_time_sec']:.1f} & "
            f"{row['predict_time_sec']:.3f} & "
            f"{row['total_wall_time_sec']:.1f}\\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def generate_statistical_table(
    results_df: pd.DataFrame,
    metric: str = "roc_auc",
    caption: str = "Statistical significance (Wilcoxon tests)",
    label: str = "tab:stats",
) -> str:
    """Generate pairwise statistical test table."""
    from tabular_gauge.evaluation.statistical import wilcoxon_pairwise

    wilcoxon_df = wilcoxon_pairwise(results_df, metric=metric)

    if wilcoxon_df.empty:
        return "Statistical tests not available."

    # Filter significant pairs
    sig_pairs = wilcoxon_df[wilcoxon_df["significant"]]

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{tabular}{llrr}")
    lines.append("\\toprule")
    lines.append("Model A & Model B & p-value & Significant\\\\\\_midrule")

    for _, row in sig_pairs.iterrows():
        sig_mark = "$\\dagger$" if row["significant"] else ""
        lines.append(f"{row['model_a']} & {row['model_b']} & {row['p_value']:.4f} & {sig_mark}\\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def generate_ranking_table(
    results_df: pd.DataFrame,
    metric: str = "roc_auc",
    caption: str = "Model rankings by Friedman test",
    label: str = "tab:ranking",
) -> str:
    """Generate ranking table from Friedman test."""
    from tabular_gauge.evaluation.statistical import average_rank_table, friedman_test

    rank_df = average_rank_table(results_df, metric=metric)
    friedman = friedman_test(results_df, metric=metric)

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{tabular}{lrr}")
    lines.append("\\toprule")
    lines.append("Model & Avg. Rank\\\\\\_midrule")

    for _, row in rank_df.sort_values("average_rank").iterrows():
        lines.append(f"{row['model']} & {row['average_rank']:.2f}\\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"\\smallskip\\\\\n")
    lines.append(f"\\textit{{Friedman test: $\\chi^2 = {friedman.get('statistic', 'N/A'):.2f}$, $p = {friedman.get('p_value', 'N/A'):.4f}$}}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def generate_all_latex_tables(
    results_df: pd.DataFrame,
    output_dir: str = "reports/tables",
):
    """Generate all LaTeX tables and save to files."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    tables = {
        "main_results.tex": generate_main_results_table(results_df),
        "summary.tex": generate_summary_table(results_df),
        "timing.tex": generate_timing_table(results_df),
        "ranking.tex": generate_ranking_table(results_df),
    }

    for filename, content in tables.items():
        with open(f"{output_dir}/{filename}", "w") as f:
            f.write(content)

        print(f"  Generated {filename}")

    return tables


# =============================================================================
# Experiment Tracking with MLflow
# =============================================================================

def setup_mlflow_tracking(experiment_name: str = "Tabular-Gauge-Benchmark"):
    """Setup MLflow for experiment tracking."""
    import mlflow

    # Set tracking URI (local by default)
    mlflow.set_experiment(experiment_name)

    return mlflow


def log_run_to_mlflow(
    mlflow,
    run_params: Dict,
    metrics: Dict,
    tags: Dict = None,
):
    """Log a single run to MLflow."""
    with mlflow.start_run(run_name=run_params.get("dataset")):
        # Log parameters
        mlflow.log_params(run_params)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log tags
        if tags:
            mlflow.set_tags(tags)


def log_benchmark_results(
    results_df: pd.DataFrame,
    experiment_name: str = "Tabular-Gauge-Benchmark",
):
    """Log all benchmark results to MLflow."""
    mlflow = setup_mlflow_tracking(experiment_name)

    # Group by model and dataset
    for (model, dataset), group in results_df.groupby(["model", "dataset"]):
        # Compute mean metrics
        mean_metrics = group.mean(numeric_only=True).to_dict()

        # Log as a run
        log_run_to_mlflow(
            mlflow,
            run_params={"model": model, "dataset": dataset},
            metrics=mean_metrics,
            tags={"model_type": model, "benchmark": "Tabular-Gauge"},
        )

    print(f"  Logged {len(results_df)} runs to MLflow")


if __name__ == "__main__":
    # Demo: Generate tables from sample data
    print("LaTeX Table Generation")
    print("=" * 40)

    # Try to load results
    try:
        df = pd.read_csv("experiments/results/aggregated/results.csv")
        print(f"Loaded {len(df)} results")

        # Generate tables
        tables = generate_all_latex_tables(df)
        print("\nGenerated tables:")
        for name in tables.keys():
            print(f"  - {name}")

    except FileNotFoundError:
        print("No results found. Run benchmark first.")