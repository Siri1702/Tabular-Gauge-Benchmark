"""
Few-Shot Learning Analysis for Tabular Foundation Models.

This module systematically evaluates model performance across different training
set sizes to understand:
1. When TabPFN stops being competitive vs tuned GBDT
2. Crossover points where performance changes dramatically
3. Data efficiency characteristics of each model

Key Research Question: At what training set size does TabPFN stop being competitive?
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from tabular_gauge.data.loader import DataLoader
from tabular_gauge.models.tabpfn_model import TabPFNWrapper
from tabular_gauge.models.xgboost_model import XGBoostWrapper
from tabular_gauge.models.lightgbm_model import LightGBMWrapper
from tabular_gauge.models.catboost_model import CatBoostWrapper
from tabular_gauge.models.mlp_model import MLPWrapper
from tabular_gauge.evaluation.metrics import compute_metrics
from tabular_gauge.models.base import RunResult


# Training sizes to test (None = use full training data)
TRAIN_SIZES = [50, 100, 200, 500, 1000, None]  # None = use all available


def subsample_train_data(X_train: np.ndarray, y_train: np.ndarray,
                         n_samples: int, random_state: int = 42) -> tuple:
    """
    Stratified subsample of training data.
    Returns (X_subsampled, y_subsampled) with n_samples.
    """
    if n_samples is None:
        return X_train, y_train

    n_samples = min(n_samples, len(y_train))

    # Stratified sampling
    np.random.seed(random_state)
    pos_indices = np.where(y_train == 1)[0]
    neg_indices = np.where(y_train == 0)[0]

    n_pos = min(len(pos_indices), int(n_samples * (len(pos_indices) / len(y_train))))
    n_neg = n_samples - n_pos

    if n_pos > 0 and n_neg > 0:
        selected_pos = np.random.choice(pos_indices, n_pos, replace=False)
        selected_neg = np.random.choice(neg_indices, n_neg, replace=False)
        selected_indices = np.concatenate([selected_pos, selected_neg])
    elif n_pos > 0:
        selected_indices = np.random.choice(pos_indices, n_samples, replace=False)
    else:
        selected_indices = np.random.choice(neg_indices, n_samples, replace=False)

    selected_indices = np.sort(selected_indices)

    return X_train[selected_indices], y_train[selected_indices]


def run_few_shot_analysis(
    dataset_cfg: dict,
    exp_cfg: dict,
    train_sizes: List[Optional[int]] = TRAIN_SIZES,
    n_seeds: int = 3,
) -> pd.DataFrame:
    """
    Run few-shot analysis on a single dataset.
    Tests each model at multiple training set sizes.

    Returns DataFrame with columns:
    - model, dataset, train_size, seed, roc_auc, pr_auc, f1_macro, etc.
    """
    loader = DataLoader(
        test_size=exp_cfg.get("test_size", 0.2),
        random_state=42,
    )
    dataset = loader.load_openml(
        dataset_id=dataset_cfg["id"],
        dataset_name=dataset_cfg["name"],
    )

    all_results = []

    for train_size in tqdm(train_sizes, desc=f"Train sizes for {dataset_cfg['name']}"):
        for seed in range(n_seeds):
            # Subsample training data if needed
            if train_size is not None:
                X_train_sub, y_train_sub = subsample_train_data(
                    dataset.X_train, dataset.y_train,
                    n_samples=train_size,
                    random_state=seed
                )
            else:
                X_train_sub, y_train_sub = dataset.X_train, dataset.y_train

            # Build models
            models = [
                TabPFNWrapper(random_state=seed),
                XGBoostWrapper(n_trials=exp_cfg.get("n_optuna_trials", 20),  # Fewer trials for speed
                              random_state=seed),
                LightGBMWrapper(n_trials=exp_cfg.get("n_optuna_trials", 20),
                               random_state=seed),
                CatBoostWrapper(n_trials=exp_cfg.get("n_optuna_trials", 20),
                               random_state=seed),
                MLPWrapper(n_trials=exp_cfg.get("n_optuna_trials", 15),
                          random_state=seed),
            ]

            for model in models:
                try:
                    result = model.run(
                        X_train_sub, y_train_sub,
                        dataset.X_test, dataset.y_test,
                        dataset_name=f"{dataset_cfg['name']}_N{train_size}",
                        seed=seed,
                    )
                    metrics = compute_metrics(result)
                    metrics["train_size"] = train_size if train_size else "full"
                    all_results.append(metrics)
                except Exception as e:
                    print(f"  Error with {model.name} at N={train_size}: {e}")

    return pd.DataFrame(all_results)


def find_crossover_points(results_df: pd.DataFrame, metric: str = "roc_auc") -> pd.DataFrame:
    """
    Find crossover points where model performance rankings change.
    Identifies critical training sizes where one model surpasses another.
    """
    # Group by train_size and compute mean metrics
    summary = results_df.groupby(["train_size", "model"])[metric].mean().reset_index()

    # Pivot to get models as columns
    pivot = summary.pivot(index="train_size", columns="model", values=metric)
    pivot = pivot.sort_index()

    # Find where TabPFN is best vs XGBoost
    crossovers = []
    train_size_order = pivot.index.tolist()

    for i in range(1, len(train_size_order)):
        prev_size = train_size_order[i-1]
        curr_size = train_size_order[i]

        if "TabPFN" in pivot.columns and "XGBoost" in pivot.columns:
            prev_tabpfn = pivot.loc[prev_size, "TabPFN"]
            prev_xgboost = pivot.loc[prev_size, "XGBoost"]
            curr_tabpfn = pivot.loc[curr_size, "TabPFN"]
            curr_xgboost = pivot.loc[curr_size, "XGBoost"]

            # Check for crossover
            if (prev_tabpfn > prev_xgboost and curr_tabpfn < curr_xgboost) or \
               (prev_tabpfn < prev_xgboost and curr_tabpfn > curr_xgboost):
                crossovers.append({
                    "train_size_prev": prev_size,
                    "train_size_curr": curr_size,
                    "tabpfn_prev": prev_tabpfn,
                    "tabpfn_curr": curr_tabpfn,
                    "xgboost_prev": prev_xgboost,
                    "xgboost_curr": curr_xgboost,
                })

    return pd.DataFrame(crossovers)


def plot_learning_curves(
    results_df: pd.DataFrame,
    metric: str = "roc_auc",
    save_path: str = "reports/figures/learning_curves.png",
    dataset_filter: str = None,
):
    """
    Plot learning curves (performance vs training set size) for each model.
    """
    if dataset_filter:
        results_df = results_df[results_df["dataset"].str.startswith(dataset_filter)]

    # Convert train_size to numeric for plotting
    results_df = results_df.copy()
    results_df["train_size_num"] = results_df["train_size"].apply(
        lambda x: x if isinstance(x, (int, float)) else float("inf")
    )

    # Filter out "full" for cleaner plots (or keep at end)

    # Group by train_size and model
    summary = results_df.groupby(["train_size_num", "model"])[metric].agg(["mean", "std"]).reset_index()

    # Get unique train sizes sorted
    train_sizes = sorted(summary["train_size_num"].unique())

    # Create plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Get unique datasets
    datasets = results_df["dataset"].str.replace("_N.*", "", regex=True).unique()

    for idx, dataset in enumerate(datasets[:6]):
        ax = axes[idx]
        dataset_data = summary[summary["dataset"].str.startswith(dataset.replace("_N", ""))]

        for model in dataset_data["model"].unique():
            model_data = dataset_data[dataset_data["model"] == model].sort_values("train_size_num")
            ax.errorbar(
                model_data["train_size_num"],
                model_data["mean"],
                yerr=model_data["std"],
                marker="o",
                label=model,
                capsize=3,
            )

        ax.set_xlabel("Training Set Size")
        ax.set_ylabel(metric.upper())
        ax.set_title(dataset.replace("_N", "").replace("-", " "))
        ax.set_xscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Learning Curves: {metric.upper()} vs Training Set Size", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved learning curves to {save_path}")


def plot_model_comparison_by_size(
    results_df: pd.DataFrame,
    metric: str = "roc_auc",
    save_path: str = "reports/figures/few_shot_comparison.png",
):
    """
    Plot model comparison at each training size.
    Shows which model wins at each data regime.
    """
    # Convert train_size
    results_df = results_df.copy()
    results_df["train_size_num"] = results_df["train_size"].apply(
        lambda x: x if isinstance(x, (int, float)) else float("inf")
    )

    # Group by train_size and model
    summary = results_df.groupby(["train_size_num", "model"])[metric].mean().reset_index()

    # Pivot for plotting
    pivot = summary.pivot(index="train_size_num", columns="model", values=metric)
    pivot = pivot.sort_index()

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(pivot.index))
    width = 0.15
    models = pivot.columns

    for i, model in enumerate(models):
        offset = (i - len(models)/2) * width
        bars = ax.bar(x + offset, pivot[model], width, label=model, alpha=0.8)

    ax.set_xlabel("Training Set Size")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Model Performance by Training Set Size ({metric.upper()})")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in pivot.index])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved few-shot comparison to {save_path}")


def compute_data_efficiency(results_df: pd.DataFrame, metric: str = "roc_auc") -> pd.DataFrame:
    """
    Compute data efficiency metrics for each model.
    Measures performance per training sample used.
    """
    results_df = results_df.copy()
    results_df["train_size_num"] = results_df["train_size"].apply(
        lambda x: x if isinstance(x, (int, float)) else 1000  # Estimate for "full"
    )

    # Compute performance per sample
    efficiency = results_df.groupby(["model", "train_size_num"])[metric].mean().reset_index()
    efficiency["performance_per_sample"] = efficiency[metric] / efficiency["train_size_num"]

    return efficiency.pivot(index="train_size_num", columns="model", values="performance_per_sample")


def run_few_shot_benchmark(
    dataset_names: List[str] = None,
    train_sizes: List[Optional[int]] = TRAIN_SIZES,
    n_seeds: int = 3,
    output_dir: str = "experiments/results/few_shot",
):
    """
    Run complete few-shot benchmark across multiple datasets.
    """
    # Load configs
    with open("configs/datasets.yaml") as f:
        dataset_cfgs = yaml.safe_load(f)["datasets"]

    with open("configs/experiment.yaml") as f:
        exp_cfg = yaml.safe_load(f)["experiment"]

    # Filter datasets
    if dataset_names:
        dataset_cfgs = [d for d in dataset_cfgs if d["name"] in dataset_names]

    # Ensure output dir exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    all_results = []

    for dataset_cfg in dataset_cfgs:
        print(f"\n{'='*60}")
        print(f"Few-shot: {dataset_cfg['name']}")
        print(f"{'='*60}")

        results_df = run_few_shot_analysis(
            dataset_cfg, exp_cfg, train_sizes, n_seeds
        )

        if not results_df.empty:
            # Save dataset results
            results_df.to_csv(f"{output_dir}/few_shot_{dataset_cfg['name']}.csv", index=False)
            all_results.append(results_df)

    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_df.to_csv(f"{output_dir}/all_few_shot_results.csv", index=False)

        # Generate visualizations
        print("\n--- Generating Visualizations ---")

        # Learning curves
        try:
            plot_learning_curves(combined_df, metric="roc_auc")
        except Exception as e:
            print(f"  Warning: Could not generate learning curves: {e}")

        # Model comparison by size
        try:
            plot_model_comparison_by_size(combined_df, metric="roc_auc")
        except Exception as e:
            print(f"  Warning: Could not generate comparison: {e}")

        # Find crossover points
        crossovers = find_crossover_points(combined_df, metric="roc_auc")
        if not crossovers.empty:
            crossovers.to_csv(f"{output_dir}/crossover_points.csv", index=False)
            print(f"\n--- Crossover Points ---")
            print(crossovers.to_string(index=False))

        # Summary statistics
        print("\n--- Few-Shot Summary ---")
        summary = combined_df.groupby(["train_size", "model"])[["roc_auc", "pr_auc", "f1_macro"]].mean()
        print(summary.to_string())

        return combined_df

    return pd.DataFrame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run few-shot learning analysis")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--train_sizes", nargs="+", type=int, default=None)
    parser.add_argument("--n_seeds", type=int, default=3)
    args = parser.parse_args()

    train_sizes = args.train_sizes if args.train_sizes else TRAIN_SIZES

    run_few_shot_benchmark(
        dataset_names=args.datasets,
        train_sizes=train_sizes,
        n_seeds=args.n_seeds,
    )