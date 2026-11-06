"""
Learning Curves Analysis - Performance vs Training Set Size.

This module provides comprehensive learning curve analysis:
- Tests models at multiple training set sizes (50, 100, 200, 500, 1000, full)
- Identifies crossover points where model rankings change
- Provides data efficiency analysis
- Generates professional visualizations
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from tabular_gauge.data.loader import DataLoader
from tabular_gauge.models.tabpfn_model import TabPFNWrapper
from tabular_gauge.models.xgboost_model import XGBoostWrapper
from tabular_gauge.models.lightgbm_model import LightGBMWrapper
from tabular_gauge.models.catboost_model import CatBoostWrapper
from tabular_gauge.models.mlp_model import MLPWrapper
from tabular_gauge.evaluation.metrics import compute_metrics


# Standard training sizes to test
DEFAULT_TRAIN_SIZES = [50, 100, 200, 500, 1000, None]  # None = use all


def stratified_subsample(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform stratified subsampling to reduce training set size.
    Maintains class balance.
    """
    if n_samples is None or n_samples >= len(y):
        return X, y

    np.random.seed(random_state)

    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]

    # Calculate proportions
    pos_ratio = len(pos_indices) / len(y)
    n_pos = min(len(pos_indices), int(n_samples * pos_ratio))
    n_neg = n_samples - n_pos

    # Sample from each class
    if n_pos > 0:
        selected_pos = np.random.choice(pos_indices, n_pos, replace=False)
    else:
        selected_pos = np.array([], dtype=int)

    if n_neg > 0:
        selected_neg = np.random.choice(neg_indices, n_neg, replace=False)
    else:
        selected_neg = np.array([], dtype=int)

    selected_indices = np.sort(np.concatenate([selected_pos, selected_neg]))

    return X[selected_indices], y[selected_indices]


def run_learning_curve_experiment(
    dataset_cfg: dict,
    exp_cfg: dict,
    train_sizes: List[Optional[int]] = DEFAULT_TRAIN_SIZES,
    n_seeds: int = 3,
) -> pd.DataFrame:
    """
    Run learning curve experiment on a single dataset.

    Tests each model at multiple training set sizes.
    """
    loader = DataLoader(
        test_size=exp_cfg.get("test_size", 0.2),
        random_state=42,  # Fixed split
    )
    dataset = loader.load_openml(
        dataset_id=dataset_cfg["id"],
        dataset_name=dataset_cfg["name"],
    )

    results = []

    for train_size in tqdm(train_sizes, desc=f"  Train sizes"):
        # Subsample if needed
        if train_size is not None:
            X_train_sub, y_train_sub = stratified_subsample(
                dataset.X_train, dataset.y_train,
                n_samples=train_size,
            )
        else:
            X_train_sub, y_train_sub = dataset.X_train, dataset.y_train

        actual_size = len(y_train_sub)

        # Run each model
        models = [
            TabPFNWrapper(random_state=0),
            XGBoostWrapper(n_trials=exp_cfg.get("n_optuna_trials", 20), random_state=0),
            LightGBMWrapper(n_trials=exp_cfg.get("n_optuna_trials", 20), random_state=0),
            CatBoostWrapper(n_trials=exp_cfg.get("n_optuna_trials", 20), random_state=0),
            MLPWrapper(n_trials=exp_cfg.get("n_optuna_trials", 15), random_state=0),
        ]

        for model in models:
            try:
                result = model.run(
                    X_train_sub, y_train_sub,
                    dataset.X_test, dataset.y_test,
                    dataset_name=f"{dataset_cfg['name']}_N{train_size or 'full'}",
                    seed=0,
                )
                metrics = compute_metrics(result)
                metrics["train_size"] = train_size if train_size else "full"
                metrics["actual_n_train"] = actual_size
                results.append(metrics)
            except Exception as e:
                print(f"    Error ({model.name}): {e}")

    return pd.DataFrame(results)


def analyze_crossover_points(
    lc_results: pd.DataFrame,
    metric: str = "roc_auc",
) -> pd.DataFrame:
    """
    Identify crossover points where model rankings change.

    Returns DataFrame with crossover events.
    """
    # Get mean performance per train_size and model
    summary = lc_results.groupby(["train_size", "model"])[metric].mean().reset_index()

    # Pivot
    pivot = summary.pivot(index="train_size", columns="model", values=metric)
    pivot = pivot.sort_index()

    crossovers = []

    sizes = list(pivot.index)
    for i in range(1, len(sizes)):
        prev_size = sizes[i - 1]
        curr_size = sizes[i]

        # Check all pairs for crossovers
        for model_a in pivot.columns:
            for model_b in pivot.columns:
                if model_a >= model_b:
                    continue

                prev_a = pivot.loc[prev_size, model_a]
                prev_b = pivot.loc[prev_size, model_b]
                curr_a = pivot.loc[curr_size, model_a]
                curr_b = pivot.loc[curr_size, model_b]

                # Check for crossover
                if (prev_a > prev_b and curr_a < curr_b) or \
                   (prev_a < prev_b and curr_a > curr_b):
                    crossovers.append({
                        "prev_train_size": prev_size,
                        "curr_train_size": curr_size,
                        "model_a": model_a,
                        "model_b": model_b,
                        f"{model_a}_prev": prev_a,
                        f"{model_b}_prev": prev_b,
                        f"{model_a}_curr": curr_a,
                        f"{model_b}_curr": curr_b,
                        "crossover": True,
                    })

    return pd.DataFrame(crossovers)


def compute_data_efficiency(
    lc_results: pd.DataFrame,
    metric: str = "roc_auc",
) -> pd.DataFrame:
    """
    Compute data efficiency: performance per training sample.

    Useful for understanding which models are most data-efficient.
    """
    # Convert "full" to actual number
    lc_results = lc_results.copy()
    lc_results["train_size_num"] = lc_results["train_size"].apply(
        lambda x: x if isinstance(x, (int, float)) else 1000  # Approx for "full"
    )

    # Compute efficiency
    efficiency = lc_results.groupby(["model", "train_size_num"])[metric].mean().reset_index()
    efficiency["efficiency"] = efficiency[metric] / efficiency["train_size_num"]

    return efficiency.pivot(index="train_size_num", columns="model", values="efficiency")


def plot_learning_curves(
    lc_results: pd.DataFrame,
    metric: str = "roc_auc",
    save_path: str = "reports/figures/learning_curves.png",
):
    """Generate learning curve visualization."""
    # Summary statistics
    summary = lc_results.groupby(["train_size", "model"])[metric].agg(["mean", "std"]).reset_index()

    # Convert train_size to numeric
    summary = summary.copy()
    summary["train_size_num"] = summary["train_size"].apply(
        lambda x: x if isinstance(x, (int, float)) else 2000  # Placeholder for "full"
    )
    summary = summary.sort_values("train_size_num")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Get unique datasets
    datasets = lc_results["dataset"].str.replace("_N.*", "", regex=True).unique()

    colors = {"TabPFN": "#FF6B6B", "XGBoost": "#4ECDC4", "LightGBM": "#45B7D1",
              "CatBoost": "#96CEB4", "MLP": "#DDA0DD"}

    for idx, dataset in enumerate(datasets[:6]):
        ax = axes[idx]
        dataset_data = summary[summary["dataset"].str.startswith(dataset.replace("_N", ""))]

        for model in dataset_data["model"].unique():
            model_data = dataset_data[dataset_data["model"] == model]
            ax.errorbar(
                model_data["train_size_num"],
                model_data["mean"],
                yerr=model_data["std"],
                marker="o",
                label=model,
                color=colors.get(model, "#333"),
                capsize=3,
            )

        ax.set_xlabel("Training Set Size")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(dataset.replace("_N", "").replace("-", " "))
        ax.set_xscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved learning curves to {save_path}")


def plot_crossover_analysis(
    crossovers: pd.DataFrame,
    save_path: str = "reports/figures/crossovers.png",
):
    """Visualize crossover points."""
    if crossovers.empty:
        print("  No crossovers found")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for _, row in crossovers.iterrows():
        ax.plot(
            [row["prev_train_size"], row["curr_train_size"]],
            [1, 0],
            marker="o",
            label=f"{row['model_a']} vs {row['model_b']}",
        )

    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Relative Performance")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved crossover analysis to {save_path}")


def run_full_learning_curve_analysis(
    dataset_names: List[str] = None,
    train_sizes: List[Optional[int]] = DEFAULT_TRAIN_SIZES,
    n_seeds: int = 3,
    output_dir: str = "experiments/results/learning_curves",
):
    """Run complete learning curve analysis across multiple datasets."""
    # Load configs
    with open("configs/datasets.yaml") as f:
        dataset_cfgs = yaml.safe_load(f)["datasets"]

    with open("configs/experiment.yaml") as f:
        exp_cfg = yaml.safe_load(f)["experiment"]

    # Filter
    if dataset_names:
        dataset_cfgs = [d for d in dataset_cfgs if d["name"] in dataset_names]

    # Ensure dirs
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    all_results = []

    for dataset_cfg in dataset_cfgs[:5]:  # Use first 5 for speed
        print(f"\n{'='*50}")
        print(f"Learning Curve: {dataset_cfg['name']}")
        print(f"{'='*50}")

        results_df = run_learning_curve_experiment(
            dataset_cfg, exp_cfg, train_sizes, n_seeds
        )

        if not results_df.empty:
            results_df.to_csv(f"{output_dir}/lc_{dataset_cfg['name']}.csv", index=False)
            all_results.append(results_df)

    # Combine
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(f"{output_dir}/all_learning_curves.csv", index=False)

        # Visualizations
        plot_learning_curves(combined)

        # Crossovers
        crossovers = analyze_crossover_points(combined)
        if not crossovers.empty:
            crossovers.to_csv(f"{output_dir}/crossovers.csv", index=False)
            plot_crossover_analysis(crossovers)

        # Summary
        print("\n--- Learning Curve Summary ---")
        summary = combined.groupby(["train_size", "model"])["roc_auc"].mean()
        print(summary.unstack().to_string())

        return combined

    return pd.DataFrame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run learning curve analysis")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--train_sizes", nargs="+", type=int, default=None)
    parser.add_argument("--n_seeds", type=int, default=3)
    args = parser.parse_args()

    train_sizes = args.train_sizes if args.train_sizes else DEFAULT_TRAIN_SIZES

    run_full_learning_curve_analysis(
        dataset_names=args.datasets,
        train_sizes=train_sizes,
        n_seeds=args.n_seeds,
    )