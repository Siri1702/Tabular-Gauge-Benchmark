"""
Optimized Crossover Point Analysis - Using Existing Larger Datasets

Uses existing benchmark datasets that are larger (N > 1000) to find
where XGBoost beats TabPFN without downloading new datasets.

Training sizes: 100, 250, 500, 750, 1000, 1500 (for spambase/phoneme), full
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

from tabular_gauge.data.loader import DataLoader
from tabular_gauge.models.tabpfn_model import TabPFNWrapper
from tabular_gauge.models.xgboost_model import XGBoostWrapper
from tabular_gauge.models.lightgbm_model import LightGBMWrapper
from tabular_gauge.models.catboost_model import CatBoostWrapper
from tabular_gauge.evaluation.metrics import compute_metrics


# Datasets with N > 1000 for crossover analysis
# Note: max training size is limited by dataset size
CROSSOVER_DATASETS = [
    {"id": 44, "name": "spambase", "domain": "nlp-derived", "n": 4601, "p": 57},
    {"id": 1489, "name": "phoneme", "domain": "audio-derived", "n": 5404, "p": 5},
    {"id": 1487, "name": "ozone-level", "domain": "environment", "n": 2536, "p": 72},
    {"id": 1067, "name": "kc1", "domain": "software", "n": 2109, "p": 21},
    {"id": 1050, "name": "sick", "domain": "healthcare", "n": 3772, "p": 37},
    {"id": 40536, "name": "australian", "domain": "finance", "n": 690, "p": 14},
]

# Finer training sizes to find crossover
# Will adapt based on each dataset's max size
BASE_TRAIN_SIZES = [50, 100, 250, 500, 750, 1000, 1500, 2000]


def get_adaptive_sizes(dataset_n):
    """Get training sizes that work for this dataset."""
    sizes = []
    for s in BASE_TRAIN_SIZES:
        if s < dataset_n * 0.8:  # Leave 20% for test
            sizes.append(s)
    sizes.append(None)  # Full dataset
    return sizes


def stratified_subsample(X, y, n_samples, random_state=42):
    """Stratified subsample maintaining class balance."""
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = X

    y_array = np.asarray(y)

    if n_samples is None or n_samples >= len(y_array):
        return X_array, y_array

    np.random.seed(random_state)

    pos_indices = np.where(y_array == 1)[0]
    neg_indices = np.where(y_array == 0)[0]

    pos_ratio = len(pos_indices) / len(y_array)
    n_pos = min(len(pos_indices), int(n_samples * pos_ratio))
    n_neg = n_samples - n_pos

    if n_pos > 0:
        selected_pos = np.random.choice(pos_indices, n_pos, replace=False)
    else:
        selected_pos = np.array([], dtype=int)

    if n_neg > 0:
        selected_neg = np.random.choice(neg_indices, n_neg, replace=False)
    else:
        selected_neg = np.array([], dtype=int)

    selected_indices = np.sort(np.concatenate([selected_pos, selected_neg]))
    return X_array[selected_indices], y_array[selected_indices]


def run_dataset_analysis(dataset_cfg, train_sizes, seed=0):
    """Run analysis on one dataset."""
    print(f"\n{'='*50}")
    print(f"Dataset: {dataset_cfg['name']} (N={dataset_cfg['n']:,}, P={dataset_cfg['p']})")
    print(f"{'='*50}")
    print(f"Training sizes to test: {[s if s else 'full' for s in train_sizes]}")

    loader = DataLoader(test_size=0.2, random_state=42)
    dataset = loader.load_openml(dataset_cfg["id"], dataset_cfg["name"])

    results = []

    for train_size in tqdm(train_sizes, desc="  Training sizes"):
        # Subsample
        if train_size is not None:
            X_train_sub, y_train_sub = stratified_subsample(
                dataset.X_train, dataset.y_train, n_samples=train_size, random_state=seed
            )
        else:
            X_train_sub, y_train_sub = dataset.X_train, dataset.y_train

        actual_size = len(y_train_sub)

        # Models to test
        models = [
            ("TabPFN", TabPFNWrapper(random_state=seed)),
            ("XGBoost", XGBoostWrapper(n_trials=15, random_state=seed)),
            ("LightGBM", LightGBMWrapper(n_trials=15, random_state=seed)),
            ("CatBoost", CatBoostWrapper(n_trials=15, random_state=seed)),
        ]

        for model_name, model in models:
            try:
                result = model.run(
                    X_train_sub, y_train_sub,
                    dataset.X_test, dataset.y_test,
                    dataset_name=f"{dataset_cfg['name']}_N{train_size}",
                    seed=seed,
                )
                metrics = compute_metrics(result)
                metrics["model"] = model_name
                metrics["train_size"] = train_size if train_size else "full"
                metrics["actual_n_train"] = actual_size
                metrics["domain"] = dataset_cfg["domain"]
                metrics["dataset"] = dataset_cfg["name"]
                results.append(metrics)

                print(f"    N={train_size or 'full':>5} | {model_name:10s} | AUC={metrics.get('roc_auc', 0):.4f}")

            except Exception as e:
                print(f"    N={train_size or 'full':>5} | {model_name:10s} | ERROR: {e}")

    return pd.DataFrame(results)


def analyze_crossover(results_df):
    """Analyze crossover points from results."""
    if results_df.empty:
        return {}

    print("\n" + "="*50)
    print("CROSSOVER ANALYSIS")
    print("="*50)

    for dataset in results_df["dataset"].unique():
        dataset_data = results_df[results_df["dataset"] == dataset]

        print(f"\n📊 {dataset}:")
        pivot = dataset_data.pivot_table(
            index="train_size",
            columns="model",
            values="roc_auc",
            aggfunc="mean"
        ).sort_index()

        print(pivot.to_string())

        # Find where XGBoost beats TabPFN
        if "TabPFN" in pivot.columns and "XGBoost" in pivot.columns:
            for size in pivot.index:
                tabpfn = pivot.loc[size, "TabPFN"]
                xgb = pivot.loc[size, "XGBoost"]
                diff = xgb - tabpfn
                if diff > 0:
                    print(f"\n🔄 CROSSOVER at N={size}")
                    print(f"   XGBoost: {xgb:.4f} beats TabPFN: {tabpfn:.4f} by {diff:.4f}")
                    break
            else:
                # Check full
                if "full" in pivot.index:
                    tabpfn_full = pivot.loc["full", "TabPFN"]
                    xgb_full = pivot.loc["full", "XGBoost"]
                    if xgb_full > tabpfn_full:
                        print(f"\n🔄 CROSSOVER at full data")
                        print(f"   XGBoost: {xgb_full:.4f} beats TabPFN: {tabpfn_full:.4f}")
                    else:
                        print(f"\n✅ TabPFN wins even at full data")


def main():
    print("="*70)
    print("OPTIMIZED CROSSOVER POINT ANALYSIS")
    print("="*70)
    print("\n🎯 Goal: Find where XGBoost beats TabPFN")
    print(f"Using existing datasets with larger training sizes")

    all_results = []

    for dataset_cfg in CROSSOVER_DATASETS:
        # Get adaptive training sizes based on dataset size
        train_sizes = get_adaptive_sizes(dataset_cfg["n"])

        results_df = run_dataset_analysis(dataset_cfg, train_sizes)

        if not results_df.empty:
            all_results.append(results_df)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)

        # Save results
        output_dir = project_root / "experiments" / "results" / "crossover"
        output_dir.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_dir / "optimized_crossover_results.csv", index=False)

        print(f"\n✅ Results saved to: {output_dir / 'optimized_crossover_results.csv'}")

        # Analyze crossovers
        analyze_crossover(combined)

        # Overall summary
        print("\n" + "="*70)
        print("OVERALL SUMMARY")
        print("="*70)

        overall = combined.pivot_table(
            index="train_size",
            columns="model",
            values="roc_auc",
            aggfunc="mean"
        ).sort_index()

        overall["TabPFN_minus_XGBoost"] = overall["TabPFN"] - overall["XGBoost"]
        print(overall.round(4).to_string())

    return combined if all_results else None


if __name__ == "__main__":
    main()