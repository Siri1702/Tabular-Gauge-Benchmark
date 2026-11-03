"""
Optimized Large-Scale Crossover Analysis

Based on few-shot findings:
- TabPFN dominates at N < 500
- Crossover zone is around N = 500-1000
- Need to test finer sizes in this range

Uses larger datasets (adult ~49K) with focused training sizes.
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from tqdm import tqdm

from tabular_gauge.data.loader import DataLoader
from tabular_gauge.models.tabpfn_model import TabPFNWrapper
from tabular_gauge.models.xgboost_model import XGBoostWrapper
from tabular_gauge.models.lightgbm_model import LightGBMWrapper
from tabular_gauge.evaluation.metrics import compute_metrics


# Large datasets - adult is best for crossover analysis
LARGE_DATASETS = [
    {"id": 1590, "name": "adult", "domain": "demographic", "n": 48842, "p": 14},
]

# OPTIMIZED training sizes based on few-shot findings
TRAIN_SIZES = [50, 100, 250, 500, 750, 1000, 1500, 2000, 3000, 5000, None]

# Sort training sizes for proper iteration order
def sort_key(x):
    if x is None:
        return 999999
    return int(x)

TRAIN_SIZES = sorted(TRAIN_SIZES, key=sort_key)


def stratified_subsample_numpy(X, y, n_samples, random_state=42):
    """
    Stratified subsample maintaining class balance.
    Returns numpy arrays to ensure consistency.
    """
    # Convert to numpy if DataFrame
    if hasattr(X, 'values'):
        X_array = X.values
    else:
        X_array = np.array(X)

    y_array = np.asarray(y)

    if n_samples is None or n_samples >= len(y_array):
        return X_array, y_array

    np.random.seed(random_state)

    pos_indices = np.where(y_array == 1)[0]
    neg_indices = np.where(y_array == 0)[0]

    if len(pos_indices) == 0 or len(neg_indices) == 0:
        # Edge case: take first n_samples
        selected_indices = np.random.choice(len(y_array), n_samples, replace=False)
        return X_array[selected_indices], y_array[selected_indices]

    pos_ratio = len(pos_indices) / len(y_array)
    n_pos = min(len(pos_indices), int(n_samples * pos_ratio))
    n_neg = n_samples - n_pos

    selected_pos = np.random.choice(pos_indices, n_pos, replace=False)
    selected_neg = np.random.choice(neg_indices, n_neg, replace=False)

    selected_indices = np.sort(np.concatenate([selected_pos, selected_neg]))
    return X_array[selected_indices], y_array[selected_indices]


def run_experiment(dataset_cfg, train_size, seed=0):
    """Run single experiment."""
    print(f"    Testing N={train_size or 'full'}...")

    loader = DataLoader(test_size=0.2, random_state=42)
    dataset = loader.load_openml(dataset_cfg["id"], dataset_cfg["name"])

    # Get original train/test as numpy arrays
    X_train_orig = dataset.X_train.values if hasattr(dataset.X_train, 'values') else dataset.X_train
    X_test_orig = dataset.X_test.values if hasattr(dataset.X_test, 'values') else dataset.X_test
    y_train_orig = dataset.y_train
    y_test_orig = dataset.y_test

    # Subsample if needed
    if train_size is not None:
        X_train_sub, y_train_sub = stratified_subsample_numpy(
            X_train_orig, y_train_orig, n_samples=train_size, random_state=seed
        )
    else:
        X_train_sub, y_train_sub = X_train_orig, y_train_orig

    actual_size = len(y_train_sub)
    results = []

    # Focus on TabPFN vs XGBoost (key comparison)
    models = [
        ("TabPFN", TabPFNWrapper(random_state=seed)),
        ("XGBoost", XGBoostWrapper(n_trials=20, random_state=seed)),
        ("LightGBM", LightGBMWrapper(n_trials=20, random_state=seed)),
    ]

    for model_name, model in models:
        try:
            result = model.run(
                X_train_sub, y_train_sub,
                X_test_orig, y_test_orig,
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
            print(f"    N={train_size or 'full':>5} | {model_name:10s} | ERROR: {str(e)[:50]}")

    return pd.DataFrame(results)


def find_crossover(pivot_df):
    """Find exact crossover point from pivot table."""
    if "TabPFN" not in pivot_df.columns or "XGBoost" not in pivot_df.columns:
        return None

    # Convert index to numeric for sorting (full = large number)
    def sort_key(x):
        if x == "full" or x is None:
            return 999999
        return int(x) if isinstance(x, (int, np.integer)) else int(x)

    sizes = sorted(pivot_df.index.tolist(), key=sort_key)

    for i, size in enumerate(sizes):
        tabpfn = pivot_df.loc[size, "TabPFN"]
        xgb = pivot_df.loc[size, "XGBoost"]

        if xgb > tabpfn:
            if i > 0:
                prev_size = sizes[i-1]
                prev_tabpfn = pivot_df.loc[prev_size, "TabPFN"]
                prev_xgb = pivot_df.loc[prev_size, "XGBoost"]
                return {
                    "crossover_at": size,
                    "prev_size": prev_size,
                    "tabpfn_before": prev_tabpfn,
                    "xgb_before": prev_xgb,
                    "tabpfn_at": tabpfn,
                    "xgb_at": xgb,
                }
            else:
                return {
                    "crossover_at": size,
                    "prev_size": None,
                    "tabpfn_at": tabpfn,
                    "xgb_at": xgb,
                }

    return {"crossover_at": None, "tabpfn_at": pivot_df.iloc[-1]["TabPFN"], "xgb_at": pivot_df.iloc[-1]["XGBoost"]}


def main():
    print("="*70)
    print("LARGE-SCALE CROSSOVER ANALYSIS")
    print("="*70)
    print(f"\n🎯 Goal: Find exact crossover point with large datasets")
    print(f"\nDatasets: {[d['name'] for d in LARGE_DATASETS]}")
    print(f"Training sizes: {TRAIN_SIZES}")
    print(f"\n📊 Based on few-shot findings:")
    print(f"   - TabPFN dominates at N < 500")
    print(f"   - Crossover expected around N = 500-1500")
    print(f"   - Testing finer sizes in this zone")

    all_results = []

    for dataset_cfg in LARGE_DATASETS:
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset_cfg['name']} (N={dataset_cfg['n']:,}, P={dataset_cfg['p']})")
        print(f"{'='*50}")

        dataset_results = []

        for train_size in tqdm(TRAIN_SIZES, desc="Training sizes"):
            results_df = run_experiment(dataset_cfg, train_size)
            if not results_df.empty:
                dataset_results.append(results_df)

        if dataset_results:
            combined = pd.concat(dataset_results, ignore_index=True)
            all_results.append(combined)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)

        # Save results
        output_dir = project_root / "experiments" / "results" / "crossover"
        output_dir.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_dir / "large_scale_crossover_results.csv", index=False)

        # Analyze
        print("\n" + "="*70)
        print("RESULTS & CROSSOVER ANALYSIS")
        print("="*70)

        for dataset in combined["dataset"].unique():
            dataset_data = combined[combined["dataset"] == dataset]

            print(f"\n📊 {dataset} - Performance by Training Size:")

            # Create pivot with numeric sorting
            def sort_key(x):
                if x == "full" or x is None:
                    return 999999
                return int(x) if isinstance(x, (int, np.integer)) else int(x)

            pivot = dataset_data.pivot_table(
                index="train_size",
                columns="model",
                values="roc_auc"
            )
            pivot = pivot.reindex(sorted(pivot.index, key=sort_key))

            # Add diff column
            pivot["TabPFN_minus_XGB"] = pivot["TabPFN"] - pivot["XGBoost"]
            print(pivot.round(4).to_string())

            # Find crossover
            crossover = find_crossover(pivot)
            print(f"\n🎯 CROSSOVER ANALYSIS:")
            if crossover and crossover["crossover_at"]:
                print(f"   🔄 CROSSOVER FOUND at N={crossover['crossover_at']}")
                if crossover["prev_size"]:
                    print(f"      Before (N={crossover['prev_size']}): TabPFN={crossover['tabpfn_before']:.4f}, XGBoost={crossover['xgb_before']:.4f}")
                    print(f"      At crossover (N={crossover['crossover_at']}): TabPFN={crossover['tabpfn_at']:.4f}, XGBoost={crossover['xgb_at']:.4f}")
                print(f"   ✅ XGBoost starts winning at N ≈ {crossover['crossover_at']}")
            else:
                print(f"   ✅ TabPFN wins at all tested sizes (up to {crossover['tabpfn_at'] if crossover else 'N/A'})")

        print(f"\n✅ Results saved to: {output_dir / 'large_scale_crossover_results.csv'}")

    return combined if all_results else None


if __name__ == "__main__":
    main()