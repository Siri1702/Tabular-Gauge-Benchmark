"""
Crossover Point Analysis - Finding Where GBDT Beats TabPFN

This script finds the EXACT crossover point where tuned GBDT models
start beating TabPFN by:
1. Adding larger datasets (adult ~49K, airlines ~63K)
2. Testing finer training sizes (250, 500, 750, 1000, 2000, 5000, full)
3. Identifying precise crossover points
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
import json

from tabular_gauge.data.loader import DataLoader
from tabular_gauge.models.tabpfn_model import TabPFNWrapper
from tabular_gauge.models.xgboost_model import XGBoostWrapper
from tabular_gauge.models.lightgbm_model import LightGBMWrapper
from tabular_gauge.models.catboost_model import CatBoostWrapper


# Larger datasets to find crossover
LARGER_DATASETS = [
    {"id": 1590, "name": "adult", "domain": "demographic", "n": 48842, "p": 14},
    {"id": 41163, "name": "airlines", "domain": "transportation", "n": 539383, "p": 7},
    {"id": 23517, "name": "epsilon", "domain": "synthetic", "n": 500000, "p": 8},
]

# Finer training sizes to pinpoint crossover
TRAIN_SIZES = [100, 250, 500, 750, 1000, 2000, 5000, None]

N_SEEDS = 1  # Quick run for now


def stratified_subsample(X, y, n_samples, random_state=42):
    """Stratified subsample maintaining class balance."""
    if isinstance(X, pd.DataFrame):
        is_dataframe = True
        X_array = X.values
    else:
        is_dataframe = False
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

    if is_dataframe:
        return X.iloc[selected_indices], y_array[selected_indices]
    else:
        return X_array[selected_indices], y_array[selected_indices]


def run_experiment(dataset_cfg, train_size, seed=0):
    """Run single experiment."""
    print(f"    Testing N={train_size or 'full'}...")

    try:
        loader = DataLoader(test_size=0.2, random_state=42)
        dataset = loader.load_openml(dataset_cfg["id"], dataset_cfg["name"])

        # Subsample
        if train_size is not None:
            X_train_sub, y_train_sub = stratified_subsample(
                dataset.X_train, dataset.y_train, n_samples=train_size, random_state=seed
            )
        else:
            X_train_sub, y_train_sub = dataset.X_train, dataset.y_train

        actual_size = len(y_train_sub)
        print(f"      Actual training size: {actual_size}")

        results = []

        # TabPFN - reduced ensemble for speed on large data
        try:
            print(f"      Running TabPFN...")
            model = TabPFNWrapper(random_state=seed)
            result = model.run(
                X_train_sub, y_train_sub,
                dataset.X_test, dataset.y_test,
                dataset_name=f"{dataset_cfg['name']}_N{train_size}",
                seed=seed,
            )
            from tabular_gauge.evaluation.metrics import compute_metrics
            metrics = compute_metrics(result)
            metrics["model"] = "TabPFN"
            metrics["actual_n_train"] = actual_size
            results.append(metrics)
        except Exception as e:
            print(f"      TabPFN error: {e}")

        # XGBoost - key comparison
        try:
            print(f"      Running XGBoost...")
            model = XGBoostWrapper(n_trials=20, random_state=seed)
            result = model.run(
                X_train_sub, y_train_sub,
                dataset.X_test, dataset.y_test,
                dataset_name=f"{dataset_cfg['name']}_N{train_size}",
                seed=seed,
            )
            metrics = compute_metrics(result)
            metrics["model"] = "XGBoost"
            metrics["actual_n_train"] = actual_size
            results.append(metrics)
        except Exception as e:
            print(f"      XGBoost error: {e}")

        return pd.DataFrame(results)

    except Exception as e:
        print(f"    Dataset error: {e}")
        return pd.DataFrame()


def find_crossover(results_df):
    """Find exact crossover point from results."""
    if results_df.empty:
        return None

    # Focus on TabPFN vs XGBoost
    pivot = results_df.pivot_table(
        index="train_size",
        columns="model",
        values="roc_auc",
        aggfunc="mean"
    )

    if "TabPFN" not in pivot.columns or "XGBoost" not in pivot.columns:
        return None

    pivot = pivot.sort_index()

    # Find crossover
    sizes = [s for s in pivot.index if s != "full"] + ["full"]
    crossover = None

    for i in range(len(sizes)):
        curr_size = sizes[i]
        if curr_size not in pivot.index:
            continue

        tabpfn = pivot.loc[curr_size, "TabPFN"]
        xgb = pivot.loc[curr_size, "XGBoost"]

        # Check if XGBoost wins at this size
        if xgb > tabpfn:
            if i > 0:
                prev_size = sizes[i-1]
                crossover = {
                    "crossover_at": curr_size,
                    "prev_size": prev_size,
                    "tabpfn_at_crossover": tabpfn,
                    "xgb_at_crossover": xgb,
                }
            else:
                crossover = {
                    "crossover_at": curr_size,
                    "prev_size": None,
                    "tabpfn_at_crossover": tabpfn,
                    "xgb_at_crossover": xgb,
                }
            break

    return crossover


def main():
    print("="*70)
    print("CROSSOVER POINT ANALYSIS")
    print("="*70)
    print("\n🎯 Goal: Find where XGBoost beats TabPFN")
    print(f"\nDatasets: {[d['name'] for d in LARGER_DATASETS]}")
    print(f"Training sizes: {TRAIN_SIZES}")

    all_results = []

    for dataset_cfg in LARGER_DATASETS:
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset_cfg['name']} (N={dataset_cfg['n']:,}, P={dataset_cfg['p']})")
        print(f"{'='*50}")

        dataset_results = []

        for train_size in tqdm(TRAIN_SIZES, desc="Training sizes"):
            results_df = run_experiment(dataset_cfg, train_size)
            if not results_df.empty:
                results_df["dataset"] = dataset_cfg["name"]
                results_df["domain"] = dataset_cfg["domain"]
                results_df["train_size"] = train_size if train_size else "full"
                dataset_results.append(results_df)

        if dataset_results:
            combined = pd.concat(dataset_results, ignore_index=True)
            all_results.append(combined)

            # Save dataset-specific results
            output_dir = project_root / "experiments" / "results" / "crossover"
            output_dir.mkdir(parents=True, exist_ok=True)
            combined.to_csv(output_dir / f"crossover_{dataset_cfg['name']}.csv", index=False)

            # Print immediate results
            print(f"\n📊 {dataset_cfg['name']} Results:")
            pivot = combined.pivot_table(index="train_size", columns="model", values="roc_auc")
            print(pivot.sort_index())

    # Combine all
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        output_dir = project_root / "experiments" / "results" / "crossover"
        combined.to_csv(output_dir / "all_crossover_results.csv", index=False)

        # Analyze crossovers
        print("\n" + "="*70)
        print("CROSSOVER ANALYSIS")
        print("="*70)

        for dataset in combined["dataset"].unique():
            dataset_data = combined[combined["dataset"] == dataset]
            crossover = find_crossover(dataset_data)

            print(f"\n📊 {dataset}:")
            pivot = dataset_data.pivot_table(index="train_size", columns="model", values="roc_auc")
            print(pivot.sort_index())

            if crossover:
                print(f"\n🔄 CROSSOVER FOUND at N={crossover['crossover_at']}")
                print(f"   TabPFN: {crossover['tabpfn_at_crossover']:.4f}")
                print(f"   XGBoost: {crossover['xgb_at_crossover']:.4f}")
            else:
                print(f"\n✅ No crossover found - TabPFN wins at all tested sizes")

        # Overall summary
        print("\n" + "="*70)
        print("OVERALL SUMMARY")
        print("="*70)

        overall = combined.pivot_table(
            index="train_size",
            columns="model",
            values="roc_auc",
            aggfunc="mean"
        )
        overall["diff"] = overall["TabPFN"] - overall["XGBoost"]
        print(overall.sort_index())

        print(f"\n✅ Results saved to: {output_dir}")

    return combined if all_results else None


if __name__ == "__main__":
    main()