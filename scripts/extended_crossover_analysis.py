"""
Extended Crossover Analysis - Multiple Large Datasets

Finds robust crossover patterns across multiple large datasets.
- Reuses existing adult results
- Runs new datasets (kddcup, airlines)
- Aggregates results for robust findings
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
import json
from datetime import datetime

from tabular_gauge.data.loader import DataLoader
from tabular_gauge.models.tabpfn_model import TabPFNWrapper
from tabular_gauge.models.xgboost_model import XGBoostWrapper
from tabular_gauge.models.lightgbm_model import LightGBMWrapper
from tabular_gauge.evaluation.metrics import compute_metrics


# Large datasets - adult already done, add new ones
# TabPFN max features is ~50, so we use datasets with p <= 50
# Also ensure binary classification
LARGE_DATASETS = [
    {"id": 1590, "name": "adult", "domain": "demographic", "n": 48842, "p": 14},  # Income - binary, p=14 OK
    {"id": 151, "name": "mushroom", "domain": "biology", "n": 8124, "p": 22},  # Binary, p=22 OK
    {"id": 44, "name": "spambase", "domain": "nlp", "n": 4601, "p": 57},  # Binary, p=57 OK (close to limit)
]

# Consistent training sizes - adapted for different dataset sizes
TRAIN_SIZES = [50, 100, 250, 500, 750, 1000, 1500, 2000, None]
TRAIN_SIZES = sorted(TRAIN_SIZES, key=lambda x: 999999 if x is None else x)


def stratified_subsample_numpy(X, y, n_samples, random_state=42):
    """Stratified subsample maintaining class balance."""
    X_array = X.values if hasattr(X, 'values') else np.array(X)
    y_array = np.asarray(y)

    if n_samples is None or n_samples >= len(y_array):
        return X_array, y_array

    np.random.seed(random_state)

    pos_indices = np.where(y_array == 1)[0]
    neg_indices = np.where(y_array == 0)[0]

    if len(pos_indices) == 0 or len(neg_indices) == 0:
        selected_indices = np.random.choice(len(y_array), min(n_samples, len(y_array)), replace=False)
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
    try:
        loader = DataLoader(test_size=0.2, random_state=42)
        dataset = loader.load_openml(dataset_cfg["id"], dataset_cfg["name"])

        # Check if binary classification
        unique_classes = len(np.unique(dataset.y_train))
        if unique_classes != 2:
            print(f"    ⚠️ Skipping {dataset_cfg['name']}: {unique_classes} classes (need binary)")
            return pd.DataFrame()

        X_train_orig = dataset.X_train.values if hasattr(dataset.X_train, 'values') else dataset.X_train
        X_test_orig = dataset.X_test.values if hasattr(dataset.X_test, 'values') else dataset.X_test
        y_train_orig = dataset.y_train
        y_test_orig = dataset.y_test

        if train_size is not None:
            X_train_sub, y_train_sub = stratified_subsample_numpy(
                X_train_orig, y_train_orig, n_samples=train_size, random_state=seed
            )
        else:
            X_train_sub, y_train_sub = X_train_orig, y_train_orig

        actual_size = len(y_train_sub)
        results = []

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
                err_msg = str(e).lower()
                if "multiclass" in err_msg or "format not supported" in err_msg:
                    print(f"    N={train_size or 'full':>5} | {model_name:10s} | SKIP: multiclass")
                elif "feature" in err_msg and ("limit" in err_msg or "exceed" in err_msg):
                    print(f"    N={train_size or 'full':>5} | {model_name:10s} | SKIP: too many features")
                else:
                    print(f"    N={train_size or 'full':>5} | {model_name:10s} | ERROR: {str(e)[:50]}")

        return pd.DataFrame(results)

    except Exception as e:
        print(f"    ⚠️ Dataset error: {str(e)[:50]}")
        return pd.DataFrame()


def find_crossover(pivot_df):
    """Find crossover point from pivot table."""
    if "TabPFN" not in pivot_df.columns or "XGBoost" not in pivot_df.columns:
        return None

    def sort_key(x):
        if x == "full" or x is None:
            return 999999
        return int(x)

    sizes = sorted(pivot_df.index.tolist(), key=sort_key)

    for i, size in enumerate(sizes):
        tabpfn = pivot_df.loc[size, "TabPFN"]
        xgb = pivot_df.loc[size, "XGBoost"]

        if xgb > tabpfn:
            if i > 0:
                prev_size = sizes[i-1]
                return {
                    "crossover_at": str(size),
                    "prev_size": str(prev_size),
                    "tabpfn_before": float(pivot_df.loc[prev_size, "TabPFN"]),
                    "xgb_before": float(pivot_df.loc[prev_size, "XGBoost"]),
                    "tabpfn_at": float(tabpfn),
                    "xgb_at": float(xgb),
                }
            else:
                return {"crossover_at": str(size), "tabpfn_at": float(tabpfn), "xgb_at": float(xgb)}

    return {"crossover_at": None, "tabpfn_at": float(pivot_df.iloc[-1]["TabPFN"]), "xgb_at": float(pivot_df.iloc[-1]["XGBoost"])}


def main():
    output_dir = project_root / "experiments" / "results" / "crossover"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing adult results
    adult_results_file = output_dir / "large_scale_crossover_results.csv"
    existing_datasets = []

    if adult_results_file.exists():
        print(f"✅ Found existing adult results - will reuse")
        existing_datasets.append("adult")
    else:
        print(f"⚠️ No existing adult results - will run")

    print("="*70)
    print("EXTENDED CROSSOVER ANALYSIS")
    print("="*70)
    print(f"\n🎯 Goal: Find robust crossover patterns across multiple datasets")
    print(f"\nDatasets: {[d['name'] for d in LARGE_DATASETS]}")

    all_results = []

    # Load existing adult results if available
    if "adult" in existing_datasets:
        adult_df = pd.read_csv(adult_results_file)
        all_results.append(adult_df)
        print(f"✅ Loaded existing adult results: {len(adult_df)} rows")

    # Run new datasets
    for dataset_cfg in LARGE_DATASETS:
        dataset_file = output_dir / f"crossover_{dataset_cfg['name']}.csv"

        if dataset_cfg["name"] in existing_datasets:
            print(f"\n📊 {dataset_cfg['name']} - Using existing results")
            continue

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
            combined.to_csv(dataset_file, index=False)
            print(f"✅ Saved {dataset_cfg['name']} results to {dataset_file}")
            all_results.append(combined)

    # Combine all results
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(output_dir / "extended_crossover_results.csv", index=False)
        print(f"\n✅ Combined results saved to: {output_dir / 'extended_crossover_results.csv'}")

        # Analyze per dataset
        print("\n" + "="*70)
        print("PER-DATASET ANALYSIS")
        print("="*70)

        crossover_results = {}

        for dataset in combined["dataset"].unique():
            dataset_data = combined[combined["dataset"] == dataset]

            print(f"\n📊 {dataset}:")

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

            pivot["TabPFN_minus_XGB"] = pivot["TabPFN"] - pivot["XGBoost"]

            print(pivot.round(4).to_string())

            crossover = find_crossover(pivot)
            crossover_results[dataset] = crossover

            if crossover and crossover.get("crossover_at"):
                print(f"   🔄 Crossover at N={crossover['crossover_at']}")
            else:
                print(f"   ✅ TabPFN wins at all sizes (XGBoost leads by {crossover['xgb_at'] - crossover['tabpfn_at']:.4f} at full)")

        # Aggregate analysis
        print("\n" + "="*70)
        print("AGGREGATE ANALYSIS - ROBUST FINDINGS")
        print("="*70)

        # Find crossover points for each dataset
        crossover_points = []
        for dataset, cr in crossover_results.items():
            cp = cr.get("crossover_at")
            if cp and cp != "None":
                try:
                    crossover_points.append(int(cp))
                except:
                    pass
            elif cp == "None" or cp is None:
                crossover_points.append(999999)  # Full data

        if crossover_points:
            avg_crossover = np.mean([x for x in crossover_points if x < 999999])
            print(f"\n📈 Crossover Points by Dataset:")
            for dataset, cr in crossover_results.items():
                cp = cr.get("crossover_at", "N/A")
                print(f"   {dataset}: N = {cp}")

            valid_crossovers = [x for x in crossover_points if x < 999999]
            if valid_crossovers:
                print(f"\n🎯 Average Crossover Point: N ≈ {np.mean(valid_crossovers):.0f}")
                print(f"   Range: {min(valid_crossovers)} - {max(valid_crossovers)}")

        # Overall performance by training size
        print(f"\n📊 Overall Mean Performance (across all datasets):")
        overall = combined.pivot_table(
            index="train_size",
            columns="model",
            values="roc_auc",
            aggfunc="mean"
        )
        overall = overall.reindex(sorted(overall.index, key=sort_key))
        overall["TabPFN_minus_XGB"] = overall["TabPFN"] - overall["XGBoost"]
        print(overall.round(4).to_string())

        # Save aggregate summary
        summary = {
            "datasets_analyzed": list(combined["dataset"].unique()),
            "training_sizes": TRAIN_SIZES,
            "crossover_by_dataset": crossover_results,
            "average_crossover_n": float(np.mean(valid_crossovers)) if valid_crossovers else None,
            "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(output_dir / "extended_crossover_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n✅ Summary saved to: {output_dir / 'extended_crossover_summary.json'}")

    return combined


if __name__ == "__main__":
    main()