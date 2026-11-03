"""
Quick Few-Shot Learning Analysis Runner

Run this script to perform few-shot learning analysis with:
- 1 seed for speed
- Selected datasets with varied characteristics
- Multiple training sizes
"""

import sys
from pathlib import Path

# Add parent directory to path
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
from tabular_gauge.models.mlp_model import MLPWrapper
from tabular_gauge.evaluation.metrics import compute_metrics


# Training sizes to test
#TRAIN_SIZES = [25, 50, 75, 100, 150, 200, 300, 500, 750, 1000, None]
TRAIN_SIZES = [25, 50, 100, 200, 500, None]

# Selected datasets with varied characteristics
DATASETS = [
    {"id": 37, "name": "diabetes", "domain": "healthcare", "n": 768, "p": 8},
    {"id": 1510, "name": "wdbc", "domain": "healthcare", "n": 569, "p": 30},
    {"id": 40536, "name": "australian", "domain": "finance", "n": 690, "p": 14},
    {"id": 31, "name": "credit-g", "domain": "finance", "n": 1000, "p": 20},
    {"id": 40945, "name": "titanic", "domain": "historical", "n": 891, "p": 11},
    {"id": 1462, "name": "banknote-auth", "domain": "security", "n": 1372, "p": 4},
]


def stratified_subsample(X, y, n_samples, random_state=42):
    """Stratified subsample maintaining class balance."""
    import pandas as pd

    if n_samples is None or n_samples >= len(y):
        return X, y

    np.random.seed(random_state)

    # Handle pandas DataFrame or numpy array
    if isinstance(X, pd.DataFrame):
        is_dataframe = True
        X_array = X.values
    else:
        is_dataframe = False
        X_array = X

    y_array = np.asarray(y)

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


def run_single_dataset(dataset_cfg, train_sizes, n_seeds=1):
    """Run few-shot analysis on a single dataset."""
    loader = DataLoader(test_size=0.2, random_state=42)
    dataset = loader.load_openml(dataset_cfg["id"], dataset_cfg["name"])

    print(f"\n[{dataset_cfg['name']}] ({dataset_cfg['domain']}) - N={dataset.n_rows}, P={dataset.n_features}")

    results = []

    for train_size in tqdm(train_sizes, desc="  Training sizes"):
        # Subsample training data
        if train_size is not None:
            X_train_sub, y_train_sub = stratified_subsample(
                dataset.X_train, dataset.y_train, n_samples=train_size
            )
        else:
            X_train_sub, y_train_sub = dataset.X_train, dataset.y_train

        actual_size = len(y_train_sub)

        # Test models
        models = [
            ("TabPFN", TabPFNWrapper(random_state=0)),
            ("XGBoost", XGBoostWrapper(n_trials=10, random_state=0)),
            ("LightGBM", LightGBMWrapper(n_trials=10, random_state=0)),
            ("CatBoost", CatBoostWrapper(n_trials=10, random_state=0)),
            ("MLP", MLPWrapper(n_trials=8, random_state=0)),
        ]

        for model_name, model in models:
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
                metrics["domain"] = dataset_cfg["domain"]
                results.append(metrics)

                print(f"    N={train_size or 'full':>4} | {model_name:10s} | AUC={metrics.get('roc_auc', 0):.4f}")

            except Exception as e:
                print(f"    N={train_size} | {model_name:10s} | ERROR: {e}")

    return pd.DataFrame(results)


def main():
    print("="*60)
    print("FEW-SHOT LEARNING ANALYSIS")
    print("="*60)
    print(f"\nDatasets: {[d['name'] for d in DATASETS]}")
    print(f"Training sizes: {TRAIN_SIZES}")
    print(f"Seeds: 1 (quick mode)")

    all_results = []

    for dataset_cfg in DATASETS:
        results_df = run_single_dataset(dataset_cfg, TRAIN_SIZES, n_seeds=1)
        if not results_df.empty:
            all_results.append(results_df)

    # Combine and save
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)

        # Save results
        output_dir = project_root / "experiments" / "results" / "few_shot"
        output_dir.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_dir / "quick_few_shot_results.csv", index=False)

        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)

        # Summary by training size
        print("\nMean ROC-AUC by Training Size:")
        summary = combined.groupby("train_size")["roc_auc"].mean()
        for size, auc in summary.items():
            print(f"  N={str(size):>5}: {auc:.4f}")

        # Summary by model
        print("\nMean ROC-AUC by Model:")
        model_summary = combined.groupby("model")["roc_auc"].mean().sort_values(ascending=False)
        for model, auc in model_summary.items():
            print(f"  {model:10s}: {auc:.4f}")

        # Crossover analysis
        print("\nFinding Crossover Points (TabPFN vs XGBoost)...")
        pivot = combined.pivot_table(index="train_size", columns="model", values="roc_auc", aggfunc="mean")
        pivot = pivot.sort_index()

        print("\n  Crossover Analysis:")
        if "TabPFN" in pivot.columns and "XGBoost" in pivot.columns:
            for size in pivot.index:
                tabpfn = pivot.loc[size, "TabPFN"]
                xgb = pivot.loc[size, "XGBoost"]
                winner = "TabPFN" if tabpfn > xgb else "XGBoost"
                diff = abs(tabpfn - xgb)
                print(f"    N={str(size):>5}: TabPFN={tabpfn:.4f}, XGBoost={xgb:.4f} -> {winner} (diff={diff:.4f})")

        print(f"\nResults saved to: {output_dir / 'quick_few_shot_results.csv'}")

        return combined

    return None


if __name__ == "__main__":
    main()