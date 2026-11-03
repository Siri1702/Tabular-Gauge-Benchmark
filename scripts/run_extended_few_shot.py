"""
Extended Few-Shot Learning Analysis Runner

Runs experiments for training sizes: 75, 150, 300, 750, 1000
These are additional to the existing: 25, 50, 100, 200, 500, full

This gives us a comprehensive learning curve with finer granularity.
"""

import sys
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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


# Extended training sizes - ADDING these to existing: 25, 50, 100, 200, 500, None
EXTENDED_TRAIN_SIZES = [75, 150, 300, 750, 1000]

# Same 6 datasets as original few-shot analysis
DATASETS = [
    {"id": 37, "name": "diabetes", "domain": "healthcare", "n": 768, "p": 8},
    {"id": 1510, "name": "wdbc", "domain": "healthcare", "n": 569, "p": 30},
    {"id": 40536, "name": "australian", "domain": "finance", "n": 690, "p": 14},
    {"id": 31, "name": "credit-g", "domain": "finance", "n": 1000, "p": 20},
    {"id": 40945, "name": "titanic", "domain": "historical", "n": 891, "p": 11},
    {"id": 1462, "name": "banknote-auth", "domain": "security", "n": 1372, "p": 4},
]

# Models to test
MODELS = [
    ("TabPFN", TabPFNWrapper(random_state=0)),
    ("XGBoost", XGBoostWrapper(n_trials=10, random_state=0)),
    ("LightGBM", LightGBMWrapper(n_trials=10, random_state=0)),
    ("CatBoost", CatBoostWrapper(n_trials=10, random_state=0)),
    ("MLP", MLPWrapper(n_trials=8, random_state=0)),
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


def run_extended_few_shot():
    """Run extended few-shot analysis for missing training sizes."""
    print("="*70)
    print("EXTENDED FEW-SHOT LEARNING ANALYSIS")
    print("="*70)
    print(f"\nExtended sizes to run: {EXTENDED_TRAIN_SIZES}")
    print(f"Datasets: {[d['name'] for d in DATASETS]}")
    print(f"Models: {[m[0] for m in MODELS]}")

    all_results = []

    for dataset_cfg in DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_cfg['name']} ({dataset_cfg['domain']})")
        print(f"{'='*60}")

        # Load dataset
        loader = DataLoader(test_size=0.2, random_state=42)
        dataset = loader.load_openml(dataset_cfg["id"], dataset_cfg["name"])

        print(f"  N={dataset.n_rows}, P={dataset.n_features}")

        for train_size in tqdm(EXTENDED_TRAIN_SIZES, desc="  Training sizes"):
            # Subsample training data
            if train_size is not None and train_size < len(dataset.y_train):
                X_train_sub, y_train_sub = stratified_subsample(
                    dataset.X_train, dataset.y_train, n_samples=train_size
                )
            else:
                X_train_sub, y_train_sub = dataset.X_train, dataset.y_train

            actual_size = len(y_train_sub)

            # Test all models
            for model_name, model in MODELS:
                try:
                    result = model.run(
                        X_train_sub, y_train_sub,
                        dataset.X_test, dataset.y_test,
                        dataset_name=f"{dataset_cfg['name']}_N{train_size}",
                        seed=0,
                    )
                    metrics = compute_metrics(result)
                    metrics["train_size"] = train_size
                    metrics["actual_n_train"] = actual_size
                    metrics["domain"] = dataset_cfg["domain"]
                    all_results.append(metrics)

                    print(f"    N={train_size:>4} | {model_name:10s} | AUC={metrics.get('roc_auc', 0):.4f}")

                except Exception as e:
                    print(f"    N={train_size:>4} | {model_name:10s} | ERROR: {e}")

    # Save extended results
    if all_results:
        extended_df = pd.DataFrame(all_results)

        # Load existing results
        existing_df = pd.read_csv(project_root / "experiments" / "results" / "few_shot" / "quick_few_shot_results.csv")
        existing_df = existing_df.rename(columns={'the above few shotmodel': 'model'})

        # Combine
        combined_df = pd.concat([existing_df, extended_df], ignore_index=True)

        # Save combined
        output_path = project_root / "experiments" / "results" / "few_shot" / "extended_few_shot_results.csv"
        combined_df.to_csv(output_path, index=False)

        print(f"\n{'='*60}")
        print("RESULTS SAVED")
        print(f"{'='*60}")
        print(f"Extended results: {len(extended_df)} new rows")
        print(f"Combined total: {len(combined_df)} rows")
        print(f"Saved to: {output_path}")

        # Summary
        print("\n### Extended Results Summary ###")
        summary = combined_df.groupby("train_size")["roc_auc"].mean()
        print(summary.sort_index().to_string())

        return combined_df

    return None


if __name__ == "__main__":
    run_extended_few_shot()
