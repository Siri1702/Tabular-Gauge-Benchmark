"""
SAP-RPT-1 Few-Shot Learning Analysis Runner

This script runs SAP-RPT-1 on the same few-shot datasets and training sizes
as the original few-shot analysis for direct comparison.

The key difference from other models: SAP-RPT-1 needs column names (DataFrame format)
as it uses semantic information from column names for in-context learning.
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
from tabular_gauge.models.sap_rpt_model import SAPRptWrapper
from tabular_gauge.evaluation.metrics import compute_metrics


# Same training sizes as original few-shot analysis
TRAIN_SIZES = [25, 50, 100, 200, 500, None]

# Same datasets as original few-shot analysis
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


def run_single_dataset_sap_rpt(dataset_cfg, train_sizes):
    """Run SAP-RPT-1 few-shot analysis on a single dataset."""
    print(f"\n[{dataset_cfg['name']}] ({dataset_cfg['domain']})")

    # Load dataset - SAP-RPT-1 needs DataFrame format to preserve column names
    loader = DataLoader(test_size=0.2, random_state=42)
    dataset = loader.load_openml(dataset_cfg["id"], dataset_cfg["name"])

    # SAP-RPT-1 requires DataFrame with column names
    # But the DataLoader returns numpy arrays. Let's check what we have
    # The loader returns Dataset object with X_train, X_test as numpy arrays

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

        # For SAP-RPT-1, we need to reconstruct a DataFrame with column names
        # Let's reload properly with DataFrame format
        # Actually, the DataLoader already does this - let's check

        # Re-load to get DataFrame format with column names
        loader_df = DataLoader(test_size=0.2, random_state=42)
        dataset_df = loader_df.load_openml(dataset_cfg["id"], dataset_cfg["name"])

        # Now subsample from the DataFrame version
        if train_size is not None:
            X_train_sub, y_train_sub = stratified_subsample(
                dataset_df.X_train, dataset_df.y_train, n_samples=train_size
            )
        else:
            X_train_sub, y_train_sub = dataset_df.X_train, dataset_df.y_train

        # SAP-RPT-1 model
        model = SAPRptWrapper(
            max_context_size=8192,
            bagging=1,
            random_state=42,
        )

        try:
            result = model.run(
                X_train_sub, y_train_sub,
                dataset_df.X_test, dataset_df.y_test,
                dataset_name=f"{dataset_cfg['name']}_N{train_size or 'full'}",
                seed=42,
            )
            metrics = compute_metrics(result)
            metrics["train_size"] = train_size if train_size else "full"
            metrics["actual_n_train"] = actual_size
            metrics["domain"] = dataset_cfg["domain"]
            results.append(metrics)

            print(f"    N={train_size or 'full':>4} | SAP-RPT-1 | AUC={metrics.get('roc_auc', 0):.4f}")

        except Exception as e:
            print(f"    N={train_size or 'full':>4} | SAP-RPT-1 | ERROR: {e}")
            import traceback
            traceback.print_exc()

    return pd.DataFrame(results)


def main():
    print("="*60)
    print("SAP-RPT-1 FEW-SHOT LEARNING ANALYSIS")
    print("="*60)
    print(f"\nDatasets: {[d['name'] for d in DATASETS]}")
    print(f"Training sizes: {TRAIN_SIZES}")

    all_results = []

    for dataset_cfg in DATASETS:
        results_df = run_single_dataset_sap_rpt(dataset_cfg, TRAIN_SIZES)
        if not results_df.empty:
            all_results.append(results_df)

    # Combine and save
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)

        # Save results
        output_dir = project_root / "experiments" / "results" / "few_shot"
        output_dir.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_dir / "sap_rpt_few_shot_results.csv", index=False)

        print("\n" + "="*60)
        print("SAP-RPT-1 FEW-SHOT RESULTS SUMMARY")
        print("="*60)

        # Summary by training size
        print("\nMean ROC-AUC by Training Size:")
        summary = combined.groupby("train_size")["roc_auc"].mean()
        for size, auc in summary.items():
            print(f"  N={str(size):>5}: {auc:.4f}")

        print(f"\nResults saved to: {output_dir / 'sap_rpt_few_shot_results.csv'}")

        return combined

    return None


if __name__ == "__main__":
    main()
