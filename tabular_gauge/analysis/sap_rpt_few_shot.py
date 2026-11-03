"""
SAP-RPT-1 Few-Shot Learning Analysis

This script runs SAP-RPT-1 on the same few-shot datasets and training sizes
as the existing benchmark to enable direct comparison.

This is critical for the publication to show:
1. How SAP-RPT-1 compares to TabPFN in few-shot regime
2. Whether SAP-RPT-1 has different few-shot characteristics
3. Complete the benchmark for publication-quality results
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
from tabular_gauge.models.sap_rpt_model import SAPRptWrapper
from tabular_gauge.evaluation.metrics import compute_metrics


# Same training sizes as original few-shot analysis
TRAIN_SIZES = [25, 50, 100, 200, 500, None]  # None = full

# Same datasets as original few-shot analysis
FEW_SHOT_DATASETS = [
    {"id": 188, "name": "diabetes"},
    {"id": 1514, "name": "wdbc"},
    {"id": 293, "name": "australian"},
    {"id": 31, "name": "credit-g"},
    {"id": 40978, "name": "titanic"},
    {"id": 1461, "name": "banknote-auth"},
]


def stratified_subsample(
    X,
    y,
    n_samples: int,
    random_state: int = 42,
):
    """Stratified subsample to reduce training set size."""
    if n_samples is None or n_samples >= len(y):
        return X, y

    np.random.seed(random_state)

    # Handle DataFrame case (for SAP-RPT-1 which needs DataFrame)
    if hasattr(X, 'iloc'):
        X_arr = X.values
        is_dataframe = True
    else:
        X_arr = X
        is_dataframe = False

    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]

    pos_ratio = len(pos_indices) / len(y)
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
        return X.iloc[selected_indices], y[selected_indices]
    return X_arr[selected_indices], y[selected_indices]


def run_sap_rpt_few_shot(
    dataset_cfg: dict,
    train_sizes: List[Optional[int]] = TRAIN_SIZES,
    output_path: str = "experiments/results/few_shot/sap_rpt_few_shot_results.csv",
) -> pd.DataFrame:
    """
    Run SAP-RPT-1 few-shot analysis on a single dataset.
    Uses the same methodology as the original few-shot analysis.
    """
    print(f"\n{'='*60}")
    print(f"SAP-RPT-1 Few-Shot: {dataset_cfg['name']}")
    print(f"{'='*60}")

    loader = DataLoader(
        test_size=0.2,
        random_state=42,  # Same split as original
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

        print(f"    N={train_size if train_size else 'full'} ({actual_size} samples)")

        # SAP-RPT-1 model
        model = SAPRptWrapper(
            max_context_size=8192,
            bagging=1,
            random_state=42,
        )

        try:
            # SAP-RPT-1 needs DataFrame with column names
            # The DataLoader returns numpy arrays, convert back to DataFrame
            # But we need to track the original column names

            # For SAP-RPT-1, we need to keep X as DataFrame to preserve column names
            # Let's reload properly
            from tabular_gauge.data.loader import DataLoader

            # Re-load to get DataFrame format
            loader_df = DataLoader(test_size=0.2, random_state=42)
            dataset_df = loader_df.load_openml(
                dataset_id=dataset_cfg["id"],
                dataset_name=dataset_cfg["name"],
            )

            # Get column names before conversion
            column_names = dataset_df.X_train.columns.tolist()

            # Subsample
            if train_size is not None:
                X_train_sub, y_train_sub = stratified_subsample(
                    dataset_df.X_train, dataset_df.y_train,
                    n_samples=train_size,
                )
            else:
                X_train_sub, y_train_sub = dataset_df.X_train, dataset_df.y_train

            actual_size = len(y_train_sub)

            # Run SAP-RPT-1
            result = model.run(
                X_train_sub, y_train_sub,
                dataset_df.X_test, dataset_df.y_test,
                dataset_name=f"{dataset_cfg['name']}_N{train_size or 'full'}",
                seed=42,
            )

            metrics = compute_metrics(result)
            metrics["train_size"] = train_size if train_size else "full"
            metrics["actual_n_train"] = actual_size
            metrics["domain"] = dataset_df.X_train.shape[1]  # Use features as proxy for domain

            # Add domain from config
            if dataset_cfg["name"] in ["diabetes", "wdbc", "sick"]:
                metrics["domain"] = "healthcare"
            elif dataset_cfg["name"] in ["credit-g", "australian"]:
                metrics["domain"] = "finance"
            elif dataset_cfg["name"] == "titanic":
                metrics["domain"] = "historical"
            elif dataset_cfg["name"] == "banknote-auth":
                metrics["domain"] = "security"
            else:
                metrics["domain"] = "other"

            results.append(metrics)
            print(f"      ROC-AUC: {metrics['roc_auc']:.4f}")

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()

    results_df = pd.DataFrame(results)

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, mode='a', header=not Path(output_path).exists())

    return results_df


def run_all_sap_rpt_few_shot():
    """Run SAP-RPT-1 on all few-shot datasets."""
    output_path = "experiments/results/few_shot/sap_rpt_few_shot_results.csv"

    # Clear/create new file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Clear the file if it exists
    if Path(output_path).exists():
        Path(output_path).unlink()

    for dataset_cfg in FEW_SHOT_DATASETS:
        run_sap_rpt_few_shot(
            dataset_cfg,
            TRAIN_SIZES,
            output_path=output_path,
        )

    print(f"\n✅ SAP-RPT-1 Few-Shot Results saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SAP-RPT-1 Few-Shot Analysis")
    parser.add_argument("--datasets", nargs="+", default=None, help="Specific datasets to run")
    parser.add_argument("--output", type=str, default="experiments/results/few_shot/sap_rpt_few_shot_results.csv")
    args = parser.parse_args()

    if args.datasets:
        datasets_to_run = [d for d in FEW_SHOT_DATASETS if d["name"] in args.datasets]
    else:
        datasets_to_run = FEW_SHOT_DATASETS

    for dataset_cfg in datasets_to_run:
        run_sap_rpt_few_shot(
            dataset_cfg,
            TRAIN_SIZES,
            output_path=args.output,
        )
