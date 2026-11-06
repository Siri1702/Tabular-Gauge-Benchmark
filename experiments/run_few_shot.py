"""
Entry point for running few-shot learning analysis.
This script provides a CLI wrapper around the few-shot analysis module.
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from typing import List, Optional
from tqdm import tqdm

# Ensure package is importable
from tabular_gauge.data.loader import DataLoader
from tabular_gauge.models.factory import ModelFactory
from tabular_gauge.evaluation.metrics import compute_metrics
from tabular_gauge.models.base import RunResult

# Import custom wrappers specifically for few-shot
from tabular_gauge.models.tabpfn_model import TabPFNWrapper
from tabular_gauge.models.xgboost_model import XGBoostWrapper
from tabular_gauge.models.lightgbm_model import LightGBMWrapper
from tabular_gauge.models.catboost_model import CatBoostWrapper
from tabular_gauge.models.mlp_model import MLPWrapper

def subsample_train_data(X_train: np.ndarray, y_train: np.ndarray,
                         n_samples: int, random_state: int = 42) -> tuple:
    """Stratified subsample of training data."""
    if n_samples is None:
        return X_train, y_train
    # Ensure inputs are numpy arrays to avoid Pandas index-based slicing errors
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    n_samples = min(n_samples, len(y_train))
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
    return X_train[np.sort(selected_indices)], y_train[np.sort(selected_indices)]

def run_few_shot_benchmark(
    dataset_names: List[str] = None,
    train_sizes: List[Optional[int]] = [50, 100, 200, 500, 1000, None],
    n_seeds: int = 3,
    output_dir: str = "experiments/results/few_shot",
):
    """Run complete few-shot benchmark across multiple datasets."""
    try:
        with open("configs/datasets.yaml") as f:
            dataset_cfgs = yaml.safe_load(f)["datasets"]
        with open("configs/experiment.yaml") as f:
            exp_cfg = yaml.safe_load(f)["experiment"]
    except Exception as e:
        print(f"ERROR: Config load failed: {e}")
        return

    exp_name = exp_cfg.get("name", "default_experiment")
    # Use a dedicated few_shot subfolder under the results base to keep things clean
    output_dir = Path(output_dir) / exp_name

    if dataset_names:
        dataset_cfgs = [d for d in dataset_cfgs if d["name"] in dataset_names]

    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    loader = DataLoader(test_size=exp_cfg.get("test_size", 0.2), random_state=42)

    for dataset_cfg in dataset_cfgs:
        name = dataset_cfg["name"]
        print(f"\n{'='*60}\nFew-Shot: {name}\n{'='*60}")

        try:
            if dataset_cfg.get("source") == "csv":
                dataset = loader.load_csv(dataset_cfg["path"], dataset_cfg["target"], name)
            else:
                dataset = loader.load_openml(dataset_cfg["id"], name)
        except Exception as e:
            print(f"  ERROR: Failed to load dataset {name}: {e}")
            continue

        dataset_results = []
        total_train_available = len(dataset.y_train)

        # Optimization: Prune train_sizes for this dataset.
        # Keep sizes only if they are smaller than the total available.
        filtered_sizes = [s for s in train_sizes if s is not None and s < total_train_available]

        # If any requested size was >= total_train_available, or if 'None' was explicitly requested,
        # we ensure the 'None' (Full) bucket is included.
        should_include_full = (None in train_sizes) or any(s is not None and s >= total_train_available for s in train_sizes)

        if should_include_full:
            filtered_sizes.append(None)

        # Final safety: If no sizes were selected (e.g. all were pruned and no None), default to Full.
        if not filtered_sizes:
            filtered_sizes = [None]

        for train_size in tqdm(filtered_sizes, desc=f"Sizes for {name}"):
            for seed in range(n_seeds):
                X_tr, y_tr = subsample_train_data(dataset.X_train, dataset.y_train, train_size, seed)

                # Use ModelFactory for consistency
                models = []
                try:
                    # We define a temporary few-shot config set
                    fs_models = ["tabpfn", "xgboost", "lightgbm", "catboost", "mlp"]
                    for mid in fs_models:
                        # In a real scenario, we'd load these from models.yaml
                        # Here we instantiate the known wrappers for brevity
                        if mid == "tabpfn": models.append(TabPFNWrapper(random_state=seed))
                        elif mid == "xgboost": models.append(XGBoostWrapper(random_state=seed))
                        elif mid == "lightgbm": models.append(LightGBMWrapper(random_state=seed))
                        elif mid == "catboost": models.append(CatBoostWrapper(random_state=seed))
                        elif mid == "mlp": models.append(MLPWrapper(random_state=seed))
                except Exception as e:
                    print(f"  Warning: Model init failed: {e}")

                for model in models:
                    try:
                        res = model.run(X_tr, y_tr, dataset.X_test, dataset.y_test, f"{name}_N{train_size}", seed)
                        metrics = compute_metrics(res)
                        metrics["train_size"] = train_size if train_size else "full"
                        dataset_results.append(metrics)
                    except Exception as e:
                        print(f"  Error {model.name} N={train_size}: {e}")
                        dataset_results.append({"model": model.name, "dataset": name, "seed": seed, "train_size": train_size, "error": str(e)})

        if dataset_results:
            res_df = pd.DataFrame(dataset_results)
            res_df.to_csv(f"{output_dir}/few_shot_{name}.csv", index=False)
            all_results.append(res_df)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(f"{output_dir}/all_few_shot_results.csv", index=False)
        print(f"\n✅ Few-Shot Analysis Complete. Results saved to {output_dir}")
        return combined

    print("\nNo results generated.")
    return pd.DataFrame()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run few-shot learning analysis")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--train_sizes", nargs="+", type=int, default=None)
    parser.add_argument("--n_seeds", type=int, default=3)
    args = parser.parse_args()

    sizes = args.train_sizes if args.train_sizes else [50, 100, 200, 500, 1000, None]
    run_few_shot_benchmark(dataset_names=args.datasets, train_sizes=sizes, n_seeds=args.n_seeds)
