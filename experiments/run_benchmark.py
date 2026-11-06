#!/usr/bin/env python3
"""
Main benchmark runner v2.
Integrated with ModelFactory and supporting Custom CSV datasets.
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

# Pin matplotlib backend to Agg to prevent Tkinter crashes on Windows
os.environ.setdefault("MPLBACKEND", "Agg")
# Silence joblib warnings
warnings.filterwarnings("ignore", message=".*resource_tracker.*")
warnings.filterwarnings("ignore", message=".*joblib_memmapping_folder.*")

# Force UTF-8 stdout/stderr for Windows compatibility
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from tabular_gauge.data.loader import DataLoader
from tabular_gauge.models.factory import ModelFactory
from tabular_gauge.evaluation.metrics import compute_metrics
from tabular_gauge.evaluation.statistical import (
    wilcoxon_pairwise,
    average_rank_table,
    friedman_test,
    nemenyi_cd_test,
    compute_bootstrap_ci,
)
from tabular_gauge.viz.leaderboard import plot_auc_heatmap, plot_average_ranks
from tabular_gauge.viz.leaderboard import plot_timing_comparison, plot_memory_comparison
from tabular_gauge.viz.statistical import (
    plot_critical_difference_diagram,
    plot_confidence_intervals,
    plot_statistical_significance_heatmap,
    plot_win_loss_matrix,
)
from tabular_gauge.viz.curves import plot_calibration_comparison
from tabular_gauge.evaluation.profiling import get_system_info

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def get_existing_datasets(exp_name: str) -> set:
    raw_dir = Path("experiments/results/raw") / exp_name
    existing = set()
    if not raw_dir.exists():
        return existing
    for json_file in raw_dir.glob("*.json"):
        # We expect cache files to be {dataset}.cache.json
        # Only add if it's a valid dataset result (not a generic config)
        if "cache" in json_file.name:
            dataset_name = json_file.stem.replace(".cache", "")
            existing.add(dataset_name)
    return existing

def load_existing_aggregated_results(exp_name: str) -> pd.DataFrame:
    agg_dir = Path("experiments/results/aggregated") / exp_name
    if not agg_dir.exists():
        return pd.DataFrame()
    results_csv = agg_dir / "results.csv"
    if results_csv.exists():
        return pd.read_csv(results_csv)
    csv_files = list(agg_dir.glob("results_*.csv"))
    if not csv_files:
        return pd.DataFrame()
    latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
    return pd.read_csv(latest_csv)

def build_models(exp_cfg: dict, seed: int, skip_models: list = None) -> list:
    """Instantiate all model wrappers via ModelFactory based on models.yaml."""
    skip = set(skip_models or [])
    try:
        with open("configs/models.yaml") as f:
            models_cfg = yaml.safe_load(f)["models"]
    except Exception as e:
        print(f"  ERROR: Could not load configs/models.yaml: {e}")
        return []

    if isinstance(models_cfg, list):
        model_entries = [(m["id"], m) for m in models_cfg if m.get("enabled", True) and m["id"] not in skip]
    else:
        model_entries = [(mid, cfg) for mid, cfg in models_cfg.items() if mid not in skip]

    initialized_models = []
    for model_id, config in model_entries:
        try:
            model = ModelFactory.create(model_id, config, seed)
            initialized_models.append(model)
        except Exception as e:
            print(f"  Warning: Could not initialize model {model_id}: {e}")
            continue
    return initialized_models

def run_single(dataset_cfg: dict, exp_cfg: dict, exp_name: str, skip_models: list = None) -> list[dict]:
    """
    Run all models x all seeds on a single dataset.
    """
    name = dataset_cfg["name"]
    print(f"\n[run_single] dataset={name} exp={exp_name}", flush=True)

    loader = DataLoader(
        test_size=exp_cfg.get("test_size", 0.2),
        random_state=42,
    )

    try:
        source = dataset_cfg.get("source", "openml")
        if source == "openml":
            dataset = loader.load_openml(dataset_id=dataset_cfg["id"], dataset_name=name)
        elif source == "csv":
            dataset = loader.load_csv(path=dataset_cfg["path"], target_col=dataset_cfg["target"], name=name)
        else:
            raise ValueError(f"Unsupported dataset source '{source}'")
    except Exception as e:
        print(f"  ERROR: Failed to load dataset {name}: {e}")
        return []

    if dataset.n_rows == 0:
        print(f"  ERROR: Dataset {name} is empty.")
        return []

    if dataset.n_rows > exp_cfg.get("max_rows", 10000):
        print(f"  Skipping {name}: {dataset.n_rows} rows > limit")
        return []

    seeds = list(range(exp_cfg.get("n_seeds", 5)))
    cache_enabled = exp_cfg.get("cache_results", True)
    cache_path = Path("experiments/results/raw") / exp_name / f"{name}.cache.json"

    cached_rows = []
    if cache_enabled and cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                cached_rows = json.load(f)
            print(f"  Loaded {len(cached_rows)} cached results for {name}")
        except Exception as e:
            print(f"  Warning: Failed to load cache for {name}: {e}")
            cached_rows = []

    cached_by_key = {(r["model"], r["seed"]): r for r in cached_rows}
    all_metrics = list(cached_rows)
    new_count = 0

    def _flush_cache():
        if not cache_enabled or not all_metrics:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            tmp_path = cache_path.with_suffix(".tmp")
            with open(tmp_path, "w") as f:
                json.dump(all_metrics, f, indent=2, default=str)
            os.replace(tmp_path, cache_path)
        except Exception as e:
            print(f"  Warning: Failed to flush cache for {name}: {e}")

    for seed in seeds:
        models = build_models(exp_cfg, seed, skip_models=skip_models)
        for model in models:
            key = (model.name, seed)
            if key in cached_by_key:
                continue

            print(f"    {model.name:12s} | seed={seed} ...", end=" ", flush=True)
            try:
                result = model.run(
                    dataset.X_train, dataset.y_train,
                    dataset.X_test,  dataset.y_test,
                    dataset_name=name,
                    seed=seed,
                )
                metrics = compute_metrics(result)
            except Exception as exc:
                print(f"ERROR: {type(exc).__name__}: {exc}")
                metrics = {
                    "model": model.name, "dataset": name,
                    "seed": seed, "error": str(exc),
                    "n_train": dataset.X_train.shape[0],
                    "n_test": dataset.X_test.shape[0],
                    "n_features": dataset.X_train.shape[1],
                    "fit_time_sec": 0.0, "predict_time_sec": 0.0,
                    "roc_auc": float("nan"), "accuracy": float("nan"),
                }

            all_metrics.append(metrics)
            cached_by_key[key] = metrics
            new_count += 1

            status = f"AUC={metrics.get('roc_auc', float('nan')):.4f}" if metrics.get("error") is None else f"ERROR"
            print(f"{status}")
            _flush_cache()

    if cache_enabled and all_metrics:
        _flush_cache()
        if new_count > 0:
            print(f"  -> Updated cache: {len(all_metrics)} rows ({new_count} new)")

    return all_metrics

def main():
    parser = argparse.ArgumentParser(description="Run Tabular-Gauge benchmark v2")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--n_seeds", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip_models", nargs="+", default=None)
    args = parser.parse_args()

    try:
        dataset_configs_raw = load_config("configs/datasets.yaml")
        dataset_cfgs_all = dataset_configs_raw["datasets"]
        exp_cfg = load_config("configs/experiment.yaml")["experiment"]
    except Exception as e:
        print(f"ERROR: Config load failed: {e}")
        sys.exit(1)

    exp_name = exp_cfg.get("name", "default_experiment")
    if args.n_seeds:
        exp_cfg["n_seeds"] = args.n_seeds

    if isinstance(dataset_cfgs_all, list):
        dataset_cfgs = [d for d in dataset_cfgs_all if d["name"] in (args.datasets or [d["name"] for d in dataset_cfgs_all])]
    else:
        requested = args.datasets or list(dataset_cfgs_all.keys())
        dataset_cfgs = []
        for rid in requested:
            if rid in dataset_cfgs_all:
                cfg = dataset_cfgs_all[rid].copy()
                cfg["name"] = rid
                dataset_cfgs.append(cfg)

    if args.all:
        if isinstance(dataset_cfgs_all, list):
            dataset_cfgs = dataset_cfgs_all
        else:
            dataset_cfgs = [{**v, "name": k} for k, v in dataset_cfgs_all.items()]

    if args.force:
        print(f"  -> --force: Clearing results for '{exp_name}'...")
        raw_dir = Path("experiments/results/raw") / exp_name
        if raw_dir.exists():
            for f in raw_dir.glob("*.json"): f.unlink()
        agg_dir = Path("experiments/results/aggregated") / exp_name
        if agg_dir.exists():
            results_csv = agg_dir / "results.csv"
            if results_csv.exists(): results_csv.unlink()
        existing_datasets = set()
        existing_df = pd.DataFrame()
    else:
        existing_datasets = get_existing_datasets(exp_name)
        existing_df = load_existing_aggregated_results(exp_name)
        if not args.datasets:
            dataset_cfgs = [d for d in dataset_cfgs if d["name"] not in existing_datasets]

    cumulative_df = existing_df.copy() if not existing_df.empty else pd.DataFrame()

    for dataset_cfg in dataset_cfgs:
        print(f"\n{'='*60}\nDataset: {dataset_cfg['name']}\n{'='*60}")
        metrics_list = run_single(dataset_cfg, exp_cfg, exp_name, skip_models=args.skip_models)

        if metrics_list:
            new_df = pd.DataFrame(metrics_list)
            if not cumulative_df.empty:
                # Deduplicate based on (model, dataset, seed)
                existing_keys = set(zip(cumulative_df['model'], cumulative_df['dataset'], cumulative_df['seed']))
                new_unique = new_df[~new_df.apply(lambda row: (row['model'], row['dataset'], row['seed']) in existing_keys, axis=1)]
                cumulative_df = pd.concat([cumulative_df, new_unique], ignore_index=True) if not new_unique.empty else cumulative_df
            else:
                cumulative_df = new_df

            agg_path = Path("experiments/results/aggregated") / exp_name / "results.csv"
            agg_path.parent.mkdir(parents=True, exist_ok=True)
            cumulative_df.to_csv(agg_path, index=False)

    if cumulative_df.empty:
        print("\nNo results available.")
        return

    print("\n--- Final Summary ---")
    from tabular_gauge.evaluation.analysis import average_ranks
    rank_df = average_ranks(cumulative_df, metric="roc_auc")
    print(rank_df.to_string(index=False))

    sys_info = get_system_info()
    for k, v in sys_info.items():
        print(f"  {k}: {v}")

    print("\n[OK] Benchmark complete.")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        sys.exit(1)
