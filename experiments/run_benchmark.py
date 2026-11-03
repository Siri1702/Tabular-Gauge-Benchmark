"""
Main benchmark runner v2.
Integrated with ModelFactory and supporting Custom CSV datasets.

Usage:
    python experiments/run_benchmark_v2.py --datasets credit-g my_custom_csv --all
    python experiments/run_benchmark_v2.py --datasets credit-g --n_seeds 5
    python experiments/run_benchmark_v2.py --all --force
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

# Remove sys.path.insert(0, ...) as the package is now installable
# sys.path.insert(0, str(Path(__file__).parent.parent))

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
        parts = json_file.stem.split("_")
        if len(parts) >= 3:
            dataset_name = "_".join(parts[2:])
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
    with open("configs/models.yaml") as f:
        models_cfg = yaml.safe_load(f)["models"]

    # If models_cfg is a list (new flexible format), we need to handle it.
    # Otherwise, it's a dict (original format).
    if isinstance(models_cfg, list):
        # Extract model_id and config from list items
        model_entries = [(m["id"], m) for m in models_cfg if m.get("enabled", True) and m["id"] not in skip]
    else:
        # Legacy dict format: keys are model_ids, values are configs
        model_entries = [(mid, cfg) for mid, cfg in models_cfg.items() if mid not in skip]

    initialized_models = []
    for model_id, config in model_entries:
        try:
            model = ModelFactory.create(model_id, config, seed)
            initialized_models.append(model)
        except (ImportError, ValueError, TypeError) as e:
            print(f"  Warning: Could not initialize model {model_id}: {e}")
            continue
    return initialized_models

def run_single(dataset_cfg: dict, exp_cfg: dict, exp_name: str, skip_models: list = None) -> list[dict]:
    """
    Run all models x all seeds on a single dataset.
    Supports both OpenML and CSV sources.
    """
    name = dataset_cfg["name"]
    print(f"[run_single] start: dataset={name} exp={exp_name} skip_models={skip_models}", flush=True)

    loader = DataLoader(
        test_size=exp_cfg.get("test_size", 0.2),
        random_state=42,
    )

    try:
        source = dataset_cfg.get("source", "openml")
        if source == "openml":
            print(f"  Loading OpenML dataset {dataset_cfg['id']} ({name})...", flush=True)
            dataset = loader.load_openml(
                dataset_id=dataset_cfg["id"],
                dataset_name=name,
            )
        elif source == "csv":
            print(f"  Loading CSV dataset from {dataset_cfg['path']} ({name})...", flush=True)
            dataset = loader.load_csv(
                path=dataset_cfg["path"],
                target_col=dataset_cfg["target"],
                name=name,
            )
        else:
            raise ValueError(f"Unsupported dataset source '{source}' for dataset {name}")
    except Exception as e:
        print(f"  ERROR: Failed to load dataset {name}: {e}")
        return []

    print(f"  Loaded {dataset.n_rows} rows, {dataset.n_features} features", flush=True)

    if dataset.n_rows > exp_cfg.get("max_rows", 10000):
        print(f"  Skipping {name}: {dataset.n_rows} rows > limit")
        return []

    seeds = list(range(exp_cfg.get("n_seeds", 5)))
    cache_enabled = exp_cfg.get("cache_results", True)
    cache_path = Path(f"experiments/results/raw/{exp_name}/{name}.cache.json")

    cached_rows = []
    if cache_enabled and cache_path.exists():
        with open(cache_path, "r") as f:
            cached_rows = json.load(f)
        print(f"  Loaded {len(cached_rows)} cached results for {name}")

    cached_by_key = {(r["model"], r["seed"]): r for r in cached_rows}
    all_metrics = list(cached_rows)
    new_count = 0

    def _flush_cache():
        if not cache_enabled or not all_metrics:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(".cache.json.tmp")
        with open(tmp_path, "w") as f:
            json.dump(all_metrics, f, indent=2, default=str)
        os.replace(tmp_path, cache_path)

    for seed in seeds:
        models = build_models(exp_cfg, seed, skip_models=skip_models)
        print(f"  [seed {seed}] {len(models)} models -> {[m.name for m in models]}")
        for model in models:
            key = (model.name, seed)
            if key in cached_by_key:
                print(f"    {model.name:12s} | seed={seed} (cached, skipped)")
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
                error_row = {
                    "model": model.name, "dataset": name,
                    "seed": seed, "error": str(exc),
                    "n_train": dataset.X_train.shape[0],
                    "n_test": dataset.X_test.shape[0],
                    "n_features": dataset.X_train.shape[1],
                    "fit_time_sec": 0.0, "predict_time_sec": 0.0,
                    "roc_auc": float("nan"), "accuracy": float("nan"),
                }
                all_metrics.append(error_row)
                cached_by_key[key] = error_row
                new_count += 1
                _flush_cache()
                continue

            all_metrics.append(metrics)
            cached_by_key[key] = metrics
            new_count += 1
            status = f"AUC={metrics.get('roc_auc', float('nan')):.4f}" if metrics.get("error") is None else f"ERROR: {metrics['error']}"
            print(f"{status} ({result.fit_time_sec:.1f}s fit)")
            _flush_cache()

    if cache_enabled and all_metrics:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "r") as f:
            # This was a bug in my previous draft; it should be write "w"
            pass
        with open(cache_path, "w") as f:
            json.dump(all_metrics, f, indent=2, default=str)
        if new_count > 0:
            print(f"  -> Wrote {len(all_metrics)} rows ({new_count} new) to {cache_path.name}")

    return all_metrics

def _generate_reports(results_df: pd.DataFrame, exp_name: str):
    if results_df.empty:
        return
    n_datasets = results_df["dataset"].nunique()
    print(f"  -> Generating reports for {n_datasets} datasets...")
    try:
        rank_df = average_rank_table(results_df, metric="roc_auc")

        # Namespaced reports directory
        report_dir = Path("reports/figures") / exp_name
        report_dir.mkdir(parents=True, exist_ok=True)

        plot_auc_heatmap(results_df, save_path=str(report_dir / "auc_heatmap.png"))
        plot_average_ranks(rank_df, save_path=str(report_dir / "average_ranks.png"))
        plot_timing_comparison(results_df, save_path=str(report_dir / "timing_comparison.png"))
        plot_memory_comparison(results_df, save_path=str(report_dir / "memory_comparison.png"))
        plot_win_loss_matrix(results_df, metric="roc_auc", save_path=str(report_dir / "win_loss_matrix.png"))
        if "ece_10" in results_df.columns:
            plot_calibration_comparison(results_df, save_path=str(report_dir / "calibration_comparison.png"))
        print(f"  -> Reports updated successfully in {report_dir}")
    except Exception as e:
        print(f"  Warning: Report generation partially failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run Tabular-Gauge benchmark v2")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--n_seeds", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip_models", nargs="+", default=None)
    args = parser.parse_args()

    # Load configs
    dataset_configs_raw = load_config("configs/datasets.yaml")
    dataset_cfgs_all = dataset_configs_raw["datasets"]
    exp_cfg = load_config("configs/experiment.yaml")["experiment"]

    # Experiment name for output namespacing
    exp_name = exp_cfg.get("name", "default_experiment")

    if args.n_seeds:
        exp_cfg["n_seeds"] = args.n_seeds

    # Handle dataset selection (handle both list and dict formats)
    if isinstance(dataset_cfgs_all, list):
        dataset_cfgs = [d for d in dataset_cfgs_all if d["name"] in (args.datasets or [d["name"] for d in dataset_cfgs_all])]
    else:
        # If dict, dataset_cfgs_all.items() gives (id, config)
        requested = args.datasets or list(dataset_cfgs_all.keys())
        dataset_cfgs = []
        for rid in requested:
            if rid in dataset_cfgs_all:
                cfg = dataset_cfgs_all[rid].copy()
                cfg["name"] = rid
                dataset_cfgs.append(cfg)

    # For --all, we take everything
    if args.all:
        if isinstance(dataset_cfgs_all, list):
            dataset_cfgs = dataset_cfgs_all
        else:
            dataset_cfgs = [{**v, "name": k} for k, v in dataset_cfgs_all.items()]

    Path("experiments/results/raw").mkdir(parents=True, exist_ok=True)
    Path("experiments/results/aggregated").mkdir(parents=True, exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    existing_datasets = get_existing_datasets(exp_name)
    existing_df = load_existing_aggregated_results(exp_name)

    if not args.force and not args.datasets:
        dataset_cfgs = [d for d in dataset_cfgs if d["name"] not in existing_datasets]

    all_results = []
    cumulative_df = existing_df.copy() if not existing_df.empty else pd.DataFrame()

    for dataset_cfg in dataset_cfgs:
        print(f"\n{'='*60}\nDataset: {dataset_cfg['name']}\n{'='*60}")
        metrics_list = run_single(dataset_cfg, exp_cfg, exp_name, skip_models=args.skip_models)
        all_results.extend(metrics_list)

        if metrics_list:
            new_df = pd.DataFrame(metrics_list)
            if not cumulative_df.empty:
                existing_keys = set(zip(cumulative_df['model'], cumulative_df['dataset'], cumulative_df['seed']))
                new_unique = new_df[~new_df.apply(lambda row: (row['model'], row['dataset'], row['seed']) in existing_keys, axis=1)]
                results_df = pd.concat([cumulative_df, new_unique], ignore_index=True) if len(new_unique) > 0 else cumulative_df
            else:
                results_df = new_df

            cumulative_df = results_df.copy()

            # Save to namespaced aggregated results
            agg_path = Path("experiments/results/aggregated") / exp_name / "results.csv"
            agg_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(agg_path, index=False)

            _generate_reports(results_df, exp_name)

    if not all_results and not dataset_cfgs:
        results_df = existing_df
    elif not all_results:
        results_df = existing_df
    else:
        results_df = cumulative_df

    if results_df.empty:
        print("No results available.")
        return

    # Final summary stats
    rank_df = average_rank_table(results_df, metric="roc_auc")
    print("\n--- Average Ranks ---\n", rank_df.to_string(index=False))

    # System Info
    sys_info = get_system_info()
    for k, v in sys_info.items():
        print(f"  {k}: {v}")

    print("\n[OK] Benchmark v2 complete.")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        sys.exit(1)
