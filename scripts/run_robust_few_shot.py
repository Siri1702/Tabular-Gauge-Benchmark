"""
Robust Few-Shot Learning Analysis - Multi-Seed Statistical Validation

This script performs few-shot learning analysis with:
- 3 seeds for statistical validation
- Mean ± standard deviation computation
- 95% confidence intervals
- Statistical significance tests (Wilcoxon)
- Finer training sizes to find crossover point
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
import json

from tabular_gauge.data.loader import DataLoader
from tabular_gauge.models.tabpfn_model import TabPFNWrapper
from tabular_gauge.models.xgboost_model import XGBoostWrapper
from tabular_gauge.models.lightgbm_model import LightGBMWrapper
from tabular_gauge.models.catboost_model import CatBoostWrapper
from tabular_gauge.models.mlp_model import MLPWrapper
from tabular_gauge.evaluation.metrics import compute_metrics


# Extended training sizes - finer granularity to find crossover
TRAIN_SIZES = [25, 50, 75, 100, 150, 200, 300, 500, 750, 1000, None]

# Selected datasets with varied characteristics
DATASETS = [
    {"id": 37, "name": "diabetes", "domain": "healthcare", "n": 768, "p": 8},
    {"id": 1510, "name": "wdbc", "domain": "healthcare", "n": 569, "p": 30},
    {"id": 40536, "name": "australian", "domain": "finance", "n": 690, "p": 14},
    {"id": 31, "name": "credit-g", "domain": "finance", "n": 1000, "p": 20},
    {"id": 40945, "name": "titanic", "domain": "historical", "n": 891, "p": 11},
    {"id": 1462, "name": "banknote-auth", "domain": "security", "n": 1372, "p": 4},
    {"id": 44, "name": "spambase", "domain": "nlp-derived", "n": 4601, "p": 57},
    {"id": 1067, "name": "kc1", "domain": "software", "n": 2109, "p": 21},
]

N_SEEDS = 3  # For statistical validation


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
        if is_dataframe:
            return X, y_array
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


def run_single_seed(dataset_cfg, train_size, seed):
    """Run a single experiment (one dataset, one train_size, one seed)."""
    loader = DataLoader(test_size=0.2, random_state=42)
    dataset = loader.load_openml(dataset_cfg["id"], dataset_cfg["name"])

    # Subsample training data
    if train_size is not None:
        X_train_sub, y_train_sub = stratified_subsample(
            dataset.X_train, dataset.y_train, n_samples=train_size, random_state=seed
        )
    else:
        X_train_sub, y_train_sub = dataset.X_train, dataset.y_train

    actual_size = len(y_train_sub)

    # Define models with reduced trials for speed
    models = [
        ("TabPFN", TabPFNWrapper(random_state=seed)),
        ("XGBoost", XGBoostWrapper(n_trials=15, random_state=seed)),
        ("LightGBM", LightGBMWrapper(n_trials=15, random_state=seed)),
        ("CatBoost", CatBoostWrapper(n_trials=15, random_state=seed)),
        ("MLP", MLPWrapper(n_trials=10, random_state=seed)),
    ]

    results = []
    for model_name, model in models:
        try:
            result = model.run(
                X_train_sub, y_train_sub,
                dataset.X_test, dataset.y_test,
                dataset_name=f"{dataset_cfg['name']}_N{train_size}",
                seed=seed,
            )
            metrics = compute_metrics(result)
            metrics["model"] = model_name
            metrics["seed"] = seed
            metrics["train_size"] = train_size if train_size else "full"
            metrics["actual_n_train"] = actual_size
            metrics["domain"] = dataset_cfg["domain"]
            metrics["dataset"] = dataset_cfg["name"]
            results.append(metrics)
        except Exception as e:
            print(f"      ERROR ({model_name}, seed={seed}): {e}")

    return pd.DataFrame(results)


def compute_statistics(results_df):
    """Compute mean, std, 95% CI for each model/train_size combination."""
    stats_df = results_df.groupby(["train_size", "model"]).agg(
        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_std=("roc_auc", "std"),
        avg_precision_mean=("avg_precision", "mean"),
        f1_macro_mean=("f1_macro", "mean"),
    ).reset_index()

    # Compute 95% CI using t-distribution
    # CI = mean ± t * (std / sqrt(n))
    n_seeds = N_SEEDS
    t_critical = stats.t.ppf(0.975, df=n_seeds-1)  # 2-tailed, 95%

    # Get std for each group
    std_df = results_df.groupby(["train_size", "model"])["roc_auc"].std().reset_index()
    std_df.columns = ["train_size", "model", "roc_auc_std"]

    # Merge and compute CI
    stats_df = stats_df.merge(std_df, on=["train_size", "model"], how="left")
    stats_df["roc_auc_ci"] = t_critical * (stats_df["roc_auc_std"] / np.sqrt(n_seeds))

    return stats_df


def find_crossover_points(summary_df):
    """Find where TabPFN is overtaken by XGBoost."""
    # Focus on TabPFN vs XGBoost
    tabpfn_df = summary_df[summary_df["model"] == "TabPFN"].copy()
    xgb_df = summary_df[summary_df["model"] == "XGBoost"].copy()

    # Merge
    comparison = tabpfn_df.merge(
        xgb_df,
        on="train_size",
        suffixes=("_tabpfn", "_xgb")
    )

    comparison["tabpfn_wins"] = comparison["roc_auc_mean_tabpfn"] > comparison["roc_auc_mean_xgb"]
    comparison["diff"] = comparison["roc_auc_mean_tabpfn"] - comparison["roc_auc_mean_xgb"]

    # Find crossover
    crossovers = []
    sizes = sorted([s for s in comparison["train_size"].unique() if s != "full"])
    sizes_with_full = sizes + ["full"]

    for i, curr_size in enumerate(sizes_with_full):
        if curr_size == "full":
            continue
        next_idx = i + 1
        if next_idx >= len(sizes_with_full):
            break
        next_size = sizes_with_full[next_idx]

        curr_row = comparison[comparison["train_size"] == curr_size]
        next_row = comparison[comparison["train_size"] == next_size]

        if curr_row["tabpfn_wins"].values[0] and not next_row["tabpfn_wins"].values[0]:
            crossovers.append({
                "crossover_at": next_size,
                "tabpfn_before": curr_row["roc_auc_mean_tabpfn"].values[0],
                "xgb_before": curr_row["roc_auc_mean_xgb"].values[0],
                "tabpfn_after": next_row["roc_auc_mean_tabpfn"].values[0],
                "xgb_after": next_row["roc_auc_mean_xgb"].values[0],
            })

    return pd.DataFrame(crossovers)


def run_analysis(n_seeds=N_SEEDS):
    """Run complete few-shot analysis."""
    print("="*70)
    print("ROBUST FEW-SHOT LEARNING ANALYSIS")
    print("="*70)
    print(f"\nDatasets: {[d['name'] for d in DATASETS]}")
    print(f"Training sizes: {TRAIN_SIZES}")
    print(f"Seeds: {n_seeds} (for statistical validation)")

    all_results = []

    for dataset_cfg in tqdm(DATASETS, desc="Datasets"):
        print(f"\n[{dataset_cfg['name']}] ({dataset_cfg['domain']})")

        for train_size in tqdm(TRAIN_SIZES, desc="  Training sizes", leave=False):
            for seed in range(n_seeds):
                results_df = run_single_seed(dataset_cfg, train_size, seed)
                if not results_df is not None and not results_df.empty:
                    all_results.append(results_df)

    # Combine all results
    combined = pd.concat(all_results, ignore_index=True)

    # Save raw results
    output_dir = project_root / "experiments" / "results" / "few_shot"
    output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_dir / "robust_few_shot_results.csv", index=False)
    print(f"\nRaw results saved to: {output_dir / 'robust_few_shot_results.csv'}")

    # Compute statistics
    stats_df = compute_statistics(combined)
    stats_df.to_csv(output_dir / "few_shot_statistics.csv", index=False)
    print(f"Statistics saved to: {output_dir / 'few_shot_statistics.csv'}")

    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    # Overall performance by training size
    print("\n📊 Mean ROC-AUC by Training Size (across all models and datasets):")
    summary_by_size = stats_df.groupby("train_size")["roc_auc_mean"].mean().sort_index()
    for size, auc in summary_by_size.items():
        print(f"  N={str(size):>5}: {auc:.4f}")

    # Model rankings
    print("\n📊 Overall Model Rankings (mean AUC across all training sizes):")
    model_summary = combined.groupby("model")["roc_auc"].mean().sort_values(ascending=False)
    for model, auc in model_summary.items():
        print(f"  {model:10s}: {auc:.4f}")

    # TabPFN vs XGBoost comparison
    print("\n🎯 TabPFN vs XGBoost by Training Size:")
    pivot = combined.pivot_table(
        index="train_size",
        columns="model",
        values="roc_auc",
        aggfunc="mean"
    )
    pivot = pivot.sort_index()

    for size in pivot.index:
        tabpfn = pivot.loc[size, "TabPFN"] if "TabPFN" in pivot.columns else 0
        xgb = pivot.loc[size, "XGBoost"] if "XGBoost" in pivot.columns else 0
        diff = tabpfn - xgb
        winner = "TabPFN ✅" if diff > 0 else "XGBoost ✅" if diff < 0 else "TIE"
        print(f"  N={str(size):>5}: TabPFN={tabpfn:.4f}, XGBoost={xgb:.4f} | Diff: {diff:+.4f} | {winner}")

    # Crossover analysis
    print("\n🔄 Crossover Point Analysis:")
    crossovers = find_crossover_points(stats_df)
    if not crossovers.empty:
        for _, row in crossovers.iterrows():
            print(f"  Crossover at N={row['crossover_at']}")
            print(f"    Before: TabPFN={row['tabpfn_before']:.4f}, XGBoost={row['xgb_before']:.4f}")
            print(f"    After:  TabPFN={row['tabpfn_after']:.4f}, XGBoost={row['xgb_after']:.4f}")
    else:
        print("  No clear crossover found - TabPFN wins at all sizes")

    # Statistical significance
    print("\n📈 Statistical Analysis:")
    for size in [25, 50, 100, 200]:
        tabpfn_scores = combined[(combined["train_size"] == size) & (combined["model"] == "TabPFN")]["roc_auc"].values
        xgb_scores = combined[(combined["train_size"] == size) & (combined["model"] == "XGBoost")]["roc_auc"].values

        if len(tabpfn_scores) > 0 and len(xgb_scores) > 0:
            # Wilcoxon signed-rank test
            try:
                stat, pval = stats.wilcoxon(tabpfn_scores, xgb_scores, alternative='greater')
                sig = "SIGNIFICANT" if pval < 0.05 else "not significant"
                print(f"  N={size}: Wilcoxon p-value={pval:.4f} ({sig})")
            except:
                print(f"  N={size}: Cannot compute (need paired samples)")

    # Save summary report
    summary_report = {
        "n_datasets": len(DATASETS),
        "n_seeds": n_seeds,
        "training_sizes": TRAIN_SIZES,
        "model_rankings": model_summary.to_dict(),
        "tabpfn_vs_xgb_by_size": pivot[["TabPFN", "XGBoost"]].to_dict(),
    }

    with open(output_dir / "analysis_summary.json", "w") as f:
        json.dump(summary_report, f, indent=2)

    print(f"\n✅ Analysis complete!")
    print(f"📁 Results saved to: {output_dir}")

    return combined, stats_df


if __name__ == "__main__":
    run_analysis(n_seeds=N_SEEDS)