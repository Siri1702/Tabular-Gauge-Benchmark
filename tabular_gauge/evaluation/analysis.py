"""
Single source of truth for every number that appears in reports/summary.md.

All functions take a results DataFrame (or dataset metadata dict) and return
either a DataFrame or a small dict. No side effects, no file writes — the
regenerate_report orchestrator takes the return value and either formats it
into the markdown document or feeds it to a viz function.

Why a separate module: keeping the analysis here means summary.md cannot
drift from the underlying data. When the underlying numbers change, the
orchestrator runs this module and the markdown regenerates.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from .statistical import (
    friedman_test,
    wilcoxon_pairwise,
    nemenyi_cd_test,
    compute_bootstrap_ci,
    average_rank_table,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ORDER = [
    "TabPFN", "SAP-RPT-1", "XGBoost", "LightGBM", "CatBoost", "MLP",
]

# Canonical model-name mapping so the analysis works regardless of how the
# source CSV spells the model names.
ALIASES = {
    "TabPFN": "TabPFN",
    "tabpfn": "TabPFN",
    "SAP-RPT-1": "SAP-RPT-1",
    "sap_rpt_oss": "SAP-RPT-1",
    "SAP-RPT": "SAP-RPT-1",
    "sap_rpt": "SAP-RPT-1",
    "XGBoost": "XGBoost",
    "xgboost": "XGBoost",
    "LightGBM": "LightGBM",
    "lightgbm": "LightGBM",
    "CatBoost": "CatBoost",
    "catboost": "CatBoost",
    "MLP": "MLP",
    "mlp": "MLP",
}


def normalise_model_name(name: str) -> str:
    """Map alternate spellings to the canonical name."""
    if name in ALIASES:
        return ALIASES[name]
    for alias, canonical in ALIASES.items():
        if alias.lower() == str(name).lower():
            return canonical
    return name


def load_csv(path: str | Path = "experiments/results/aggregated/results.csv") -> pd.DataFrame:
    """Load the aggregated results CSV and normalise the model column."""
    df = pd.read_csv(path)
    df["model"] = df["model"].map(normalise_model_name)
    return df


def load_dataset_meta(path: str | Path = "configs/datasets.yaml") -> pd.DataFrame:
    """Load dataset metadata (N, p, domain, imbalance ...) as a DataFrame."""
    cfg = yaml.safe_load(open(path))
    rows = []
    for d in cfg["datasets"]:
        rows.append({
            "dataset":    d["name"],
            "openml_id":  d["id"],
            "n_rows":     d["n_rows"],
            "n_features": d["n_features"],
            "target":     d["target"],
            "domain":     d.get("domain", "unknown"),
            "notes":      d.get("notes", ""),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-metric summary tables
# ---------------------------------------------------------------------------

def headline_table(df: pd.DataFrame, metric: str = "roc_auc") -> pd.DataFrame:
    """
    One-row-per-model summary for the headline metric.

    Columns: model, mean, std, ci_lower, ci_upper, n_datasets
    """
    ci = compute_bootstrap_ci(df, metric=metric, n_bootstrap=1000)
    ci = ci.rename(columns={"mean": "mean", "std": "std",
                             "ci_lower": "ci_lower", "ci_upper": "ci_upper",
                             "model": "model"})
    rows = []
    for model in MODEL_ORDER:
        sub = ci[ci["model"] == model]
        if sub.empty:
            continue
        n = sub["n_samples"].iloc[0] if "n_samples" in sub.columns else 0
        rows.append({
            "model":    model,
            "mean":     sub["mean"].iloc[0],
            "std":      sub["std"].iloc[0],
            "ci_lower": sub["ci_lower"].iloc[0],
            "ci_upper": sub["ci_upper"].iloc[0],
            "n_samples": n,
            "n_datasets": df[df["model"] == model]["dataset"].nunique(),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        # Fallback when bootstrap did not find a row for the model name
        gb = df.groupby("model")[metric].agg(["mean", "std", "count"]).reset_index()
        gb.columns = ["model", "mean", "std", "n_datasets"]
        gb["n_samples"] = gb["n_datasets"]
        gb["ci_lower"] = gb["mean"] - 2 * gb["std"]
        gb["ci_upper"] = gb["mean"] + 2 * gb["std"]
        return gb
    return out.sort_values("mean", ascending=False).reset_index(drop=True)


def per_dataset_table(df: pd.DataFrame, metric: str = "roc_auc") -> pd.DataFrame:
    """
    Rows = datasets, columns = models, cells = mean metric across seeds.

    Sorted by TabPFN column descending.
    """
    pivot = (
        df.groupby(["dataset", "model"])[metric]
        .mean()
        .unstack("model")
    )
    # Reorder columns to MODEL_ORDER and add any missing models as NaN
    for m in MODEL_ORDER:
        if m not in pivot.columns:
            pivot[m] = np.nan
    pivot = pivot[MODEL_ORDER]
    if "TabPFN" in pivot.columns:
        pivot = pivot.sort_values("TabPFN", ascending=False)
    return pivot.round(4)


def average_ranks(df: pd.DataFrame, metric: str = "roc_auc") -> pd.DataFrame:
    """
    Per-model average rank across datasets (1 = best on each dataset).
    Lower is better.
    """
    return average_rank_table(df, metric=metric, higher_is_better=True)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibration_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean ECE-10 and mean MCE per model (the two metrics reported in Section 4.4).
    """
    rows = []
    for model in MODEL_ORDER:
        sub = df[df["model"] == model]
        if sub.empty or "ece_10" not in sub.columns:
            continue
        rows.append({
            "model":               model,
            "mean_ece_10":         sub["ece_10"].mean(),
            "mean_ece_15":         sub.get("ece_15", pd.Series(dtype=float)).mean(),
            "mean_max_cal_err":    sub["max_calibration_error"].mean(),
            "n_datasets":          sub["dataset"].nunique(),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("mean_ece_10").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Timing and memory
# ---------------------------------------------------------------------------

def timing_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-model mean timing breakdown: tuning, fit, predict, total wall time,
    throughput, ms per test row.
    """
    rows = []
    for model in MODEL_ORDER:
        sub = df[df["model"] == model]
        if sub.empty:
            continue
        rows.append({
            "model":                   model,
            "tuning_time_sec":         sub["tuning_time_sec"].mean(),
            "fit_time_sec":            sub["fit_time_sec"].mean(),
            "predict_time_sec":        sub["predict_time_sec"].mean(),
            "total_wall_time_sec":     sub["total_wall_time_sec"].mean(),
            "throughput_rows_per_sec": sub["throughput_rows_per_sec"].mean(),
            "ms_per_test_row":         sub["ms_per_test_row"].mean()
                                       if "ms_per_test_row" in sub.columns else np.nan,
        })
    out = pd.DataFrame(rows).sort_values("total_wall_time_sec").reset_index(drop=True)
    if not out.empty:
        # Speedup vs the first non-zero model (typically XGBoost)
        xgb = out[out["model"] == "XGBoost"]
        if not xgb.empty:
            ref = xgb["total_wall_time_sec"].iloc[0]
            if ref > 0:
                out["speedup_vs_xgboost"] = (ref / out["total_wall_time_sec"]).round(2)
    return out


def memory_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-model mean peak and delta memory.
    """
    if "peak_memory_mb" not in df.columns:
        return pd.DataFrame()
    rows = []
    for model in MODEL_ORDER:
        sub = df[df["model"] == model]
        if sub.empty:
            continue
        rows.append({
            "model":             model,
            "peak_memory_mb":    sub["peak_memory_mb"].mean(),
            "memory_delta_mb":   sub["memory_delta_mb"].mean(),
        })
    return pd.DataFrame(rows).sort_values("peak_memory_mb").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def friedman_summary(df: pd.DataFrame, metric: str = "roc_auc") -> dict:
    """Wrap friedman_test and flatten to a small dict for embedding in markdown."""
    result = friedman_test(df, metric=metric, alpha=0.05)
    out = {
        "statistic":   result.get("statistic"),
        "f_statistic": result.get("f_statistic"),
        "p_value":     result.get("p_value"),
        "significant": result.get("significant"),
        "n_datasets":  result.get("n_datasets"),
        "n_models":    result.get("n_models"),
        "avg_ranks":   result.get("avg_ranks"),
    }
    return out


def nemenyi_table(df: pd.DataFrame, metric: str = "roc_auc") -> tuple[pd.DataFrame, float]:
    """Pairwise Nemenyi post-hoc tests. Returns (df, cd)."""
    ndf = nemenyi_cd_test(df, metric=metric, alpha=0.05)
    cd = float(ndf["critical_difference"].iloc[0]) if not ndf.empty else 0.0
    return ndf.reset_index(drop=True), cd


def wilcoxon_table(df: pd.DataFrame, metric: str = "roc_auc") -> pd.DataFrame:
    """Pairwise Wilcoxon signed-rank tests."""
    return wilcoxon_pairwise(df, metric=metric, alpha=0.05).reset_index(drop=True)


def bootstrap_ci(df: pd.DataFrame, metric: str = "roc_auc") -> pd.DataFrame:
    """Bootstrap 95% CIs for each model on the given metric."""
    return compute_bootstrap_ci(df, metric=metric, n_bootstrap=1000)


# ---------------------------------------------------------------------------
# Win/Loss
# ---------------------------------------------------------------------------

def win_loss_matrix(df: pd.DataFrame, metric: str = "roc_auc") -> dict[str, pd.DataFrame]:
    """
    Compute wins/losses/ties across datasets (mean-per-seed) for each model pair.

    Returns dict with keys 'wins', 'losses', 'ties'; each is a square DataFrame.
    """
    pivot = (
        df.groupby(["dataset", "model"])[metric]
        .mean()
        .unstack("model")
    )
    pivot = pivot[MODEL_ORDER]
    models = list(pivot.columns)
    wins   = pd.DataFrame(0, index=models, columns=models, dtype=int)
    losses = pd.DataFrame(0, index=models, columns=models, dtype=int)
    ties   = pd.DataFrame(0, index=models, columns=models, dtype=int)
    for _, row in pivot.iterrows():
        for m1 in models:
            for m2 in models:
                if m1 == m2:
                    continue
                v1, v2 = row[m1], row[m2]
                if pd.isna(v1) or pd.isna(v2):
                    continue
                if v1 > v2:
                    wins.loc[m1, m2] += 1
                elif v1 < v2:
                    losses.loc[m1, m2] += 1
                else:
                    ties.loc[m1, m2] += 1
    return {"wins": wins, "losses": losses, "ties": ties}


# ---------------------------------------------------------------------------
# Dataset-characteristic analysis
# ---------------------------------------------------------------------------

DATASET_SIZE_BINS = [
    (0,    1000,  "Small (N<1000)"),
    (1000, 3000,  "Medium (1000-3000)"),
    (3000, 1_000_000, "Large (>3000)"),
]

DATASET_FEATURE_BINS = [
    (0,   10, "Low (p<10)"),
    (10,  50, "Medium (10-50)"),
    (50,  1000, "High (>50)"),
]


def _bin_dataset(meta: pd.DataFrame) -> pd.DataFrame:
    """Add `size_bin` and `feature_bin` columns to dataset metadata."""
    meta = meta.copy()
    meta["size_bin"] = pd.cut(
        meta["n_rows"],
        bins=[b[0] for b in DATASET_SIZE_BINS] + [DATASET_SIZE_BINS[-1][1]],
        labels=[b[2] for b in DATASET_SIZE_BINS],
        right=False,
    )
    meta["feature_bin"] = pd.cut(
        meta["n_features"],
        bins=[b[0] for b in DATASET_FEATURE_BINS] + [DATASET_FEATURE_BINS[-1][1]],
        labels=[b[2] for b in DATASET_FEATURE_BINS],
        right=False,
    )
    return meta


def characteristic_table(
    df: pd.DataFrame,
    meta: pd.DataFrame,
    metric: str = "roc_auc",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute the Section 4.7 dataset-characteristic tables.

    Returns 5 DataFrames (in order):
        1) per-dataset mean AUC table
        2) per-size-bin mean AUC per model
        3) per-feature-bin mean AUC per model
        4) per-domain mean AUC per model
        5) per-dataset std across seeds (5-seed stability)
    """
    meta = _bin_dataset(meta).set_index("dataset")
    per_dataset = (
        df.groupby(["dataset", "model"])[metric]
        .mean()
        .unstack("model")
        [MODEL_ORDER]
        .round(4)
    )
    # Add dataset metadata
    per_dataset = per_dataset.join(meta[["n_rows", "n_features", "domain",
                                         "size_bin", "feature_bin"]], how="left")

    by_size = (
        per_dataset.groupby("size_bin", observed=True)[MODEL_ORDER]
        .mean()
        .round(4)
    )
    by_features = (
        per_dataset.groupby("feature_bin", observed=True)[MODEL_ORDER]
        .mean()
        .round(4)
    )
    by_domain = (
        per_dataset.groupby("domain", observed=True)[MODEL_ORDER]
        .mean()
        .round(4)
    )

    # Stability across seeds: per-(dataset, model) std of the metric across seeds.
    stability = (
        df.groupby(["dataset", "model"])[metric]
        .std()
        .unstack("model")
        [MODEL_ORDER]
        .round(4)
    )

    return per_dataset, by_size, by_features, by_domain, stability


# ---------------------------------------------------------------------------
# Summary by dataset-size quantiles (for plots)
# ---------------------------------------------------------------------------

def metric_correlation_with_size(df: pd.DataFrame, meta: pd.DataFrame,
                                  metric: str = "roc_auc") -> pd.DataFrame:
    """
    For each model, compute the Pearson correlation between dataset size (N)
    and that model's mean AUC across datasets.

    A negative correlation implies "performs worse on larger datasets" and
    a positive correlation implies "performs better on larger datasets".
    """
    per = (
        df.groupby(["dataset", "model"])[metric]
        .mean()
        .unstack("model")
        [MODEL_ORDER]
    )
    out = {}
    n_map = meta.set_index("dataset")["n_rows"]
    for model in MODEL_ORDER:
        if model not in per.columns:
            continue
        y = per[model].values
        x = np.array([n_map.get(ds, np.nan) for ds in per.index], dtype=float)
        mask = ~np.isnan(y) & ~np.isnan(x)
        if mask.sum() < 3:
            out[model] = float("nan")
            continue
        out[model] = float(np.corrcoef(x[mask], y[mask])[0, 1])
    return pd.DataFrame({"model": list(out.keys()),
                          "corr_N_vs_" + metric: list(out.values())})


# ---------------------------------------------------------------------------
# All-metric summary (useful for appendix)
# ---------------------------------------------------------------------------

def all_metric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per model, one column per metric. Mean across all rows for that model.
    """
    metric_cols = [
        "roc_auc", "avg_precision", "f1_macro", "f1_weighted",
        "brier_score", "log_loss_val",
        "ece_10", "ece_15", "max_calibration_error",
        "matthews_corrcoef", "balanced_accuracy", "accuracy",
        "sensitivity", "specificity", "ppv",
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]
    rows = []
    for model in MODEL_ORDER:
        sub = df[df["model"] == model]
        if sub.empty:
            continue
        row = {"model": model}
        for c in metric_cols:
            row[f"{c}_mean"] = sub[c].mean()
            row[f"{c}_std"]  = sub[c].std()
        rows.append(row)
    return pd.DataFrame(rows).set_index("model").round(4)
