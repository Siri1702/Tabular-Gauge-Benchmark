"""
Statistical significance testing across multiple datasets and models.
Implements Wilcoxon, Friedman, Nemenyi tests and produces data for CD diagrams.

Reference: Demsar, J. (2006). Statistical comparisons of classifiers over
multiple data sets. Journal of Machine Learning Research, 7, 1-30.
"""

import numpy as np
import pandas as pd
from scipy import stats
import scikit_posthocs as sp
from itertools import combinations
from typing import Optional
from tabular_gauge.evaluation.metrics import METRIC_DIRECTIONS

def friedman_test(
    results_df: pd.DataFrame,
    metric: str = "roc_auc",
    alpha: float = 0.05,
) -> dict:
    """
    Run Friedman test to determine if there are significant differences
    among multiple models across multiple datasets.
    """
    pivot = (
        results_df
        .groupby(["dataset", "model"])[metric]
        .mean()
        .unstack("model")
    )
    pivot = pivot.dropna()
    if len(pivot) < 2:
        return {"error": "Insufficient datasets for Friedman test"}
    n_datasets = len(pivot)
    n_models = len(pivot.columns)
    if n_models < 3:
        return {"error": "Friedman test requires at least 3 models"}

    direction = METRIC_DIRECTIONS.get(metric, "higher")
    is_ascending = (direction == "lower")
    ranked = pivot.rank(axis=1, ascending=is_ascending, method="average")
    avg_ranks = ranked.mean()

    k = n_models
    n = n_datasets
    rank_sums_sq = (ranked.sum() ** 2).sum()
    chi_sq = (12 * n * rank_sums_sq / (k * (k + 1))) - 3 * n * (k + 1)

    if n * (k - 1) - chi_sq > 0:
        f_stat = (n - 1) * chi_sq / (n * (k - 1) - chi_sq)
        p_value = 1 - stats.f.cdf(f_stat, k - 1, (k - 1) * (n - 1))
    else:
        p_value = 1 - stats.chi2.cdf(chi_sq, k - 1)

    return {
        "statistic": float(chi_sq),
        "f_statistic": float(f_stat) if n * (k - 1) - chi_sq > 0 else None,
        "p_value": float(p_value),
        "significant": p_value < alpha,
        "n_datasets": n_datasets,
        "n_models": n_models,
        "avg_ranks": avg_ranks.to_dict(),
    }

def nemenyi_cd_test(
    results_df: pd.DataFrame,
    metric: str = "roc_auc",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Nemenyi post-hoc test for pairwise comparisons after Friedman test.
    """
    pivot = (
        results_df
        .groupby(["dataset", "model"])[metric]
        .mean()
        .unstack("model")
    ).dropna()
    n_datasets = len(pivot)
    n_models = len(pivot.columns)
    if n_models < 3 or n_datasets < 2:
        return pd.DataFrame()

    direction = METRIC_DIRECTIONS.get(metric, "higher")
    is_ascending = (direction == "lower")
    ranked = pivot.rank(axis=1, ascending=is_ascending, method="average")
    avg_ranks = ranked.mean()

    q_alpha = {
        3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164
    }.get(n_models, 3.0)
    cd = q_alpha * np.sqrt(n_models * (n_models + 1) / (6 * n_datasets))

    rows = []
    for m1, m2 in combinations(pivot.columns, 2):
        rank_diff = abs(avg_ranks[m1] - avg_ranks[m2])
        significant = rank_diff > cd
        if rank_diff > 0:
            t_stat = rank_diff / np.sqrt(n_models * (n_models + 1) / (6 * n_datasets))
            p_value = 1 - stats.norm.cdf(t_stat)
        else:
            p_value = 1.0
        rows.append({
            "model_a": m1, "model_b": m2, "mean_rank_a": avg_ranks[m1],
            "mean_rank_b": avg_ranks[m2], "rank_diff": rank_diff,
            "critical_difference": cd, "significant": significant, "p_value": p_value,
        })
    return pd.DataFrame(rows).sort_values("rank_diff", ascending=False)

def compute_bootstrap_ci(
    results_df: pd.DataFrame,
    metric: str = "roc_auc",
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> pd.DataFrame:
    alpha = 1 - confidence
    model_metrics = results_df.groupby("model")[metric].apply(list).to_dict()
    rows = []
    for model, values in model_metrics.items():
        if len(values) < 2:
            if len(values) == 1:
                rows.append({"model": model, "mean": np.mean(values), "std": 0.0,
                             "ci_lower": np.mean(values), "ci_upper": np.mean(values), "n_samples": 1})
            continue
        values = np.array(values)
        n = len(values)
        np.random.seed(42)
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(values, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))
        bootstrap_means = np.array(bootstrap_means)
        ci_lower = np.percentile(bootstrap_means, alpha / 2 * 100)
        ci_upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
        rows.append({"model": model, "mean": np.mean(values), "std": np.std(values),
                    "ci_lower": ci_lower, "ci_upper": ci_upper, "n_samples": n})
    return pd.DataFrame(rows).sort_values("mean", ascending=False)

def wilcoxon_pairwise(
    results_df: pd.DataFrame,
    metric: str = "roc_auc",
    alpha: float = 0.05,
) -> pd.DataFrame:
    pivot = (
        results_df
        .groupby(["dataset", "model"])[metric]
        .mean()
        .unstack("model")
    )
    models = pivot.columns.tolist()
    rows = []
    for m1, m2 in combinations(models, 2):
        paired = pivot[[m1, m2]].dropna()
        if len(paired) < 5: continue
        stat, p = stats.wilcoxon(paired[m1], paired[m2], alternative="two-sided")
        rows.append({
            "model_a": m1, "model_b": m2, "n_datasets": len(paired),
            "mean_a": paired[m1].mean(), "mean_b": paired[m2].mean(),
            "statistic": stat, "p_value": p, "significant": p < alpha,
            "better": m1 if paired[m1].mean() > paired[m2].mean() else m2,
        })
    df = pd.DataFrame(rows)
    if df.empty: return df
    return pd.DataFrame(rows).sort_values("p_value")

def average_rank_table(
    results_df: pd.DataFrame,
    metric: str = "roc_auc",
    higher_is_better: Optional[bool] = None,
) -> pd.DataFrame:
    pivot = (
        results_df
        .groupby(["dataset", "model"])[metric]
        .mean()
        .unstack("model")
    )
    if higher_is_better is not None:
        is_ascending = not higher_is_better
    else:
        direction = METRIC_DIRECTIONS.get(metric, "higher")
        is_ascending = (direction == "lower")
    ranked = pivot.rank(axis=1, ascending=is_ascending, method="average")
    avg_ranks = ranked.mean().sort_values()
    return pd.DataFrame({
        "model": avg_ranks.index,
        "average_rank": avg_ranks.values,
        "n_datasets": (~pivot.isna()).sum().values,
    })
