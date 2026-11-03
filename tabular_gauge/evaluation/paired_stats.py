"""
Paired statistical tests for tabular benchmark comparison.

Why these matter:
    The existing statistical.py module aggregates (dataset, model) cells by
    taking the mean across seeds. That gives *unpaired* Wilcoxon and bootstrap
    CIs. When we have multiple seeds per cell, the right thing to do is pair
    on the seed so the within-cell variance is removed. That is the version
    that the AutoML benchmark community uses (cf. Demšar 2006 + paired
    bootstrap in TabRepo / TabArena write-ups).

Public functions:
    paired_bootstrap_diff(scores_a, scores_b, n_resamples=10_000, seed=0)
        -> dict(mean, ci_low, ci_high, prob_positive)
    cliffs_delta(scores_a, scores_b)
        -> float in [-1, 1]
    holm_bonferroni(pvalues)
        -> np.ndarray of adjusted p-values, same order as input
    per_n_paired_pvalues(df, n_col, model_col, ref, metric)
        -> DataFrame of paired Wilcoxon p-values per N

Conventions:
    Higher metric is better (ROC-AUC, F1, accuracy, etc.). All functions
    return positive numbers indicating "a > b" effect. Pairing is positional;
    the two inputs must be 1D arrays of equal length.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations


def paired_bootstrap_diff(
    scores_a: np.ndarray | list[float],
    scores_b: np.ndarray | list[float],
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict:
    """
    Bootstrap the difference (a - b) of two paired samples.

    Returns the mean diff, the percentile CI, and the probability that
    `a - b > 0` (i.e. the posterior probability that `a` is better than `b`).
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("paired_bootstrap_diff expects two equal-length 1D arrays")
    if a.size < 2:
        # Single observation: return the point estimate, no CI.
        diff = a - b
        return {
            "mean": float(diff.mean()),
            "ci_low": float(diff.mean()),
            "ci_high": float(diff.mean()),
            "prob_positive": float(diff.mean() > 0),
            "n_pairs": int(a.size),
        }

    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_resamples, a.size))
    boot = diff[idx].mean(axis=1)

    alpha = 1.0 - confidence
    ci_low = float(np.percentile(boot, 100 * alpha / 2))
    ci_high = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return {
        "mean": float(diff.mean()),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "prob_positive": float((boot > 0).mean()),
        "n_pairs": int(a.size),
    }


def cliffs_delta(
    scores_a: np.ndarray | list[float],
    scores_b: np.ndarray | list[float],
) -> float:
    """
    Cliff's δ effect size for paired samples (Romano et al. 2006).
    Defined as (#(a_i > b_j) - #(a_i < b_j)) / (n * m).

    For paired samples we use the simpler symmetric form on differences:
        δ = mean(sign(diff)) where ties count as 0.
    Range: [-1, 1]. |δ| < 0.147 is "negligible", < 0.33 small,
    < 0.474 medium, otherwise large.
    """
    diff = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    if diff.size == 0:
        return 0.0
    n = diff.size
    pos = int((diff > 0).sum())
    neg = int((diff < 0).sum())
    return (pos - neg) / n


def holm_bonferroni(pvalues: np.ndarray | list[float]) -> np.ndarray:
    """
    Holm-Bonferroni step-down correction. Family-wise error controlled
    at the level max(input pvalues). Returns adjusted p-values in the
    same order as input; values are monotonically non-decreasing.
    """
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(n, dtype=float)
    running_max = 0.0
    for i in range(n):
        factor = n - i
        adj = ranked[i] * factor
        running_max = max(running_max, adj)
        adjusted[i] = min(running_max, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = adjusted
    return out


def per_n_paired_pvalues(
    df: pd.DataFrame,
    metric: str = "roc_auc",
    n_col: str = "train_size",
    model_col: str = "model",
    ref_model: str = "TabPFN",
    alternative: str = "two-sided",
) -> pd.DataFrame:
    """
    For each value of n_col, run a paired Wilcoxon signed-rank test of every
    model against `ref_model`. Pairing is by (dataset, seed): both models
    must have observations on the same (dataset, seed) cell.

    Returns DataFrame with columns:
        train_size, model, n_pairs, mean_ref, mean_other, diff_mean,
        wilcoxon_stat, wilcoxon_p, cliffs_delta, prob_positive,
        holm_p_adj, significant
    """
    rows = []
    for n_value, sub in df.groupby(n_col):
        ref_vals = sub[sub[model_col] == ref_model].set_index(["dataset", "seed"])[metric]
        for model in sorted(sub[model_col].unique()):
            if model == ref_model:
                continue
            other_vals = sub[sub[model_col] == model].set_index(["dataset", "seed"])[metric]
            paired_idx = ref_vals.index.intersection(other_vals.index)
            if len(paired_idx) < 2:
                continue
            a = ref_vals.loc[paired_idx].values
            b = other_vals.loc[paired_idx].values
            try:
                w_stat, w_p = stats.wilcoxon(a, b, alternative=alternative)
            except ValueError:
                # all differences are zero
                w_stat, w_p = 0.0, 1.0
            boot = paired_bootstrap_diff(a, b, n_resamples=2000)
            rows.append({
                "train_size": n_value,
                "model": model,
                "n_pairs": len(paired_idx),
                "mean_ref": float(a.mean()),
                "mean_other": float(b.mean()),
                "diff_mean": float(a.mean() - b.mean()),
                "wilcoxon_stat": float(w_stat),
                "wilcoxon_p": float(w_p),
                "cliffs_delta": cliffs_delta(a, b),
                "prob_positive_ref": boot["prob_positive"],
                "ci_low": boot["ci_low"],
                "ci_high": boot["ci_high"],
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["holm_p_adj"] = holm_bonferroni(out["wilcoxon_p"].values)
    out["significant"] = out["holm_p_adj"] < 0.05
    return out.sort_values(["train_size", "wilcoxon_p"])


def per_dataset_paired_pvalues(
    df: pd.DataFrame,
    metric: str = "roc_auc",
    dataset_col: str = "dataset",
    model_col: str = "model",
    ref_model: str = "TabPFN",
    alternative: str = "two-sided",
) -> pd.DataFrame:
    """
    For each dataset, paired Wilcoxon across seeds of ref vs every other
    model. Same schema as per_n_paired_pvalues but sliced by dataset.
    """
    rows = []
    for d, sub in df.groupby(dataset_col):
        ref_vals = sub[sub[model_col] == ref_model].set_index("seed")[metric]
        for model in sorted(sub[model_col].unique()):
            if model == ref_model:
                continue
            other_vals = sub[sub[model_col] == model].set_index("seed")[metric]
            paired_idx = ref_vals.index.intersection(other_vals.index)
            if len(paired_idx) < 2:
                continue
            a = ref_vals.loc[paired_idx].values
            b = other_vals.loc[paired_idx].values
            try:
                w_stat, w_p = stats.wilcoxon(a, b, alternative=alternative)
            except ValueError:
                w_stat, w_p = 0.0, 1.0
            boot = paired_bootstrap_diff(a, b, n_resamples=2000)
            rows.append({
                "dataset": d,
                "model": model,
                "n_pairs": len(paired_idx),
                "mean_ref": float(a.mean()),
                "mean_other": float(b.mean()),
                "diff_mean": float(a.mean() - b.mean()),
                "wilcoxon_stat": float(w_stat),
                "wilcoxon_p": float(w_p),
                "cliffs_delta": cliffs_delta(a, b),
                "prob_positive_ref": boot["prob_positive"],
                "ci_low": boot["ci_low"],
                "ci_high": boot["ci_high"],
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["holm_p_adj"] = holm_bonferroni(out["wilcoxon_p"].values)
    out["significant"] = out["holm_p_adj"] < 0.05
    return out.sort_values(["dataset", "wilcoxon_p"])


__all__ = [
    "paired_bootstrap_diff",
    "cliffs_delta",
    "holm_bonferroni",
    "per_n_paired_pvalues",
    "per_dataset_paired_pvalues",
]
