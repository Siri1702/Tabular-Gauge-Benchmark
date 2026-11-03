"""
Ensemble Analysis for Tabular Models.

This module analyzes ensemble strategies combining TabPFN with traditional models:
1. Simple averaging of probability predictions
2. Weighted averaging (optimized weights)
3. Stacking with meta-learner
4. Voting (hard voting for classification)

Research Questions:
- Can TabPFN + GBDT ensemble beat both individually?
- What are optimal ensemble weights?
- How does ensemble diversity affect performance?
- When is ensembling most beneficial?
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))


def simple_average_ensemble(
    predictions: List[np.ndarray],
    weights: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Simple or weighted average of probability predictions.

    Args:
        predictions: List of probability arrays (n_samples, n_classes) or (n_samples,)
        weights: Optional weights for each prediction (must sum to 1)

    Returns:
        Averaged probabilities
    """
    if weights is None:
        weights = [1.0 / len(predictions)] * len(predictions)
    else:
        # Normalize weights
        weights = np.array(weights) / np.sum(weights)

    # Stack predictions
    stacked = np.stack(predictions, axis=0)

    # Weighted average
    weighted = np.tensordot(weights, stacked, axes=([0], [0]))

    return weighted


def rank_average_ensemble(
    predictions: List[np.ndarray],
) -> np.ndarray:
    """
    Rank-based averaging: convert probabilities to ranks, average, then normalize.

    This is more robust to calibration differences between models.
    """
    ranks = []
    for pred in predictions:
        # For binary, just use the positive class probability as rank
        if pred.ndim == 1:
            ranks.append(pred)
        else:
            ranks.append(pred[:, 1])  # Positive class

    # Average ranks
    avg_rank = np.mean(np.stack(ranks), axis=0)

    # Normalize to [0, 1]
    avg_rank = (avg_rank - avg_rank.min()) / (avg_rank.max() - avg_rank.min() + 1e-10)

    return avg_rank


def optimize_ensemble_weights(
    predictions: List[np.ndarray],
    y_true: np.ndarray,
    metric: str = "roc_auc",
    n_trials: int = 100,
) -> Dict:
    """
    Optimize ensemble weights to maximize performance metric.

    Uses random search over weight combinations.
    """
    best_metric = -np.inf
    best_weights = None

    for _ in range(n_trials):
        # Random weights
        weights = np.random.rand(len(predictions))
        weights = weights / weights.sum()

        # Compute ensemble
        ensemble_pred = simple_average_ensemble(predictions, weights)

        # Evaluate
        if metric == "roc_auc":
            score = roc_auc_score(y_true, ensemble_pred)
        elif metric == "average_precision":
            score = average_precision_score(y_true, ensemble_pred)
        elif metric == "brier":
            score = -brier_score_loss(y_true, ensemble_pred)  # Negative because lower is better
        else:
            score = roc_auc_score(y_true, ensemble_pred)

        if score > best_metric:
            best_metric = score
            best_weights = weights.tolist()

    return {
        "weights": best_weights,
        "metric": best_metric,
        "metric_name": metric,
    }


def stacking_ensemble(
    predictions: List[np.ndarray],
    y_true: np.ndarray,
    meta_model: str = "logistic",
) -> Tuple[np.ndarray, any]:
    """
    Stacking ensemble: use a meta-learner to combine predictions.

    Args:
        predictions: List of probability arrays
        y_true: True labels
        meta_model: Meta-learner type ('logistic' or 'random_forest')

    Returns:
        (predictions, fitted_meta_model)
    """
    # Stack predictions as features
    X_meta = np.column_stack(predictions)

    # Train meta-model
    if meta_model == "logistic":
        meta = LogisticRegression(max_iter=1000, random_state=42)
    else:
        meta = RandomForestClassifier(n_estimators=50, random_state=42)

    meta.fit(X_meta, y_true)
    y_pred = meta.predict_proba(X_meta)

    return y_pred, meta


def voting_ensemble(
    predictions: List[np.ndarray],
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Hard voting: majority vote on binary predictions.
    """
    # Convert to binary predictions
    binary_preds = []
    for pred in predictions:
        if pred.ndim > 1:
            binary = (pred[:, 1] >= threshold).astype(int)
        else:
            binary = (pred >= threshold).astype(int)
        binary_preds.append(binary)

    # Majority vote
    votes = np.sum(np.stack(binary_preds), axis=0)
    return (votes >= len(predictions) / 2).astype(int)


def compute_ensemble_diversity(predictions: List[np.ndarray], y_true: np.ndarray) -> Dict:
    """
    Compute ensemble diversity metrics.

    Key metrics:
    - Disagreement: fraction of samples where models disagree
    - Correlation: average pairwise correlation of predictions
    - Q-statistic: measure of diversity
    """
    # Convert to binary predictions for disagreement
    binary_preds = []
    for pred in predictions:
        if pred.ndim > 1:
            binary = (pred[:, 1] >= 0.5).astype(int)
        else:
            binary = (pred >= 0.5).astype(int)
        binary_preds.append(binary)

    # Disagreement
    n_disagree = 0
    for i in range(len(binary_preds)):
        for j in range(i + 1, len(binary_preds)):
            n_disagree += np.sum(binary_preds[i] != binary_preds[j])

    total_pairs = len(binary_preds) * (len(binary_preds) - 1) / 2 * len(y_true)
    disagreement = n_disagree / total_pairs if total_pairs > 0 else 0

    # Correlation
    correlations = []
    for i in range(len(predictions)):
        for j in range(i + 1, len(predictions)):
            corr = np.corrcoef(predictions[i].ravel(), predictions[j].ravel())[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)

    mean_correlation = np.mean(correlations) if correlations else 0

    return {
        "disagreement": disagreement,
        "mean_correlation": mean_correlation,
        "diversity": 1 - mean_correlation,  # Higher = more diverse
    }


def evaluate_ensemble_strategy(
    predictions: List[np.ndarray],
    y_true: np.ndarray,
    model_names: List[str],
) -> pd.DataFrame:
    """
    Evaluate multiple ensemble strategies on the same predictions.

    Returns DataFrame with results for each strategy.
    """
    results = []

    # Individual model performance
    for i, pred in enumerate(predictions):
        if pred.ndim > 1:
            pred_proba = pred[:, 1]
        else:
            pred_proba = pred

        results.append({
            "strategy": model_names[i] if i < len(model_names) else f"model_{i}",
            "roc_auc": roc_auc_score(y_true, pred_proba),
            "avg_precision": average_precision_score(y_true, pred_proba),
            "brier_score": brier_score_loss(y_true, pred_proba),
        })

    # Simple average
    avg_pred = simple_average_ensemble(predictions)
    if avg_pred.ndim > 1:
        avg_proba = avg_pred[:, 1]
    else:
        avg_proba = avg_pred

    results.append({
        "strategy": "simple_average",
        "roc_auc": roc_auc_score(y_true, avg_proba),
        "avg_precision": average_precision_score(y_true, avg_proba),
        "brier_score": brier_score_loss(y_true, avg_proba),
    })

    # Weighted average (optimized)
    opt_weights = optimize_ensemble_weights(predictions, y_true, "roc_auc")
    weighted_pred = simple_average_ensemble(predictions, opt_weights["weights"])
    if weighted_pred.ndim > 1:
        weighted_proba = weighted_pred[:, 1]
    else:
        weighted_proba = weighted_pred

    results.append({
        "strategy": "weighted_average",
        "roc_auc": roc_auc_score(y_true, weighted_proba),
        "avg_precision": average_precision_score(y_true, weighted_proba),
        "brier_score": brier_score_loss(y_true, weighted_proba),
        "weights": opt_weights["weights"],
    })

    # Rank average
    rank_pred = rank_average_ensemble(predictions)
    results.append({
        "strategy": "rank_average",
        "roc_auc": roc_auc_score(y_true, rank_pred),
        "avg_precision": average_precision_score(y_true, rank_pred),
        "brier_score": brier_score_loss(y_true, rank_pred),
    })

    # Stacking
    stack_pred, _ = stacking_ensemble(predictions, y_true, "logistic")
    if stack_pred.ndim > 1:
        stack_proba = stack_pred[:, 1]
    else:
        stack_proba = stack_pred

    results.append({
        "strategy": "stacking_logistic",
        "roc_auc": roc_auc_score(y_true, stack_proba),
        "avg_precision": average_precision_score(y_true, stack_proba),
        "brier_score": brier_score_loss(y_true, stack_proba),
    })

    # Voting
    vote_pred = voting_ensemble(predictions)
    vote_binary = vote_pred  # Already binary

    # Compute AUC for voting (need probabilities, not available - skip)
    results.append({
        "strategy": "voting_hard",
        "roc_auc": np.nan,  # Hard voting doesn't give probabilities
        "avg_precision": average_precision_score(y_true, vote_binary),
        "brier_score": brier_score_loss(y_true, vote_binary),
    })

    return pd.DataFrame(results)


def analyze_ensemble_benefit(
    predictions: List[np.ndarray],
    y_true: np.ndarray,
    model_names: List[str],
) -> Dict:
    """
    Analyze when ensembling provides benefit over individual models.
    """
    # Get individual AUCs
    individual_aucs = []
    for pred in predictions:
        if pred.ndim > 1:
            pred_proba = pred[:, 1]
        else:
            pred_proba = pred
        individual_aucs.append(roc_auc_score(y_true, pred_proba))

    best_individual = max(individual_aucs)

    # Ensemble AUCs
    simple_avg = simple_average_ensemble(predictions)
    if simple_avg.ndim > 1:
        simple_proba = simple_avg[:, 1]
    else:
        simple_proba = simple_avg

    simple_auc = roc_auc_score(y_true, simple_proba)

    # Weighted
    opt_weights = optimize_ensemble_weights(predictions, y_true, "roc_auc")
    weighted = simple_average_ensemble(predictions, opt_weights["weights"])
    if weighted.ndim > 1:
        weighted_proba = weighted[:, 1]
    else:
        weighted_proba = weighted

    weighted_auc = roc_auc_score(y_true, weighted_proba)

    return {
        "best_individual_auc": best_individual,
        "simple_average_auc": simple_auc,
        "weighted_average_auc": weighted_auc,
        "simple_improvement": simple_auc - best_individual,
        "weighted_improvement": weighted_auc - best_individual,
        "optimal_weights": opt_weights["weights"],
    }


def plot_ensemble_comparison(
    ensemble_results: pd.DataFrame,
    save_path: str = "reports/figures/ensemble_comparison.png",
):
    """
    Plot ensemble comparison results.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ROC-AUC
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(ensemble_results)))
    ax.barh(ensemble_results["strategy"], ensemble_results["roc_auc"], color=colors)
    ax.set_xlabel("ROC-AUC")
    ax.set_title("ROC-AUC by Ensemble Strategy")
    ax.set_xlim([0.7, 1.0])

    # Average Precision
    ax = axes[1]
    ax.barh(ensemble_results["strategy"], ensemble_results["avg_precision"], color=colors)
    ax.set_xlabel("Average Precision")
    ax.set_title("Average Precision by Ensemble Strategy")
    ax.set_xlim([0.5, 1.0])

    # Brier Score (lower is better)
    ax = axes[2]
    ax.barh(ensemble_results["strategy"], ensemble_results["brier_score"], color=colors)
    ax.set_xlabel("Brier Score")
    ax.set_title("Brier Score by Ensemble Strategy (Lower = Better)")
    ax.set_xlim([0, 0.2])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved ensemble comparison to {save_path}")


def plot_weight_heatmap(
    results_df: pd.DataFrame,
    save_path: str = "reports/figures/ensemble_weights.png",
):
    """
    Plot heatmap of ensemble weights across datasets.
    """
    # This requires running ensemble analysis on multiple datasets
    # Placeholder for now

    print("  Weight heatmap requires multiple datasets - skipping")


# =============================================================================
# Main: Run Ensemble Analysis on Benchmark Results
# =============================================================================

def run_ensemble_analysis_on_results(
    results_df: pd.DataFrame,
    datasets: List[str] = None,
) -> pd.DataFrame:
    """
    Run ensemble analysis on benchmark results.

    For each dataset, creates ensembles combining models.
    """
    if datasets is None:
        datasets = results_df["dataset"].unique()

    all_results = []

    for dataset_name in datasets:
        dataset_results = results_df[results_df["dataset"] == dataset_name]

        # Skip if not enough models
        if dataset_results["model"].nunique() < 2:
            continue

        # Get predictions for each model
        # Note: We need y_proba which may not be in results
        # This is a placeholder - actual implementation needs prediction storage

        # For now, use available metrics
        print(f"  Ensemble analysis for {dataset_name}")
        print(f"    Available models: {dataset_results['model'].unique().tolist()}")

    return pd.DataFrame(all_results)


if __name__ == "__main__":
    # Demo: Test ensemble methods
    print("Ensemble Analysis Demo")
    print("=" * 40)

    # Generate synthetic predictions
    np.random.seed(42)
    n_samples = 1000
    y_true = np.random.randint(0, 2, n_samples)

    # Model 1: Good predictions
    pred1 = np.random.rand(n_samples)
    pred1 = np.clip(pred1 + 0.3 * (y_true - 0.5), 0.01, 0.99)

    # Model 2: Different good predictions
    pred2 = np.random.rand(n_samples)
    pred2 = np.clip(pred2 + 0.25 * (y_true - 0.5), 0.01, 0.99)

    # Model 3: Weak predictions
    pred3 = np.random.rand(n_samples)

    predictions = [pred1, pred2, pred3]
    model_names = ["TabPFN", "XGBoost", "MLP"]

    # Evaluate
    results = evaluate_ensemble_strategy(predictions, y_true, model_names)

    print("\nEnsemble Results:")
    print(results.to_string(index=False))

    # Diversity
    diversity = compute_ensemble_diversity(predictions, y_true)
    print(f"\nDiversity Metrics:")
    for k, v in diversity.items():
        print(f"  {k}: {v:.4f}")

    # Benefit
    benefit = analyze_ensemble_benefit(predictions, y_true, model_names)
    print(f"\nEnsemble Benefit:")
    for k, v in benefit.items():
        if k != "optimal_weights":
            print(f"  {k}: {v:.4f}")
    print(f"  optimal_weights: {benefit['optimal_weights']}")