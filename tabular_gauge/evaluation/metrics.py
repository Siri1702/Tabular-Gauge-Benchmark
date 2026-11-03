"""
Computes all evaluation metrics from a RunResult.
Returns a flat dict ready for pandas DataFrame construction.
Supports both binary and multi-class classification.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    accuracy_score,
)
# ... (existing imports)
from tabular_gauge.models.base import RunResult  # noqa — import from base

# Centralized mapping of metric optimization directions
# "higher" -> Higher value is better (e.g., AUC)
# "lower" -> Lower value is better (e.g., Loss, Time)
METRIC_DIRECTIONS = {
    "roc_auc": "higher",
    "avg_precision": "higher",
    "f1_macro": "higher",
    "f1_weighted": "higher",
    "accuracy": "higher",
    "balanced_accuracy": "higher",
    "matthews_corrcoef": "higher",
    "precision_class_0": "higher",
    "precision_class_1": "higher",
    "recall_class_0": "higher",
    "recall_class_1": "higher",
    "f1_class_0": "higher",
    "f1_class_1": "higher",
    "brier_score": "lower",
    "log_loss_val": "lower",
    "ece_10": "lower",
    "ece_15": "lower",
    "max_calibration_error": "lower",
    "fit_time_sec": "lower",
    "predict_time_sec": "lower",
    "total_time_sec": "lower",
    "tuning_time_sec": "lower",
    "total_wall_time_sec": "lower",
    "peak_memory_mb": "lower",
    "memory_delta_mb": "lower",
}

def is_binary(y_true: np.ndarray) -> bool:
    """Check if the problem is binary classification."""
    unique_labels = np.unique(y_true)
    return len(unique_labels) == 2


def get_num_classes(y_true: np.ndarray) -> int:
    """Get the number of classes in the dataset."""
    return len(np.unique(y_true))


def _safe_log_loss(y_true: np.ndarray, y_proba: np.ndarray, labels: list) -> float:
    """
    Safely compute log loss, handling cases where y_proba has fewer classes than y_true.
    """
    n_classes_proba = y_proba.shape[1] if y_proba.ndim == 2 and y_proba.shape[1] > 1 else 1
    n_classes_true = len(labels)

    # If y_proba has fewer classes than y_true, we need to filter y_true to only include
    # classes that exist in y_proba
    if n_classes_proba < n_classes_true:
        # Filter labels to only those that have probability columns
        valid_labels = [l for l in labels if l < n_classes_proba]
        if not valid_labels:
            return np.nan

        # Filter y_true to only include samples with valid labels
        mask = np.isin(y_true, valid_labels)
        y_filtered = y_true[mask]
        y_proba_filtered = y_proba[mask]

        if len(y_filtered) == 0:
            return np.nan

        try:
            return log_loss(y_filtered, y_proba_filtered, labels=valid_labels)
        except (ValueError, TypeError):
            return np.nan
    else:
        try:
            return log_loss(y_true, y_proba, labels=labels)
        except (ValueError, TypeError):
            return np.nan


def compute_ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    Measures the difference between predicted confidence and actual accuracy.
    Supports both binary and multi-class classification.
    For multi-class, computes ECE for each class and returns the average.
    """
    n_classes = get_num_classes(y_true)

    # Determine number of classes from y_proba
    if y_proba.ndim == 2 and y_proba.shape[1] > 1:
        n_classes_proba = y_proba.shape[1]
    else:
        n_classes_proba = n_classes

    # Use MAXIMUM for conservative detection
    effective_n_classes = max(n_classes, n_classes_proba)

    # Handle multi-class: y_proba should be (n_samples, n_classes)
    if effective_n_classes > 2 and y_proba.ndim == 2 and y_proba.shape[1] > 1:
        # Get unique labels present in y_true that also have probability columns
        unique_labels = np.unique(y_true)
        available_labels = [l for l in unique_labels if l < n_classes_proba]

        if not available_labels:
            available_labels = list(range(n_classes_proba))

        # Multi-class ECE: compute per-class and average
        ece_values = []
        for class_idx in available_labels:
            # Get probability for this class
            proba_class = y_proba[:, int(class_idx)]
            # Binary target for this class
            y_true_binary = (y_true == class_idx).astype(int)
            ece_values.append(_compute_ece_binary(y_true_binary, proba_class, n_bins))

        if not ece_values:
            return np.nan
        return np.mean(ece_values)
    else:
        # Binary case: ensure 1D array
        y_proba_1d = _flatten_proba_for_binary(y_proba)
        return _compute_ece_binary(y_true, y_proba_1d, n_bins)


def _flatten_proba_for_binary(y_proba: np.ndarray) -> np.ndarray:
    """
    Flatten y_proba to 1D for binary classification.
    Handles the case where binary models return 2D probabilities (n_samples, 2).
    In that case, extract only the positive class probability (column 1).
    """
    if y_proba.ndim == 2 and y_proba.shape[1] == 2:
        # Binary model returning both class probabilities - take positive class
        return y_proba[:, 1]
    else:
        return y_proba.ravel() if y_proba.ndim > 1 else y_proba


def _compute_ece_binary(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    """Helper function to compute ECE for binary classification."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]

        in_bin = (y_proba >= bin_lower) & (y_proba < bin_upper)
        bin_count = np.sum(in_bin)

        if bin_count > 0:
            bin_accuracy = np.mean(y_true[in_bin])
            bin_confidence = np.mean(y_proba[in_bin])
            ece += (bin_count / len(y_true)) * np.abs(bin_accuracy - bin_confidence)

    return ece


def compute_calibration_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """
    Compute calibration metrics: ECE, MCE (Maximum Calibration Error).
    Supports both binary and multi-class classification.
    """
    n_classes = get_num_classes(y_true)

    # Determine number of classes from y_proba
    if y_proba.ndim == 2 and y_proba.shape[1] > 1:
        n_classes_proba = y_proba.shape[1]
    else:
        n_classes_proba = n_classes

    # Use MAXIMUM for conservative detection
    effective_n_classes = max(n_classes, n_classes_proba)

    # Handle multi-class case
    if effective_n_classes > 2 and y_proba.ndim == 2 and y_proba.shape[1] > 1:
        y_proba_clipped = np.clip(y_proba, 1e-7, 1 - 1e-7)

        # Get unique labels present in y_true that also have probability columns
        unique_labels = np.unique(y_true)
        available_labels = [l for l in unique_labels if l < n_classes_proba]

        if not available_labels:
            available_labels = list(range(n_classes_proba))

        ece_10_values = []
        ece_15_values = []
        mce_values = []

        for class_idx in available_labels:
            proba_class = y_proba_clipped[:, int(class_idx)]
            y_true_binary = (y_true == class_idx).astype(int)

            # ECE with 10 bins
            ece_10_values.append(_compute_ece_binary(y_true_binary, proba_class, n_bins=10))
            # ECE with 15 bins
            ece_15_values.append(_compute_ece_binary(y_true_binary, proba_class, n_bins=15))

            # MCE for this class
            bin_edges = np.linspace(0, 1, 15 + 1)
            max_cal_error = 0.0
            for i in range(len(bin_edges) - 1):
                in_bin = (proba_class >= bin_edges[i]) & (proba_class < bin_edges[i + 1])
                if np.sum(in_bin) > 0:
                    bin_acc = np.mean(y_true_binary[in_bin])
                    bin_conf = np.mean(proba_class[in_bin])
                    max_cal_error = max(max_cal_error, abs(bin_acc - bin_conf))
            mce_values.append(max_cal_error)

        if not ece_10_values:
            return {
                "ece_10": np.nan,
                "ece_15": np.nan,
                "max_calibration_error": np.nan,
            }

        return {
            "ece_10": np.mean(ece_10_values),
            "ece_15": np.mean(ece_15_values),
            "max_calibration_error": np.mean(mce_values),
        }
    else:
        # Binary case
        y_proba_1d = _flatten_proba_for_binary(y_proba)
        y_proba_clipped = np.clip(y_proba_1d, 1e-7, 1 - 1e-7)

        ece_10 = compute_ece(y_true, y_proba_clipped, n_bins=10)
        ece_15 = compute_ece(y_true, y_proba_clipped, n_bins=15)

        # Maximum Calibration Error
        bin_edges = np.linspace(0, 1, 15 + 1)
        max_calibration_error = 0.0
        for i in range(len(bin_edges) - 1):
            in_bin = (y_proba_clipped >= bin_edges[i]) & (y_proba_clipped < bin_edges[i + 1])
            if np.sum(in_bin) > 0:
                bin_acc = np.mean(y_true[in_bin])
                bin_conf = np.mean(y_proba_clipped[in_bin])
                max_calibration_error = max(max_calibration_error, abs(bin_acc - bin_conf))

        return {
            "ece_10": ece_10,
            "ece_15": ece_15,
            "max_calibration_error": max_calibration_error,
        }


def compute_roc_pr_curves(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """
    Compute ROC curve and PR curve metrics.
    Returns threshold-independent metrics.
    Supports both binary and multi-class classification.
    """
    n_classes = get_num_classes(y_true)

    # Determine number of classes from y_proba
    if y_proba.ndim == 2 and y_proba.shape[1] > 1:
        n_classes_proba = y_proba.shape[1]
    else:
        n_classes_proba = n_classes

    # Use MAXIMUM for conservative detection
    effective_n_classes = max(n_classes, n_classes_proba)

    # Handle multi-class case
    if effective_n_classes > 2 and y_proba.ndim == 2 and y_proba.shape[1] > 1:
        # Multi-class: compute One-vs-Rest ROC-AUC and PR-AUC
        roc_auc_values = []
        pr_auc_values = []

        # Get unique labels present in y_true that also have probability columns
        unique_labels = np.unique(y_true)
        available_labels = [l for l in unique_labels if l < n_classes_proba]

        if not available_labels:
            available_labels = list(range(n_classes_proba))

        for class_idx in available_labels:
            y_true_binary = (y_true == class_idx).astype(int)
            proba_class = y_proba[:, int(class_idx)]

            try:
                # ROC curve
                fpr, tpr, _ = roc_curve(y_true_binary, proba_class)
                roc_auc_values.append(auc(fpr, tpr))
            except ValueError:
                roc_auc_values.append(np.nan)

            try:
                # PR curve
                precision, recall, _ = precision_recall_curve(y_true_binary, proba_class)
                pr_auc_values.append(auc(recall, precision))
            except ValueError:
                pr_auc_values.append(np.nan)

        # Return macro-average for multi-class
        return {
            "roc_auc": np.nanmean(roc_auc_values),
            "pr_auc": np.nanmean(pr_auc_values),
            "optimal_threshold": np.nan,  # Not meaningful for multi-class
            "optimal_tpr": np.nan,
            "optimal_fpr": np.nan,
        }
    else:
        # Binary case
        y_proba_1d = _flatten_proba_for_binary(y_proba)

        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_proba_1d)
        roc_auc = auc(fpr, tpr)

        # PR curve
        precision, recall, _ = precision_recall_curve(y_true, y_proba_1d)
        pr_auc = auc(recall, precision)

        # Find optimal threshold (Youden's J statistic)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = fpr[optimal_idx] if optimal_idx < len(fpr) else 0.5

        return {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "optimal_threshold": optimal_threshold,
            "optimal_tpr": tpr[optimal_idx] if optimal_idx < len(tpr) else np.nan,
            "optimal_fpr": fpr[optimal_idx] if optimal_idx < len(fpr) else np.nan,
        }


def compute_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute per-class precision, recall, and F1.
    Supports both binary and multi-class classification.
    For multi-class, returns metrics for all classes.
    """
    n_classes = get_num_classes(y_true)

    # Per-class precision and recall
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    result = {}

    if n_classes > 2:
        # Multi-class: return metrics for all classes
        for i in range(n_classes):
            result[f"precision_class_{i}"] = precision[i] if i < len(precision) else np.nan
            result[f"recall_class_{i}"] = recall[i] if i < len(recall) else np.nan
            result[f"f1_class_{i}"] = f1[i] if i < len(f1) else np.nan
    else:
        # Binary case: return class_0 and class_1
        result = {
            "precision_class_0": precision[0] if len(precision) > 0 else np.nan,
            "precision_class_1": precision[1] if len(precision) > 1 else np.nan,
            "recall_class_0": recall[0] if len(recall) > 0 else np.nan,
            "recall_class_1": recall[1] if len(recall) > 1 else np.nan,
            "f1_class_0": f1[0] if len(f1) > 0 else np.nan,
            "f1_class_1": f1[1] if len(f1) > 1 else np.nan,
        }

    return result


def compute_fairness_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                              sensitive_attribute: np.ndarray = None) -> dict:
    """
    Compute fairness metrics if sensitive attribute is available.

    Metrics computed:
    - Demographic Parity: P(Y_pred=1)
    - Equalized Odds: TPR and FPR by group
    - Balanced Accuracy
    - Matthews Correlation Coefficient
    """
    if sensitive_attribute is None:
        # Return basic fairness-adjacent metrics without group info
        return {
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "matthews_corrcoef": matthews_corrcoef(y_true, y_pred),
            "demographic_parity": np.mean(y_pred),
            "tpr_overall": np.sum((y_true == 1) & (y_pred == 1)) / max(np.sum(y_true == 1), 1),
            "fpr_overall": np.sum((y_true == 0) & (y_pred == 1)) / max(np.sum(y_true == 0), 1),
        }

    # Compute group-wise metrics
    unique_groups = np.unique(sensitive_attribute)
    tpr_by_group = []
    fpr_by_group = []
    pp_by_group = []

    for group in unique_groups:
        mask = sensitive_attribute == group
        y_true_g = y_true[mask]
        y_pred_g = y_pred[mask]

        if np.sum(y_true_g == 1) > 0:
            tpr_g = np.sum((y_true_g == 1) & (y_pred_g == 1)) / np.sum(y_true_g == 1)
        else:
            tpr_g = np.nan

        if np.sum(y_true_g == 0) > 0:
            fpr_g = np.sum((y_true_g == 0) & (y_pred_g == 1)) / np.sum(y_true_g == 0)
        else:
            fpr_g = np.nan

        pp_g = np.mean(y_pred_g)

        tpr_by_group.append(tpr_g)
        fpr_by_group.append(fpr_g)
        pp_by_group.append(pp_g)

    # Compute disparities (max - min across groups)
    tpr_disparity = np.nanmax(tpr_by_group) - np.nanmin(tpr_by_group)
    fpr_disparity = np.nanmax(fpr_by_group) - np.nanmin(fpr_by_group)
    pp_disparity = np.nanmax(pp_by_group) - np.nanmin(pp_by_group)

    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "matthews_corrcoef": matthews_corrcoef(y_true, y_pred),
        "demographic_parity": np.mean(y_pred),
        "demographic_parity_disparity": pp_disparity,
        "tpr_disparity": tpr_disparity,
        "fpr_disparity": fpr_disparity,
    }


def compute_metrics(result: RunResult) -> dict:
    """
    Given a RunResult, compute all metrics and return as a flat dict.
    Returns NaN for metrics that cannot be computed (e.g., error run).

    Supports both binary and multi-class classification.

    Includes:
    - Core metrics (ROC-AUC, PR-AUC, F1, Brier, Log Loss)
    - Calibration metrics (ECE, MCE)
    - ROC/PR curve metrics
    - Per-class metrics
    - Fairness-related metrics
    """
    base = {
        "model": result.model_name,
        "dataset": result.dataset_name,
        "seed": result.seed,
        "n_train": result.n_train,
        "n_test": result.n_test,
        "n_features": result.n_features,
        "fit_time_sec": result.fit_time_sec,
        "predict_time_sec": result.predict_time_sec,
        "total_time_sec": result.fit_time_sec + result.predict_time_sec,
        "tuning_time_sec": result.tuning_time_sec,
        "total_wall_time_sec": result.total_wall_time_sec,
        "throughput_rows_per_sec": result.throughput_rows_per_sec,
        "ms_per_test_row": (result.predict_time_sec / result.n_test) * 1000 if result.n_test > 0 else None,
        "peak_memory_mb": result.peak_memory_mb,
        "memory_delta_mb": result.memory_delta_mb,
        "gpu_memory_mb": result.gpu_memory_mb,
        "error": result.error,
    }

    # Get number of classes for dynamic handling
    n_classes = get_num_classes(result.y_test)

    # Additional validation: check actual class distribution in y
    # Some datasets might have labels beyond 0/1 even for binary tasks
    unique_y = np.unique(result.y_test)
    min_y, max_y = unique_y.min(), unique_y.max()

    if result.error is not None:
        # Fill all metrics with NaN for failed runs
        base.update({k: np.nan for k in [
            "roc_auc", "avg_precision", "f1_macro", "f1_weighted", "pr_auc",
            "brier_score", "log_loss_val", "balanced_accuracy", "accuracy",
            "ece_10", "ece_15", "max_calibration_error",
            "sensitivity", "specificity", "ppv",
            "peak_memory_mb", "memory_delta_mb", "gpu_memory_mb",
        ]})
        return base

    y = result.y_test
    yp = result.y_proba
    yh = result.y_pred

    # Determine number of classes from y_proba (if 2D, use columns; otherwise use unique y values)
    if yp.ndim == 2 and yp.shape[1] > 1:
        n_classes_proba = yp.shape[1]
    else:
        n_classes_proba = n_classes

    # Robust detection of binary vs multi-class
    # Use the MAXIMUM to be conservative - if either y or y_proba suggests multi-class, treat as multi-class
    # This prevents errors from treating multi-class as binary
    effective_n_classes = max(n_classes, n_classes_proba)

    # Determine if binary or multi-class based on effective number of classes
    # Note: n_classes == 1 (single class in test set) is a degenerate case, not binary
    is_binary_labels = n_classes == 2
    is_multiclass = effective_n_classes > 2 or not is_binary_labels

    if is_multiclass:
        # Multi-class metrics
        yp_clipped = np.clip(yp, 1e-7, 1 - 1e-7)
        unique_labels = np.unique(y)

        # Limit to classes that have probability columns
        # If y_proba has fewer columns than unique labels, use only available classes
        available_labels = [l for l in unique_labels if l < n_classes_proba]

        if not available_labels:
            available_labels = list(range(n_classes_proba))

        # Core metrics for multi-class
        try:
            roc_auc = roc_auc_score(y, yp_clipped, labels=available_labels, multi_class='ovr', average='macro')
        except (ValueError, TypeError):
            roc_auc = np.nan

        try:
            # For multi-class average_precision_score, we need to compute per-class and average
            ap_values = []
            for label in available_labels:
                y_binary = (y == label).astype(int)
                try:
                    ap = average_precision_score(y_binary, yp_clipped[:, int(label)])
                    ap_values.append(ap)
                except ValueError:
                    ap_values.append(np.nan)
            avg_precision = np.nanmean(ap_values)
        except Exception:
            avg_precision = np.nan

        base.update({
            "roc_auc": roc_auc,
            "avg_precision": avg_precision,
            "f1_macro": f1_score(y, yh, average="macro", zero_division=0),
            "f1_weighted": f1_score(y, yh, average="weighted", zero_division=0),
            "brier_score": np.nan,  # Brier score not defined for multi-class
            "log_loss_val": _safe_log_loss(y, yp_clipped, available_labels),
            "accuracy": accuracy_score(y, yh),
        })

        # Confusion matrix - compute per-class metrics instead of binary TN/FP/TP/FN
        cm = confusion_matrix(y, yh)
        base.update({
            "sensitivity": np.nan,  # Not defined for multi-class
            "specificity": np.nan,
            "ppv": np.nan,
        })

        # Calibration metrics
        calibration = compute_calibration_metrics(y, yp_clipped)
        base.update(calibration)

        # ROC/PR curve metrics
        curve_metrics = compute_roc_pr_curves(y, yp)
        base.update(curve_metrics)

        # Per-class metrics (returns all classes for multi-class)
        per_class = compute_per_class_metrics(y, yh)
        base.update(per_class)

        # Fairness metrics for multi-class
        fairness = compute_fairness_metrics_multiclass(y, yh)
        base.update(fairness)

    else:
        # Binary classification metrics
        yp_1d = _flatten_proba_for_binary(yp)
        yp_clipped = np.clip(yp_1d, 1e-7, 1 - 1e-7)

        # Core metrics - wrapped in try/except for robustness
        try:
            roc_auc = roc_auc_score(y, yp_clipped)
        except (ValueError, TypeError):
            roc_auc = np.nan

        try:
            avg_precision = average_precision_score(y, yp_clipped)
        except (ValueError, TypeError):
            avg_precision = np.nan

        # Binary-specific metrics that require y to be truly binary
        # If y has more than 2 unique values, these will fail
        try:
            brier_score = brier_score_loss(y, yp_clipped)
        except (ValueError, TypeError):
            brier_score = np.nan

        try:
            log_loss_val = log_loss(y, yp_clipped)
        except (ValueError, TypeError):
            log_loss_val = np.nan

        base.update({
            "roc_auc":       roc_auc,
            "avg_precision": avg_precision,
            "f1_macro":      f1_score(y, yh, average="macro", zero_division=0),
            "f1_weighted":   f1_score(y, yh, average="weighted", zero_division=0),
            "brier_score":   brier_score,
            "log_loss_val":  log_loss_val,
            "accuracy":     accuracy_score(y, yh),
        })

        # Confusion matrix derived metrics
        unique_labels = np.unique(np.concatenate([y, yh]))
        if len(unique_labels) == 2:
            labels = [unique_labels.min(), unique_labels.max()]
        else:
            labels = [0, 1]
        cm = confusion_matrix(y, yh, labels=labels)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Handle case where not all classes are present
            tn = fp = fn = tp = 0
            if cm.shape[0] == 2 and cm.shape[1] == 2:
                tn, fp, fn, tp = cm.ravel()

        base.update({
            "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else np.nan,  # recall
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
            "ppv":         tp / (tp + fp) if (tp + fp) > 0 else np.nan,  # precision
        })

        # Calibration metrics
        calibration = compute_calibration_metrics(y, yp_clipped)
        base.update(calibration)

        # ROC/PR curve metrics
        curve_metrics = compute_roc_pr_curves(y, yp)
        base.update(curve_metrics)

        # Per-class metrics
        per_class = compute_per_class_metrics(y, yh)
        base.update(per_class)

        # Fairness metrics (without sensitive attribute - basic version)
        fairness = compute_fairness_metrics(y, yh)
        base.update(fairness)

    return base


def compute_fairness_metrics_multiclass(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute fairness metrics for multi-class classification.
    """
    n_classes = get_num_classes(y_true)

    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "matthews_corrcoef": matthews_corrcoef(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "demographic_parity": np.nan,  # Not defined for multi-class
        "tpr_overall": np.nan,
        "fpr_overall": np.nan,
    }