"""
Meta-Feature Extraction for Dataset Characterization.

This module extracts interpretable meta-features from datasets that can be used
to predict which model will perform best for a given dataset.

Key Meta-Features:
1. Dimensionality: n_samples, n_features, n_samples/n_features ratio
2. Class Balance: class imbalance ratio, entropy
3. Feature Types: categorical count, numerical count, mixed count
4. Feature Cardinality: mean/max/min cardinality for categoricals
5. Feature Distributions: skewness, kurtosis for numericals
6. Correlation: feature-target correlation, inter-feature correlation
7. Complexity: PCA variance explained, intrinsic dimensionality estimate

Research Questions:
- Which meta-features best predict model performance?
- Can we build a meta-model that recommends the best model?
- What dataset characteristics favor TabPFN vs GBDT?
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def compute_class_balance(y: np.ndarray) -> Dict[str, float]:
    """
    Compute class balance metrics.
    """
    unique, counts = np.unique(y, return_counts=True)
    n_classes = len(unique)

    if n_classes < 2:
        return {
            "class_balance_ratio": np.nan,
            "class_entropy": 0.0,
            "imbalance_ratio": np.nan,
            "n_classes": n_classes,
        }

    proportions = counts / counts.sum()

    # Balance ratio: ratio of minority to majority class
    balance_ratio = counts.min() / counts.max()

    # Entropy (normalized to [0, 1])
    class_entropy = entropy(proportions) / np.log(2)

    # Imbalance ratio: majority / minority
    imbalance_ratio = counts.max() / counts.min()

    return {
        "class_balance_ratio": balance_ratio,
        "class_entropy": class_entropy,
        "imbalance_ratio": imbalance_ratio,
        "n_classes": n_classes,
    }


def compute_feature_cardinality(X: np.ndarray, categorical_mask: np.ndarray = None) -> Dict[str, float]:
    """
    Compute cardinality statistics for categorical features.
    """
    if categorical_mask is None:
        # Assume all features are numerical
        return {
            "mean_cardinality": 0.0,
            "max_cardinality": 0.0,
            "min_cardinality": 0.0,
            "n_categorical": 0,
            "categorical_ratio": 0.0,
        }

    cardinalities = []
    n_categorical = 0

    for i, is_categorical in enumerate(categorical_mask):
        if is_categorical:
            n_categorical += 1
            unique_vals = np.unique(X[:, i])
            cardinalities.append(len(unique_vals))

    if not cardinalities:
        cardinalities = [0]

    return {
        "mean_cardinality": np.mean(cardinalities),
        "max_cardinality": np.max(cardinalities),
        "min_cardinality": np.min(cardinalities),
        "n_categorical": n_categorical,
        "categorical_ratio": n_categorical / X.shape[1] if X.shape[1] > 0 else 0.0,
    }


def compute_numerical_statistics(X: np.ndarray) -> Dict[str, float]:
    """
    Compute statistics for numerical features.
    """
    # Flatten and compute overall statistics
    X_flat = X.flatten()

    # Remove infinities and NaNs
    X_clean = X_flat[np.isfinite(X_flat)]

    if len(X_clean) == 0:
        return {
            "mean_skewness": 0.0,
            "mean_kurtosis": 0.0,
            "mean_nunique_ratio": 0.0,
            "pct_constant_features": 0.0,
        }

    # Per-feature statistics
    skewnesses = []
    kurtoses = []
    unique_ratios = []
    constant_features = 0

    for i in range(X.shape[1]):
        col = X[:, i]
        col_clean = col[np.isfinite(col)]

        if len(col_clean) > 1:
            # Skewness and kurtosis
            if np.std(col_clean) > 1e-10:
                skewnesses.append(stats.skew(col_clean))
                kurtoses.append(stats.kurtosis(col_clean))

            # Unique ratio
            unique_ratios.append(len(np.unique(col_clean)) / len(col_clean))

            # Check for constant features
            if len(np.unique(col_clean)) == 1:
                constant_features += 1

    return {
        "mean_skewness": np.mean(skewnesses) if skewnesses else 0.0,
        "mean_kurtosis": np.mean(kurtoses) if kurtoses else 0.0,
        "mean_nunique_ratio": np.mean(unique_ratios) if unique_ratios else 0.0,
        "pct_constant_features": constant_features / X.shape[1] if X.shape[1] > 0 else 0.0,
    }


def compute_correlation_metrics(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Compute correlation-based features.
    """
    # Feature-target correlations
    correlations = []
    for i in range(X.shape[1]):
        col = X[:, i]
        mask = np.isfinite(col)
        if mask.sum() > 10 and np.std(col[mask]) > 1e-10:
            corr = np.abs(np.corrcoef(col[mask], y[mask])[0, 1])
            if not np.isnan(corr):
                correlations.append(corr)

    if not correlations:
        correlations = [0.0]

    # Inter-feature correlations (mean absolute pairwise)
    X_clean = X[:, np.std(X, axis=0) > 1e-10]
    if X_clean.shape[1] > 1:
        corr_matrix = np.corrcoef(X_clean.T)
        # Get upper triangle (excluding diagonal)
        upper = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        mean_inter_feature_corr = np.mean(np.abs(upper))
        max_inter_feature_corr = np.max(np.abs(upper))
    else:
        mean_inter_feature_corr = 0.0
        max_inter_feature_corr = 0.0

    return {
        "mean_feature_target_corr": np.mean(correlations),
        "max_feature_target_corr": np.max(correlations),
        "mean_inter_feature_corr": mean_inter_feature_corr,
        "max_inter_feature_corr": max_inter_feature_corr,
    }


def compute_intrinsic_dimensionality(X: np.ndarray, max_components: int = 20) -> Dict[str, float]:
    """
    Estimate intrinsic dimensionality using PCA.
    """
    # Center and scale
    X_centered = X - X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-10] = 1.0
    X_scaled = X_centered / std

    # PCA
    n_components = min(max_components, X.shape[0], X.shape[1])
    if n_components < 2:
        return {
            "pca_95_var_components": np.nan,
            "pca_99_var_components": np.nan,
            "intrinsic_dim_ratio": np.nan,
        }

    pca = PCA(n_components=n_components)
    try:
        pca.fit(X_scaled)
    except:
        return {
            "pca_95_var_components": np.nan,
            "pca_99_var_components": np.nan,
            "intrinsic_dim_ratio": np.nan,
        }

    cumvar = np.cumsum(pca.explained_variance_ratio_)

    # Components needed for 95% and 99% variance
    n_95 = np.searchsorted(cumvar, 0.95) + 1
    n_99 = np.searchsorted(cumvar, 0.99) + 1

    return {
        "pca_95_var_components": n_95,
        "pca_99_var_components": n_99,
        "intrinsic_dim_ratio": n_95 / X.shape[1],
    }


def compute_sample_density(X: np.ndarray) -> Dict[str, float]:
    """
    Compute sample density metrics.
    """
    n, p = X.shape

    # Samples per feature
    samples_per_feature = n / p if p > 0 else np.nan

    # Log samples (for regression-friendly)
    log_n = np.log10(n) if n > 0 else np.nan

    # Log features
    log_p = np.log10(p) if p > 0 else np.nan

    # Sample density: n / (p * log(p))
    density = n / (p * np.log(p)) if p > 1 else np.nan

    return {
        "samples_per_feature": samples_per_feature,
        "log_n_samples": log_n,
        "log_n_features": log_p,
        "sample_density": density,
    }


def extract_meta_features(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str = "unknown",
    categorical_mask: np.ndarray = None,
) -> Dict[str, float]:
    """
    Extract all meta-features from a dataset.

    Returns a dictionary of meta-feature_name: value pairs.
    """
    n_samples, n_features = X.shape

    # Basic dimensionality
    meta = {
        "dataset_name": dataset_name,
        "n_samples": n_samples,
        "n_features": n_features,
    }

    # Class balance
    meta.update(compute_class_balance(y))

    # Feature cardinality
    meta.update(compute_feature_cardinality(X, categorical_mask))

    # Numerical statistics
    meta.update(compute_numerical_statistics(X))

    # Correlation metrics
    meta.update(compute_correlation_metrics(X, y))

    # Intrinsic dimensionality
    meta.update(compute_intrinsic_dimensionality(X))

    # Sample density
    meta.update(compute_sample_density(X))

    return meta


def extract_meta_features_from_df(
    df: pd.DataFrame,
    target_col: str,
    dataset_name: str = "unknown",
) -> Dict[str, float]:
    """
    Extract meta-features from a pandas DataFrame.
    Automatically detects categorical columns.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col].values

    # Detect categorical columns
    categorical_mask = np.array([
        X[col].dtype == "object" or str(X[col].dtype) == "category"
        for col in X.columns
    ])

    # Encode categoricals for numerical processing
    X_encoded = X.copy()
    for col in X.columns:
        if X[col].dtype == "object" or str(X[col].dtype) == "category":
            X_encoded[col] = pd.factorize(X[col])[0]

    X_encoded = X_encoded.values.astype(np.float64)

    return extract_meta_features(X_encoded, y, dataset_name, categorical_mask)


def create_meta_features_dataframe(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a dataframe of meta-features from benchmark results.

    This computes meta-features for each dataset in the results.
    """
    datasets = results_df["dataset"].unique()
    meta_features_list = []

    for dataset_name in datasets:
        # Get sample of data for this dataset
        dataset_results = results_df[results_df["dataset"] == dataset_name]

        # We need the actual data to compute meta-features
        # For now, use available metadata from results
        meta = {
            "dataset": dataset_name,
            "n_train": dataset_results["n_train"].iloc[0],
            "n_test": dataset_results["n_test"].iloc[0],
            "n_features": dataset_results["n_features"].iloc[0],
        }

        # Compute derived features
        n = meta["n_train"]
        p = meta["n_features"]
        meta["samples_per_feature"] = n / p if p > 0 else np.nan
        meta["log_n"] = np.log10(n) if n > 0 else np.nan

        meta_features_list.append(meta)

    return pd.DataFrame(meta_features_list)


# =============================================================================
# Meta-Learning Model
# =============================================================================

def build_meta_dataset(
    meta_features_df: pd.DataFrame,
    results_df: pd.DataFrame,
    metric: str = "roc_auc",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build dataset for meta-learning: predict best model from meta-features.

    Returns:
    - X_meta: DataFrame of meta-features
    - y_meta: DataFrame with best model per dataset
    """
    # Get best model per dataset
    best_models = results_df.groupby("dataset").apply(
        lambda x: x.loc[x[metric].idxmax(), "model"]
    ).reset_index()
    best_models.columns = ["dataset", "best_model"]

    # Merge with meta-features
    meta_df = meta_features_df.merge(best_models, on="dataset")

    # Separate features and target
    feature_cols = [c for c in meta_df.columns if c not in ["dataset", "best_model"]]
    X_meta = meta_df[feature_cols]
    y_meta = meta_df["best_model"]

    return X_meta, y_meta


def train_meta_model(
    X_meta: pd.DataFrame,
    y_meta: pd.DataFrame,
    model_type: str = "random_forest",
):
    """
    Train a meta-model to predict best model from meta-features.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_meta)

    # Handle missing values
    X_clean = X_meta.fillna(X_meta.median())

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    if model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = LogisticRegression(max_iter=1000, random_state=42)

    model.fit(X_scaled, y_encoded)

    return model, scaler, le


def predict_best_model(
    model,
    scaler: StandardScaler,
    le,
    meta_features: Dict[str, float],
    feature_names: List[str],
) -> str:
    """
    Predict the best model for a new dataset.
    """
    # Extract features in correct order
    X = np.array([[meta_features.get(f, np.nan) for f in feature_names]])

    # Handle missing
    X_clean = np.nan_to_num(X, nan=0)

    # Scale
    X_scaled = scaler.transform(X_clean)

    # Predict
    pred = model.predict(X_scaled)
    return le.inverse_transform(pred)[0]


# =============================================================================
# Analysis: Which Meta-Features Predict Model Performance?
# =============================================================================

def analyze_feature_importance(
    X_meta: pd.DataFrame,
    y_meta: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyze which meta-features are most important for predicting model performance.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    # Prepare data
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_meta)
    X_clean = X_meta.fillna(X_meta.median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    # Train RF for feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y_encoded)

    # Get importance
    importance = pd.DataFrame({
        "feature": X_meta.columns,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    return importance


def analyze_model_preferences(
    results_df: pd.DataFrame,
    meta_features_df: pd.DataFrame,
    metric: str = "roc_auc",
) -> Dict[str, pd.DataFrame]:
    """
    Analyze which meta-features each model prefers.
    Returns a dictionary of model -> preferences DataFrame.
    """
    merged = meta_features_df.merge(
        results_df[["dataset", "model", metric]],
        on="dataset"
    )

    preferences = {}
    feature_cols = [c for c in meta_features_df.columns if c not in ["dataset"]]

    for model in merged["model"].unique():
        model_data = merged[merged["model"] == model]

        # Compute mean meta-features for datasets where this model wins
        pref = model_data[feature_cols].mean()
        preferences[model] = pref

    return preferences


def find_decision_boundaries(
    results_df: pd.DataFrame,
    meta_features_df: pd.DataFrame,
    metric: str = "roc_auc",
) -> Dict[str, Dict]:
    """
    Find decision boundaries that separate model performance.

    Returns rules like:
    - "If n_samples < 500, use TabPFN"
    - "If n_features > 50, use TabPFN"
    """
    merged = meta_features_df.merge(
        results_df[["dataset", "model", metric]],
        on="dataset"
    )

    boundaries = {}

    # Find best model for each dataset
    best_per_dataset = merged.groupby("dataset").apply(
        lambda x: x.loc[x[metric].idxmax()]
    )

    # Simple thresholds
    for col in ["n_samples", "n_features", "samples_per_feature"]:
        if col in meta_features_df.columns:
            # Bin the feature
            values = meta_features_df[col].dropna()
            if len(values) > 0:
                q25, q50, q75 = np.percentile(values, [25, 50, 75])

                # Check if this correlates with wins
                for threshold, model_name in [
                    (500, "TabPFN"),
                    (1000, "TabPFN"),
                    (50, "XGBoost"),
                ]:
                    # Count wins above/below threshold
                    wins_above = best_per_dataset[best_per_dataset[col] >= threshold]["model"].value_counts()
                    wins_below = best_per_dataset[best_per_dataset[col] < threshold]["model"].value_counts()

                    if model_name in wins_above.index or model_name in wins_below.index:
                        key = f"{col}_{threshold}"
                        boundaries[key] = {
                            "feature": col,
                            "threshold": threshold,
                            "model": model_name,
                        }

    return boundaries


if __name__ == "__main__":
    # Demo: Extract meta-features from a sample dataset
    print("Meta-Feature Extraction Demo")
    print("=" * 40)

    # Create sample data
    np.random.seed(42)
    X = np.random.randn(500, 20)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)

    # Add some categorical-like features
    X[:, 5] = np.random.choice([1, 2, 3, 4], 500)
    X[:, 10] = np.random.choice([0, 1], 500)

    categorical_mask = np.array([False] * 20)
    categorical_mask[5] = True
    categorical_mask[10] = True

    # Extract features
    meta = extract_meta_features(X, y, "demo_dataset", categorical_mask)

    for k, v in meta.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")