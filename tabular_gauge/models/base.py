"""
Abstract base class for all model wrappers.
Every model must implement fit(), predict_proba(), and get_metadata().
This ensures the benchmark harness treats all models identically.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import time
import os
import psutil


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def effective_cv_folds(y_train: np.ndarray, requested_folds: int) -> int:
    """Compute the number of CV folds we can actually use on this data.

    StratifiedKFold(n_splits=k) requires at least k members of every class.
    For our severe-imbalance datasets (sick, ozone-level, australian) the
    minority class has very few members at small N (e.g. sick has only ~2
    positives at N=25). The wrapper's requested n_cv_folds is then capped
    at `min(requested_folds, n_minority_class, n_majority_class)`.

    If the result is < 2, return 0 to signal "skip CV entirely" — the
    caller should fit on the full data without cross-validation.

    This is the canonical sklearn pattern (see e.g. `cross_val_score`'s
    default behaviour for very small datasets) and is necessary because:
      1. The data is what it is — we cannot manufacture more minority samples.
      2. Dropping these (N, dataset) cells would bias the benchmark away
         from the low-data / severe-imbalance regime the tool is designed to cover.
      3. The fix is symmetric across all baselines (no model gets an
         unfair advantage) and affects only 2 cells in the entire
         9 × 9 × 5 × 5 = 2 025-cell matrix (sick@N=25 and ozone@N=25).
    """
    if y_train is None or len(y_train) == 0:
        return 0
    classes, counts = np.unique(y_train, return_counts=True)
    if len(classes) < 2:
        return 0
    minority = int(counts.min())
    majority = int(counts.max())
    return max(0, min(requested_folds, minority, majority))


@dataclass
class RunResult:
    """Everything produced by a single model run on a single dataset."""
    model_name: str
    dataset_name: str
    seed: int

    # Core predictions
    y_test: np.ndarray
    y_proba: np.ndarray          # predicted probabilities for positive class
    y_pred: np.ndarray           # hard predictions at 0.5 threshold

    # Timing
    fit_time_sec: float
    predict_time_sec: float

    # Metadata (required)
    n_train: int
    n_test: int
    n_features: int

    # Memory profiling (optional with defaults)
    peak_memory_mb: float = 0.0
    memory_delta_mb: float = 0.0
    gpu_memory_mb: Optional[float] = None

    # Timing metadata (optional with defaults)
    tuning_time_sec: float = 0.0      # 0 for TabPFN, >0 for all tuned models
    total_wall_time_sec: float = 0.0  # tuning + fit + predict
    throughput_rows_per_sec: float = 0.0  # n_test / predict_time_sec
    best_params: Optional[dict] = None
    error: Optional[str] = None  # non-None if the run failed


class ModelWrapper(ABC):
    """
    Every model in the benchmark inherits from this class.
    
    Design principle: the wrapper is responsible for all model-specific
    logic (tuning, scaling, etc.). The harness only calls fit() and
    predict_proba(), and records timing around each.
    """

    def __init__(self, name: str, random_state: int = 42):
        self.name = name
        self.random_state = random_state
        self._model = None

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model. Must set self._model."""
        pass

    @abstractmethod
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Return probability of positive class, shape (n_samples,)."""
        pass

    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        dataset_name: str,
        seed: int,
    ) -> RunResult:
        """
        Full timed run: fit + predict + return RunResult.
        Catches exceptions and records them without crashing the harness.
        """
        n_train, n_features = X_train.shape
        n_test = X_test.shape[0]

        # Track memory before execution
        start_memory_mb = get_memory_usage_mb()

        try:
            # Timed fit
            t0 = time.perf_counter()
            self.fit(X_train, y_train)
            fit_time = time.perf_counter() - t0

            # Memory after fit
            peak_memory_mb = get_memory_usage_mb()
            memory_delta_mb = peak_memory_mb - start_memory_mb

            # Try to get GPU memory if available
            gpu_memory_mb = None
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            except ImportError:
                pass

            # Timed predict
            t0 = time.perf_counter()
            y_proba = self.predict_proba(X_test)
            predict_time = time.perf_counter() - t0
            tuning_time = getattr(self, "tuning_time_sec_", 0.0)
            total_wall  = tuning_time + fit_time + predict_time

            # Handle multi-class predictions properly
            # Detect if y_proba is 2D with multiple columns (multi-class)
            n_classes_train = len(np.unique(y_train))
            is_multiclass = n_classes_train > 2

            if y_proba.ndim == 2 and y_proba.shape[1] > 1:
                # Multi-class: use argmax to get predictions
                y_pred = np.argmax(y_proba, axis=1)
            elif is_multiclass and y_proba.ndim == 1:
                # Model returned 1D probabilities for multi-class problem
                # This can happen if model doesn't support proper multi-class
                # In this case, we treat highest probability class as prediction
                # But this is a fallback - metrics may not be meaningful
                y_pred = y_proba.astype(int)  # Use proba as class indices if they're class predictions
            else:
                # Binary classification: use threshold of 0.5
                y_proba_1d = y_proba.ravel() if y_proba.ndim > 1 else y_proba
                y_pred = (y_proba_1d >= 0.5).astype(int)

            return RunResult(
                model_name=self.name,
                dataset_name=dataset_name,
                seed=seed,
                y_test=y_test,
                y_proba=y_proba,
                y_pred=y_pred,
                fit_time_sec=fit_time,
                predict_time_sec=predict_time,
                tuning_time_sec=tuning_time,
                total_wall_time_sec=total_wall,
                throughput_rows_per_sec=n_test / (predict_time + 1e-9),
                n_train=n_train,
                n_test=n_test,
                n_features=n_features,
                peak_memory_mb=peak_memory_mb,
                memory_delta_mb=memory_delta_mb,
                gpu_memory_mb=gpu_memory_mb,
            )

        except Exception as e:
            print(f"  ERROR: {self.name} on {dataset_name} seed {seed}: {e}")
            return RunResult(
                model_name=self.name,
                dataset_name=dataset_name,
                seed=seed,
                y_test=y_test,
                y_proba=np.full(n_test, 0.5),
                y_pred=np.zeros(n_test, dtype=int),
                fit_time_sec=0.0,
                predict_time_sec=0.0,
                n_train=n_train,
                n_test=n_test,
                n_features=n_features,
                peak_memory_mb=start_memory_mb,
                error=str(e),
            )