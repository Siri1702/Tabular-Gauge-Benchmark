"""
TabPFN wrapper.

Critical design choices:
- TabPFN is ZERO-SHOT: we call fit() but it doesn't train, it stores context
- n_ensemble_configurations controls the number of augmented forward passes
- For datasets > tabpfn_max_rows, we subsample the training context
- TabPFN v2 handles missing values internally — we pass preprocessed data anyway

Research Notes:
- TabPFN has a maximum context window of ~8000 samples
- TabPFN works best with N < 10,000 samples (sweet spot)
- TabPFN max features is ~50
- TabPFN is optimized for binary classification
"""

import os
import numpy as np
from dotenv import load_dotenv

# Load API keys from .env file if it exists
load_dotenv()

try:
    from tabpfn import TabPFNClassifier
    try:
        from tabpfn.constants import ModelVersion
    except ImportError:
        ModelVersion = None
    _USE_LOCAL_TABPFN = True
except ImportError:
    from tabpfn_client import TabPFNClassifier
    ModelVersion = None
    _USE_LOCAL_TABPFN = False

# Allow TabPFN to run on CPU even when the dataset has more than 1000 samples.
os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

from .base import ModelWrapper


class TabPFNWrapper(ModelWrapper):
    """
    Wrapper for TabPFN 3.5.
    Leverages the massively expanded context window (1M rows, 20k features).
    """
    def __init__(
        self,
        n_ensemble_configurations: int = 8,
        max_context_size: int = 1_000_000,
        random_state: int = 42,
        device: str = "cpu",
    ):
        super().__init__(name="TabPFN_3.5", random_state=random_state)
        self.n_ensemble_configurations = n_ensemble_configurations
        self.max_context_size = max_context_size
        self.device = device

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        n_original = len(X_train)

        if len(X_train) > self.max_context_size:
            X_train, y_train = self._subsample_balanced(
                X_train, y_train, self.max_context_size
            )

        self.tuning_time_sec_ = 0.0
        self.context_used_ = len(X_train)
        self.context_original_ = n_original
        self.was_subsampled_ = n_original > self.max_context_size

        if _USE_LOCAL_TABPFN and ModelVersion:
            version = ModelVersion.V3 if hasattr(ModelVersion, 'V3') else ModelVersion.V2_6
            try:
                self._model = TabPFNClassifier.create_default_for_version(version)
                if hasattr(self._model, 'n_estimators'):
                    self._model.n_estimators = self.n_ensemble_configurations
                os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
                if hasattr(self._model, 'device'):
                    self._model.device = "cpu"
                if hasattr(self._model, 'ignore_pretraining_limits'):
                    self._model.ignore_pretraining_limits = True
            except Exception:
                self._model = TabPFNClassifier(
                    n_estimators=self.n_ensemble_configurations,
                    device="cpu",
                    random_state=self.random_state,
                    ignore_pretraining_limits=True,
                )
        else:
            self._model = TabPFNClassifier(
                n_estimators=self.n_ensemble_configurations,
            )
        self._model.fit(X_train, y_train)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        proba = self._model.predict_proba(X_test)
        if proba.shape[1] == 2:
            return proba[:, 1]
        return proba.max(axis=1)

    def get_metadata(self):
        return {
            "context_used": getattr(self, 'context_used_', None),
            "context_original": getattr(self, 'context_original_', None),
            "was_subsampled": getattr(self, 'was_subsampled_', False),
            "max_context_size": self.max_context_size,
            "n_ensemble_configurations": self.n_ensemble_configurations,
        }

    def _subsample_balanced(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.RandomState(self.random_state)
        classes, counts = np.unique(y, return_counts=True)
        per_class = n // len(classes)
        indices = []
        for cls in classes:
            cls_idx = np.where(y == cls)[0]
            take = min(per_class, len(cls_idx))
            indices.append(rng.choice(cls_idx, size=take, replace=False))
        idx = np.concatenate(indices)
        rng.shuffle(idx)
        return X[idx], y[idx]



class TabPFN35Wrapper(TabPFNWrapper):
    """
    Wrapper for TabPFN 3.5.
    Leverages the massively expanded context window (1M rows, 20k features).
    """

    def __init__(
        self,
        n_ensemble_configurations: int = 8,
        max_context_size: int = 1_000_000,
        random_state: int = 42,
    ):
        super().__init__(
            n_ensemble_configurations=n_ensemble_configurations,
            max_context_size=max_context_size,
            random_state=random_state
        )
        self.name = "TabPFN_3.5"

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        n_original = len(X_train)

        # TabPFN 3.5 has a huge window, but we still respect max_context_size
        if len(X_train) > self.max_context_size:
            X_train, y_train = self._subsample_balanced(
                X_train, y_train, self.max_context_size
            )

        self.tuning_time_sec_ = 0.0
        self.context_used_ = len(X_train)
        self.context_original_ = n_original
        self.was_subsampled_ = n_original > self.max_context_size

        if _USE_LOCAL_TABPFN and ModelVersion:
            # Use the factory method to create the best available version.
            # Based on current installed package, we use V3 if available, otherwise V2_6.
            version = ModelVersion.V3 if hasattr(ModelVersion, 'V3') else ModelVersion.V2_6
            try:
                self._model = TabPFNClassifier.create_default_for_version(version)
                if hasattr(self._model, 'n_estimators'):
                    self._model.n_estimators = self.n_ensemble_configurations

                # TabPFN 3.5 (V3) often requires explicit device placement
                # and may have different internal safety checks for CPU.
                # We force CPU and set the env var explicitly.
                os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"

                if hasattr(self._model, 'device'):
                    self._model.device = "cpu"

                # Some TabPFN versions require ignore_pretraining_limits
                # to be passed to the constructor or set as a property.
                if hasattr(self._model, 'ignore_pretraining_limits'):
                    self._model.ignore_pretraining_limits = True

            except Exception as e:
                self._model = TabPFNClassifier(
                    n_estimators=self.n_ensemble_configurations,
                    device="cpu",
                    random_state=self.random_state,
                    ignore_pretraining_limits=True,
                )
        else:
            # Fallback for client or old versions
            self._model = TabPFNClassifier(
                n_estimators=self.n_ensemble_configurations,
            )

        self._model.fit(X_train, y_train)
