"""
sklearn MLP wrapper with StandardScaler (required for gradient-based models).
Uses Optuna for hidden layer architecture and regularisation search.
"""

import warnings

import numpy as np
import optuna
import time
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from .base import ModelWrapper, effective_cv_folds

# sklearn's MLPClassifier raises a ConvergenceWarning when training stops at
# max_iter without meeting `tol`. On small-N / few-shot cells MLP frequently
# won't hit the default tol (1e-4) within 200 iterations, but the model is
# still well-defined and the cross-val score is meaningful. Suppress the
# warning so the few-shot sweep log stays clean; the metric values themselves
# are unaffected.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Architecture choices as discrete options
ARCHITECTURES = [
    (64,), (128,), (256,),
    (128, 64), (256, 128), (256, 128, 64),
]


class MLPWrapper(ModelWrapper):

    def __init__(self, n_trials: int = 30, n_cv_folds: int = 3,
                 random_state: int = 42, timeout: int = 90):
        super().__init__(name="MLP", random_state=random_state)
        self.n_trials = n_trials
        self.n_cv_folds = n_cv_folds
        self.timeout = timeout
        self.best_params_ = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        # Compute effective CV folds: min(requested, n_minority, n_majority).
        # For severe-imbalance + small-N (e.g. sick@N=25) this drops from 3
        # to 2; if even 2 folds is impossible, we skip CV entirely and use
        # the library's default hyperparameters.
        n_cv = effective_cv_folds(y_train, self.n_cv_folds)

        # sklearn's MLPClassifier with early_stopping=True runs an internal
        # stratified train_test_split per fit. That internal split fails
        # with "minimum groups for any class cannot be less than 2" when
        # the minority class is small (e.g. sick@N=25 has 2 positives and
        # the post-CV training fold leaves only 1 positive in the val
        # split). We disable early_stopping for the entire MLP wrapper —
        # training-time cost is comparable on a 200-epoch budget, and the
        # max_iter / tol params still prevent runaway fits.
        use_early_stopping = False

        if n_cv < 2:
            self.tuning_time_sec_ = 0.0
            self.best_params_ = {}
            self._model = Pipeline([
                ("scaler", StandardScaler()),
                ("mlp", MLPClassifier(
                    hidden_layer_sizes=(64,),
                    random_state=self.random_state,
                )),
            ])
            self._model.fit(X_train, y_train)
            return

        def objective(trial):
            arch_idx = trial.suggest_int("arch_idx", 0, len(ARCHITECTURES) - 1)
            params = {
                "hidden_layer_sizes": ARCHITECTURES[arch_idx],
                "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
                "learning_rate_init": trial.suggest_float(
                    "learning_rate_init", 1e-4, 1e-2, log=True
                ),
                # Generous iteration budget so the optimizer actually converges
                # on small-N cells (which it did not on the previous 200..500
                # range for e.g. titanic@N=25). The `tol` term below lets the
                # solver stop early when the loss plateau is hit.
                "max_iter": trial.suggest_int("max_iter", 400, 1000),
                "tol": trial.suggest_float("tol", 1e-5, 1e-3, log=True),
                "early_stopping": use_early_stopping,
                "random_state": self.random_state,
            }
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("mlp", MLPClassifier(**params)),
            ])
            cv = StratifiedKFold(n_splits=n_cv, shuffle=True,
                                 random_state=self.random_state)
            scores = cross_val_score(pipe, X_train, y_train,
                                     cv=cv, scoring="roc_auc", n_jobs=-1)
            return scores.mean()

        t_tune_start = time.perf_counter()

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        self.tuning_time_sec_ = time.perf_counter() - t_tune_start
        self.best_params_ = study.best_params

        arch = ARCHITECTURES[self.best_params_.pop("arch_idx")]
        self._model = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=arch,
                early_stopping=use_early_stopping,
                random_state=self.random_state,
                **self.best_params_,
            )),
        ])
        self._model.fit(X_train, y_train)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X_test)[:, 1]