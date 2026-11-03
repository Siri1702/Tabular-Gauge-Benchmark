import numpy as np
import optuna
import os
import tempfile
import time
import uuid
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from .base import ModelWrapper, effective_cv_folds

optuna.logging.set_verbosity(optuna.logging.WARNING)


class CatBoostWrapper(ModelWrapper):

    def __init__(self, n_trials: int = 50, n_cv_folds: int = 3,
                 random_state: int = 42, timeout: int = 120):
        super().__init__(name="CatBoost", random_state=random_state)
        self.n_trials = n_trials
        self.n_cv_folds = n_cv_folds
        self.timeout = timeout
        self.best_params_ = None
        # Each instance gets its own catboost_info working dir.
        # Without this, concurrent CatBoost runs collide on the
        # default "catboost_info" path and crash with
        # "Can't create train working dir: catboost_info".
        self.train_dir_ = os.path.join(
            tempfile.gettempdir(),
            f"catboost_info_{os.getpid()}_{uuid.uuid4().hex[:8]}",
        )
        os.makedirs(self.train_dir_, exist_ok=True)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        # Compute effective CV folds: min(requested, n_minority, n_majority).
        # For severe-imbalance + small-N (e.g. sick@N=25) this drops from 3
        # to 2; if even 2 folds is impossible, we skip CV entirely and use
        # the library's default hyperparameters.
        n_cv = effective_cv_folds(y_train, self.n_cv_folds)

        if n_cv < 2:
            self.tuning_time_sec_ = 0.0
            self.best_params_ = {}
            self._model = CatBoostClassifier(
                random_seed=self.random_state,
                verbose=0,
                train_dir=self.train_dir_,
            )
            self._model.fit(X_train, y_train)
            return

        def objective(trial):
            params = {
                "iterations": trial.suggest_int("iterations", 100, 1000),
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
                "border_count": trial.suggest_int("border_count", 32, 255),
                "random_seed": self.random_state,
                "verbose": 0,
                "train_dir": self.train_dir_,
            }
            clf = CatBoostClassifier(**params)
            cv = StratifiedKFold(n_splits=n_cv, shuffle=True,
                                 random_state=self.random_state)
            scores = cross_val_score(clf, X_train, y_train,
                                     cv=cv, scoring="roc_auc", n_jobs=1)
            return scores.mean()

        t_tune_start = time.perf_counter()

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout, n_jobs=1)
        self.tuning_time_sec_ = time.perf_counter() - t_tune_start
        self.best_params_ = study.best_params

        self._model = CatBoostClassifier(
            **self.best_params_,
            random_seed=self.random_state,
            verbose=0,
            train_dir=self.train_dir_,
        )
        self._model.fit(X_train, y_train)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X_test)[:, 1]