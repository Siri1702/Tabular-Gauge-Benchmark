"""
XGBoost wrapper with Optuna hyperparameter tuning.

Design: we run a full Optuna study for each (dataset, seed) pair.
This is expensive but honest — it's the apples-to-apples comparison
that TabPFN (zero-shot) has to beat.
"""

import numpy as np
import optuna
import time
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from .base import ModelWrapper, effective_cv_folds

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


class XGBoostWrapper(ModelWrapper):

    def __init__(
        self,
        n_trials: int = 50,
        n_cv_folds: int = 3,
        random_state: int = 42,
        timeout: int = 120,     # seconds per tuning run
    ):
        super().__init__(name="XGBoost", random_state=random_state)
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

        if n_cv < 2:
            # Cannot do CV — fit with default hyperparameters. This is the
            # only sensible choice when there are < 2 members of the minority
            # class (it would happen for sick@N=25 and ozone@N=25 in the
            # 9-dataset Act-2 list).
            self.tuning_time_sec_ = 0.0
            self.best_params_ = {}
            self._model = xgb.XGBClassifier(
                eval_metric="logloss",
                random_state=self.random_state,
                n_jobs=-1,
            )
            self._model.fit(X_train, y_train)
            return

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.0),
                "eval_metric": "logloss",
                "random_state": self.random_state,
                "n_jobs": -1,
            }
            clf = xgb.XGBClassifier(**params)
            cv = StratifiedKFold(n_splits=n_cv, shuffle=True,
                                 random_state=self.random_state)
            scores = cross_val_score(clf, X_train, y_train,
                                     cv=cv, scoring="roc_auc", n_jobs=-1)
            return scores.mean()

        t_tune_start = time.perf_counter()

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout, n_jobs=-1)
        self.tuning_time_sec_ = time.perf_counter() - t_tune_start

        self.best_params_ = study.best_params
        self._model = xgb.XGBClassifier(
            **self.best_params_,
            eval_metric="logloss",
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._model.fit(X_train, y_train)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X_test)[:, 1]