import numpy as np
import optuna
import time
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from .base import ModelWrapper, effective_cv_folds

optuna.logging.set_verbosity(optuna.logging.WARNING)


class LightGBMWrapper(ModelWrapper):

    def __init__(self, n_trials: int = 50, n_cv_folds: int = 3,
                 random_state: int = 42, timeout: int = 120):
        super().__init__(name="LightGBM", random_state=random_state)
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
            self.tuning_time_sec_ = 0.0
            self.best_params_ = {}
            self._model = lgb.LGBMClassifier(
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1,
            )
            self._model.fit(X_train, y_train)
            return

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 255),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "random_state": self.random_state,
                "n_jobs": -1,
                "verbose": -1,
            }
            clf = lgb.LGBMClassifier(**params)
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

        self._model = lgb.LGBMClassifier(
            **self.best_params_,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        self._model.fit(X_train, y_train)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X_test)[:, 1]