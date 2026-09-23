from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

try:
    from scipy import sparse
except ImportError:  # pragma: no cover
    sparse = None

try:
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMRegressor = None


def make_estimator(model_name: str, seed: int, params: dict | None = None):
    params = dict(params or {})
    if model_name == "lightgbm":
        if LGBMRegressor is None:
            raise ImportError("lightgbm is not available in the active environment.")
        defaults = dict(
            objective="regression",
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
        )
        defaults.update(params)
        return LGBMRegressor(**defaults)
    if model_name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError:  # pragma: no cover
            XGBRegressor = None
        else:
            defaults = dict(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=8,
                subsample=0.9,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=seed,
                n_jobs=1,
            )
            defaults.update(params)
            return XGBRegressor(**defaults)
        raise ImportError("xgboost is not available in the active environment.")
    if model_name == "mlp":
        defaults = dict(
            hidden_layer_sizes=(512, 256),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=128,
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=seed,
        )
        defaults.update(params)
        return MLPRegressor(**defaults)
    if model_name == "random_forest":
        defaults = dict(
            n_estimators=500,
            criterion="squared_error",
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=1,
        )
        defaults.update(params)
        return RandomForestRegressor(**defaults)
    raise ValueError(f"Unsupported model '{model_name}'.")


@dataclass
class PredictionResult:
    mean: np.ndarray
    std: np.ndarray
    member_predictions: np.ndarray


class BootstrapEnsembleRegressor:
    def __init__(
        self,
        model_name: str = "lightgbm",
        n_members: int = 5,
        seed: int = 42,
        n_jobs: int = -1,
        estimator_params: dict | None = None,
    ) -> None:
        self.model_name = model_name
        self.n_members = n_members
        self.seed = seed
        self.n_jobs = n_jobs
        self.estimator_params = dict(estimator_params or {})
        self.members = []
        self.member_n_iter_: list[int] = []
        self.member_early_stopped_: list[bool] = []

    def _prepare_matrix(self, X):
        if self.model_name == "mlp" and sparse is not None and sparse.issparse(X):
            return X.toarray()
        return X

    def fit(self, X, y) -> "BootstrapEnsembleRegressor":
        y = np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.seed)
        seeds = rng.integers(0, 2**31 - 1, size=self.n_members)
        X_prepared = self._prepare_matrix(X)
        bootstrap_indices = [
            np.random.default_rng(int(member_seed)).choice(len(y), size=len(y), replace=True)
            for member_seed in seeds
        ]

        def _fit_member(member_seed: int, sample_idx: np.ndarray):
            estimator = make_estimator(self.model_name, int(member_seed), self.estimator_params)
            estimator.fit(X_prepared[sample_idx], y[sample_idx])
            return estimator

        self.members = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_member)(int(member_seed), sample_idx)
            for member_seed, sample_idx in zip(seeds, bootstrap_indices, strict=False)
        )
        self.member_n_iter_ = [
            int(getattr(member, "n_iter_", 0)) for member in self.members
        ]
        self.member_early_stopped_ = [
            bool(
                self.model_name == "mlp"
                and n_iter < int(
                    self.estimator_params.get(
                        "max_iter",
                        300,
                    )
                )
            )
            for n_iter in self.member_n_iter_
        ]
        return self

    def predict(self, X) -> PredictionResult:
        if not self.members:
            raise RuntimeError("Model must be fit before predict().")
        X_prepared = self._prepare_matrix(X)
        preds = np.vstack([member.predict(X_prepared) for member in self.members])
        return PredictionResult(
            mean=preds.mean(axis=0),
            std=preds.std(axis=0),
            member_predictions=preds,
        )
