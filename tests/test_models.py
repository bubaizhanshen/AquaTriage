from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

import ecoood.models as models
from ecoood.models import BootstrapEnsembleRegressor, make_estimator


def test_lightgbm_row_subsampling_is_enabled_when_available() -> None:
    estimator = make_estimator("lightgbm", seed=123)
    if estimator.__class__.__name__ == "LGBMRegressor":
        params = estimator.get_params()
        assert params["objective"] == "regression"
        assert params["subsample"] == 0.9
        assert params["subsample_freq"] == 1


def test_random_forest_uses_explicit_squared_error_criterion() -> None:
    estimator = make_estimator("random_forest", seed=123)
    assert estimator.get_params()["criterion"] == "squared_error"


def test_mlp_ensemble_handles_sparse_inputs() -> None:
    rng = np.random.default_rng(123)
    X_dense = rng.normal(size=(40, 16)).astype(float)
    X = sparse.csr_matrix(X_dense)
    y = X_dense[:, 0] * 0.5 - X_dense[:, 1] * 0.25 + rng.normal(scale=0.05, size=40)

    model = BootstrapEnsembleRegressor(model_name="mlp", n_members=2, seed=123, n_jobs=1).fit(X, y)
    pred = model.predict(X)

    assert pred.mean.shape == (40,)
    assert pred.std.shape == (40,)
    assert np.isfinite(pred.mean).all()


def test_unknown_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported model"):
        make_estimator("typo", seed=123)


def test_missing_lightgbm_is_not_silently_replaced(monkeypatch) -> None:
    monkeypatch.setattr(models, "LGBMRegressor", None)
    with pytest.raises(ImportError, match="lightgbm"):
        make_estimator("lightgbm", seed=123)
