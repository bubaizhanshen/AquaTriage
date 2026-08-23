from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ecoood.ood import CalibrationRiskScorer


def test_calibration_risk_accepts_prespecified_binary_labels() -> None:
    features = pd.DataFrame({"distance": [0.0, 0.2, 0.8, 1.0]})
    labels = np.array([False, False, True, True])

    scorer = CalibrationRiskScorer().fit_labels(features, labels)
    scores = scorer.predict(features)

    assert scorer.positive_count_ == 2
    assert scores.shape == (4,)
    assert scores[-1] > scores[0]


def test_calibration_risk_rejects_misaligned_labels() -> None:
    features = pd.DataFrame({"distance": [0.0, 1.0]})

    with pytest.raises(ValueError, match="align"):
        CalibrationRiskScorer().fit_labels(features, np.array([True]))


def test_calibration_risk_uses_constant_prevalence_for_single_class() -> None:
    features = pd.DataFrame({"distance": [0.0, 0.5, 1.0]})

    scorer = CalibrationRiskScorer().fit_labels(
        features, np.array([False, False, False])
    )

    assert scorer.positive_rate_ == 0.0
    np.testing.assert_array_equal(scorer.predict(features), np.zeros(3))
