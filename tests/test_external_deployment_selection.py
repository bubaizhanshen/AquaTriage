import numpy as np
import pandas as pd
import pytest

from scripts.analyze_external_deployment_selection import (
    build_chemical_panel,
    normalize_prediction_columns,
    random_reference,
    routed_metrics,
)


CANDIDATES = {
    "support": "support_score",
    "similarity": "similarity_score",
}


def test_normalize_prediction_columns_supports_frozen_score_name() -> None:
    predictions = pd.DataFrame({"ecoood_score": [0.2, 0.8]})

    normalized = normalize_prediction_columns(predictions)

    assert normalized["prediction_error_risk_score"].tolist() == [0.2, 0.8]
    assert "prediction_error_risk_score" not in predictions


def test_normalize_prediction_columns_maps_historical_alias() -> None:
    predictions = pd.DataFrame({"ecoood_score": [0.2, 0.8]})

    normalized = normalize_prediction_columns(predictions)

    assert normalized["prediction_error_risk_score"].tolist() == [0.2, 0.8]


def test_build_chemical_panel_uses_endpoint_cutoffs_and_chemical_aggregation() -> None:
    predictions = pd.DataFrame(
        {
            "case_id": ["a1", "a2", "a3", "b1"],
            "chemical_id": ["A", "A", "A", "B"],
            "endpoint": ["fish", "fish", "algae", "fish"],
            "target_log_molar": [-6.0, -5.0, -4.0, -3.0],
            "y_pred": [-4.0, -4.0, -5.0, -4.0],
            "support_score": [0.2, 0.4, 0.8, 0.1],
            "similarity_score": [0.3, 0.5, 0.7, 0.2],
        }
    )

    panel = build_chemical_panel(
        predictions,
        {"fish": -4.5, "algae": -4.5},
        candidate_columns=CANDIDATES,
    ).set_index("chemical_id")

    assert bool(panel.loc["A", "true_high_concern"])
    assert bool(panel.loc["A", "pred_high_concern"])
    assert not bool(panel.loc["B", "true_high_concern"])
    assert panel.loc["A", "n_endpoints"] == 2
    assert panel.loc["A", "n_cases"] == 3
    assert np.isclose(panel.loc["A", "support_score"], 0.8)
    assert np.isclose(
        panel.loc["A", "directional_underprediction_loss"],
        1.5,
    )


def test_random_reference_keeps_the_requested_review_workload() -> None:
    evaluation = pd.DataFrame(
        {
            "chemical_id": [f"C{i}" for i in range(8)],
            "true_high_concern": [True, False, True, False] * 2,
            "pred_high_concern": [False, False, True, False] * 2,
        }
    )

    metrics = random_reference(
        evaluation,
        review_fraction=0.25,
        draws=200,
        seed=42,
    )

    assert metrics["n_chemicals"] == 8
    assert metrics["review_n"] == 2
    assert 0 <= metrics["false_negative_capture"] <= 1


@pytest.mark.parametrize("fraction, count, capture", [(0.0, 0, 0.0), (0.01, 0, 0.0), (1.0, 4, 1.0)])
def test_random_reference_budget_boundaries(fraction, count, capture) -> None:
    evaluation = pd.DataFrame({
        "chemical_id": ["a", "b", "c", "d"],
        "true_high_concern": [True, False, True, False],
        "pred_high_concern": [False, False, True, False],
    })
    metrics = random_reference(evaluation, review_fraction=fraction, draws=20, seed=42)
    assert metrics["review_n"] == count
    assert metrics["false_negative_capture"] == capture
    if fraction == 1:
        assert np.isnan(metrics["lower_priority_false_omission_rate"])
    if count == 0:
        assert np.isnan(metrics["reviewed_high_concern_fraction"])


def test_routed_metrics_counts_high_concern_left_at_low_priority() -> None:
    routed = pd.DataFrame(
        {
            "true_high_concern": [True, True, False, False],
            "pred_high_concern": [False, True, False, False],
            "reviewed": [False, False, True, False],
            "route": [
                "lower_priority",
                "screen_now",
                "withhold_review",
                "lower_priority",
            ],
        }
    )

    metrics = routed_metrics(routed)

    assert metrics["n_false_negative"] == 1
    assert metrics["n_high_concern_omitted"] == 1
    assert np.isclose(metrics["lower_priority_false_omission_rate"], 0.5)
    assert np.isclose(metrics["false_negative_capture"], 0.0)
