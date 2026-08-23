from __future__ import annotations

import pandas as pd

from ecoood.application import apply_ecoood_protocol, select_reliability_signal


CANDIDATES = {
    "block_normalized_knn": "distance_score",
    "ensemble_sd_risk": "uncertainty_score",
}


def _development() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "chemical_id": [f"D{i}" for i in range(8)],
            "pred_high_concern": [False, False, False, True, False, True, False, True],
            "true_high_concern": [True, True, False, True, False, True, False, False],
            "y_true": [-6.0, -5.5, -3.0, -5.0, -2.0, -4.8, -2.5, -3.5],
            "y_pred": [-3.0, -3.2, -2.8, -4.8, -2.1, -4.7, -2.4, -4.2],
            "distance_score": [0.95, 0.90, 0.10, 0.20, 0.15, 0.25, 0.05, 0.30],
            "uncertainty_score": [0.10, 0.20, 0.95, 0.90, 0.30, 0.40, 0.80, 0.70],
        }
    )


def _queue() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "chemical_id": [f"Q{i}" for i in range(8)],
            "pred_high_concern": [False, True, False, True, False, False, True, False],
            "input_eligible": [True, True, True, True, True, True, True, False],
            "distance_score": [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, float("nan")],
            "uncertainty_score": [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, float("nan")],
        }
    )


def test_false_negative_selection_uses_fixed_candidate_library() -> None:
    selected, metrics, source = select_reliability_signal(
        _development(),
        candidate_columns=CANDIDATES,
        objective="false_negative_capture",
        review_fraction=0.25,
    )

    assert selected == "block_normalized_knn"
    assert source == "development_labels"
    selected_row = metrics.loc[metrics["selected"]].iloc[0]
    assert selected_row["false_negative_capture"] == 1.0


def test_application_routes_ineligible_and_reviewed_chemicals() -> None:
    result = apply_ecoood_protocol(
        _queue(),
        development=_development(),
        candidate_columns=CANDIDATES,
        review_fraction=0.25,
    )

    assert result.selected_signal == "block_normalized_knn"
    routed = result.routed_queue.set_index("chemical_id")
    assert routed.loc["Q0", "screening_action"] == "withhold_review"
    assert routed.loc["Q1", "screening_action"] == "prioritize_testing"
    assert routed.loc["Q3", "screening_action"] == "screen_now"
    assert routed.loc["Q7", "screening_action"] == "withhold_review"
    assert routed["route"].equals(routed["screening_action"])
    assert not routed.loc["Q7", "reviewed"]


def test_application_uses_benchmark_default_without_local_labels() -> None:
    result = apply_ecoood_protocol(
        _queue(),
        candidate_columns=CANDIDATES,
        review_fraction=0.25,
    )

    assert result.selected_signal == "block_normalized_knn"
    assert result.selection_source == "benchmark_default_no_local_labels"
