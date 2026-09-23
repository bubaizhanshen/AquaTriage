from __future__ import annotations

import pandas as pd
import pytest

from ecoood.application import apply_ecoood_protocol, evaluate_candidate_signals, select_reliability_signal, route_screening_queue


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
            "directional_underprediction_loss": [
                3.0,
                2.3,
                0.2,
                0.2,
                0.0,
                0.1,
                0.1,
                0.0,
            ],
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


def test_directional_underprediction_selection_uses_full_workload_curve() -> None:
    selected, metrics, source = select_reliability_signal(
        _development(),
        candidate_columns=CANDIDATES,
        objective="directional_underprediction_area",
        review_fraction=0.25,
    )

    assert selected == "block_normalized_knn"
    assert source == "development_labels"
    selected_row = metrics.loc[metrics["selected"]].iloc[0]
    assert selected_row["directional_underprediction_capture_area"] > 0.5


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


@pytest.mark.parametrize("fraction, count", [(0.0, 0), (0.01, 0), (0.25, 2), (1.0, 7)])
def test_application_budget_includes_endpoints(fraction: float, count: int) -> None:
    result = apply_ecoood_protocol(
        _queue(), candidate_columns=CANDIDATES, review_fraction=fraction,
    )
    assert int(result.routed_queue["reviewed"].sum()) == count
    rejected = result.routed_queue.iloc[-1]
    assert rejected["screening_action"] == "withhold_review"
    assert not rejected["reviewed"]


@pytest.mark.parametrize("fraction, count, capture", [(0.0, 0, 0.0), (0.01, 0, 0.0), (1.0, 8, 1.0)])
def test_candidate_evaluation_respects_budget(fraction: float, count: int, capture: float) -> None:
    metrics = evaluate_candidate_signals(
        _development(), candidate_columns=CANDIDATES, review_fraction=fraction,
    )
    assert (metrics["review_count"] == count).all()
    assert (metrics["false_negative_capture"] == capture).all()
    if fraction == 1.0:
        assert metrics["lower_priority_false_omission_rate"].isna().all()


@pytest.mark.parametrize("fraction", [-0.1, 1.1, float("nan"), float("inf")])
def test_application_rejects_invalid_budget(fraction: float) -> None:
    with pytest.raises(ValueError, match="review_fraction"):
        apply_ecoood_protocol(_queue(), candidate_columns=CANDIDATES, review_fraction=fraction)


@pytest.mark.parametrize("quantile", [0.0, 1.0])
def test_error_quantile_still_requires_open_interval(quantile: float) -> None:
    with pytest.raises(ValueError, match="high_error_quantile"):
        evaluate_candidate_signals(
            _development(), candidate_columns=CANDIDATES, high_error_quantile=quantile,
        )


@pytest.mark.parametrize("fraction", [0, .01, .25, .5, .75, 1])
def test_low_concern_first_preserves_budget_and_eligibility(fraction: float) -> None:
    q = _queue()
    q.loc[q.pred_high_concern, "distance_score"] = 99.
    r = route_screening_queue(q, selected_signal="block_normalized_knn",
                              candidate_columns=CANDIDATES, review_fraction=fraction,
                              review_policy="low_concern_first")
    budget = round(7 * fraction)
    assert r.reviewed.sum() == budget
    low = r.input_eligible & ~r.pred_high_concern
    assert (r.reviewed & low).sum() == min(budget, low.sum())
    assert not (r.reviewed & ~r.input_eligible).any()
    if budget <= low.sum():
        assert not (r.route == "prioritize_testing").any()


def test_development_and_evaluation_use_same_review_policy() -> None:
    d = _development()
    d.loc[d.pred_high_concern, "distance_score"] = 99.
    kwargs = dict(candidate_columns=CANDIDATES, review_fraction=.25,
                  review_policy="low_concern_first")
    metrics = evaluate_candidate_signals(d, **kwargs).set_index("signal")
    r = route_screening_queue(d, selected_signal="block_normalized_knn", **kwargs)
    fn = d.true_high_concern & ~d.pred_high_concern
    assert metrics.loc['block_normalized_knn', 'false_negative_capture'] == r.loc[fn, 'reviewed'].mean()
    assert metrics.loc['block_normalized_knn', 'false_negative_capture'] == 1.


def test_policy_ties_use_chemical_id_not_row_order() -> None:
    q = _queue()
    q.loc[q.input_eligible, 'distance_score'] = 1.
    r = route_screening_queue(q.iloc[::-1], selected_signal='block_normalized_knn',
                              candidate_columns=CANDIDATES, review_policy='low_concern_first')
    assert set(r.loc[r.reviewed, 'chemical_id']) == {'Q0','Q2'}


def test_unknown_policy_and_development_overlap_rejected() -> None:
    with pytest.raises(ValueError, match='review policy'):
        apply_ecoood_protocol(_queue(), candidate_columns=CANDIDATES, review_policy='unknown')
    with pytest.raises(ValueError, match='disjoint'):
        apply_ecoood_protocol(_development(), development=_development(), candidate_columns=CANDIDATES)


def test_primary_cli_selects_threshold_and_low_concern_first(tmp_path, monkeypatch) -> None:
    import sys
    import json
    from scripts.apply_ecoood import main
    q = _queue().rename(columns={'distance_score':'prediction_distance'})
    q.loc[q.pred_high_concern, 'prediction_distance'] = 99.
    queue_path = tmp_path/'queue.csv'
    q.to_csv(queue_path,index=False)
    out = tmp_path/'out'
    monkeypatch.setattr(sys,'argv',['apply_ecoood','--queue',str(queue_path),
        '--candidate','threshold_proximity=prediction_distance','--output-dir',str(out)])
    main()
    summary = json.loads((out/'application_summary.json').read_text())
    assert summary['review_policy'] == 'low_concern_first'
    assert summary['selected_signal'] == 'threshold_proximity'
    assert summary['reviewed_scoreable_chemicals'] == 2
    assert summary['route_counts'].get('prioritize_testing',0) == 0
