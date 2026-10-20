import io
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ecoood.review_presentation import (
    endpoint_plot_table, review_queue_view, run_summary, summary_text,
)
from ecoood.review_workflow import export_csv


def completed_run(task="hazard_omission", rule="threshold_proximity"):
    return SimpleNamespace(
        chemicals=pd.DataFrame({
            "chemical_id": ["a", "b"], "rank": [1, 2],
            "selection_basis": ["Predicted-low-concern queue"] * 2,
            "selected_for_review": [True, False],
            "margin_log10": [0.2, 0.8], "block_knn": [0.7, 0.3],
            "similarity_ad": [0.4, 0.2], "ensemble_sd": [0.1, 0.05],
            "predicted_high_concern": [False, False],
        }),
        endpoints=pd.DataFrame({
            "chemical_id": ["a", "b"],
            "endpoint": ["fish_96h_lc50", "algae_72_96h_ec50"],
            "prediction_mg_l": [10**0.2, 10**0.8],
            "case_count": [2, 1],
        }),
        manifest={
            "task": task, "rule": rule, "eligible_chemicals": 2, "review_count": 1,
            "cutoff_mg_l": 1, "realized_review_fraction": 0.5,
            "eligible_records": 3, "rejected_records": 0,
            "reference": {"records": 100, "chemicals": 40, "reference_sha256": "ref"},
            "protocol": "test", "input_sha256": "input",
        },
    )


@pytest.mark.parametrize("rule", [
    "threshold_proximity", "block_knn", "similarity_ad", "ensemble_sd",
])
def test_view_uses_completed_run_score_and_does_not_mutate(rule):
    run = completed_run(rule=rule)
    original = run.chemicals.copy(deep=True)
    view = review_queue_view(run)
    score = "margin_log10" if rule == "threshold_proximity" else rule
    np.testing.assert_array_equal(view.review_score, original[score])
    np.testing.assert_allclose(view.criterion_ratio, run.endpoints.prediction_mg_l)
    assert view.iloc[0].represented_endpoints == "Fish LC50 (96 h)"
    assert "Rank 1 of 2; 1 review place." in view.iloc[0].review_reason
    pd.testing.assert_frame_equal(run.chemicals, original)


def test_accuracy_view_has_no_criterion_interpretation():
    run = completed_run(task="general_error", rule="block_knn")
    view = review_queue_view(run)
    assert "criterion_position" not in view
    assert "criterion_ratio" not in view
    assert "Concentration criterion" not in run_summary(run).Setting.values


def test_missing_endpoints_are_explicit_and_not_predicted():
    table = endpoint_plot_table(completed_run(), "a")
    assert len(table) == 3
    missing = table[table.prediction_mg_l.isna()]
    assert len(missing) == 2
    assert missing.status.eq("No eligible input case").all()
    assert missing.case_count.isna().all()
    assert table.iloc[0].prediction_mg_l == pytest.approx(10**0.2)


def test_summary_units_denominator_and_protocol():
    run = completed_run()
    text = summary_text(run).decode()
    assert "Workload denominator: All eligible chemicals" in text
    assert "Concentration criterion: 1 mg/L" in text
    assert "Missing endpoints: Not assigned a prediction" in text
    assert "Protocol: test" in text
    assert "Reference identifier: ref" in text


def test_csv_export_escapes_text_not_negative_numbers():
    data = pd.DataFrame({"name": ["=1+1", " @SUM(A1)", "normal"], "value": [-3.2, 0, 2]})
    original = data.copy(deep=True)
    exported = pd.read_csv(io.BytesIO(export_csv(data)))
    assert exported.name.tolist() == ["'=1+1", "' @SUM(A1)", "normal"]
    np.testing.assert_array_equal(exported.value, data.value)
    pd.testing.assert_frame_equal(data, original)
