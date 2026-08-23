from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.summarize_benchmark_audits import _redraw_random_review
from scripts.summarize_multiseed_screening import summarize_fixed_workload


def test_fixed_workload_compares_methods_at_identical_review_count() -> None:
    frame = pd.DataFrame(
        {
            "seed": [40] * 8,
            "model": ["lightgbm"] * 8,
            "split": ["temporal"] * 8,
            "chemical_id": [f"c{i}" for i in range(8)],
            "chemical_name": [f"C{i}" for i in range(8)],
            "casrn": [str(i) for i in range(8)],
            "chemical_class": ["test"] * 8,
            "min_true_tox": [-8.0, -7.5, -7.0, -6.5, -6.0, -5.5, -5.0, -4.5],
            "min_pred_tox": [-5.0, -5.1, -7.1, -6.6, -6.0, -5.5, -5.0, -4.5],
            "max_ecoood": [0.99, 0.98, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            "max_input_distance": [0.1, 0.2, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            "max_similarity_risk": [0.2, 0.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            "endpoint_breadth": [1] * 8,
            "row_count": [1] * 8,
        }
    )

    _, summary = summarize_fixed_workload(frame, burdens=(0.25,))

    assert summary["review_count"].nunique() == 1
    assert summary["review_count"].iloc[0] == 2
    risk = summary.loc[
        summary["method"] == "Prediction-error risk score"
    ].iloc[0]
    assert risk["rescued_false_negatives"] == 2
    assert risk["rescued_baseline_misses"] == 2
    assert risk["lower_priority_queue_size"] == 6
    assert risk["lower_priority_false_omission_rate"] == 0
    assert risk["high_concern_left_lower_priority_fraction"] == 0
    assert (
        risk["lower_priority_false_reassurance"]
        == risk["lower_priority_false_omission_rate"]
    )


def test_random_review_is_redrawn_within_a_bootstrap_sample() -> None:
    values = np.column_stack(
        [
            np.zeros(8, dtype=bool),
            np.array([True, True, False, False, False, False, False, False]),
            np.zeros(8, dtype=bool),
            np.zeros(8, dtype=bool),
            np.array([True, True, True, False, False, False, False, False]),
            np.array([False, False, True, False, False, False, False, False]),
        ]
    )
    redrawn = _redraw_random_review(
        values,
        rng=np.random.default_rng(20260731),
        review_burden=0.25,
    )

    assert redrawn[:, 3].sum() == 2
    assert np.array_equal(
        redrawn[:, 0],
        (~redrawn[:, 5]) & (~redrawn[:, 3]),
    )
    assert np.array_equal(
        redrawn[:, 2],
        redrawn[:, 1] & redrawn[:, 3],
    )
