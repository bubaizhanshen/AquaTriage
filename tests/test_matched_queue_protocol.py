import numpy as np

from scripts.audit_matched_queue_protocol import evaluate, METHODS


def panel():
    values = {
        "chemical_id": np.array([str(i) for i in range(8)]),
        "pred_high": np.array([False] * 4 + [True] * 4),
        "true_high": np.array([True, False, True, False] + [True] * 4),
    }
    for col in METHODS.values():
        values[col] = np.arange(8, 0, -1, dtype=float)
    return values


def test_same_queue_random_uses_low_queue_denominator():
    result = evaluate(panel())
    assert result[("single", "random_low_queue", "capture")] == .5
    assert result[("single", "random_low_queue", "FOR")] == .5


def test_continued_threshold_uses_same_total_budget():
    result = evaluate(panel())
    assert result[("combined", "threshold", "capture")] == 1
    assert np.isnan(result[("combined", "threshold", "FOR")])
    assert result[("second", "threshold", "residual_capture")] == 1


def test_missing_false_negatives_does_not_become_zero_capture():
    a = panel()
    a["true_high"][:] = False
    result = evaluate(a)
    assert np.isnan(result[("single", "threshold", "capture")])
    assert np.isnan(result[("second", "random_low_queue", "residual_capture")])


def test_interval_comparator_obeys_the_same_low_queue_budget():
    a = panel()
    a['interval_cross_score'] = np.arange(8, dtype=float)
    result = evaluate(a)
    # Higher scores on predicted-high chemicals must not consume the quota.
    assert result[("single", "interval", "capture")] == .5
    assert result[("combined", "interval", "capture")] == 1
