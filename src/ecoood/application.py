from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from .evaluation import risk_coverage


DEFAULT_CANDIDATE_COLUMNS: dict[str, str] = {
    "block_normalized_knn_distinct_chemical": (
        "ad_equal_block_distance_distinct_chemical"
    ),
    "block_normalized_knn": "ad_equal_block_distance",
    "block_normalized_knn_plus_sd": "equal_block_knn_plus_sd_risk",
    "prediction_error_risk": "prediction_error_risk_score",
    "direction_aligned_risk": "direction_aligned_risk_score",
    "ensemble_sd_risk": "ensemble_sd_risk",
    "similarity_ad": "ad_similarity",
}

TASK_ALIGNED_CANDIDATE_COLUMNS = {
    "threshold_proximity": "prediction_distance",
    "interval_crossing": "interval_cross_score",
    **DEFAULT_CANDIDATE_COLUMNS,
}
REVIEW_POLICIES = {"all_queue", "low_concern_first"}

SUPPORTED_OBJECTIVES = {
    "directional_underprediction_area",
    "false_negative_capture",
    "largest_error_capture",
    "overall_error_ranking",
}


@dataclass(frozen=True)
class EcoOODApplicationResult:
    selected_signal: str
    selection_source: str
    selection_metrics: pd.DataFrame
    routed_queue: pd.DataFrame


def _coerce_bool(values: pd.Series, *, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    if pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors="raise")
        if not numeric.isin([0, 1]).all():
            raise ValueError(f"{column} must contain only boolean or 0/1 values.")
        return numeric.astype(bool)
    normalized = values.astype(str).str.strip().str.casefold()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    if not normalized.isin(mapping).all():
        raise ValueError(f"{column} must contain boolean-like values.")
    return normalized.map(mapping).astype(bool)


def _validate_fraction(value: float, *, name: str, inclusive: bool = False) -> float:
    value = float(value)
    valid = 0 <= value <= 1 if inclusive else 0 < value < 1
    if not valid:
        bounds = "[0, 1]" if inclusive else "(0, 1)"
        raise ValueError(f"{name} must be in {bounds}.")
    return value


def _validate_chemical_frame(frame: pd.DataFrame, *, chemical_id_column: str) -> None:
    if chemical_id_column not in frame:
        raise ValueError(f"Missing chemical identifier column: {chemical_id_column}")
    if frame[chemical_id_column].isna().any():
        raise ValueError(f"{chemical_id_column} contains missing values.")
    if frame[chemical_id_column].duplicated().any():
        raise ValueError(
            "EcoOOD application input must contain one row per chemical. "
            "Aggregate case-level predictions within endpoint and then across endpoints first."
        )


def _validate_candidates(
    frame: pd.DataFrame,
    candidate_columns: Mapping[str, str],
    *,
    mask: pd.Series | np.ndarray | None = None,
) -> None:
    if not candidate_columns:
        raise ValueError("At least one candidate reliability measure is required.")
    missing = [column for column in candidate_columns.values() if column not in frame]
    if missing:
        raise ValueError(
            "Missing candidate score columns: " + ", ".join(sorted(set(missing)))
        )
    checked = frame if mask is None else frame.loc[np.asarray(mask, dtype=bool)]
    for column in candidate_columns.values():
        numeric = pd.to_numeric(checked[column], errors="coerce")
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(
                f"Candidate score column {column} contains nonfinite values for "
                "scoreable chemicals."
            )


def _review_mask(
    frame: pd.DataFrame,
    *,
    score_column: str,
    review_fraction: float,
    chemical_id_column: str,
    review_policy: str = "all_queue",
    pred_high_concern_column: str = "pred_high_concern",
) -> pd.Series:
    review_count = int(round(len(frame) * review_fraction))
    review_count = min(review_count, len(frame))
    ordered = _ordered_queue(frame, score_column=score_column,
                             chemical_id_column=chemical_id_column,
                             review_policy=review_policy,
                             pred_high_concern_column=pred_high_concern_column)
    selected = ordered.head(review_count).index
    return frame.index.to_series().isin(selected)


def _ordered_queue(
    frame: pd.DataFrame, *, score_column: str, chemical_id_column: str,
    review_policy: str, pred_high_concern_column: str,
) -> pd.DataFrame:
    if review_policy not in REVIEW_POLICIES:
        raise ValueError(f"Unknown review policy: {review_policy}")
    ordered = frame.assign(
        _selected_score=pd.to_numeric(frame[score_column], errors="raise"),
        _chemical_sort_id=frame[chemical_id_column].astype(str),
    )
    columns = ["_selected_score", "_chemical_sort_id"]
    ascending = [False, True]
    if review_policy == "low_concern_first":
        ordered = ordered.assign(_pred_high=_coerce_bool(
            frame[pred_high_concern_column], column=pred_high_concern_column))
        columns.insert(0, "_pred_high")
        ascending.insert(0, True)
    return ordered.sort_values(columns, ascending=ascending, kind="mergesort")


def _false_negative_metrics(
    true_high: np.ndarray,
    pred_high: np.ndarray,
    reviewed: np.ndarray,
) -> dict[str, float]:
    false_negative = true_high & ~pred_high
    left_at_low_priority = false_negative & ~reviewed
    lower_priority = ~pred_high & ~reviewed
    capture = (
        float(np.mean(reviewed[false_negative]))
        if false_negative.any()
        else float("nan")
    )
    high_concern_left = (
        float(left_at_low_priority.sum() / true_high.sum())
        if true_high.any()
        else float("nan")
    )
    false_omission_rate = (
        float(left_at_low_priority.sum() / lower_priority.sum())
        if lower_priority.any()
        else float("nan")
    )
    return {
        "false_negative_count": int(false_negative.sum()),
        "false_negative_capture": capture,
        "high_concern_left_low_priority_fraction": high_concern_left,
        "lower_priority_false_omission_rate": false_omission_rate,
    }


def evaluate_candidate_signals(
    development: pd.DataFrame,
    *,
    candidate_columns: Mapping[str, str] = DEFAULT_CANDIDATE_COLUMNS,
    objective: str = "false_negative_capture",
    review_fraction: float = 0.25,
    chemical_id_column: str = "chemical_id",
    pred_high_concern_column: str = "pred_high_concern",
    true_high_concern_column: str = "true_high_concern",
    y_true_column: str = "y_true",
    y_pred_column: str = "y_pred",
    directional_loss_column: str = "directional_underprediction_loss",
    high_error_quantile: float = 0.90,
    review_policy: str = "all_queue",
) -> pd.DataFrame:
    if review_policy not in REVIEW_POLICIES:
        raise ValueError(f"Unknown review policy: {review_policy}")
    review_fraction = _validate_fraction(review_fraction, name="review_fraction", inclusive=True)
    high_error_quantile = _validate_fraction(
        high_error_quantile,
        name="high_error_quantile",
    )
    if objective not in SUPPORTED_OBJECTIVES:
        raise ValueError(
            f"Unsupported objective {objective!r}; choose from {sorted(SUPPORTED_OBJECTIVES)}."
        )
    _validate_chemical_frame(development, chemical_id_column=chemical_id_column)
    _validate_candidates(development, candidate_columns)

    if pred_high_concern_column not in development:
        raise ValueError(f"Missing predicted concern column: {pred_high_concern_column}")
    pred_high = _coerce_bool(
        development[pred_high_concern_column],
        column=pred_high_concern_column,
    ).to_numpy()

    true_high: np.ndarray | None = None
    if true_high_concern_column in development:
        true_high = _coerce_bool(
            development[true_high_concern_column],
            column=true_high_concern_column,
        ).to_numpy()
    if objective == "false_negative_capture" and true_high is None:
        raise ValueError(
            f"{true_high_concern_column} is required for false-negative selection."
        )

    directional_loss: np.ndarray | None = None
    if objective == "directional_underprediction_area":
        if directional_loss_column not in development:
            raise ValueError(
                "Directional-underprediction selection requires column: "
                f"{directional_loss_column}"
            )
        directional_loss = pd.to_numeric(
            development[directional_loss_column],
            errors="raise",
        ).to_numpy(dtype=float)
        if not np.isfinite(directional_loss).all() or (directional_loss < 0).any():
            raise ValueError(
                f"{directional_loss_column} must contain finite nonnegative values."
            )

    y_true: np.ndarray | None = None
    y_pred: np.ndarray | None = None
    if objective in {"largest_error_capture", "overall_error_ranking"}:
        missing = [
            column
            for column in (y_true_column, y_pred_column)
            if column not in development
        ]
        if missing:
            raise ValueError(
                "Error-ranking objectives require columns: " + ", ".join(missing)
            )
        y_true = pd.to_numeric(development[y_true_column], errors="raise").to_numpy()
        y_pred = pd.to_numeric(development[y_pred_column], errors="raise").to_numpy()

    rows: list[dict[str, object]] = []
    for signal, score_column in candidate_columns.items():
        score = pd.to_numeric(development[score_column], errors="raise").to_numpy()
        reviewed = _review_mask(
            development,
            score_column=score_column,
            review_fraction=review_fraction,
            chemical_id_column=chemical_id_column,
            review_policy=review_policy,
            pred_high_concern_column=pred_high_concern_column,
        ).to_numpy()
        row: dict[str, object] = {
            "signal": signal,
            "score_column": score_column,
            "objective": objective,
            "review_fraction": review_fraction,
            "review_count": int(reviewed.sum()),
            "review_policy": review_policy,
        }
        if true_high is not None:
            row.update(_false_negative_metrics(true_high, pred_high, reviewed))
        if y_true is not None and y_pred is not None:
            absolute_error = np.abs(y_true - y_pred)
            high_error = absolute_error >= np.quantile(
                absolute_error,
                high_error_quantile,
            )
            row["largest_error_capture"] = (
                float(np.mean(reviewed[high_error]))
                if high_error.any()
                else float("nan")
            )
            _, _, aurc = risk_coverage(y_true, y_pred, score)
            row["aurc"] = aurc
        if directional_loss is not None:
            chemical_ids = development[chemical_id_column].astype(str).to_numpy()
            ordered = (np.lexsort((chemical_ids, -score, pred_high))
                       if review_policy == "low_concern_first"
                       else np.lexsort((chemical_ids, -score)))
            total_loss = float(directional_loss.sum())
            if total_loss > 0:
                cumulative_capture = np.r_[
                    0.0,
                    np.cumsum(directional_loss[ordered]) / total_loss,
                ]
                workload = np.arange(len(cumulative_capture), dtype=float) / len(
                    directional_loss
                )
                row["directional_underprediction_capture_area"] = float(
                    np.trapezoid(cumulative_capture, workload)
                )
                row["directional_underprediction_capture"] = float(
                    directional_loss[reviewed].sum() / total_loss
                )
            else:
                row["directional_underprediction_capture_area"] = float("nan")
                row["directional_underprediction_capture"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def select_reliability_signal(
    development: pd.DataFrame,
    *,
    candidate_columns: Mapping[str, str] = DEFAULT_CANDIDATE_COLUMNS,
    objective: str = "false_negative_capture",
    review_fraction: float = 0.25,
    default_signal: str = "block_normalized_knn",
    chemical_id_column: str = "chemical_id",
    pred_high_concern_column: str = "pred_high_concern",
    true_high_concern_column: str = "true_high_concern",
    y_true_column: str = "y_true",
    y_pred_column: str = "y_pred",
    directional_loss_column: str = "directional_underprediction_loss",
    high_error_quantile: float = 0.90,
    review_policy: str = "all_queue",
) -> tuple[str, pd.DataFrame, str]:
    if default_signal not in candidate_columns:
        raise ValueError(f"Unknown default signal: {default_signal}")
    metrics = evaluate_candidate_signals(
        development,
        candidate_columns=candidate_columns,
        objective=objective,
        review_fraction=review_fraction,
        chemical_id_column=chemical_id_column,
        pred_high_concern_column=pred_high_concern_column,
        true_high_concern_column=true_high_concern_column,
        y_true_column=y_true_column,
        y_pred_column=y_pred_column,
        directional_loss_column=directional_loss_column,
        high_error_quantile=high_error_quantile,
        review_policy=review_policy,
    )
    candidate_order = {
        signal: position for position, signal in enumerate(candidate_columns)
    }
    metrics["candidate_order"] = metrics["signal"].map(candidate_order)

    selection_source = "development_labels"
    if objective == "directional_underprediction_area":
        if metrics["directional_underprediction_capture_area"].notna().any():
            ranked = metrics.assign(
                _primary=metrics[
                    "directional_underprediction_capture_area"
                ].fillna(-np.inf),
                _secondary=metrics[
                    "directional_underprediction_capture"
                ].fillna(-np.inf),
            ).sort_values(
                ["_primary", "_secondary", "candidate_order"],
                ascending=[False, False, True],
                kind="mergesort",
            )
        else:
            ranked = metrics.sort_values("candidate_order", kind="mergesort")
            selected = default_signal
            selection_source = "benchmark_default_no_directional_loss"
            metrics["selected"] = metrics["signal"].eq(selected)
            return selected, metrics, selection_source
    elif objective == "false_negative_capture":
        if metrics["false_negative_capture"].notna().any():
            ranked = metrics.assign(
                _primary=metrics["false_negative_capture"].fillna(-np.inf),
                _secondary=metrics[
                    "high_concern_left_low_priority_fraction"
                ].fillna(np.inf),
                _tertiary=metrics[
                    "lower_priority_false_omission_rate"
                ].fillna(np.inf),
            ).sort_values(
                ["_primary", "_secondary", "_tertiary", "candidate_order"],
                ascending=[False, True, True, True],
                kind="mergesort",
            )
        else:
            ranked = metrics.sort_values("candidate_order", kind="mergesort")
            selected = default_signal
            selection_source = "benchmark_default_no_false_negatives"
            metrics["selected"] = metrics["signal"].eq(selected)
            return selected, metrics, selection_source
    elif objective == "largest_error_capture":
        ranked = metrics.assign(
            _primary=metrics["largest_error_capture"].fillna(-np.inf),
            _secondary=metrics["aurc"].fillna(np.inf),
        ).sort_values(
            ["_primary", "_secondary", "candidate_order"],
            ascending=[False, True, True],
            kind="mergesort",
        )
    else:
        ranked = metrics.assign(
            _primary=metrics["aurc"].fillna(np.inf),
            _secondary=metrics["largest_error_capture"].fillna(-np.inf),
        ).sort_values(
            ["_primary", "_secondary", "candidate_order"],
            ascending=[True, False, True],
            kind="mergesort",
        )

    selected = str(ranked.iloc[0]["signal"])
    metrics["selected"] = metrics["signal"].eq(selected)
    return selected, metrics, selection_source


def route_screening_queue(
    queue: pd.DataFrame,
    *,
    selected_signal: str,
    candidate_columns: Mapping[str, str] = DEFAULT_CANDIDATE_COLUMNS,
    review_fraction: float = 0.25,
    chemical_id_column: str = "chemical_id",
    pred_high_concern_column: str = "pred_high_concern",
    eligible_column: str = "input_eligible",
    review_policy: str = "all_queue",
) -> pd.DataFrame:
    if review_policy not in REVIEW_POLICIES:
        raise ValueError(f"Unknown review policy: {review_policy}")
    review_fraction = _validate_fraction(review_fraction, name="review_fraction", inclusive=True)
    _validate_chemical_frame(queue, chemical_id_column=chemical_id_column)
    if selected_signal not in candidate_columns:
        raise ValueError(f"Unknown selected signal: {selected_signal}")
    score_column = candidate_columns[selected_signal]
    if pred_high_concern_column not in queue:
        raise ValueError(f"Missing predicted concern column: {pred_high_concern_column}")

    routed = queue.copy()
    pred_high = _coerce_bool(
        routed[pred_high_concern_column],
        column=pred_high_concern_column,
    )
    eligible = (
        _coerce_bool(routed[eligible_column], column=eligible_column)
        if eligible_column in routed
        else pd.Series(True, index=routed.index)
    )
    _validate_candidates(
        routed,
        {selected_signal: score_column},
        mask=eligible,
    )
    routed[eligible_column] = eligible.to_numpy()
    routed["selected_signal"] = selected_signal
    routed["selected_score"] = pd.to_numeric(
        routed[score_column],
        errors="raise",
    )
    routed["review_fraction"] = review_fraction
    routed["review_policy"] = review_policy
    routed["review_rank"] = np.nan
    routed["reviewed"] = False

    eligible_frame = routed.loc[eligible].copy()
    if not eligible_frame.empty:
        ordered = _ordered_queue(
            eligible_frame, score_column="selected_score",
            chemical_id_column=chemical_id_column, review_policy=review_policy,
            pred_high_concern_column=pred_high_concern_column,
        )
        routed.loc[ordered.index, "review_rank"] = np.arange(1, len(ordered) + 1)
        review_count = int(round(len(ordered) * review_fraction))
        review_count = min(review_count, len(ordered))
        routed.loc[ordered.head(review_count).index, "reviewed"] = True

    action = pd.Series("lower_priority", index=routed.index, dtype=object)
    action.loc[eligible & pred_high & ~routed["reviewed"]] = "screen_now"
    action.loc[eligible & ~pred_high & routed["reviewed"]] = "withhold_review"
    action.loc[eligible & pred_high & routed["reviewed"]] = "prioritize_testing"
    action.loc[~eligible] = "withhold_review"
    routed["screening_action"] = action
    routed["route"] = action

    reason = pd.Series("lower predicted concern and outside review subset", index=routed.index)
    reason.loc[eligible & pred_high & ~routed["reviewed"]] = (
        "higher predicted concern and outside review subset"
    )
    reason.loc[eligible & ~pred_high & routed["reviewed"]] = (
        "lower predicted concern but selected for reliability review"
    )
    reason.loc[eligible & pred_high & routed["reviewed"]] = (
        "higher predicted concern and selected for reliability review"
    )
    reason.loc[~eligible] = "identity or molecular input did not pass eligibility checks"
    routed["route_reason"] = reason
    return routed


def apply_ecoood_protocol(
    queue: pd.DataFrame,
    *,
    development: pd.DataFrame | None = None,
    candidate_columns: Mapping[str, str] = DEFAULT_CANDIDATE_COLUMNS,
    objective: str = "false_negative_capture",
    review_fraction: float = 0.25,
    default_signal: str = "block_normalized_knn",
    chemical_id_column: str = "chemical_id",
    pred_high_concern_column: str = "pred_high_concern",
    true_high_concern_column: str = "true_high_concern",
    y_true_column: str = "y_true",
    y_pred_column: str = "y_pred",
    directional_loss_column: str = "directional_underprediction_loss",
    eligible_column: str = "input_eligible",
    high_error_quantile: float = 0.90,
    review_policy: str = "all_queue",
) -> EcoOODApplicationResult:
    if review_policy not in REVIEW_POLICIES:
        raise ValueError(f"Unknown review policy: {review_policy}")
    if default_signal not in candidate_columns:
        raise ValueError(f"Unknown default signal: {default_signal}")
    if objective not in SUPPORTED_OBJECTIVES:
        raise ValueError(
            f"Unsupported objective {objective!r}; choose from {sorted(SUPPORTED_OBJECTIVES)}."
        )
    queue_eligible = (
        _coerce_bool(queue[eligible_column], column=eligible_column)
        if eligible_column in queue
        else pd.Series(True, index=queue.index)
    )
    _validate_candidates(queue, candidate_columns, mask=queue_eligible)
    if development is None:
        selected_signal = default_signal
        selection_source = "benchmark_default_no_local_labels"
        selection_metrics = pd.DataFrame(
            [
                {
                    "signal": signal,
                    "score_column": column,
                    "objective": objective,
                    "selected": signal == selected_signal,
                    "selection_source": selection_source,
                    "review_policy": review_policy,
                }
                for signal, column in candidate_columns.items()
            ]
        )
    else:
        overlap = set(queue[chemical_id_column].astype(str)) & set(
            development[chemical_id_column].astype(str))
        if overlap:
            raise ValueError("Development and evaluation chemicals must be disjoint.")
        _validate_candidates(development, candidate_columns)
        selected_signal, selection_metrics, selection_source = (
            select_reliability_signal(
                development,
                candidate_columns=candidate_columns,
                objective=objective,
                review_fraction=review_fraction,
                default_signal=default_signal,
                chemical_id_column=chemical_id_column,
                pred_high_concern_column=pred_high_concern_column,
                true_high_concern_column=true_high_concern_column,
                y_true_column=y_true_column,
                y_pred_column=y_pred_column,
                directional_loss_column=directional_loss_column,
                high_error_quantile=high_error_quantile,
                review_policy=review_policy,
            )
        )
        selection_metrics["selection_source"] = selection_source

    routed = route_screening_queue(
        queue,
        selected_signal=selected_signal,
        candidate_columns=candidate_columns,
        review_fraction=review_fraction,
        chemical_id_column=chemical_id_column,
        pred_high_concern_column=pred_high_concern_column,
        eligible_column=eligible_column,
        review_policy=review_policy,
    )
    return EcoOODApplicationResult(
        selected_signal=selected_signal,
        selection_source=selection_source,
        selection_metrics=selection_metrics,
        routed_queue=routed,
    )
