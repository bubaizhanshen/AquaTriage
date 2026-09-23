#!/usr/bin/env python3
"""Compare simple screening rules with EcoOOD reliability rankings.

This is a ranking-only analysis. It does not refit a toxicity predictor or a
reliability model. The primary endpoint is chemical-level false-negative
capture at a common review workload; lower-priority false-omission rate and
the full workload curve are reported alongside it.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ecoood.schema import DEFAULT_SCHEMA  # noqa: E402
from ecoood.splits import build_split  # noqa: E402
from scripts.run_echa_pmra_external_validation import (  # noqa: E402
    calibration_split_by_chemical,
)


RELIABILITY_COLUMNS = {
    "block_normalized_knn": "ad_equal_block_distance",
    "similarity_ad": "ad_similarity_risk",
    "prediction_error_risk": "prediction_error_risk_score",
    "ensemble_sd_risk": "ensemble_sd_risk",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal-data", type=Path, required=True)
    parser.add_argument("--internal-predictions", type=Path, required=True)
    parser.add_argument("--external-train", type=Path, required=True)
    parser.add_argument("--external-predictions", type=Path, required=True)
    parser.add_argument(
        "--internal-cutoffs",
        type=Path,
        default=None,
        help="Locked seed/split/endpoint concern cutoffs for the prediction version.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 42, 43, 44])
    parser.add_argument("--review-fraction", type=float, default=0.25)
    return parser.parse_args()


def _review_count(n: int, fraction: float) -> int:
    if n == 0 or fraction <= 0:
        return 0
    if fraction >= 1:
        return n
    return min(n, max(0, int(round(n * fraction))))


def _safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _endpoint_cutoffs(frame: pd.DataFrame, target: str) -> dict[str, float]:
    medians = (
        frame.groupby(["chemical_id", "endpoint"], dropna=False, as_index=False)[target]
        .median()
    )
    return {
        str(endpoint): float(group[target].quantile(0.25))
        for endpoint, group in medians.groupby("endpoint", sort=True)
    }


def _aggregate_predictions(
    predictions: pd.DataFrame,
    cutoffs: dict[str, float],
    *,
    true_column: str,
    pred_column: str,
    interval_lower_column: str | None = None,
    interval_upper_column: str | None = None,
) -> pd.DataFrame:
    required = {"chemical_id", "endpoint", true_column, pred_column}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError("Missing prediction columns: " + ", ".join(missing))

    frame = predictions.copy()
    aliases = {
        "ecoood_score": "prediction_error_risk_score",
        "ad_similarity": "ad_similarity_risk",
    }
    for source, target in aliases.items():
        if target not in frame and source in frame:
            frame[target] = frame[source]
    frame["concern_cutoff"] = frame["endpoint"].map(cutoffs)
    if frame["concern_cutoff"].isna().any():
        raise ValueError("Some endpoints have no calibration concern cutoff.")
    endpoint = (
        frame.groupby(["chemical_id", "endpoint"], dropna=False, as_index=False)
        .agg(
            true_tox=(true_column, "median"),
            pred_tox=(pred_column, "median"),
            concern_cutoff=("concern_cutoff", "first"),
            **{
                name: (column, "median")
                for name, column in RELIABILITY_COLUMNS.items()
                if column in frame.columns
            },
            **({
                "interval_lower": (interval_lower_column, "median"),
                "interval_upper": (interval_upper_column, "median"),
            } if interval_lower_column in frame and interval_upper_column in frame else {}),
        )
    )
    endpoint["true_high"] = endpoint["true_tox"] <= endpoint["concern_cutoff"]
    endpoint["pred_high"] = endpoint["pred_tox"] <= endpoint["concern_cutoff"]
    endpoint["false_negative"] = endpoint["true_high"] & ~endpoint["pred_high"]
    endpoint["prediction_distance"] = (
        -(endpoint["pred_tox"] - endpoint["concern_cutoff"]).abs()
    )
    endpoint["low_prediction_distance"] = endpoint["prediction_distance"].where(
        ~endpoint["pred_high"], -np.inf
    )
    if "interval_lower" in endpoint:
        endpoint["interval_crosses"] = (
            (endpoint["interval_lower"] <= endpoint["concern_cutoff"])
            & (endpoint["interval_upper"] >= endpoint["concern_cutoff"])
        )
        cross_depth = endpoint["concern_cutoff"] - endpoint["interval_lower"]
        endpoint["interval_cross_score"] = np.where(
            endpoint["interval_crosses"], 1.0 + np.maximum(cross_depth, 0.0),
            -np.maximum(endpoint["interval_lower"] - endpoint["concern_cutoff"], 0.0),
        )
        endpoint["low_interval_cross_score"] = endpoint[
            "interval_cross_score"
        ].where(~endpoint["pred_high"], -np.inf)

    score_columns = [
        "prediction_distance",
        "low_prediction_distance",
        "interval_cross_score",
        "low_interval_cross_score",
        *[name for name in RELIABILITY_COLUMNS if name in endpoint],
    ]
    score_columns = [name for name in score_columns if name in endpoint]
    aggregations: dict[str, tuple[str, str]] = {
        "true_high": ("true_high", "max"),
        "pred_high": ("pred_high", "max"),
        "false_negative": ("false_negative", "max"),
        "n_endpoints": ("endpoint", "nunique"),
    }
    aggregations.update({name: (name, "max") for name in score_columns})
    return endpoint.groupby("chemical_id", as_index=False).agg(**aggregations)


def _rank_mask(
    panel: pd.DataFrame,
    score_column: str,
    fraction: float,
    *,
    low_only: bool,
) -> np.ndarray:
    n = len(panel)
    count = _review_count(n, fraction)
    mask = np.zeros(n, dtype=bool)
    if count == 0:
        return mask
    frame = panel.copy()
    if low_only:
        frame = frame.loc[~frame["pred_high"].astype(bool)].copy()
    frame = frame.assign(
        _score=pd.to_numeric(frame[score_column], errors="coerce").fillna(-np.inf),
        _id=frame["chemical_id"].astype(str),
    ).sort_values(["_score", "_id"], ascending=[False, True], kind="mergesort")
    selected = frame.index.to_numpy()[:count]
    mask[selected] = True
    if len(selected) < count:
        # Only relevant when a low-only queue is smaller than the requested
        # burden; complete the workload with the unselected rows.
        remaining = panel.loc[~mask].assign(
            _score=pd.to_numeric(panel.loc[~mask, score_column], errors="coerce").fillna(-np.inf),
            _id=panel.loc[~mask, "chemical_id"].astype(str),
        ).sort_values(["_score", "_id"], ascending=[False, True], kind="mergesort")
        mask[remaining.index.to_numpy()[: count - len(selected)]] = True
    return mask


def _metrics(panel: pd.DataFrame, reviewed: np.ndarray) -> dict[str, float | int]:
    true_high = panel["true_high"].to_numpy(dtype=bool)
    pred_high = panel["pred_high"].to_numpy(dtype=bool)
    false_negative = true_high & ~pred_high
    lower_priority = ~pred_high & ~reviewed
    omitted = true_high & lower_priority
    return {
        "n_chemicals": int(len(panel)),
        "review_n": int(reviewed.sum()),
        "n_predicted_low": int((~pred_high).sum()),
        "n_true_high": int(true_high.sum()),
        "n_false_negative": int(false_negative.sum()),
        "rescued_false_negative": int((false_negative & reviewed).sum()),
        "false_negative_capture": _safe_rate(
            int((false_negative & reviewed).sum()), int(false_negative.sum())
        ),
        "high_concern_left_low_priority_fraction": _safe_rate(
            int(omitted.sum()), int(true_high.sum())
        ),
        "lower_priority_false_omission_rate": _safe_rate(
            int(omitted.sum()), int(lower_priority.sum())
        ),
        "low_queue_capacity_sufficient": bool((~pred_high).sum() >= reviewed.sum()),
    }


def _random_metrics(panel: pd.DataFrame, fraction: float) -> dict[str, float | int]:
    n = len(panel)
    count = _review_count(n, fraction)
    true_high = panel["true_high"].to_numpy(dtype=bool)
    pred_high = panel["pred_high"].to_numpy(dtype=bool)
    false_negative = true_high & ~pred_high
    p = count / n if n else float("nan")
    lower_n = int((~pred_high).sum())
    return {
        "n_chemicals": n,
        "review_n": count,
        "n_predicted_low": lower_n,
        "n_true_high": int(true_high.sum()),
        "n_false_negative": int(false_negative.sum()),
        "rescued_false_negative": float(false_negative.sum() * p),
        "false_negative_capture": p if false_negative.any() else float("nan"),
        "high_concern_left_low_priority_fraction": (
            float((false_negative.sum() * (1 - p)) / true_high.sum())
            if true_high.any() else float("nan")
        ),
        "lower_priority_false_omission_rate": (
            float(false_negative.sum() / lower_n) if lower_n else float("nan")
        ),
        "low_queue_capacity_sufficient": bool(lower_n >= count),
    }


def _evaluate_panel(
    panel: pd.DataFrame,
    *,
    dataset: str,
    seed: int,
    split: str,
    review_fraction: float,
) -> list[dict[str, object]]:
    methods = {
        "block_normalized_knn": "block_normalized_knn",
        "similarity_ad": "similarity_ad",
        "prediction_error_risk": "prediction_error_risk",
        "ensemble_sd_risk": "ensemble_sd_risk",
        "prediction_threshold_proximity": "prediction_distance",
        "interval_crossing": "interval_cross_score",
    }
    rows: list[dict[str, object]] = []
    for policy in ("all_queue", "predicted_low_priority_first"):
        for method, score in methods.items():
            if score not in panel:
                continue
            reviewed = _rank_mask(
                panel,
                score,
                review_fraction,
                low_only=policy == "predicted_low_priority_first",
            )
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "split": split,
                    "policy": policy,
                    "method": method,
                    "review_fraction": review_fraction,
                    **_metrics(panel, reviewed),
                }
            )
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "split": split,
                "policy": policy,
                "method": "random_review",
                "review_fraction": review_fraction,
                **_random_metrics(panel, review_fraction),
            }
        )
    return rows


def _load_cutoff_table(path: Path) -> dict[tuple[int, str], dict[str, float]]:
    table = pd.read_csv(path)
    required = {"seed", "split", "endpoint", "cutoff"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError("Internal cutoff table is missing: " + ", ".join(missing))
    result: dict[tuple[int, str], dict[str, float]] = {}
    for (seed, split), group in table.groupby(["seed", "split"], dropna=False):
        result[(int(seed), str(split))] = {
            str(row.endpoint): float(row.cutoff) for row in group.itertuples(index=False)
        }
    return result


def _internal_panels(
    data: pd.DataFrame,
    predictions: pd.DataFrame,
    seeds: list[int],
    cutoff_table: dict[tuple[int, str], dict[str, float]] | None = None,
) -> list[tuple[str, int, pd.DataFrame]]:
    rows: list[tuple[str, int, pd.DataFrame]] = []
    for seed in seeds:
        for split in sorted(predictions.loc[predictions["seed"].eq(seed), "split"].unique()):
            if cutoff_table is None:
                split_indices = build_split(data, split=split, schema=DEFAULT_SCHEMA, seed=seed)
                calibration = data.iloc[split_indices.calib]
                cutoffs = _endpoint_cutoffs(calibration, DEFAULT_SCHEMA.target)
            else:
                try:
                    cutoffs = cutoff_table[(seed, split)]
                except KeyError as exc:
                    raise ValueError(f"No locked cutoffs for seed={seed}, split={split}.") from exc
            subset = predictions.loc[
                predictions["seed"].eq(seed) & predictions["split"].eq(split)
            ].copy()
            panel = _aggregate_predictions(
                subset,
                cutoffs,
                true_column="y_true",
                pred_column="y_pred",
                interval_lower_column="endpoint_interval_lower",
                interval_upper_column="endpoint_interval_upper",
            )
            rows.append(("internal", seed, split, panel))
    return rows


def _external_panels(train: pd.DataFrame, predictions: pd.DataFrame, seeds: list[int]) -> list[tuple[str, int, pd.DataFrame]]:
    rows: list[tuple[str, int, pd.DataFrame]] = []
    for seed in seeds:
        _, calibration = calibration_split_by_chemical(train, seed=seed)
        cutoffs = _endpoint_cutoffs(calibration, DEFAULT_SCHEMA.target)
        subset = predictions.loc[predictions["seed"].eq(seed)].copy()
        if "interval_width" in subset:
            subset["interval_lower"] = subset["y_pred"] - subset["interval_width"] / 2
            subset["interval_upper"] = subset["y_pred"] + subset["interval_width"] / 2
        panel = _aggregate_predictions(
            subset,
            cutoffs,
            true_column="target_log_molar",
            pred_column="y_pred",
            interval_lower_column="interval_lower",
            interval_upper_column="interval_upper",
        )
        rows.append(("external", seed, "expanded", panel))
    return rows


def main() -> None:
    args = parse_args()
    if not 0 <= args.review_fraction <= 1:
        raise ValueError("review-fraction must be in [0, 1].")
    internal_data = pd.read_csv(args.internal_data, low_memory=False)
    internal_predictions = pd.read_csv(args.internal_predictions, low_memory=False)
    external_train = pd.read_csv(args.external_train, low_memory=False)
    external_predictions = pd.read_csv(args.external_predictions, low_memory=False)
    cutoff_table = _load_cutoff_table(args.internal_cutoffs) if args.internal_cutoffs else None

    all_rows: list[dict[str, object]] = []
    for dataset, seed, split, panel in [
        *[
            (dataset, seed, split, panel)
            for dataset, seed, split, panel in _internal_panels(
                internal_data, internal_predictions, args.seeds, cutoff_table
            )
        ],
        *[(dataset, seed, split, panel) for dataset, seed, split, panel in _external_panels(external_train, external_predictions, args.seeds)],
    ]:
        all_rows.extend(
            _evaluate_panel(
                panel,
                dataset=dataset,
                seed=seed,
                split=split,
                review_fraction=args.review_fraction,
            )
        )
    detailed = pd.DataFrame(all_rows)
    summary = (
        detailed.groupby(["dataset", "split", "policy", "method"], as_index=False)
        .agg(
            n_runs=("seed", "nunique"),
            review_n_mean=("review_n", "mean"),
            n_predicted_low_mean=("n_predicted_low", "mean"),
            false_negative_capture_mean=("false_negative_capture", "mean"),
            false_negative_capture_sd=("false_negative_capture", "std"),
            high_concern_left_low_priority_fraction_mean=(
                "high_concern_left_low_priority_fraction", "mean"
            ),
            high_concern_left_low_priority_fraction_sd=(
                "high_concern_left_low_priority_fraction", "std"
            ),
            lower_priority_false_omission_rate_mean=(
                "lower_priority_false_omission_rate", "mean"
            ),
            lower_priority_false_omission_rate_sd=(
                "lower_priority_false_omission_rate", "std"
            ),
            low_queue_capacity_sufficient_fraction=(
                "low_queue_capacity_sufficient", "mean"
            ),
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(args.output_dir / "stage1_decision_baselines_detailed.csv", index=False)
    summary.to_csv(args.output_dir / "stage1_decision_baselines_summary.csv", index=False)
    (args.output_dir / "stage1_notes.txt").write_text(
        "Ranking-only analysis; no predictor or reliability model was refit. "
        "Internal thresholds use the supplied locked seed/split cutoff table when provided; "
        "otherwise they are rebuilt from each split's calibration chemicals. External "
        "thresholds use the chemical-disjoint calibration split used by the "
        "external validation pipeline. Prediction-threshold proximity ranks "
        "endpoint predictions closest to the endpoint-specific cutoff; interval "
        "crossing ranks intervals that cross the cutoff first. The low-priority "
        "policy restricts ranking to predicted-low chemicals while the requested "
        "review burden remains the same.\n"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
