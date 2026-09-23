"""Evaluate chemical-level review rankings over the complete 0-100% workload.

The analysis uses the frozen endpoint-relative chemical panel. It does not
refit toxicity predictors or reliability models. For every split and seed, all
integer review-set sizes from zero to the full eligible queue are evaluated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PRIMARY_SPLITS = ["chemical_random", "scaffold", "temporal", "species"]
METHOD_COLUMNS = {
    "Prediction-error risk score": "max_prediction_error_risk_score",
    "Ensemble-SD risk": "max_ensemble_sd_risk",
    "Block-normalized kNN + SD risk": "max_equal_block_knn_plus_sd_risk",
    "Block-normalized kNN distance": "max_equal_block_distance",
    "Distinct-chemical block-normalized kNN distance": (
        "max_equal_block_distance_distinct_chemical"
    ),
    "Similarity AD": "max_similarity_risk",
}
METHOD_ORDER = [*METHOD_COLUMNS, "Random review"]
DISPLAY_FRACTIONS = np.linspace(0.0, 1.0, 11)


def safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def ranked_rows(group: pd.DataFrame, method: str, score_column: str) -> list[dict[str, object]]:
    frame = group.reset_index(drop=True)
    n_chemicals = len(frame)
    order = frame.sort_values(
        [score_column, "chemical_id"],
        ascending=[False, True],
        kind="mergesort",
    ).index.to_numpy()
    true_high = frame["true_high_concern_endpoint_relative"].to_numpy(dtype=bool)
    pred_high = frame["pred_high_concern_endpoint_relative"].to_numpy(dtype=bool)
    baseline_miss = true_high & ~pred_high
    cumulative_rescued = np.concatenate(
        ([0], np.cumsum(baseline_miss[order], dtype=int))
    )
    predicted_lower = ~pred_high
    cumulative_predicted_lower_reviewed = np.concatenate(
        ([0], np.cumsum(predicted_lower[order], dtype=int))
    )
    n_true_high = int(true_high.sum())
    n_baseline_miss = int(baseline_miss.sum())

    rows: list[dict[str, object]] = []
    for review_count in range(n_chemicals + 1):
        rescued = int(cumulative_rescued[review_count])
        high_left = n_baseline_miss - rescued
        lower_priority_n = int(
            predicted_lower.sum()
            - cumulative_predicted_lower_reviewed[review_count]
        )
        rows.append(
            {
                "seed": int(frame["seed"].iloc[0]),
                "split": str(frame["split"].iloc[0]),
                "method": method,
                "n_chemicals": n_chemicals,
                "review_count": review_count,
                "review_fraction": review_count / n_chemicals,
                "high_concern_chemicals": n_true_high,
                "baseline_false_negative_chemicals": n_baseline_miss,
                "rescued_false_negative_chemicals": rescued,
                "false_negative_capture": safe_rate(rescued, n_baseline_miss),
                "high_concern_left_lower_priority_fraction": safe_rate(
                    high_left, n_true_high
                ),
                "lower_priority_false_omission_rate": safe_rate(
                    high_left,
                    lower_priority_n,
                ),
            }
        )
    return rows


def random_expectation_rows(group: pd.DataFrame) -> list[dict[str, object]]:
    frame = group.reset_index(drop=True)
    n_chemicals = len(frame)
    true_high = frame["true_high_concern_endpoint_relative"].to_numpy(dtype=bool)
    pred_high = frame["pred_high_concern_endpoint_relative"].to_numpy(dtype=bool)
    n_true_high = int(true_high.sum())
    n_baseline_miss = int((true_high & ~pred_high).sum())
    initial_left_fraction = safe_rate(n_baseline_miss, n_true_high)
    initial_false_omission_rate = safe_rate(n_baseline_miss, int((~pred_high).sum()))

    rows: list[dict[str, object]] = []
    for review_count in range(n_chemicals + 1):
        review_fraction = review_count / n_chemicals
        rows.append(
            {
                "seed": int(frame["seed"].iloc[0]),
                "split": str(frame["split"].iloc[0]),
                "method": "Random review",
                "n_chemicals": n_chemicals,
                "review_count": review_count,
                "review_fraction": review_fraction,
                "high_concern_chemicals": n_true_high,
                "baseline_false_negative_chemicals": n_baseline_miss,
                "rescued_false_negative_chemicals": n_baseline_miss * review_fraction,
                "false_negative_capture": (
                    review_fraction if n_baseline_miss else float("nan")
                ),
                "high_concern_left_lower_priority_fraction": (
                    initial_left_fraction * (1.0 - review_fraction)
                    if np.isfinite(initial_left_fraction)
                    else float("nan")
                ),
                "lower_priority_false_omission_rate": (
                    initial_false_omission_rate
                    if review_count < n_chemicals
                    else float("nan")
                ),
            }
        )
    return rows


def full_curves(panel: pd.DataFrame) -> pd.DataFrame:
    subset = panel.loc[
        panel["model"].eq("lightgbm") & panel["split"].isin(PRIMARY_SPLITS)
    ].copy()
    duplicate = subset.duplicated(["seed", "split", "chemical_id"], keep=False)
    if duplicate.any():
        raise ValueError("Chemical panel contains duplicate chemical rows within a split and seed.")

    rows: list[dict[str, object]] = []
    for (_, _), group in subset.groupby(["seed", "split"], sort=True):
        for method, score_column in METHOD_COLUMNS.items():
            rows.extend(ranked_rows(group, method, score_column))
        rows.extend(random_expectation_rows(group))
    return pd.DataFrame(rows)


def display_summary(curves: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for target in DISPLAY_FRACTIONS:
        selected = []
        for (_, _, _), group in curves.groupby(["seed", "split", "method"], sort=False):
            desired_count = int(round(target * int(group["n_chemicals"].iloc[0])))
            match = group.loc[group["review_count"].eq(desired_count)].copy()
            match["nominal_review_fraction"] = target
            selected.append(match)
        rows.append(pd.concat(selected, ignore_index=True))
    deciles = pd.concat(rows, ignore_index=True)

    split_means = (
        deciles.groupby(
            ["nominal_review_fraction", "split", "method"], as_index=False
        )[
            [
                "false_negative_capture",
                "high_concern_left_lower_priority_fraction",
                "lower_priority_false_omission_rate",
            ]
        ]
        .mean()
    )
    return (
        split_means.groupby(["nominal_review_fraction", "method"], as_index=False)
        .agg(
            false_negative_capture_mean=("false_negative_capture", "mean"),
            false_negative_capture_sd_across_splits=("false_negative_capture", "std"),
            high_concern_left_lower_priority_fraction_mean=(
                "high_concern_left_lower_priority_fraction",
                "mean",
            ),
            high_concern_left_lower_priority_fraction_sd_across_splits=(
                "high_concern_left_lower_priority_fraction",
                "std",
            ),
            lower_priority_false_omission_rate_mean=(
                "lower_priority_false_omission_rate",
                "mean",
            ),
            lower_priority_false_omission_rate_sd_across_splits=(
                "lower_priority_false_omission_rate",
                "std",
            ),
            n_splits=("split", "nunique"),
        )
        .assign(
            method=lambda frame: pd.Categorical(
                frame["method"], categories=METHOD_ORDER, ordered=True
            )
        )
        .sort_values(["nominal_review_fraction", "method"])
        .reset_index(drop=True)
    )


def curve_area_summary(curves: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (seed, split, method), group in curves.groupby(
        ["seed", "split", "method"], sort=True
    ):
        group = group.sort_values("review_fraction")
        rows.append(
            {
                "seed": seed,
                "split": split,
                "method": method,
                "capture_curve_area": np.trapezoid(
                    group["false_negative_capture"], group["review_fraction"]
                ),
                "remaining_high_concern_curve_area": np.trapezoid(
                    group["high_concern_left_lower_priority_fraction"],
                    group["review_fraction"],
                ),
            }
        )
    areas = pd.DataFrame(rows)
    split_means = (
        areas.groupby(["split", "method"], as_index=False)[
            ["capture_curve_area", "remaining_high_concern_curve_area"]
        ]
        .mean()
    )
    return (
        split_means.groupby("method", as_index=False)
        .agg(
            capture_curve_area_mean=("capture_curve_area", "mean"),
            capture_curve_area_sd_across_splits=("capture_curve_area", "std"),
            remaining_high_concern_curve_area_mean=(
                "remaining_high_concern_curve_area", "mean"
            ),
            remaining_high_concern_curve_area_sd_across_splits=(
                "remaining_high_concern_curve_area", "std"
            ),
            n_splits=("split", "nunique"),
        )
        .assign(
            method=lambda frame: pd.Categorical(
                frame["method"], categories=METHOD_ORDER, ordered=True
            )
        )
        .sort_values("method")
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--panel",
        type=Path,
        required=True,
        help="Chemical-level panel produced by summarize_multiseed_screening.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for workload curves and summaries.",
    )
    args = parser.parse_args()

    panel = pd.read_csv(args.panel)
    curves = full_curves(panel)
    display = display_summary(curves)
    areas = curve_area_summary(curves)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    curves.to_csv(args.output_dir / "full_review_workload_curves.csv", index=False)
    display.to_csv(args.output_dir / "full_review_workload_decile_summary.csv", index=False)
    areas.to_csv(args.output_dir / "full_review_workload_area_summary.csv", index=False)

    quarter_rows = []
    for (_, _, _), group in curves.groupby(["seed", "split", "method"], sort=False):
        review_count = int(round(0.25 * int(group["n_chemicals"].iloc[0])))
        quarter_rows.append(group.loc[group["review_count"].eq(review_count)])
    quarter = pd.concat(quarter_rows, ignore_index=True)
    quarter.to_csv(args.output_dir / "review_workload_25pct_detailed.csv", index=False)
    (
        quarter.groupby(["split", "method"], as_index=False)
        .agg(
            false_negative_capture_mean=("false_negative_capture", "mean"),
            false_negative_capture_sd=("false_negative_capture", "std"),
            high_concern_left_mean=(
                "high_concern_left_lower_priority_fraction",
                "mean",
            ),
            high_concern_left_sd=(
                "high_concern_left_lower_priority_fraction",
                "std",
            ),
            lower_priority_for_mean=(
                "lower_priority_false_omission_rate",
                "mean",
            ),
            lower_priority_for_sd=(
                "lower_priority_false_omission_rate",
                "std",
            ),
            n_seeds=("seed", "nunique"),
        )
        .to_csv(args.output_dir / "review_workload_25pct_summary.csv", index=False)
    )


if __name__ == "__main__":
    main()
