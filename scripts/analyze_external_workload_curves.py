from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecoood.schema import DEFAULT_SCHEMA  # noqa: E402
from scripts.analyze_external_deployment_selection import (  # noqa: E402
    CANDIDATE_COLUMNS,
    build_chemical_panel,
    normalize_prediction_columns,
)
from scripts.audit_benchmark_integrity import (  # noqa: E402
    strict_input_eligibility_audit,
)
from scripts.run_echa_pmra_external_validation import (  # noqa: E402
    calibration_split_by_chemical,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute chemical-level external review-workload curves for all "
            "predefined reliability measures."
        )
    )
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=101)
    return parser.parse_args()


def _review_count(n: int, fraction: float) -> int:
    if fraction <= 0.0:
        return 0
    if fraction >= 1.0:
        return n
    return min(n, max(0, int(round(n * fraction))))


def _rank_review_mask(panel: pd.DataFrame, score_column: str, fraction: float) -> np.ndarray:
    n = len(panel)
    count = _review_count(n, fraction)
    reviewed = np.zeros(n, dtype=bool)
    if count == 0:
        return reviewed
    ordered = (
        panel.assign(
            _score=pd.to_numeric(panel[score_column], errors="raise"),
            _id=panel["chemical_id"].astype(str),
            _row=np.arange(n),
        )
        .sort_values(["_score", "_id", "_row"], ascending=[False, True, True], kind="mergesort")
    )
    reviewed[ordered.index.to_numpy()[:count]] = True
    return reviewed


def _metrics(panel: pd.DataFrame, reviewed: np.ndarray) -> dict[str, float | int]:
    true_high = panel["true_high_concern"].to_numpy(dtype=bool)
    predicted_high = panel["pred_high_concern"].to_numpy(dtype=bool)
    false_negative = true_high & ~predicted_high
    lower_priority = ~predicted_high & ~reviewed
    omitted = true_high & lower_priority
    return {
        "n_chemicals": int(len(panel)),
        "review_n": int(reviewed.sum()),
        "review_fraction_realized": float(reviewed.mean()) if len(panel) else float("nan"),
        "n_measured_high_concern": int(true_high.sum()),
        "n_false_negative": int(false_negative.sum()),
        "n_lower_priority": int(lower_priority.sum()),
        "n_high_concern_omitted": int(omitted.sum()),
        "lower_priority_false_omission_rate": (
            float(omitted.sum() / lower_priority.sum())
            if lower_priority.any()
            else float("nan")
        ),
        "high_concern_left_low_priority_fraction": (
            float(omitted.sum() / true_high.sum())
            if true_high.any()
            else float("nan")
        ),
        "false_negative_capture": (
            float((false_negative & reviewed).sum() / false_negative.sum())
            if false_negative.any()
            else float("nan")
        ),
    }


def _random_expected(panel: pd.DataFrame, fraction: float) -> dict[str, float | int]:
    """Expected random-review values at the same integer review count.

    The expected capture and high-concern-left fractions are exact under uniform
    sampling without replacement. The false-omission ratio uses the corresponding
    expected numerator and denominator and is reported as a reference, not as a
    simulated confidence interval.
    """
    n = len(panel)
    count = _review_count(n, fraction)
    p = count / n if n else float("nan")
    true_high = panel["true_high_concern"].to_numpy(dtype=bool)
    predicted_high = panel["pred_high_concern"].to_numpy(dtype=bool)
    false_negative = true_high & ~predicted_high
    lower_base = (~predicted_high).sum()
    omitted_base = (true_high & ~predicted_high).sum()
    expected_lower = lower_base * (1.0 - p)
    return {
        "n_chemicals": int(n),
        "review_n": int(count),
        "review_fraction_realized": float(p),
        "n_measured_high_concern": int(true_high.sum()),
        "n_false_negative": int(false_negative.sum()),
        "n_lower_priority": float(lower_base * (1.0 - p)),
        "n_high_concern_omitted": float(omitted_base * (1.0 - p)),
        "lower_priority_false_omission_rate": (
            float(omitted_base / lower_base)
            if expected_lower > 0
            else float("nan")
        ),
        "high_concern_left_low_priority_fraction": (
            float((1.0 - p) * omitted_base / true_high.sum())
            if true_high.any()
            else float("nan")
        ),
        "false_negative_capture": float(p) if false_negative.any() else float("nan"),
    }


def main() -> None:
    args = parse_args()
    if args.grid_size < 2:
        raise ValueError("grid-size must be at least 2.")

    train_source = pd.read_csv(args.train_path)
    train, _ = strict_input_eligibility_audit(train_source)
    predictions = normalize_prediction_columns(pd.read_csv(args.predictions))
    required = {"seed", "case_id", "chemical_id", "endpoint", "target_log_molar", "y_pred"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError("Predictions are missing required columns: " + ", ".join(missing))

    fractions = np.linspace(0.0, 1.0, args.grid_size)
    rows: list[dict[str, object]] = []
    for seed, seed_predictions in predictions.groupby("seed", sort=True):
        _, calibration = calibration_split_by_chemical(train, seed=int(seed))
        cutoffs = (
            calibration.groupby([DEFAULT_SCHEMA.chemical_id, DEFAULT_SCHEMA.endpoint], as_index=False)[
                DEFAULT_SCHEMA.target
            ]
            .median()
            .groupby(DEFAULT_SCHEMA.endpoint)[DEFAULT_SCHEMA.target]
            .quantile(0.25)
            .to_dict()
        )
        panel = build_chemical_panel(seed_predictions, cutoffs)
        for method, score_column in {
            **CANDIDATE_COLUMNS,
            "random_review": None,
        }.items():
            for fraction in fractions:
                if method == "random_review":
                    metrics = _random_expected(panel, float(fraction))
                else:
                    reviewed = _rank_review_mask(panel, score_column, float(fraction))
                    metrics = _metrics(panel, reviewed)
                rows.append(
                    {
                        "seed": int(seed),
                        "method": method,
                        "score_column": score_column or "",
                        "review_fraction_requested": float(fraction),
                        **metrics,
                    }
                )
            # Areas are computed after all methods have been written so that
            # the analytic random-review reference receives the same summary.

    curves = pd.DataFrame(rows)
    area_rows: list[dict[str, object]] = []
    for (seed, method), frame in curves.groupby(["seed", "method"], sort=True):
        ordered = frame.sort_values("review_fraction_requested")
        area_rows.append(
            {
                "seed": int(seed),
                "method": method,
                "false_negative_capture_area": float(
                    np.trapezoid(
                        ordered["false_negative_capture"].to_numpy(dtype=float),
                        ordered["review_fraction_requested"].to_numpy(dtype=float),
                    )
                ),
            }
        )
    areas = pd.DataFrame(area_rows)
    curves = curves.merge(areas, on=["seed", "method"], how="left")

    seed_summary = (
        curves.groupby(["seed", "method", "review_fraction_requested"], as_index=False)
        .agg(
            false_negative_capture=("false_negative_capture", "mean"),
            high_concern_left_low_priority_fraction=(
                "high_concern_left_low_priority_fraction",
                "mean",
            ),
            lower_priority_false_omission_rate=(
                "lower_priority_false_omission_rate",
                "mean",
            ),
        )
    )
    summary = (
        seed_summary.groupby(["method", "review_fraction_requested"], as_index=False)
        .agg(
            false_negative_capture_mean=("false_negative_capture", "mean"),
            false_negative_capture_sd=("false_negative_capture", "std"),
            high_concern_left_low_priority_fraction_mean=(
                "high_concern_left_low_priority_fraction",
                "mean",
            ),
            high_concern_left_low_priority_fraction_sd=(
                "high_concern_left_low_priority_fraction",
                "std",
            ),
            lower_priority_false_omission_rate_mean=(
                "lower_priority_false_omission_rate",
                "mean",
            ),
            lower_priority_false_omission_rate_sd=(
                "lower_priority_false_omission_rate",
                "std",
            ),
        )
    )
    area_summary = (
        curves[["seed", "method", "false_negative_capture_area"]]
        .drop_duplicates()
        .groupby("method", as_index=False)
        .agg(
            false_negative_capture_area_mean=("false_negative_capture_area", "mean"),
            false_negative_capture_area_sd=("false_negative_capture_area", "std"),
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    curves.to_csv(args.output_dir / "external_workload_curves_all.csv", index=False)
    seed_summary.to_csv(args.output_dir / "external_workload_curves_seed_summary.csv", index=False)
    summary.to_csv(args.output_dir / "external_workload_curves_summary.csv", index=False)
    area_summary.to_csv(args.output_dir / "external_workload_capture_area_summary.csv", index=False)
    (args.output_dir / "external_workload_notes.txt").write_text(
        "Requested workload grid includes 0% and 100%; 0% selects no chemicals and "
        "100% selects all chemicals. Random-review rows are analytic expectations "
        "under uniform sampling without replacement. Curves are chemical-level and "
        "use calibration-derived endpoint-relative concern cutoffs.\n"
    )
    print(area_summary.to_string(index=False))


if __name__ == "__main__":
    main()
