from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from ecoood.schema import DEFAULT_SCHEMA
from scripts.run_echa_pmra_external_validation import (
    RELIABILITY_METHODS,
    calibration_split_by_chemical,
)


REVIEW_FRACTION = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate chemical-level false reassurance in an external prediction set "
            "using calibration-derived endpoint thresholds."
        )
    )
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-draws", type=int, default=10000)
    return parser.parse_args()


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def calibration_cutoffs(frame: pd.DataFrame) -> dict[str, float]:
    medians = frame.groupby(
        [DEFAULT_SCHEMA.chemical_id, DEFAULT_SCHEMA.endpoint],
        as_index=False,
    )[DEFAULT_SCHEMA.target].median()
    return (
        medians.groupby(DEFAULT_SCHEMA.endpoint)[DEFAULT_SCHEMA.target]
        .quantile(0.25)
        .to_dict()
    )


def chemical_panel(
    predictions: pd.DataFrame,
    cutoffs: dict[str, float],
) -> pd.DataFrame:
    endpoint = predictions.groupby(
        ["chemical_id", "endpoint"],
        as_index=False,
    ).agg(
        target_log_molar=("target_log_molar", "median"),
        y_pred=("y_pred", "median"),
        **{
            score_col: (score_col, "median")
            for score_col in RELIABILITY_METHODS.values()
        },
    )
    endpoint["concern_cutoff"] = endpoint["endpoint"].map(cutoffs)
    if endpoint["concern_cutoff"].isna().any():
        missing = sorted(
            endpoint.loc[endpoint["concern_cutoff"].isna(), "endpoint"].unique()
        )
        raise ValueError(f"Missing calibration cutoff for endpoint(s): {missing}")
    endpoint["measured_high_concern"] = (
        endpoint["target_log_molar"] <= endpoint["concern_cutoff"]
    )
    endpoint["predicted_high_concern"] = (
        endpoint["y_pred"] <= endpoint["concern_cutoff"]
    )
    return endpoint.groupby("chemical_id", as_index=False).agg(
        measured_high_concern=("measured_high_concern", "max"),
        predicted_high_concern=("predicted_high_concern", "max"),
        **{
            score_col: (score_col, "max")
            for score_col in RELIABILITY_METHODS.values()
        },
    )


def review_metrics(panel: pd.DataFrame, reviewed: np.ndarray) -> dict[str, float]:
    measured = panel["measured_high_concern"].to_numpy(dtype=bool)
    predicted = panel["predicted_high_concern"].to_numpy(dtype=bool)
    false_negative = measured & ~predicted
    lower_priority = ~predicted & ~reviewed
    omitted = measured & lower_priority
    return {
        "n_chemicals": int(len(panel)),
        "review_n": int(reviewed.sum()),
        "n_measured_high_concern": int(measured.sum()),
        "n_false_negative": int(false_negative.sum()),
        "n_lower_priority": int(lower_priority.sum()),
        "n_high_concern_omitted": int(omitted.sum()),
        "lower_priority_for": (
            float(omitted.sum() / lower_priority.sum())
            if lower_priority.any()
            else float("nan")
        ),
        "high_concern_left_fraction": (
            float(omitted.sum() / measured.sum())
            if measured.any()
            else float("nan")
        ),
        "false_negative_capture": (
            float((false_negative & reviewed).sum() / false_negative.sum())
            if false_negative.any()
            else float("nan")
        ),
    }


def evaluate_signal(panel: pd.DataFrame, score_col: str) -> dict[str, float]:
    review_n = max(1, int(round(REVIEW_FRACTION * len(panel))))
    order = panel.sort_values(
        [score_col, "chemical_id"],
        ascending=[False, True],
        kind="mergesort",
    ).index[:review_n]
    reviewed = panel.index.isin(order)
    return review_metrics(panel, reviewed)


def random_reference(
    panel: pd.DataFrame,
    *,
    seed: int,
    draws: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    review_n = max(1, int(round(REVIEW_FRACTION * len(panel))))
    rows: list[dict[str, float]] = []
    for _ in range(draws):
        reviewed = np.zeros(len(panel), dtype=bool)
        reviewed[rng.choice(len(panel), size=review_n, replace=False)] = True
        rows.append(review_metrics(panel, reviewed))
    return pd.DataFrame(rows).mean(numeric_only=True).to_dict()


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train_path)
    predictions = pd.read_csv(args.predictions)
    rows: list[dict[str, object]] = []
    cutoff_rows: list[dict[str, object]] = []

    for seed, seed_predictions in predictions.groupby("seed", sort=True):
        _, calibration = calibration_split_by_chemical(train, seed=int(seed))
        cutoffs = calibration_cutoffs(calibration)
        cutoff_rows.extend(
            {
                "seed": int(seed),
                "endpoint": endpoint,
                "concern_cutoff": cutoff,
            }
            for endpoint, cutoff in cutoffs.items()
        )
        panel = chemical_panel(seed_predictions, cutoffs)
        for method, score_col in RELIABILITY_METHODS.items():
            rows.append(
                {
                    "seed": int(seed),
                    "method": method,
                    **evaluate_signal(panel, score_col),
                }
            )
        rows.append(
            {
                "seed": int(seed),
                "method": "random_review",
                **random_reference(
                    panel,
                    seed=stable_seed("external_random", seed),
                    draws=args.random_draws,
                ),
            }
        )

    detailed = pd.DataFrame(rows)
    value_cols = [column for column in detailed if column not in {"seed", "method"}]
    summary = (
        detailed.groupby("method", as_index=False)[value_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        column
        if isinstance(column, str)
        else "_".join(part for part in column if part)
        for column in summary.columns.to_flat_index()
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(args.output_dir / "external_false_reassurance_all.csv", index=False)
    summary.to_csv(args.output_dir / "external_false_reassurance_summary.csv", index=False)
    pd.DataFrame(cutoff_rows).to_csv(
        args.output_dir / "external_concern_cutoffs.csv",
        index=False,
    )
    print(summary.sort_values("lower_priority_for_mean").to_string(index=False))


if __name__ == "__main__":
    main()
