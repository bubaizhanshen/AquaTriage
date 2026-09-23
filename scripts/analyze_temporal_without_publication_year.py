#!/usr/bin/env python3
"""Refit the temporal benchmark without publication year as a model input."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ecoood.pipeline import ExperimentConfig, run_single_experiment
from ecoood.schema import DEFAULT_SCHEMA


DEFAULT_SEEDS = (40, 41, 42, 43, 44)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--ensemble-n-jobs", type=int, default=5)
    parser.add_argument("--review-fraction", type=float, default=0.25)
    return parser.parse_args()


def chemical_for(
    predictions: pd.DataFrame,
    score_column: str,
    review_fraction: float,
) -> float:
    endpoint = predictions.groupby(
        ["chemical_id", "chemical_name", "casrn", "chemical_class", "endpoint"],
        dropna=False,
        as_index=False,
    ).agg(
        endpoint_true=("y_true", "median"),
        endpoint_pred=("y_pred", "median"),
        endpoint_score=(score_column, "median"),
    )
    endpoint["cutoff"] = endpoint.groupby("endpoint")["endpoint_true"].transform(
        lambda values: float(values.quantile(0.25))
    )
    endpoint["true_high"] = endpoint["endpoint_true"] <= endpoint["cutoff"]
    endpoint["pred_high"] = endpoint["endpoint_pred"] <= endpoint["cutoff"]
    chemical = endpoint.groupby(
        ["chemical_id", "chemical_name", "casrn", "chemical_class"],
        dropna=False,
        as_index=False,
    ).agg(
        true_high=("true_high", "max"),
        pred_high=("pred_high", "max"),
        score=("endpoint_score", "max"),
    )
    review_n = max(1, int(round(review_fraction * len(chemical))))
    order = chemical.sort_values(
        ["score", "chemical_id"],
        ascending=[False, True],
        kind="mergesort",
    ).index.to_numpy()
    reviewed = np.zeros(len(chemical), dtype=bool)
    reviewed[order[:review_n]] = True
    lower_priority = ~chemical["pred_high"].to_numpy(dtype=bool) & ~reviewed
    if not lower_priority.any():
        return float("nan")
    return float(
        np.mean(chemical["true_high"].to_numpy(dtype=bool)[lower_priority])
    )


def main() -> None:
    args = parse_args()
    if not 0 < args.review_fraction < 1:
        raise ValueError("--review-fraction must be between 0 and 1.")
    source = pd.read_csv(args.data)

    # The split builder converts publication year to numeric before ordering.
    # The feature builder includes only numeric context fields, so storing this
    # column as text preserves the chronological split while excluding it from
    # the predictor and every reliability input.
    without_year = source.copy()
    without_year[DEFAULT_SCHEMA.study_year] = without_year[
        DEFAULT_SCHEMA.study_year
    ].astype("string")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = args.output_dir / "predictions"
    prediction_dir.mkdir(exist_ok=True)
    rows: list[dict[str, float | int]] = []
    for seed in args.seeds:
        metrics, predictions, _ = run_single_experiment(
            without_year,
            config=ExperimentConfig(
                split="temporal",
                model_name="lightgbm",
                seed=seed,
                n_members=args.members,
                ensemble_n_jobs=args.ensemble_n_jobs,
            ),
        )
        predictions.to_csv(prediction_dir / f"seed_{seed}.csv", index=False)
        rows.append(
            {
                "seed": seed,
                "rmse": metrics["rmse"],
                "coverage": metrics["coverage"],
                "aurc_ecoood": metrics["aurc"],
                "for_block_normalized_knn": chemical_for(
                    predictions,
                    "ad_equal_block_distance",
                    args.review_fraction,
                ),
                "for_ecoood": chemical_for(
                    predictions,
                    "prediction_error_risk_score",
                    args.review_fraction,
                ),
            }
        )

    detailed = pd.DataFrame(rows)
    summary = detailed.agg(["mean", "std"]).T.reset_index()
    summary.columns = ["metric", "mean", "std"]
    detailed.to_csv(args.output_dir / "temporal_without_year_detailed.csv", index=False)
    summary.to_csv(args.output_dir / "temporal_without_year_summary.csv", index=False)


if __name__ == "__main__":
    main()
