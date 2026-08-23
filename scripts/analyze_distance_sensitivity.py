#!/usr/bin/env python3
"""Compare revised and legacy prediction-error risk distance definitions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ecoood.schema import DEFAULT_SCHEMA
from ecoood.splits import build_split


SPLITS = ("chemical_random", "scaffold", "temporal", "species", "chemical_class")
PRIMARY_FOR_SPLITS = SPLITS[:4]
METHODS = {
    "revised_tanimoto_k5": ("ecoood", "ecoood_score"),
    "legacy": ("prediction_error_risk_legacy", "prediction_error_risk_legacy"),
    "revised_cosine_k5": (
        "prediction_error_risk_cosine_k5",
        "prediction_error_risk_cosine_k5",
    ),
    "revised_tanimoto_k1": (
        "prediction_error_risk_tanimoto_k1",
        "prediction_error_risk_tanimoto_k1",
    ),
    "revised_tanimoto_k3": (
        "prediction_error_risk_tanimoto_k3",
        "prediction_error_risk_tanimoto_k3",
    ),
    "revised_tanimoto_k10": (
        "prediction_error_risk_tanimoto_k10",
        "prediction_error_risk_tanimoto_k10",
    ),
    "block_normalized_knn": (
        "ad_equal_block_distance",
        "ad_equal_block_distance",
    ),
    "block_normalized_knn_distinct_chemical": (
        "ad_equal_block_distance_distinct_chemical",
        "ad_equal_block_distance_distinct_chemical",
    ),
    "input_space_knn": ("ad_distance_to_model", "ad_distance_to_model"),
    "similarity_ad": ("ad_similarity", "ad_similarity"),
    "ensemble_sd_risk": ("ensemble_sd_risk", "ensemble_sd_risk"),
}
IDENTIFIERS = ("chemical_id", "chemical_name", "casrn", "chemical_class")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 42, 43, 44])
    parser.add_argument("--model", default="lightgbm")
    parser.add_argument("--review-fraction", type=float, default=0.25)
    return parser.parse_args()


def calibration_cutoffs(frame: pd.DataFrame) -> dict[str, float]:
    medians = frame.groupby(
        [DEFAULT_SCHEMA.chemical_id, DEFAULT_SCHEMA.endpoint],
        dropna=False,
        as_index=False,
    ).agg(endpoint_true=(DEFAULT_SCHEMA.target, "median"))
    return {
        str(endpoint): float(group["endpoint_true"].quantile(0.25))
        for endpoint, group in medians.groupby(DEFAULT_SCHEMA.endpoint, sort=True)
    }


def chemical_panel(
    predictions: pd.DataFrame,
    cutoffs: dict[str, float],
    score_columns: list[str],
) -> pd.DataFrame:
    endpoint = predictions.groupby(
        [*IDENTIFIERS, "endpoint"],
        dropna=False,
        as_index=False,
    ).agg(
        endpoint_true=("y_true", "median"),
        endpoint_pred=("y_pred", "median"),
        **{
            f"endpoint_{column}": (column, "median")
            for column in score_columns
        },
    )
    endpoint["cutoff"] = endpoint["endpoint"].map(cutoffs)
    endpoint["true_high"] = endpoint["endpoint_true"] <= endpoint["cutoff"]
    endpoint["pred_high"] = endpoint["endpoint_pred"] <= endpoint["cutoff"]
    return endpoint.groupby(list(IDENTIFIERS), dropna=False, as_index=False).agg(
        true_high=("true_high", "max"),
        pred_high=("pred_high", "max"),
        **{
            column: (f"endpoint_{column}", "max")
            for column in score_columns
        },
    )


def false_omission_rate(
    panel: pd.DataFrame,
    score_column: str,
    review_fraction: float,
) -> float:
    review_n = max(1, int(round(review_fraction * len(panel))))
    order = panel.sort_values(
        [score_column, "chemical_id"],
        ascending=[False, True],
        kind="mergesort",
    ).index.to_numpy()
    reviewed = np.zeros(len(panel), dtype=bool)
    reviewed[order[:review_n]] = True
    lower_priority = ~panel["pred_high"].to_numpy(dtype=bool) & ~reviewed
    if not lower_priority.any():
        return float("nan")
    return float(panel.loc[lower_priority, "true_high"].mean())


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.data, low_memory=False)
    rows: list[dict[str, object]] = []
    score_columns = [prediction_column for _, prediction_column in METHODS.values()]
    for seed in args.seeds:
        for split_name in SPLITS:
            result_dir = args.prediction_root / f"seed_{seed}" / "structured" / split_name / args.model
            if not result_dir.exists():
                result_dir = args.prediction_root / f"seed_{seed}" / split_name / args.model
            score_summary = pd.read_csv(result_dir / "ood_score_summary.csv").set_index("method")
            predictions = pd.read_csv(result_dir / "predictions.csv", low_memory=False)
            split = build_split(data, split_name, schema=DEFAULT_SCHEMA, seed=seed)
            cutoffs = calibration_cutoffs(data.loc[split.calib])
            panel = (
                chemical_panel(predictions, cutoffs, score_columns)
                if split_name in PRIMARY_FOR_SPLITS
                else None
            )
            for label, (score_method, prediction_column) in METHODS.items():
                rows.append(
                    {
                        "seed": seed,
                        "split": split_name,
                        "method": label,
                        "aurc": float(score_summary.loc[score_method, "aurc"]),
                        "for": (
                            false_omission_rate(panel, prediction_column, args.review_fraction)
                            if panel is not None
                            else float("nan")
                        ),
                    }
                )
    detailed = pd.DataFrame(rows)
    summary = detailed.groupby("method", as_index=False).agg(
        aurc_mean=("aurc", "mean"),
        aurc_sd=("aurc", "std"),
        aurc_n=("aurc", "count"),
        for_mean=("for", "mean"),
        for_sd=("for", "std"),
        for_n=("for", "count"),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(args.output_dir / "distance_sensitivity_detailed.csv", index=False)
    summary.to_csv(args.output_dir / "distance_sensitivity_summary.csv", index=False)
    print(summary.sort_values("aurc_mean").to_string(index=False))


if __name__ == "__main__":
    main()
