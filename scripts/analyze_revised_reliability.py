#!/usr/bin/env python3
"""Build distance-revision tables from completed multiseed predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SPLITS = ("chemical_random", "scaffold", "temporal", "species", "chemical_class")
METHOD_COLUMNS = {
    "Prediction-error risk score": "prediction_error_risk_score",
    "Ensemble-SD risk": "ensemble_sd_risk",
    "Block-normalized kNN + SD risk": "equal_block_knn_plus_sd_risk",
    "Distinct-chemical block-normalized kNN distance": (
        "ad_equal_block_distance_distinct_chemical"
    ),
    "Similarity AD": "ad_similarity",
}
COMPONENTS = (
    "d_chem_knn",
    "d_chem_mahal",
    "d_species_tax",
    "d_context",
    "context_missing_fraction",
    "d_mech",
    "bioactivity_missing_fraction",
    "u_model",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 42, 43, 44])
    parser.add_argument("--model", default="lightgbm")
    parser.add_argument("--review-fraction", type=float, default=0.25)
    return parser.parse_args()


def result_dir(root: Path, seed: int, split: str, model: str) -> Path:
    structured = root / f"seed_{seed}" / "structured" / split / model
    return structured if structured.exists() else root / f"seed_{seed}" / split / model


def load_predictions(args: argparse.Namespace) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for seed in args.seeds:
        for split in SPLITS:
            frame = pd.read_csv(
                result_dir(args.prediction_root, seed, split, args.model) / "predictions.csv",
                low_memory=False,
            )
            frame["seed"] = seed
            frame["split"] = split
            frame["model"] = args.model
            frame["abs_error"] = np.abs(frame["y_true"] - frame["y_pred"])
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def endpoint_panel(frame: pd.DataFrame, quantile: float) -> pd.DataFrame:
    identifiers = ["seed", "split", "model", "chemical_id", "endpoint"]
    score_columns = [column for column in METHOD_COLUMNS.values() if column in frame]
    endpoint = frame.groupby(identifiers, dropna=False, as_index=False).agg(
        endpoint_true=("y_true", "median"),
        endpoint_pred=("y_pred", "median"),
        **{f"endpoint_{column}": (column, "median") for column in score_columns},
    )
    endpoint["cutoff"] = endpoint.groupby(
        ["seed", "split", "model", "endpoint"], dropna=False
    )["endpoint_true"].transform(lambda values: float(values.quantile(quantile)))
    endpoint["true_high"] = endpoint["endpoint_true"] <= endpoint["cutoff"]
    endpoint["pred_high"] = endpoint["endpoint_pred"] <= endpoint["cutoff"]
    return endpoint.groupby(
        ["seed", "split", "model", "chemical_id"], dropna=False, as_index=False
    ).agg(
        true_high=("true_high", "max"),
        pred_high=("pred_high", "max"),
        **{column: (f"endpoint_{column}", "max") for column in score_columns},
    )


def threshold_sensitivity(
    predictions: pd.DataFrame,
    review_fraction: float,
) -> pd.DataFrame:
    detailed: list[dict[str, object]] = []
    for quantile in np.arange(0.10, 0.301, 0.025):
        panel = endpoint_panel(predictions, float(quantile))
        for (seed, split), group in panel.groupby(["seed", "split"], sort=True):
            review_n = max(1, int(round(review_fraction * len(group))))
            for method, column in METHOD_COLUMNS.items():
                if column not in group:
                    continue
                order = group.sort_values(
                    [column, "chemical_id"],
                    ascending=[False, True],
                    kind="mergesort",
                ).index.to_numpy()
                reviewed = group.index.to_series().isin(order[:review_n]).to_numpy()
                lower = ~group["pred_high"].to_numpy(dtype=bool) & ~reviewed
                value = (
                    float(group.loc[lower, "true_high"].mean())
                    if lower.any()
                    else float("nan")
                )
                detailed.append(
                    {
                        "seed": seed,
                        "split": split,
                        "method": method,
                        "high_concern_quantile": float(quantile),
                        "review_fraction": review_fraction,
                        "lower_priority_false_omission_rate": value,
                    }
                )
    frame = pd.DataFrame(detailed)
    summary = frame.groupby(
        ["split", "method", "high_concern_quantile", "review_fraction"],
        as_index=False,
    ).agg(
        lower_priority_false_reassurance_mean=(
            "lower_priority_false_omission_rate", "mean"
        ),
        lower_priority_false_reassurance_std=(
            "lower_priority_false_omission_rate", "std"
        ),
        n_defined=("lower_priority_false_omission_rate", "count"),
    )
    return frame, summary


def component_associations(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (seed, split), group in predictions.groupby(["seed", "split"], sort=True):
        for component in COMPONENTS:
            pair = group[[component, "abs_error"]].dropna()
            rho = (
                pair[component].corr(pair["abs_error"], method="spearman")
                if pair[component].nunique() > 1 and pair["abs_error"].nunique() > 1
                else float("nan")
            )
            rows.append(
                {"seed": seed, "split": split, "component": component, "rho": rho}
            )
    return pd.DataFrame(rows)


def component_correlations(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (seed, split), group in predictions.groupby(["seed", "split"], sort=True):
        corr = group[list(COMPONENTS)].corr(method="spearman")
        for left_index, left in enumerate(COMPONENTS):
            for right in COMPONENTS[left_index + 1 :]:
                rows.append(
                    {
                        "seed": seed,
                        "split": split,
                        "component_left": left,
                        "component_right": right,
                        "rho": corr.loc[left, right],
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    predictions = load_predictions(args)
    detailed, summary = threshold_sensitivity(predictions, args.review_fraction)
    associations = component_associations(predictions)
    correlations = component_correlations(predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(args.output_dir / "threshold_sensitivity_detailed.csv", index=False)
    summary.to_csv(
        args.output_dir / "fixed_workload_high_concern_threshold_sensitivity.csv",
        index=False,
    )
    associations.to_csv(args.output_dir / "component_error_associations.csv", index=False)
    correlations.to_csv(args.output_dir / "component_correlations.csv", index=False)


if __name__ == "__main__":
    main()
