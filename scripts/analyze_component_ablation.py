#!/usr/bin/env python3
"""Summarize leave-one-component-out prediction-error risk sensitivities."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ecoood.schema import DEFAULT_SCHEMA
from ecoood.splits import build_split


SEEDS = (40, 41, 42, 43, 44)
SPLITS = ("chemical_random", "scaffold", "temporal", "species", "chemical_class")
PRIMARY_FOR_SPLITS = SPLITS[:4]
COMPONENTS = {
    "d_chem_knn": "Distinct-chemical fingerprint Tanimoto kNN distance",
    "d_chem_mahal": "Unique-chemical shrinkage Mahalanobis distance",
    "d_species_tax": "Taxonomic-lineage distance",
    "d_context": "Endpoint-restricted context Gower kNN distance",
    "context_missing_fraction": "Context missing fraction",
    "d_mech": "Bioactivity-proxy nan-Euclidean kNN distance",
    "bioactivity_missing_fraction": "Bioactivity-proxy missing fraction",
    "u_model": "Ensemble SD",
}
IDENTIFIERS = ("chemical_id", "chemical_name", "casrn", "chemical_class")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    if endpoint["cutoff"].isna().any():
        missing = sorted(endpoint.loc[endpoint["cutoff"].isna(), "endpoint"].unique())
        raise ValueError(f"Missing calibration cutoff for endpoint(s): {missing}")
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
    ranked = panel.reset_index(drop=True)
    review_n = max(1, int(round(review_fraction * len(ranked))))
    order = ranked.sort_values(
        [score_column, "chemical_id"],
        ascending=[False, True],
        kind="mergesort",
    ).index.to_numpy()
    reviewed = np.zeros(len(ranked), dtype=bool)
    reviewed[order[:review_n]] = True
    lower_priority = ~ranked["pred_high"].to_numpy(dtype=bool) & ~reviewed
    if not lower_priority.any():
        return float("nan")
    return float(
        ranked.loc[lower_priority, "true_high"].to_numpy(dtype=bool).mean()
    )


def load_aurc(root: Path, model: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    methods = {
        "full": "ecoood",
        **{
            component: f"ecoood_minus_component_{component}"
            for component in COMPONENTS
        },
    }
    for seed in SEEDS:
        for split in SPLITS:
            path = root / f"seed_{seed}" / "structured" / split / model / "ood_score_summary.csv"
            if not path.exists():
                path = root / f"seed_{seed}" / split / model / "ood_score_summary.csv"
            scores = pd.read_csv(path).set_index("method")
            full = float(scores.loc[methods["full"], "aurc"])
            rows.append(
                {
                    "seed": seed,
                    "split": split,
                    "component": "full",
                    "component_label": "Full eight-signal score",
                    "aurc": full,
                    "delta_aurc": 0.0,
                }
            )
            for component, label in COMPONENTS.items():
                value = float(scores.loc[methods[component], "aurc"])
                rows.append(
                    {
                        "seed": seed,
                        "split": split,
                        "component": component,
                        "component_label": label,
                        "aurc": value,
                        "delta_aurc": value - full,
                    }
                )
    return pd.DataFrame(rows)


def load_for(
    data: pd.DataFrame,
    root: Path,
    model: str,
    review_fraction: float,
) -> pd.DataFrame:
    score_columns = [
        "ecoood_score",
        *[f"ecoood_minus_component_{component}" for component in COMPONENTS],
    ]
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        for split_name in PRIMARY_FOR_SPLITS:
            split = build_split(data, split_name, schema=DEFAULT_SCHEMA, seed=seed)
            cutoffs = calibration_cutoffs(data.loc[split.calib])
            path = root / f"seed_{seed}" / "structured" / split_name / model / "predictions.csv"
            if not path.exists():
                path = root / f"seed_{seed}" / split_name / model / "predictions.csv"
            predictions = pd.read_csv(path)
            expected = data.loc[split.test, DEFAULT_SCHEMA.target].to_numpy(dtype=float)
            observed = predictions["y_true"].to_numpy(dtype=float)
            if len(expected) != len(observed) or not np.allclose(
                expected, observed, rtol=0.0, atol=1e-12
            ):
                raise ValueError(f"Prediction rows do not match {split_name}, seed {seed}")
            panel = chemical_panel(predictions, cutoffs, score_columns)
            full = false_omission_rate(panel, "ecoood_score", review_fraction)
            rows.append(
                {
                    "seed": seed,
                    "split": split_name,
                    "component": "full",
                    "component_label": "Full eight-signal score",
                    "for": full,
                    "delta_for": 0.0,
                }
            )
            for component, label in COMPONENTS.items():
                value = false_omission_rate(
                    panel,
                    f"ecoood_minus_component_{component}",
                    review_fraction,
                )
                rows.append(
                    {
                        "seed": seed,
                        "split": split_name,
                        "component": component,
                        "component_label": label,
                        "for": value,
                        "delta_for": value - full,
                    }
                )
    return pd.DataFrame(rows)


def summarize(aurc: pd.DataFrame, omission: pd.DataFrame) -> pd.DataFrame:
    aurc_summary = aurc.groupby(
        ["component", "component_label"], as_index=False
    ).agg(
        aurc_mean=("aurc", "mean"),
        aurc_sd=("aurc", "std"),
        delta_aurc_mean=("delta_aurc", "mean"),
        delta_aurc_sd=("delta_aurc", "std"),
        aurc_n=("aurc", "count"),
    )
    for_summary = omission.groupby(
        ["component", "component_label"], as_index=False
    ).agg(
        for_mean=("for", "mean"),
        for_sd=("for", "std"),
        delta_for_mean=("delta_for", "mean"),
        delta_for_sd=("delta_for", "std"),
        for_n=("for", "count"),
    )
    result = aurc_summary.merge(
        for_summary,
        on=["component", "component_label"],
        how="outer",
    )
    order = {"full": 0, **{component: i + 1 for i, component in enumerate(COMPONENTS)}}
    result["order"] = result["component"].map(order)
    return result.sort_values("order").drop(columns="order").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.data)
    aurc = load_aurc(args.prediction_root, args.model)
    omission = load_for(
        data,
        args.prediction_root,
        args.model,
        args.review_fraction,
    )
    summary = summarize(aurc, omission)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aurc.to_csv(args.output_dir / "component_ablation_aurc_detailed.csv", index=False)
    omission.to_csv(args.output_dir / "component_ablation_for_detailed.csv", index=False)
    summary.to_csv(args.output_dir / "component_ablation_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
