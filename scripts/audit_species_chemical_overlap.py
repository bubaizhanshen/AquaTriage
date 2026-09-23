"""Audit chemical overlap within the species-holdout evaluation.

The species split changes the organism domain and can also change the chemical
mixture. This audit separates held-out species cases whose chemicals were seen
in the corresponding training fold from cases whose chemicals were not seen.
It reuses frozen predictions and does not refit the predictor or reliability
scores.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ecoood.evaluation import regression_metrics, risk_coverage
from ecoood.schema import DEFAULT_SCHEMA
from ecoood.splits import build_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Separate species-holdout results by training chemical overlap."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--revision-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 42, 43, 44])
    parser.add_argument("--model", default="lightgbm")
    return parser.parse_args()


def summarize_subset(
    frame: pd.DataFrame,
    *,
    seed: int,
    subset: str,
) -> dict[str, object]:
    y_true = frame["y_true"].to_numpy(dtype=float)
    y_pred = frame["y_pred"].to_numpy(dtype=float)
    covered = (frame["y_true"] >= frame["interval_lower"]) & (
        frame["y_true"] <= frame["interval_upper"]
    )
    risk_column = (
        "prediction_error_risk_score"
        if "prediction_error_risk_score" in frame.columns
        else "ecoood_score"
    )
    _, _, aurc = risk_coverage(
        y_true,
        y_pred,
        frame[risk_column].to_numpy(dtype=float),
    )
    return {
        "seed": seed,
        "subset": subset,
        "n_cases": int(len(frame)),
        "n_chemicals": int(frame["chemical_id"].nunique()),
        **regression_metrics(y_true, y_pred),
        "coverage": float(covered.mean()),
        "ecoood_ranked_aurc": aurc,
    }


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.data)
    data = data.loc[data[DEFAULT_SCHEMA.target].notna()].reset_index(drop=True)
    rows: list[dict[str, object]] = []
    annotated: list[pd.DataFrame] = []

    for seed in args.seeds:
        split = build_split(data, "species", schema=DEFAULT_SCHEMA, seed=seed)
        train_chemicals = set(
            data.loc[split.train, DEFAULT_SCHEMA.chemical_id].astype(str)
        )
        test_chemicals = (
            data.loc[split.test, DEFAULT_SCHEMA.chemical_id]
            .astype(str)
            .reset_index(drop=True)
        )
        prediction_path = (
            args.revision_root
            / f"seed_{seed}"
            / "structured"
            / "species"
            / args.model
            / "predictions.csv"
        )
        predictions = pd.read_csv(prediction_path)
        predicted_chemicals = predictions["chemical_id"].astype(str).reset_index(drop=True)
        if not predicted_chemicals.equals(test_chemicals):
            raise ValueError(
                f"Species prediction rows do not match reconstructed split for seed {seed}."
            )
        predictions["seed"] = seed
        predictions["chemical_seen_in_training"] = predicted_chemicals.isin(
            train_chemicals
        )
        annotated.append(predictions)
        rows.append(summarize_subset(predictions, seed=seed, subset="all"))
        for label, mask in [
            ("chemical seen in training", predictions["chemical_seen_in_training"]),
            ("chemical unseen in training", ~predictions["chemical_seen_in_training"]),
        ]:
            subset = predictions.loc[mask]
            if len(subset) >= 2:
                rows.append(summarize_subset(subset, seed=seed, subset=label))

    all_rows = pd.DataFrame(rows)
    numeric = [
        column
        for column in all_rows.columns
        if column not in {"seed", "subset"}
        and pd.api.types.is_numeric_dtype(all_rows[column])
    ]
    summary = all_rows.groupby("subset", as_index=False)[numeric].agg(["mean", "std"])
    summary.columns = [
        column
        if isinstance(column, str)
        else "_".join(part for part in column if part)
        for column in summary.columns.to_flat_index()
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows.to_csv(args.output_dir / "species_chemical_overlap_all_seeds.csv", index=False)
    summary.to_csv(
        args.output_dir / "species_chemical_overlap_summary.csv",
        index=False,
    )
    pd.concat(annotated, ignore_index=True).to_csv(
        args.output_dir / "species_predictions_with_chemical_overlap.csv",
        index=False,
    )
    print(f"Wrote species chemical-overlap audit to {args.output_dir}")


if __name__ == "__main__":
    main()
