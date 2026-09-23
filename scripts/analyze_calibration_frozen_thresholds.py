#!/usr/bin/env python3
"""Evaluate fixed-workload screening with calibration-derived concern cutoffs."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from ecoood.schema import DEFAULT_SCHEMA
from ecoood.splits import build_split


DEFAULT_SEEDS = (40, 41, 42, 43, 44)
DEFAULT_SPLITS = (
    "chemical_random",
    "scaffold",
    "temporal",
    "species",
    "chemical_class",
)
METHODS = {
    # The release prediction files use ``ecoood_score`` as the stable data
    # column name.  The manuscript calls this candidate measure the
    # prediction-error risk score; keep the display name separate from the
    # storage alias so older exports remain readable.
    "Prediction-error risk score": "ecoood_score",
    "Ensemble-SD risk": "ensemble_sd_risk",
    "Block-normalized kNN + SD risk": "equal_block_knn_plus_sd_risk",
    "Block-normalized kNN distance": "ad_equal_block_distance",
    "Similarity AD": "ad_similarity",
}
IDENTIFIERS = ("chemical_id", "chemical_name", "casrn", "chemical_class")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--model", default="lightgbm")
    parser.add_argument("--review-fraction", type=float, default=0.25)
    parser.add_argument("--random-replicates", type=int, default=500)
    return parser.parse_args()


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


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
            for column in METHODS.values()
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
            for column in METHODS.values()
        },
    )


def false_omission_rate(
    panel: pd.DataFrame,
    reviewed: np.ndarray,
) -> float:
    predicted_high = panel["pred_high"].to_numpy(dtype=bool)
    lower_priority = ~predicted_high & ~reviewed
    if not lower_priority.any():
        return float("nan")
    true_high = panel["true_high"].to_numpy(dtype=bool)
    return float(np.mean(true_high[lower_priority]))


def ranked_for(
    panel: pd.DataFrame,
    score_column: str,
    review_fraction: float,
) -> float:
    panel = panel.reset_index(drop=True)
    review_n = max(1, int(round(review_fraction * len(panel))))
    order = panel.sort_values(
        [score_column, "chemical_id"],
        ascending=[False, True],
        kind="mergesort",
    ).index.to_numpy()
    reviewed = np.zeros(len(panel), dtype=bool)
    reviewed[order[:review_n]] = True
    return false_omission_rate(panel, reviewed)


def random_for(
    panel: pd.DataFrame,
    *,
    review_fraction: float,
    replicates: int,
    seed: int,
) -> float:
    panel = panel.reset_index(drop=True)
    review_n = max(1, int(round(review_fraction * len(panel))))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        reviewed = np.zeros(len(panel), dtype=bool)
        reviewed[rng.choice(len(panel), size=review_n, replace=False)] = True
        values.append(false_omission_rate(panel, reviewed))
    return float(np.nanmean(values))


def main() -> None:
    args = parse_args()
    if not 0 < args.review_fraction < 1:
        raise ValueError("--review-fraction must be between 0 and 1.")
    source = pd.read_csv(args.data)
    rows: list[dict[str, object]] = []
    cutoff_rows: list[dict[str, object]] = []

    for split_name in args.splits:
        for seed in args.seeds:
            split = build_split(source, split_name, schema=DEFAULT_SCHEMA, seed=seed)
            cutoffs = calibration_cutoffs(source.loc[split.calib])
            cutoff_rows.extend(
                {
                    "split": split_name,
                    "seed": seed,
                    "endpoint": endpoint,
                    "cutoff": value,
                }
                for endpoint, value in cutoffs.items()
            )
            prediction_path = (
                args.prediction_root
                / f"seed_{seed}"
                / "structured"
                / split_name
                / args.model
                / "predictions.csv"
            )
            predictions = pd.read_csv(prediction_path)
            expected = source.loc[split.test, DEFAULT_SCHEMA.target].to_numpy(dtype=float)
            observed = predictions["y_true"].to_numpy(dtype=float)
            if len(expected) != len(observed) or not np.allclose(
                expected, observed, rtol=0.0, atol=1e-12
            ):
                raise ValueError(
                    f"Prediction rows do not match the frozen {split_name} split. "
                    "Use the release split assignments or the reference RDKit 2024.03.2 environment."
                )
            panel = chemical_panel(predictions, cutoffs)

            for method, column in METHODS.items():
                rows.append(
                    {
                        "split": split_name,
                        "seed": seed,
                        "method": method,
                        "n_chemicals": len(panel),
                        "lower_priority_false_omission_rate": ranked_for(
                            panel,
                            column,
                            args.review_fraction,
                        ),
                    }
                )
            rows.append(
                {
                    "split": split_name,
                    "seed": seed,
                    "method": "Random review",
                    "n_chemicals": len(panel),
                    "lower_priority_false_omission_rate": random_for(
                        panel,
                        review_fraction=args.review_fraction,
                        replicates=args.random_replicates,
                        seed=stable_seed(
                            seed,
                            split_name,
                            "calibration_frozen_random",
                        ),
                    ),
                }
            )

    detailed = pd.DataFrame(rows)
    summary = detailed.groupby(["split", "method"], as_index=False).agg(
        n_instances=("seed", "size"),
        n_chemicals_mean=("n_chemicals", "mean"),
        for_mean=("lower_priority_false_omission_rate", "mean"),
        for_sd=("lower_priority_false_omission_rate", "std"),
        n_defined=("lower_priority_false_omission_rate", "count"),
    )
    four_shifts = detailed[detailed["split"].isin(DEFAULT_SPLITS[:4])]
    per_seed = four_shifts.groupby(["seed", "method"], as_index=False)[
        "lower_priority_false_omission_rate"
    ].mean()
    macro = per_seed.groupby("method", as_index=False).agg(
        four_shift_for_mean=("lower_priority_false_omission_rate", "mean"),
        four_shift_for_sd=("lower_priority_false_omission_rate", "std"),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(args.output_dir / "calibration_cutoff_detailed.csv", index=False)
    summary.to_csv(args.output_dir / "calibration_cutoff_summary.csv", index=False)
    macro.to_csv(args.output_dir / "calibration_cutoff_four_shift.csv", index=False)
    pd.DataFrame(cutoff_rows).to_csv(
        args.output_dir / "calibration_cutoffs.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
