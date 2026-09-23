#!/usr/bin/env python3
"""Quantify block-normalized support-distance increments on fixed predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecoood.schema import DEFAULT_SCHEMA  # noqa: E402
from ecoood.splits import build_split  # noqa: E402
from scripts.analyze_stage1_decision_baselines import (  # noqa: E402
    _endpoint_cutoffs,
    _load_cutoff_table,
)
from scripts.analyze_stage4_paired_uncertainty import (  # noqa: E402
    _evaluate_arrays,
)


PRIMARY_SPLITS = ("chemical_random", "scaffold", "temporal", "species")
BLOCK_SETS = (
    "full_five_blocks",
    "chemical_only",
    "without_fingerprint",
    "without_descriptor",
    "without_species",
    "without_context",
    "without_bioactivity",
    "support_without_chemical",
)
COMPARISONS = (
    ("full_five_blocks", "chemical_only"),
    ("full_five_blocks", "without_fingerprint"),
    ("full_five_blocks", "without_descriptor"),
    ("full_five_blocks", "without_species"),
    ("full_five_blocks", "without_context"),
    ("full_five_blocks", "without_bioactivity"),
    ("full_five_blocks", "support_without_chemical"),
    ("full_five_blocks", "prediction_threshold_proximity"),
    ("full_five_blocks", "random_review"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--row-distances", type=Path, required=True)
    parser.add_argument("--internal-cutoffs", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 42, 43, 44])
    parser.add_argument("--review-fraction", type=float, default=0.25)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=20260920)
    return parser.parse_args()


def _chemical_panel(
    pred: pd.DataFrame,
    distances: pd.DataFrame,
    cutoffs: dict[str, float],
) -> pd.DataFrame:
    frame = pred.reset_index(drop=True).copy()
    frame["row_index"] = np.arange(len(frame))
    distance_wide = distances.pivot(index="row_index", columns="block_set", values="distance")
    distance_wide = distance_wide.reindex(frame.row_index)
    frame = frame.join(distance_wide.reset_index(drop=True))
    frame["concern_cutoff"] = frame["endpoint"].map(cutoffs)
    if frame["concern_cutoff"].isna().any():
        raise ValueError("Some test endpoints have no calibration concern cutoff.")
    endpoint = (
        frame.groupby(["chemical_id", "endpoint"], as_index=False)
        .agg(
            pred_tox=("y_pred", "median"),
            true_tox=("y_true", "median"),
            concern_cutoff=("concern_cutoff", "first"),
            **{name: (name, "median") for name in BLOCK_SETS},
        )
    )
    endpoint["true_high"] = endpoint.true_tox <= endpoint.concern_cutoff
    endpoint["pred_high"] = endpoint.pred_tox <= endpoint.concern_cutoff
    endpoint["prediction_threshold_proximity"] = (
        -(endpoint.pred_tox - endpoint.concern_cutoff).abs()
    )
    chemical = endpoint.groupby("chemical_id", as_index=False).agg(
        true_high=("true_high", "max"),
        pred_high=("pred_high", "max"),
        **{name: (name, "max") for name in BLOCK_SETS},
        prediction_threshold_proximity=("prediction_threshold_proximity", "max"),
    )
    return chemical


def _arrays(panel: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    score_columns = {
        name: name for name in BLOCK_SETS if name in panel.columns
    }
    score_columns["prediction_threshold_proximity"] = "prediction_threshold_proximity"
    arrays = {
        "chemical_id": panel.chemical_id.astype(str).to_numpy(),
        "true_high": panel.true_high.to_numpy(dtype=bool),
        "pred_high": panel.pred_high.to_numpy(dtype=bool),
        **{
            column: pd.to_numeric(panel[column], errors="coerce").to_numpy(dtype=float)
            for column in score_columns.values()
        },
    }
    return arrays, score_columns


def _comparison_rows(
    evaluated: dict[str, dict[str, float | int]],
    *,
    dataset: str,
    split: str,
    seed: int,
    bootstrap_id: int | None,
) -> list[dict[str, object]]:
    rows = []
    for method, reference in COMPARISONS:
        if method not in evaluated or reference not in evaluated:
            continue
        for metric in (
            "false_negative_capture",
            "high_concern_left_low_priority_fraction",
            "lower_priority_false_omission_rate",
        ):
            value = float(evaluated[method][metric])
            reference_value = float(evaluated[reference][metric])
            difference = value - reference_value if np.isfinite(value) and np.isfinite(reference_value) else np.nan
            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "seed": seed,
                    "bootstrap_id": bootstrap_id,
                    "method": method,
                    "reference": reference,
                    "metric": metric,
                    "difference": difference,
                }
            )
    return rows


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(["method", "reference", "metric"], dropna=False):
        values = group.difference.dropna().to_numpy(dtype=float)
        if not len(values):
            continue
        rows.append(
            {
                "method": keys[0],
                "reference": keys[1],
                "metric": keys[2],
                "n_bootstrap_values": len(values),
                "mean_difference": np.mean(values),
                "median_difference": np.median(values),
                "ci_lower_95": np.quantile(values, 0.025),
                "ci_upper_95": np.quantile(values, 0.975),
                "probability_capture_difference_gt_zero": (
                    np.mean(values > 0) if keys[2] == "false_negative_capture" else np.nan
                ),
                "probability_omission_difference_lt_zero": (
                    np.mean(values < 0) if keys[2] != "false_negative_capture" else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.bootstrap_reps < 100:
        raise ValueError("bootstrap-reps must be at least 100.")
    data = pd.read_csv(args.data, low_memory=False)
    predictions = pd.read_csv(args.predictions, low_memory=False)
    row_distances = pd.read_csv(args.row_distances, low_memory=False)
    cutoff_table = _load_cutoff_table(args.internal_cutoffs) if args.internal_cutoffs else None
    exact_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(args.random_seed)

    for seed in args.seeds:
        for split in PRIMARY_SPLITS:
            if cutoff_table is None:
                split_indices = build_split(data, split=split, schema=DEFAULT_SCHEMA, seed=seed)
                calibration = data.iloc[split_indices.calib]
                cutoffs = _endpoint_cutoffs(calibration, DEFAULT_SCHEMA.target)
            else:
                try:
                    cutoffs = cutoff_table[(seed, split)]
                except KeyError as exc:
                    raise ValueError(f"No locked cutoffs for seed={seed}, split={split}.") from exc
            pred = predictions.loc[
                predictions.seed.eq(seed) & predictions.split.eq(split)
            ].reset_index(drop=True)
            distances = row_distances.loc[
                row_distances.seed.eq(seed) & row_distances.split.eq(split)
            ].copy()
            panel = _chemical_panel(pred, distances, cutoffs)
            arrays, score_columns = _arrays(panel)
            evaluated = _evaluate_arrays(
                arrays, score_columns, args.review_fraction, low_only=True
            )
            exact_rows.extend(
                _comparison_rows(
                    evaluated,
                    dataset="internal",
                    split=split,
                    seed=seed,
                    bootstrap_id=None,
                )
            )
            n = len(panel)
            for bootstrap_id in range(args.bootstrap_reps):
                sampled_index = rng.integers(0, n, size=n)
                sampled = {name: values[sampled_index] for name, values in arrays.items()}
                evaluated_sample = _evaluate_arrays(
                    sampled, score_columns, args.review_fraction, low_only=True
                )
                bootstrap_rows.extend(
                    _comparison_rows(
                        evaluated_sample,
                        dataset="internal",
                        split=split,
                        seed=seed,
                        bootstrap_id=bootstrap_id,
                    )
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(exact_rows).to_csv(
        args.output_dir / "stage6_support_increment_exact.csv", index=False
    )
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(args.output_dir / "stage6_support_increment_bootstrap.csv", index=False)
    _summary(bootstrap).to_csv(
        args.output_dir / "stage6_support_increment_summary.csv", index=False
    )
    (args.output_dir / "stage6_notes.txt").write_text(
        "Fixed-prediction chemical-level paired bootstrap across the four primary internal shifts. "
        "The row-level distances were generated with the current strict feature configuration and "
        "the full-model training RMS scales. All comparisons use a 25% predicted-low-priority-first "
        "policy. Positive differences favor the first method; negative omission differences favor it. "
        "The analysis tests ranking increment, not predictive accuracy or causal importance of any input block.\n"
    )
    print(_summary(bootstrap).to_string(index=False))


if __name__ == "__main__":
    main()
