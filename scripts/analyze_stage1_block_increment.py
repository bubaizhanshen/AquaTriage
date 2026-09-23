#!/usr/bin/env python3
"""Measure the information increment of feature blocks for block-normalized kNN.

The toxicity predictions are treated as fixed. For each internal split, the
script rebuilds only the feature representation used by the support-distance
ranking. All retained blocks use the full-model training RMS scales and the
same five-block normalization, so a leave-one-block-out change is an
information comparison rather than a retuned distance metric.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ecoood.ad import _mean_knn_distance  # noqa: E402
from ecoood.features import EcoFeatureBuilder  # noqa: E402
from ecoood.schema import DEFAULT_SCHEMA  # noqa: E402
from ecoood.splits import build_split  # noqa: E402
from scripts.analyze_stage1_decision_baselines import (  # noqa: E402
    _aggregate_predictions,
    _endpoint_cutoffs,
    _metrics,
    _rank_mask,
)


BLOCK_NAMES = ("fingerprint", "descriptor", "species", "context", "bioactivity_proxy")
FEATURE_ATTRIBUTES = {"bioactivity_proxy": "mechanism"}
BLOCK_SETS = {
    "full_five_blocks": BLOCK_NAMES,
    "chemical_only": ("fingerprint", "descriptor"),
    "without_fingerprint": tuple(name for name in BLOCK_NAMES if name != "fingerprint"),
    "without_descriptor": tuple(name for name in BLOCK_NAMES if name != "descriptor"),
    "without_species": tuple(name for name in BLOCK_NAMES if name != "species"),
    "without_context": tuple(name for name in BLOCK_NAMES if name != "context"),
    "without_bioactivity": tuple(
        name for name in BLOCK_NAMES if name != "bioactivity_proxy"
    ),
    "support_without_chemical": ("species", "context", "bioactivity_proxy"),
}
PRIMARY_SPLITS = ("chemical_random", "scaffold", "temporal", "species")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 42, 43, 44])
    parser.add_argument("--review-fraction", type=float, default=0.25)
    return parser.parse_args()


def _block_matrix(bundle, names: tuple[str, ...], scales: dict[str, float]) -> sparse.csr_matrix:
    blocks = []
    for name in names:
        block = getattr(bundle, FEATURE_ATTRIBUTES.get(name, name))
        if not sparse.issparse(block):
            block = sparse.csr_matrix(np.asarray(block, dtype=np.float32))
        blocks.append(block.multiply(1.0 / (scales[name] * np.sqrt(len(BLOCK_NAMES)))))
    if not blocks:
        return sparse.csr_matrix((bundle.full.shape[0], 0), dtype=np.float32)
    return sparse.hstack(blocks, format="csr")


def _scales(train_bundle) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in BLOCK_NAMES:
        block = getattr(train_bundle, FEATURE_ATTRIBUTES.get(name, name))
        if not sparse.issparse(block):
            block = sparse.csr_matrix(np.asarray(block, dtype=np.float32))
        if block.shape[1] == 0:
            result[name] = 1.0
            continue
        squared_norm = np.asarray(block.multiply(block).sum(axis=1)).ravel()
        value = float(np.sqrt(np.mean(squared_norm)))
        result[name] = value if np.isfinite(value) and value > 1e-12 else 1.0
    return result


def _chemical_panel(
    predictions: pd.DataFrame,
    cutoffs: dict[str, float],
    score_columns: list[str],
) -> pd.DataFrame:
    frame = predictions.copy()
    frame["concern_cutoff"] = frame["endpoint"].map(cutoffs)
    endpoint = (
        frame.groupby(["chemical_id", "endpoint"], as_index=False)
        .agg(
            true_tox=("y_true", "median"),
            pred_tox=("y_pred", "median"),
            concern_cutoff=("concern_cutoff", "first"),
            **{column: (column, "median") for column in score_columns},
        )
    )
    endpoint["true_high"] = endpoint["true_tox"] <= endpoint["concern_cutoff"]
    endpoint["pred_high"] = endpoint["pred_tox"] <= endpoint["concern_cutoff"]
    endpoint["false_negative"] = endpoint["true_high"] & ~endpoint["pred_high"]
    return endpoint.groupby("chemical_id", as_index=False).agg(
        true_high=("true_high", "max"),
        pred_high=("pred_high", "max"),
        false_negative=("false_negative", "max"),
        **{column: (column, "max") for column in score_columns},
    )


def _selected_distance_scores(train_bundle, query_bundle) -> dict[str, np.ndarray]:
    scales = _scales(train_bundle)
    train_matrices = {
        name: _block_matrix(train_bundle, names, scales)
        for name, names in BLOCK_SETS.items()
    }
    query_matrices = {
        name: _block_matrix(query_bundle, names, scales)
        for name, names in BLOCK_SETS.items()
    }
    return {
        name: _mean_knn_distance(train_matrices[name], query_matrices[name], metric="euclidean")
        for name in BLOCK_SETS
    }


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.data, low_memory=False)
    predictions = pd.read_csv(args.predictions, low_memory=False)
    rows: list[dict[str, object]] = []
    distance_rows: list[pd.DataFrame] = []

    for seed in args.seeds:
        for split in PRIMARY_SPLITS:
            split_indices = build_split(data, split=split, schema=DEFAULT_SCHEMA, seed=seed)
            train = data.iloc[split_indices.train].reset_index(drop=True)
            calibration = data.iloc[split_indices.calib].reset_index(drop=True)
            test = data.iloc[split_indices.test].reset_index(drop=True)
            pred = predictions.loc[
                predictions["seed"].eq(seed) & predictions["split"].eq(split)
            ].reset_index(drop=True)
            if len(test) != len(pred):
                raise ValueError(f"Prediction/test length mismatch for seed={seed}, split={split}.")
            for left, right in (("chemical_id", "chemical_id"), ("endpoint", "endpoint")):
                if not test[left].astype(str).equals(pred[right].astype(str)):
                    raise ValueError(f"Prediction/test order mismatch for seed={seed}, split={split}, field={left}.")
            if not np.allclose(test[DEFAULT_SCHEMA.target].to_numpy(), pred["y_true"].to_numpy(), equal_nan=True):
                raise ValueError(f"Prediction/test targets mismatch for seed={seed}, split={split}.")

            builder = EcoFeatureBuilder(
                schema=DEFAULT_SCHEMA,
                include_study_year=False,
                recompute_rdkit_logp=True,
                allow_legacy_structure_placeholder=False,
            )
            train_bundle = builder.fit_transform(train)
            test_bundle = builder.transform(test)
            scores = _selected_distance_scores(train_bundle, test_bundle)
            for name, values in scores.items():
                distance_rows.append(
                    pd.DataFrame(
                        {
                            "seed": seed,
                            "split": split,
                            "row_index": np.arange(len(values)),
                            "block_set": name,
                            "distance": values,
                        }
                    )
                )

            cutoffs = _endpoint_cutoffs(calibration, DEFAULT_SCHEMA.target)
            score_columns = list(scores)
            pred_with_scores = pred.copy()
            for name, values in scores.items():
                pred_with_scores[name] = values
            panel = _chemical_panel(pred_with_scores, cutoffs, score_columns)
            for policy in ("all_queue", "predicted_low_priority_first"):
                for name in score_columns:
                    reviewed = _rank_mask(
                        panel,
                        name,
                        args.review_fraction,
                        low_only=policy == "predicted_low_priority_first",
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "split": split,
                            "policy": policy,
                            "block_set": name,
                            "review_fraction": args.review_fraction,
                            **_metrics(panel, reviewed),
                        }
                    )

    detailed = pd.DataFrame(rows)
    summary = (
        detailed.groupby(["split", "policy", "block_set"], as_index=False)
        .agg(
            n_runs=("seed", "nunique"),
            false_negative_capture_mean=("false_negative_capture", "mean"),
            false_negative_capture_sd=("false_negative_capture", "std"),
            high_concern_left_low_priority_fraction_mean=(
                "high_concern_left_low_priority_fraction", "mean"
            ),
            high_concern_left_low_priority_fraction_sd=(
                "high_concern_left_low_priority_fraction", "std"
            ),
            lower_priority_false_omission_rate_mean=(
                "lower_priority_false_omission_rate", "mean"
            ),
            lower_priority_false_omission_rate_sd=(
                "lower_priority_false_omission_rate", "std"
            ),
        )
    )
    distances = pd.concat(distance_rows, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(args.output_dir / "stage1_block_increment_detailed.csv", index=False)
    summary.to_csv(args.output_dir / "stage1_block_increment_summary.csv", index=False)
    distances.to_csv(args.output_dir / "stage1_block_increment_row_distances.csv", index=False)
    (args.output_dir / "stage1_block_increment_notes.txt").write_text(
        "Distances use the full-model training RMS scale for each of five blocks "
        "and retain the sqrt(5) normalization for every subset. Predictions are "
        "fixed; only support-distance rankings are recomputed. The current full "
        "five-block implementation is included as a consistency reference.\n"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
