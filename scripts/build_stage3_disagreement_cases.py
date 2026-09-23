#!/usr/bin/env python3
"""Build fixed-policy disagreement tables for kNN and similarity AD."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ecoood.schema import DEFAULT_SCHEMA  # noqa: E402
from ecoood.splits import build_split  # noqa: E402
from scripts.analyze_stage1_decision_baselines import (  # noqa: E402
    _endpoint_cutoffs,
    _load_cutoff_table,
    _rank_mask,
)
from scripts.run_echa_pmra_external_validation import (  # noqa: E402
    calibration_split_by_chemical,
)


SUPPORT_COLUMNS = {
    "d_chem": "d_chem",
    "d_species": "d_species",
    "d_context": "d_context",
    "d_mech": "d_mech",
    "context_missing_fraction": "context_missing_fraction",
    "bioactivity_missing_fraction": "bioactivity_missing_fraction",
    "model_std": "model_std",
    "n_cases": "case_id",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal-data", type=Path, required=True)
    parser.add_argument("--internal-predictions", type=Path, required=True)
    parser.add_argument("--external-train", type=Path, required=True)
    parser.add_argument("--external-predictions", type=Path, required=True)
    parser.add_argument("--internal-cutoffs", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 42, 43, 44])
    parser.add_argument("--review-fraction", type=float, default=0.25)
    return parser.parse_args()


def _panel(frame: pd.DataFrame, cutoffs: dict[str, float], *, true: str, pred: str) -> pd.DataFrame:
    frame = frame.copy()
    if "ad_similarity_risk" not in frame and "ad_similarity" in frame:
        frame["ad_similarity_risk"] = frame["ad_similarity"]
    frame["concern_cutoff"] = frame["endpoint"].map(cutoffs)
    frame["source_group"] = (
        frame["case_id"].astype(str).str.split("__", n=1).str[0]
        if "case_id" in frame else "internal"
    )
    aggs: dict[str, tuple[str, str]] = {
        "true_tox": (true, "median"),
        "pred_tox": (pred, "median"),
        "concern_cutoff": ("concern_cutoff", "first"),
        "n_cases": ("case_id", "nunique") if "case_id" in frame else ("chemical_id", "size"),
    }
    for label, column in [("knn", "ad_equal_block_distance"), ("similarity", "ad_similarity_risk")]:
        if column in frame:
            aggs[label] = (column, "median")
    for label, column in SUPPORT_COLUMNS.items():
        if column in frame and label != "n_cases":
            aggs[label] = (column, "median")
    endpoint = frame.groupby(["chemical_id", "endpoint"], as_index=False).agg(**aggs)
    endpoint["true_high"] = endpoint.true_tox <= endpoint.concern_cutoff
    endpoint["pred_high"] = endpoint.pred_tox <= endpoint.concern_cutoff
    endpoint["false_negative"] = endpoint.true_high & ~endpoint.pred_high
    endpoint["prediction_distance"] = -(endpoint.pred_tox - endpoint.concern_cutoff).abs()
    endpoint["knn"] = endpoint["knn"] if "knn" in endpoint else np.nan
    endpoint["similarity"] = endpoint["similarity"] if "similarity" in endpoint else np.nan
    chemical = endpoint.groupby("chemical_id", as_index=False).agg(
        true_high=("true_high", "max"),
        pred_high=("pred_high", "max"),
        false_negative=("false_negative", "max"),
        n_endpoints=("endpoint", "nunique"),
        **{column: (column, "max") for column in [
            "knn", "similarity", "prediction_distance", "d_chem", "d_species",
            "d_context", "d_mech", "context_missing_fraction",
            "bioactivity_missing_fraction", "model_std", "n_cases",
        ] if column in endpoint},
    )
    source = frame.groupby("chemical_id", as_index=False).agg(source_group=("source_group", "first"))
    return chemical.merge(source, on="chemical_id", how="left")


def _append_dataset_rows(
    rows: list[dict[str, object]],
    frame: pd.DataFrame,
    *,
    dataset: str,
    seed: int,
    split: str,
    cutoffs: dict[str, float],
    true: str,
    pred: str,
    review_fraction: float,
) -> None:
    panel = _panel(frame, cutoffs, true=true, pred=pred)
    for method in ("knn", "similarity"):
        reviewed = _rank_mask(
            panel,
            method,
            review_fraction,
            low_only=True,
        )
        panel[f"reviewed_{method}"] = reviewed
    baseline_fn = panel["false_negative"].astype(bool)
    categories = np.full(len(panel), "not_baseline_false_negative", dtype=object)
    categories[baseline_fn & panel.reviewed_knn & panel.reviewed_similarity] = "both_rescued"
    categories[baseline_fn & panel.reviewed_knn & ~panel.reviewed_similarity] = "knn_only_rescued"
    categories[baseline_fn & ~panel.reviewed_knn & panel.reviewed_similarity] = "similarity_only_rescued"
    categories[baseline_fn & ~panel.reviewed_knn & ~panel.reviewed_similarity] = "both_omitted"
    panel["disagreement_category"] = categories
    panel.insert(0, "dataset", dataset)
    panel.insert(1, "seed", seed)
    panel.insert(2, "split", split)
    rows.extend(panel.to_dict("records"))


def main() -> None:
    args = parse_args()
    internal_data = pd.read_csv(args.internal_data, low_memory=False)
    internal_predictions = pd.read_csv(args.internal_predictions, low_memory=False)
    external_train = pd.read_csv(args.external_train, low_memory=False)
    external_predictions = pd.read_csv(args.external_predictions, low_memory=False)
    cutoff_table = _load_cutoff_table(args.internal_cutoffs) if args.internal_cutoffs else None
    rows: list[dict[str, object]] = []

    for seed in args.seeds:
        for split in sorted(internal_predictions.loc[internal_predictions.seed.eq(seed), "split"].unique()):
            if cutoff_table is None:
                split_indices = build_split(internal_data, split=split, schema=DEFAULT_SCHEMA, seed=seed)
                calibration = internal_data.iloc[split_indices.calib]
                cutoffs = _endpoint_cutoffs(calibration, DEFAULT_SCHEMA.target)
            else:
                try:
                    cutoffs = cutoff_table[(seed, split)]
                except KeyError as exc:
                    raise ValueError(f"No locked cutoffs for seed={seed}, split={split}.") from exc
            frame = internal_predictions.loc[
                internal_predictions.seed.eq(seed) & internal_predictions.split.eq(split)
            ].copy()
            _append_dataset_rows(
                rows,
                frame,
                dataset="internal",
                seed=seed,
                split=split,
                cutoffs=cutoffs,
                true="y_true",
                pred="y_pred",
                review_fraction=args.review_fraction,
            )
        _, calibration = calibration_split_by_chemical(external_train, seed=seed)
        cutoffs = _endpoint_cutoffs(calibration, DEFAULT_SCHEMA.target)
        frame = external_predictions.loc[external_predictions.seed.eq(seed)].copy()
        _append_dataset_rows(
            rows,
            frame,
            dataset="external",
            seed=seed,
            split="expanded",
            cutoffs=cutoffs,
            true="target_log_molar",
            pred="y_pred",
            review_fraction=args.review_fraction,
        )

    detailed = pd.DataFrame(rows)
    aggregation = {
        "n_rows": ("chemical_id", "size"),
        "n_unique_chemicals": ("chemical_id", "nunique"),
        "mean_n_endpoints": ("n_endpoints", "mean"),
        "mean_knn": ("knn", "mean"),
        "mean_similarity": ("similarity", "mean"),
        "mean_d_chem": ("d_chem", "mean"),
        "mean_d_species": ("d_species", "mean"),
        "mean_d_context": ("d_context", "mean"),
        "mean_d_mech": ("d_mech", "mean"),
        "mean_context_missing": ("context_missing_fraction", "mean"),
        "mean_bioactivity_missing": ("bioactivity_missing_fraction", "mean"),
        "mean_model_std": ("model_std", "mean"),
    }
    # Older frozen prediction exports did not persist the two missingness
    # diagnostics.  Keep the disagreement analysis usable while retaining the
    # diagnostics whenever the export contains them.
    aggregation = {
        name: spec for name, spec in aggregation.items() if spec[0] in detailed.columns
    }
    summary = detailed.groupby(
        ["dataset", "split", "disagreement_category"], as_index=False
    ).agg(**aggregation)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(args.output_dir / "stage3_disagreement_cases_detailed.csv", index=False)
    summary.to_csv(args.output_dir / "stage3_disagreement_cases_summary.csv", index=False)
    (args.output_dir / "stage3_notes.txt").write_text(
        "Categories are defined among baseline false-negative chemicals under a "
        "predicted-low-priority-first policy at the requested review fraction. "
        "The support columns are descriptive associations, not mechanistic claims.\n"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
