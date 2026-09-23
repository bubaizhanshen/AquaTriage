#!/usr/bin/env python3
"""Summarize external fixed-workload results by source and endpoint.

This is a descriptive stratification of the locked expanded external panel. It
does not tune a method or select strata after seeing performance; every source
and endpoint is reported with its chemical count and event count.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_stage1_decision_baselines import (  # noqa: E402
    _endpoint_cutoffs,
    _metrics,
    _rank_mask,
)
from scripts.run_echa_pmra_external_validation import (  # noqa: E402
    calibration_split_by_chemical,
)
from ecoood.schema import DEFAULT_SCHEMA  # noqa: E402


METHODS = {
    "prediction_threshold_proximity": "prediction_distance",
    "block_normalized_knn": "block_normalized_knn",
    "similarity_ad": "similarity_ad",
    "ensemble_sd_risk": "ensemble_sd_risk",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-train", type=Path, required=True)
    parser.add_argument("--external-panel", type=Path, required=True)
    parser.add_argument("--external-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review-fraction", type=float, default=0.25)
    parser.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 42, 43, 44])
    return parser.parse_args()


def _chemical_panel(predictions: pd.DataFrame, cutoffs: dict[str, float]) -> pd.DataFrame:
    frame = predictions.copy()
    frame["source_group"] = frame["case_id"].astype(str).str.split("__", n=1).str[0]
    frame["concern_cutoff"] = frame["endpoint"].map(cutoffs)
    endpoint = (
        frame.groupby(["chemical_id", "endpoint"], as_index=False, dropna=False)
        .agg(
            true_tox=("target_log_molar", "median"),
            pred_tox=("y_pred", "median"),
            concern_cutoff=("concern_cutoff", "first"),
            source_group=("source_group", "first"),
            case_row_count=("case_row_count", "median"),
            document_count=("document_count", "median"),
            case_spread_log_molar=("case_spread_log_molar", "median"),
            standard_species_flag=("standard_species_flag", "max"),
            freshwater_keyword_flag=("freshwater_keyword_flag", "max"),
            endpoint_cases=("case_id", "nunique"),
            **{
                name: (column, "median")
                for name, column in {
                    "block_normalized_knn": "ad_equal_block_distance",
                    "similarity_ad": "ad_similarity_risk",
                    "ensemble_sd_risk": "ensemble_sd_risk",
                }.items()
            },
        )
    )
    endpoint["true_high"] = endpoint["true_tox"] <= endpoint["concern_cutoff"]
    endpoint["pred_high"] = endpoint["pred_tox"] <= endpoint["concern_cutoff"]
    endpoint["false_negative"] = endpoint["true_high"] & ~endpoint["pred_high"]
    endpoint["prediction_distance"] = -(endpoint["pred_tox"] - endpoint["concern_cutoff"]).abs()
    chemical = (
        endpoint.groupby(["chemical_id", "source_group"], as_index=False, dropna=False)
        .agg(
            true_high=("true_high", "max"),
            pred_high=("pred_high", "max"),
            false_negative=("false_negative", "max"),
            n_endpoints=("endpoint", "nunique"),
            **{
                column: (column, "max")
                for column in [
                    "prediction_distance",
                    "block_normalized_knn",
                    "similarity_ad",
                    "ensemble_sd_risk",
                    "case_row_count",
                    "document_count",
                    "case_spread_log_molar",
                    "standard_species_flag",
                    "freshwater_keyword_flag",
                ]
            },
        )
    )
    return chemical, endpoint


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.external_train, low_memory=False)
    panel = pd.read_csv(args.external_panel, low_memory=False)
    predictions = pd.read_csv(args.external_predictions, low_memory=False)
    rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    for seed in args.seeds:
        _, calibration = calibration_split_by_chemical(train, seed=seed)
        cutoffs = _endpoint_cutoffs(calibration, DEFAULT_SCHEMA.target)
        subset = predictions.loc[predictions["seed"].eq(seed)].copy()
        chemical, endpoint = _chemical_panel(subset, cutoffs)
        # Chemical-level metrics by source. The global review budget is applied
        # within each stratum only for descriptive comparison, with n reported.
        for source, group in chemical.groupby("source_group", dropna=False):
            group = group.reset_index(drop=True)
            for method, score in METHODS.items():
                reviewed = _rank_mask(group, score, args.review_fraction, low_only=True)
                metrics = _metrics(group, reviewed)
                rows.append({
                    "seed": seed,
                    "stratum": "source",
                    "stratum_value": source,
                    "method": method,
                    **metrics,
                })
        # Endpoint composition and event counts are reported without applying
        # a second within-endpoint review ranking.
        for endpoint_name, group in endpoint.groupby("endpoint", dropna=False):
            event_rows.append({
                "seed": seed,
                "stratum": "endpoint",
                "stratum_value": endpoint_name,
                "n_endpoint_chemical_pairs": int(len(group)),
                "n_chemicals": int(group.chemical_id.nunique()),
                "n_true_high_endpoint_pairs": int(group.true_high.sum()),
                "n_false_negative_endpoint_pairs": int(group.false_negative.sum()),
                "mean_case_row_count": float(group.case_row_count.mean()),
                "mean_document_count": float(group.document_count.mean()),
                "mean_case_spread_log_molar": float(group.case_spread_log_molar.mean()),
            })
        # Add panel-level source/data-quality descriptors once per seed.
        for source, group in chemical.groupby("source_group", dropna=False):
            rows.append({
                "seed": seed,
                "stratum": "source_descriptor",
                "stratum_value": source,
                "method": "descriptor",
                "n_chemicals": int(len(group)),
                "n_true_high": int(group.true_high.sum()),
                "n_false_negative": int(group.false_negative.sum()),
                "mean_case_row_count": float(group.case_row_count.mean()),
                "mean_document_count": float(group.document_count.mean()),
                "mean_case_spread_log_molar": float(group.case_spread_log_molar.mean()),
                "standard_species_fraction": float(group.standard_species_flag.mean()),
                "freshwater_keyword_fraction": float(group.freshwater_keyword_flag.mean()),
            })
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "external_strata_metrics_all.csv", index=False)
    pd.DataFrame(event_rows).to_csv(out / "external_endpoint_composition_all.csv", index=False)
    metrics = pd.DataFrame(rows).query("stratum == 'source'")
    metrics.groupby(["stratum_value", "method"], as_index=False).mean(numeric_only=True).to_csv(
        out / "external_strata_metrics_summary.csv", index=False
    )
    descriptors = pd.DataFrame(rows).query("stratum == 'source_descriptor'")
    descriptors.groupby(["stratum_value", "method"], as_index=False).mean(numeric_only=True).to_csv(
        out / "external_source_descriptors_summary.csv", index=False
    )
    pd.DataFrame(event_rows).groupby(["stratum_value"], as_index=False).mean(numeric_only=True).to_csv(
        out / "external_endpoint_composition_summary.csv", index=False
    )
    (out / "external_strata_notes.txt").write_text(
        "Source-stratified metrics are descriptive and use a 25% review fraction within each source stratum; "
        "the source-specific denominator and chemical count are reported. They are not independent external "
        "validations and should not be used to claim superiority for a source with few chemicals. Endpoint "
        "tables report composition and event counts; they do not retune thresholds or ranking rules.\n"
    )


if __name__ == "__main__":
    main()
