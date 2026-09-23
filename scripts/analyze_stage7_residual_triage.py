#!/usr/bin/env python3
"""Test whether reliability signals add value after threshold-first review.

The first-stage queue is fixed by prediction proximity to the calibration-derived
endpoint concern cutoff. A second, equal-sized review queue is then ranked using
each reliability signal on the remaining predicted-low-priority chemicals.
This is a conditional incremental analysis, not a replacement for the primary
fixed-workload comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_stage1_decision_baselines import (  # noqa: E402
    _external_panels,
    _internal_panels,
    _load_cutoff_table,
    _metrics,
    _rank_mask,
)


PRIMARY_SPLITS = ("chemical_random", "scaffold", "temporal", "species")
METHODS = {
    "block_normalized_knn": "block_normalized_knn",
    "similarity_ad": "similarity_ad",
    "prediction_error_risk": "prediction_error_risk",
    "ensemble_sd_risk": "ensemble_sd_risk",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal-data", type=Path, required=True)
    parser.add_argument("--internal-predictions", type=Path, required=True)
    parser.add_argument("--external-train", type=Path, required=True)
    parser.add_argument("--external-predictions", type=Path, required=True)
    parser.add_argument("--internal-cutoffs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 42, 43, 44])
    parser.add_argument("--review-fraction", type=float, default=0.25)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=20260920)
    return parser.parse_args()


def _expected_random(residual: pd.DataFrame, additional_n: int) -> dict[str, float | int]:
    n = len(residual)
    false_negative = residual["false_negative"].astype(bool)
    p = additional_n / n if n else np.nan
    n_fn = int(false_negative.sum())
    rescued = n_fn * p if np.isfinite(p) else np.nan
    return {
        "n_residual_chemicals": n,
        "additional_review_n": additional_n,
        "residual_false_negative_n": n_fn,
        "residual_rescued_false_negative": rescued,
        "residual_false_negative_capture": p if n_fn else np.nan,
        "residual_high_concern_left_low_priority_fraction": 1.0 - p if n_fn else np.nan,
    }


def _evaluate_panel(panel: pd.DataFrame, *, dataset: str, split: str, seed: int, fraction: float) -> list[dict[str, object]]:
    panel = panel.reset_index(drop=True).copy()
    base_review = _rank_mask(panel, "prediction_distance", fraction, low_only=True)
    predicted_low = ~panel["pred_high"].astype(bool)
    residual = panel.loc[predicted_low & ~base_review].reset_index(drop=True)
    base_n = int(base_review.sum())
    additional_n = min(base_n, len(residual))
    additional_fraction = additional_n / len(residual) if len(residual) else 0.0
    rows: list[dict[str, object]] = []
    random_values = _expected_random(residual, additional_n)
    rows.append({
        "dataset": dataset,
        "split": split,
        "seed": seed,
        "method": "random_second_stage",
        "initial_review_n": base_n,
        "initial_review_fraction": fraction,
        "additional_review_fraction_of_residual": additional_fraction,
        **random_values,
    })
    for method, score in METHODS.items():
        if score not in residual.columns:
            continue
        reviewed = _rank_mask(residual, score, additional_fraction, low_only=False)
        fn = residual["false_negative"].astype(bool).to_numpy()
        rescued = int((fn & reviewed).sum())
        rows.append({
            "dataset": dataset,
            "split": split,
            "seed": seed,
            "method": method,
            "initial_review_n": base_n,
            "initial_review_fraction": fraction,
            "additional_review_fraction_of_residual": additional_fraction,
            "n_residual_chemicals": int(len(residual)),
            "additional_review_n": int(reviewed.sum()),
            "residual_false_negative_n": int(fn.sum()),
            "residual_rescued_false_negative": rescued,
            "residual_false_negative_capture": rescued / int(fn.sum()) if fn.any() else np.nan,
            "residual_high_concern_left_low_priority_fraction": (int(fn.sum()) - rescued) / int(fn.sum()) if fn.any() else np.nan,
        })
    return rows


def main() -> None:
    args = parse_args()
    internal_data = pd.read_csv(args.internal_data, low_memory=False)
    internal_predictions = pd.read_csv(args.internal_predictions, low_memory=False)
    external_train = pd.read_csv(args.external_train, low_memory=False)
    external_predictions = pd.read_csv(args.external_predictions, low_memory=False)
    cutoff_table = _load_cutoff_table(args.internal_cutoffs)
    panels = [
        (dataset, seed, split, panel)
        for dataset, seed, split, panel in _internal_panels(
            internal_data, internal_predictions, args.seeds, cutoff_table
        )
        if split in PRIMARY_SPLITS
    ]
    panels.extend(_external_panels(external_train, external_predictions, args.seeds))
    rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(args.random_seed)
    for dataset, seed, split, panel in panels:
        rows.extend(_evaluate_panel(panel, dataset=dataset, split=split, seed=seed, fraction=args.review_fraction))
        for bootstrap_id in range(args.bootstrap_reps):
            sampled_index = rng.integers(0, len(panel), size=len(panel))
            sampled = panel.iloc[sampled_index].reset_index(drop=True)
            for result in _evaluate_panel(sampled, dataset=dataset, split=split, seed=seed, fraction=args.review_fraction):
                result["bootstrap_id"] = bootstrap_id
                bootstrap_rows.append(result)
    detailed = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(args.output_dir / "stage7_residual_triage_detailed.csv", index=False)
    summary = (
        detailed.groupby(["dataset", "split", "method"], as_index=False)
        .agg(
            n_panels=("seed", "nunique"),
            initial_review_n_mean=("initial_review_n", "mean"),
            n_residual_chemicals_mean=("n_residual_chemicals", "mean"),
            additional_review_n_mean=("additional_review_n", "mean"),
            residual_false_negative_n_mean=("residual_false_negative_n", "mean"),
            residual_rescued_false_negative_mean=("residual_rescued_false_negative", "mean"),
            residual_false_negative_capture_mean=("residual_false_negative_capture", "mean"),
            residual_false_negative_capture_sd=("residual_false_negative_capture", "std"),
            residual_high_concern_left_low_priority_fraction_mean=("residual_high_concern_left_low_priority_fraction", "mean"),
        )
    )
    summary.to_csv(args.output_dir / "stage7_residual_triage_summary.csv", index=False)
    main = detailed.loc[detailed.dataset.eq("internal")].copy()
    main["summary_group"] = "internal_main_four_shifts"
    ext = detailed.loc[detailed.dataset.eq("external")].copy()
    ext["summary_group"] = "external_expanded"
    grouped = pd.concat([main, ext], ignore_index=True)
    grouped_summary = (
        grouped.groupby(["summary_group", "method"], as_index=False)
        .agg(
            n_panels=("seed", "count"),
            residual_false_negative_capture_mean=("residual_false_negative_capture", "mean"),
            residual_false_negative_capture_sd=("residual_false_negative_capture", "std"),
            residual_high_concern_left_low_priority_fraction_mean=("residual_high_concern_left_low_priority_fraction", "mean"),
        )
    )
    grouped_summary.to_csv(args.output_dir / "stage7_residual_triage_group_summary.csv", index=False)
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(args.output_dir / "stage7_residual_triage_bootstrap.csv", index=False)
    bootstrap_main = bootstrap.loc[bootstrap.dataset.eq("internal")].copy()
    bootstrap_main["summary_group"] = "internal_main_four_shifts"
    bootstrap_ext = bootstrap.loc[bootstrap.dataset.eq("external")].copy()
    bootstrap_ext["summary_group"] = "external_expanded"
    bootstrap_grouped = pd.concat([bootstrap_main, bootstrap_ext], ignore_index=True)
    random = bootstrap_grouped.loc[bootstrap_grouped.method.eq("random_second_stage"), ["summary_group", "dataset", "split", "seed", "bootstrap_id", "residual_false_negative_capture"]].rename(
        columns={"residual_false_negative_capture": "random_capture"}
    )
    comparison = bootstrap_grouped.merge(random, on=["summary_group", "dataset", "split", "seed", "bootstrap_id"], how="left")
    comparison = comparison.loc[comparison.method.ne("random_second_stage")].copy()
    comparison["capture_difference_vs_random"] = comparison["residual_false_negative_capture"] - comparison["random_capture"]
    comparison_summary = (
        comparison.groupby(["summary_group", "method"], as_index=False)
        .agg(
            n_bootstrap_values=("capture_difference_vs_random", "count"),
            mean_capture_difference_vs_random=("capture_difference_vs_random", "mean"),
            median_capture_difference_vs_random=("capture_difference_vs_random", "median"),
            ci_lower_95=("capture_difference_vs_random", lambda x: x.quantile(0.025)),
            ci_upper_95=("capture_difference_vs_random", lambda x: x.quantile(0.975)),
            probability_difference_gt_zero=("capture_difference_vs_random", lambda x: (x > 0).mean()),
        )
    )
    comparison_summary.to_csv(args.output_dir / "stage7_residual_triage_bootstrap_summary.csv", index=False)
    (args.output_dir / "stage7_notes.txt").write_text(
        "The first review stage ranks predicted-low-priority chemicals by proximity to the locked "
        "calibration-derived concern cutoff. The second stage reviews an equal number of remaining "
        "predicted-low-priority chemicals using each reliability signal. Results are conditional on "
        "the threshold-first policy and should be interpreted as incremental triage evidence, not as "
        "an independent validation of a reliability detector.\n"
    )
    print(grouped_summary.to_string(index=False))


if __name__ == "__main__":
    main()
