#!/usr/bin/env python3
"""Estimate paired chemical-level uncertainty for fixed-workload rankings.

This analysis reuses locked predictions and calibration-derived concern
thresholds. It does not refit a predictor or a reliability model. Within each
seed/split panel, chemicals are resampled with replacement and every method is
reevaluated on the same resampled panel. The resulting paired differences are
reported separately from fit-level variation; bootstrap replicates are not
treated as independent validation datasets.
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
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecoood.schema import DEFAULT_SCHEMA  # noqa: E402
from scripts.analyze_stage1_decision_baselines import (  # noqa: E402
    _internal_panels,
    _external_panels,
    _load_cutoff_table,
    _review_count,
)


METHOD_SCORES = {
    "block_normalized_knn": "block_normalized_knn",
    "similarity_ad": "similarity_ad",
    "prediction_error_risk": "prediction_error_risk",
    "ensemble_sd_risk": "ensemble_sd_risk",
    "prediction_threshold_proximity": "prediction_distance",
    "interval_crossing": "interval_cross_score",
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
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=20260920)
    return parser.parse_args()


def _fast_review_mask(
    scores: np.ndarray,
    chemical_ids: np.ndarray,
    predicted_high: np.ndarray,
    fraction: float,
    *,
    low_only: bool = True,
) -> np.ndarray:
    """Match _rank_mask without repeated DataFrame sorting."""
    n = len(scores)
    count = _review_count(n, fraction)
    reviewed = np.zeros(n, dtype=bool)
    if count == 0 or n == 0:
        return reviewed
    numeric = np.asarray(scores, dtype=float).copy()
    numeric[~np.isfinite(numeric)] = -np.inf
    ids = np.asarray(chemical_ids, dtype=str)
    eligible = np.flatnonzero(~predicted_high) if low_only else np.arange(n)
    order_low = eligible[np.lexsort((ids[eligible], -numeric[eligible]))]
    selected = order_low[:count]
    reviewed[selected] = True
    if len(selected) < count:
        remaining = np.flatnonzero(~reviewed)
        order_remaining = remaining[np.lexsort((ids[remaining], -numeric[remaining]))]
        reviewed[order_remaining[: count - len(selected)]] = True
    return reviewed


def _fast_metrics(
    true_high: np.ndarray,
    predicted_high: np.ndarray,
    reviewed: np.ndarray,
) -> dict[str, float | int]:
    false_negative = true_high & ~predicted_high
    lower_priority = ~predicted_high & ~reviewed
    omitted = true_high & lower_priority
    n_true = int(true_high.sum())
    n_false_negative = int(false_negative.sum())
    n_lower = int(lower_priority.sum())
    rescued = int((false_negative & reviewed).sum())
    return {
        "n_chemicals": int(len(true_high)),
        "review_n": int(reviewed.sum()),
        "n_predicted_low": int((~predicted_high).sum()),
        "n_true_high": n_true,
        "n_false_negative": n_false_negative,
        "rescued_false_negative": rescued,
        "false_negative_capture": float(rescued / n_false_negative)
        if n_false_negative else np.nan,
        "high_concern_left_low_priority_fraction": float(omitted.sum() / n_true)
        if n_true else np.nan,
        "lower_priority_false_omission_rate": float(omitted.sum() / n_lower)
        if n_lower else np.nan,
        "low_queue_capacity_sufficient": bool((~predicted_high).sum() >= reviewed.sum()),
    }


def _fast_random_metrics(
    true_high: np.ndarray,
    predicted_high: np.ndarray,
    fraction: float,
) -> dict[str, float | int]:
    n = len(true_high)
    count = _review_count(n, fraction)
    p = count / n if n else np.nan
    false_negative = int((true_high & ~predicted_high).sum())
    n_true = int(true_high.sum())
    n_lower = int((~predicted_high).sum())
    return {
        "n_chemicals": n,
        "review_n": count,
        "n_predicted_low": n_lower,
        "n_true_high": n_true,
        "n_false_negative": false_negative,
        "rescued_false_negative": float(false_negative * p) if np.isfinite(p) else np.nan,
        "false_negative_capture": float(p) if false_negative else np.nan,
        "high_concern_left_low_priority_fraction": (
            float(false_negative * (1 - p) / n_true) if n_true else np.nan
        ),
        "lower_priority_false_omission_rate": (
            float(false_negative / n_lower) if n_lower else np.nan
        ),
        "low_queue_capacity_sufficient": bool(n_lower >= count),
    }


def _evaluate_arrays(
    arrays: dict[str, np.ndarray],
    score_columns: dict[str, str],
    fraction: float,
    *,
    low_only: bool = True,
) -> dict[str, dict[str, float | int]]:
    """Evaluate all available methods on pre-extracted NumPy arrays."""
    true_high = arrays["true_high"].astype(bool)
    predicted_high = arrays["pred_high"].astype(bool)
    chemical_ids = arrays["chemical_id"].astype(str)
    results: dict[str, dict[str, float | int]] = {}
    for method, score_column in score_columns.items():
        reviewed = _fast_review_mask(
            arrays[score_column], chemical_ids, predicted_high, fraction,
            low_only=low_only,
        )
        results[method] = _fast_metrics(true_high, predicted_high, reviewed)
    results["random_review"] = _fast_random_metrics(true_high, predicted_high, fraction)
    return results


def _panel_bootstrap(
    panel: pd.DataFrame,
    *,
    dataset: str,
    split: str,
    seed: int,
    fraction: float,
    reps: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return exact panel metrics and paired chemical bootstrap rows."""
    panel = panel.reset_index(drop=True)
    score_columns = {
        method: score
        for method, score in METHOD_SCORES.items()
        if score in panel.columns
    }
    base_arrays = {
        "chemical_id": panel["chemical_id"].astype(str).to_numpy(),
        "true_high": panel["true_high"].to_numpy(dtype=bool),
        "pred_high": panel["pred_high"].to_numpy(dtype=bool),
        **{
            column: pd.to_numeric(panel[column], errors="coerce").to_numpy(dtype=float)
            for column in score_columns.values()
        },
    }
    exact = _evaluate_arrays(base_arrays, score_columns, fraction)
    exact_rows = []
    for method, metrics in exact.items():
        exact_rows.append(
            {
                "dataset": dataset,
                "split": split,
                "seed": seed,
                "method": method,
                **metrics,
            }
        )

    methods = list(exact)
    ref_methods = ["block_normalized_knn", "random_review"]
    bootstrap_rows: list[dict[str, object]] = []
    n = len(panel)
    if n == 0:
        return exact_rows, bootstrap_rows

    for bootstrap_id in range(reps):
        sampled_index = rng.integers(0, n, size=n)
        sampled = {name: values[sampled_index] for name, values in base_arrays.items()}
        evaluated = _evaluate_arrays(sampled, score_columns, fraction)
        for method in methods:
            metrics = evaluated[method]
            for reference in ref_methods:
                if method == reference or reference not in evaluated:
                    continue
                reference_metrics = evaluated[reference]
                for metric in (
                    "false_negative_capture",
                    "high_concern_left_low_priority_fraction",
                    "lower_priority_false_omission_rate",
                ):
                    value = float(metrics[metric])
                    reference_value = float(reference_metrics[metric])
                    if np.isfinite(value) and np.isfinite(reference_value):
                        difference = value - reference_value
                    else:
                        difference = np.nan
                    bootstrap_rows.append(
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
    return exact_rows, bootstrap_rows


def _summarize_bootstrap(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_columns + ["method", "reference", "metric"], dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(group_columns + ["method", "reference", "metric"], keys))
        values = group["difference"].dropna().to_numpy(dtype=float)
        if not len(values):
            summary = {
                "n_bootstrap_values": 0,
                "mean_difference": np.nan,
                "median_difference": np.nan,
                "ci_lower_95": np.nan,
                "ci_upper_95": np.nan,
                "probability_capture_difference_gt_zero": np.nan,
                "probability_omission_difference_lt_zero": np.nan,
            }
        else:
            summary = {
                "n_bootstrap_values": int(len(values)),
                "mean_difference": float(np.mean(values)),
                "median_difference": float(np.median(values)),
                "ci_lower_95": float(np.quantile(values, 0.025)),
                "ci_upper_95": float(np.quantile(values, 0.975)),
                "probability_capture_difference_gt_zero": float(np.mean(values > 0))
                if key_map["metric"] == "false_negative_capture" else np.nan,
                "probability_omission_difference_lt_zero": float(np.mean(values < 0))
                if key_map["metric"] != "false_negative_capture" else np.nan,
            }
        rows.append({**key_map, **summary})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if not 0 <= args.review_fraction <= 1:
        raise ValueError("review-fraction must be in [0, 1].")
    if args.bootstrap_reps < 100:
        raise ValueError("bootstrap-reps must be at least 100.")

    internal_data = pd.read_csv(args.internal_data, low_memory=False)
    internal_predictions = pd.read_csv(args.internal_predictions, low_memory=False)
    external_train = pd.read_csv(args.external_train, low_memory=False)
    external_predictions = pd.read_csv(args.external_predictions, low_memory=False)
    cutoff_table = _load_cutoff_table(args.internal_cutoffs) if args.internal_cutoffs else None

    panels = [
        *[
            (dataset, seed, split, panel)
            for dataset, seed, split, panel in _internal_panels(
                internal_data, internal_predictions, args.seeds, cutoff_table
            )
        ],
        *[
            (dataset, seed, split, panel)
            for dataset, seed, split, panel in _external_panels(
                external_train, external_predictions, args.seeds
            )
        ],
    ]
    exact_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(args.random_seed)
    for dataset, seed, split, panel in panels:
        exact, bootstrap = _panel_bootstrap(
            panel,
            dataset=dataset,
            split=split,
            seed=seed,
            fraction=args.review_fraction,
            reps=args.bootstrap_reps,
            rng=rng,
        )
        exact_rows.extend(exact)
        bootstrap_rows.extend(bootstrap)

    exact_frame = pd.DataFrame(exact_rows)
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exact_frame.to_csv(args.output_dir / "stage4_exact_panel_metrics.csv", index=False)
    bootstrap_frame.to_csv(args.output_dir / "stage4_paired_bootstrap_differences.csv", index=False)

    # The primary internal summary gives each seed/split panel equal weight.
    internal_main = bootstrap_frame.loc[
        bootstrap_frame.dataset.eq("internal")
        & bootstrap_frame.split.isin(["chemical_random", "scaffold", "temporal", "species"])
    ].copy()
    external = bootstrap_frame.loc[bootstrap_frame.dataset.eq("external")].copy()
    summary_frames = []
    if not internal_main.empty:
        internal_main = internal_main.assign(summary_group="internal_main_four_shifts")
        summary_frames.append(internal_main)
    if not external.empty:
        external = external.assign(summary_group="external_expanded")
        summary_frames.append(external)
    if summary_frames:
        summary_input = pd.concat(summary_frames, ignore_index=True)
        summary = _summarize_bootstrap(summary_input, ["summary_group"])
    else:
        summary = pd.DataFrame()
    summary.to_csv(args.output_dir / "stage4_paired_bootstrap_summary.csv", index=False)

    split_summary = _summarize_bootstrap(
        bootstrap_frame.loc[
            bootstrap_frame.split.isin(["chemical_random", "scaffold", "temporal", "species"])
            | bootstrap_frame.dataset.eq("external")
        ],
        ["dataset", "split"],
    )
    split_summary.to_csv(args.output_dir / "stage4_paired_bootstrap_by_split.csv", index=False)

    (args.output_dir / "stage4_notes.txt").write_text(
        "Chemical-level paired bootstrap for a fixed 25% predicted-low-priority-first review policy. "
        "Predictions, calibration-derived concern cutoffs, candidate scores, and tie-breaking were fixed. "
        "Each bootstrap replicate resampled chemical rows within a seed/split panel and evaluated all "
        "methods on the same resample. Replicates are uncertainty estimates conditional on the locked panels, "
        "not independent validation datasets. Summary groups give equal weight to seed/split panels; fit-level "
        "variation should be reported separately. Positive capture differences favor the method; negative "
        "omission differences favor the method.\n"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
