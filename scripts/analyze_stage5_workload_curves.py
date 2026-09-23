#!/usr/bin/env python3
"""Evaluate fixed ranking rules across the complete review-workload range."""

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

from scripts.analyze_stage1_decision_baselines import (  # noqa: E402
    _external_panels,
    _internal_panels,
    _load_cutoff_table,
)
from scripts.analyze_stage4_paired_uncertainty import (  # noqa: E402
    METHOD_SCORES,
    _evaluate_arrays,
)


PRIMARY_SPLITS = ("chemical_random", "scaffold", "temporal", "species")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal-data", type=Path, required=True)
    parser.add_argument("--internal-predictions", type=Path, required=True)
    parser.add_argument("--external-train", type=Path, required=True)
    parser.add_argument("--external-predictions", type=Path, required=True)
    parser.add_argument("--internal-cutoffs", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 42, 43, 44])
    parser.add_argument("--grid-size", type=int, default=101)
    return parser.parse_args()


def _arrays(panel: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    score_columns = {
        method: score for method, score in METHOD_SCORES.items() if score in panel.columns
    }
    arrays = {
        "chemical_id": panel["chemical_id"].astype(str).to_numpy(),
        "true_high": panel["true_high"].to_numpy(dtype=bool),
        "pred_high": panel["pred_high"].to_numpy(dtype=bool),
        **{
            column: pd.to_numeric(panel[column], errors="coerce").to_numpy(dtype=float)
            for column in score_columns.values()
        },
    }
    return arrays, score_columns


def _rows_for_panel(
    panel: pd.DataFrame,
    *,
    dataset: str,
    split: str,
    seed: int,
    fractions: np.ndarray,
) -> list[dict[str, object]]:
    arrays, score_columns = _arrays(panel.reset_index(drop=True))
    rows: list[dict[str, object]] = []
    for policy, low_only in (
        ("all_queue", False),
        ("predicted_low_priority_first", True),
    ):
        for fraction in fractions:
            results = _evaluate_arrays(
                arrays, score_columns, float(fraction), low_only=low_only
            )
            for method, metrics in results.items():
                rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "seed": seed,
                        "policy": policy,
                        "review_fraction": float(fraction),
                        "method": method,
                        **metrics,
                    }
                )
    return rows


def main() -> None:
    args = parse_args()
    if args.grid_size < 11:
        raise ValueError("grid-size must be at least 11.")
    fractions = np.linspace(0.0, 1.0, args.grid_size)
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
    rows: list[dict[str, object]] = []
    for dataset, seed, split, panel in panels:
        rows.extend(
            _rows_for_panel(
                panel,
                dataset=dataset,
                split=split,
                seed=seed,
                fractions=fractions,
            )
        )
    detailed = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(args.output_dir / "stage5_workload_curves_detailed.csv", index=False)

    main = detailed.loc[
        detailed.dataset.eq("internal")
        & detailed.split.isin(PRIMARY_SPLITS)
    ].copy()
    main["summary_group"] = "internal_main_four_shifts"
    external = detailed.loc[detailed.dataset.eq("external")].copy()
    external["summary_group"] = "external_expanded"
    grouped = pd.concat([main, external], ignore_index=True)
    grouped["panel_id"] = (
        grouped["dataset"].astype(str)
        + "__"
        + grouped["split"].astype(str)
        + "__"
        + grouped["seed"].astype(str)
    )
    summary = (
        grouped.groupby(["summary_group", "policy", "review_fraction", "method"], as_index=False)
        .agg(
            n_panels=("panel_id", "nunique"),
            false_negative_capture_mean=("false_negative_capture", "mean"),
            false_negative_capture_sd=("false_negative_capture", "std"),
            high_concern_left_low_priority_fraction_mean=(
                "high_concern_left_low_priority_fraction", "mean"
            ),
            lower_priority_false_omission_rate_mean=(
                "lower_priority_false_omission_rate", "mean"
            ),
        )
    )
    summary.to_csv(args.output_dir / "stage5_workload_curves_summary.csv", index=False)

    # The helper above expects split-level data; write direct summary AUCs to avoid
    # treating summary groups as independent splits.
    auc_rows = []
    for keys, group in summary.groupby(["summary_group", "policy", "method"], dropna=False):
        group = group.sort_values("review_fraction")
        for metric in (
            "false_negative_capture_mean",
            "high_concern_left_low_priority_fraction_mean",
            "lower_priority_false_omission_rate_mean",
        ):
            x = group.review_fraction.to_numpy(dtype=float)
            y = group[metric].to_numpy(dtype=float)
            valid = np.isfinite(y)
            auc_rows.append(
                {
                    "summary_group": keys[0],
                    "policy": keys[1],
                    "method": keys[2],
                    "metric": metric,
                    "auc": float(np.trapezoid(y[valid], x[valid]))
                    if valid.sum() >= 2 else np.nan,
                    "n_grid_points": int(valid.sum()),
                }
            )
    pd.DataFrame(auc_rows).to_csv(args.output_dir / "stage5_workload_curves_auc.csv", index=False)
    (args.output_dir / "stage5_notes.txt").write_text(
        "Complete workload curves use fixed predictions, calibration-derived endpoint cutoffs, "
        "chemical-level aggregation, and the same deterministic tie-breaking as the 25% baseline. "
        "The threshold-proximity rule is a task-aligned screening baseline, not an OOD detector. "
        "Summary means give equal weight to seed/split panels; curve AUC is descriptive and does not "
        "replace the prespecified 25% operating point.\n"
    )
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
