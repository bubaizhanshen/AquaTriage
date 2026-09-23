#!/usr/bin/env python3
"""Compare external signal-selection objectives on the same locked splits."""

from __future__ import annotations

import argparse
from pathlib import Path
import hashlib
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ecoood.application import select_reliability_signal  # noqa: E402
from scripts.analyze_external_deployment_selection import (  # noqa: E402
    CANDIDATE_COLUMNS,
    build_chemical_panel,
    calibration_cutoffs,
    normalize_prediction_columns,
    random_reference,
    route_with_signal,
    routed_metrics,
)
from scripts.run_echa_pmra_external_validation import (  # noqa: E402
    calibration_split_by_chemical,
)


OBJECTIVES = (
    "directional_underprediction_area",
    "false_negative_capture",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--development-size", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--review-fraction", type=float, default=0.25)
    return parser.parse_args()


def _seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train_path, low_memory=False)
    predictions = normalize_prediction_columns(pd.read_csv(args.predictions, low_memory=False))
    chemical_ids = sorted(predictions["chemical_id"].astype(str).unique())
    if args.development_size >= len(chemical_ids):
        raise ValueError("development-size must leave an evaluation queue.")

    all_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    frequency_rows: list[dict[str, object]] = []
    for seed in sorted(predictions["seed"].unique()):
        _, calibration = calibration_split_by_chemical(train, seed=int(seed))
        cutoffs = calibration_cutoffs(calibration)
        seed_predictions = predictions.loc[predictions["seed"].eq(seed)].copy()
        panel = build_chemical_panel(seed_predictions, cutoffs)
        for repeat in range(args.repeats):
            rng = np.random.default_rng(_seed("stage2_selection", args.development_size, repeat))
            development_ids = set(rng.choice(chemical_ids, args.development_size, replace=False))
            development = panel.loc[panel["chemical_id"].astype(str).isin(development_ids)].reset_index(drop=True)
            evaluation = panel.loc[~panel["chemical_id"].astype(str).isin(development_ids)].reset_index(drop=True)
            for objective in OBJECTIVES:
                selected, selection_metrics, source = select_reliability_signal(
                    development,
                    candidate_columns=CANDIDATE_COLUMNS,
                    objective=objective,
                    review_fraction=args.review_fraction,
                )
                audit_rows.append(
                    {
                        "seed": int(seed),
                        "repeat": repeat,
                        "objective": objective,
                        "selected_signal": selected,
                        "selection_source": source,
                        "development_false_negative_n": int(development["true_high_concern"].sum() - (development["true_high_concern"] & development["pred_high_concern"]).sum()),
                        "selected_development_metric": float(selection_metrics.loc[selection_metrics["selected"], "false_negative_capture"].iloc[0]),
                    }
                )
                selected_metrics = routed_metrics(
                    route_with_signal(evaluation, selected, review_fraction=args.review_fraction)
                )
                all_rows.append(
                    {
                        "seed": int(seed),
                        "repeat": repeat,
                        "objective": objective,
                        "comparison": "selected_measure",
                        "selected_signal": selected,
                        **selected_metrics,
                    }
                )

            fixed = {
                "fixed_block_normalized_knn": "block_normalized_knn",
                "fixed_similarity_ad": "similarity_ad",
            }
            for label, signal in fixed.items():
                metrics = routed_metrics(
                    route_with_signal(evaluation, signal, review_fraction=args.review_fraction)
                )
                all_rows.append(
                    {
                        "seed": int(seed),
                        "repeat": repeat,
                        "objective": "fixed_comparator",
                        "comparison": label,
                        "selected_signal": signal,
                        **metrics,
                    }
                )
            all_rows.append(
                {
                    "seed": int(seed),
                    "repeat": repeat,
                    "objective": "fixed_comparator",
                    "comparison": "random_review",
                    "selected_signal": "",
                    **random_reference(
                        evaluation,
                        review_fraction=args.review_fraction,
                        draws=1000,
                        seed=_seed("stage2_random", seed, args.development_size, repeat),
                    ),
                }
            )

    detailed = pd.DataFrame(all_rows)
    audit = pd.DataFrame(audit_rows)
    summary = (
        detailed.groupby(["objective", "comparison"], as_index=False)
        .agg(
            n_runs=("seed", "nunique"),
            false_negative_capture_mean=("false_negative_capture", "mean"),
            false_negative_capture_sd=("false_negative_capture", "std"),
            lower_priority_false_omission_rate_mean=("lower_priority_false_omission_rate", "mean"),
            lower_priority_false_omission_rate_sd=("lower_priority_false_omission_rate", "std"),
        )
    )
    frequency = (
        audit.groupby(["objective", "selected_signal"], as_index=False)
        .size()
        .rename(columns={"size": "n_selected"})
    )
    frequency["selection_fraction"] = frequency["n_selected"] / args.repeats / predictions["seed"].nunique()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(args.output_dir / "stage2_selection_objectives_detailed.csv", index=False)
    summary.to_csv(args.output_dir / "stage2_selection_objectives_summary.csv", index=False)
    audit.to_csv(args.output_dir / "stage2_selection_objectives_audit.csv", index=False)
    frequency.to_csv(args.output_dir / "stage2_selection_objectives_frequency.csv", index=False)
    (args.output_dir / "stage2_notes.txt").write_text(
        "The same chemical-level development/evaluation draws are used for both "
        "selection objectives. No evaluation labels enter selection. The random "
        "reference is estimated by 1000 within-draw random reviews.\n"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
