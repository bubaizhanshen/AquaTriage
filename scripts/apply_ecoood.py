#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecoood.application import (  # noqa: E402
    DEFAULT_CANDIDATE_COLUMNS,
    TASK_ALIGNED_CANDIDATE_COLUMNS,
    REVIEW_POLICIES,
    SUPPORTED_OBJECTIVES,
    apply_ecoood_protocol,
)


def _candidate(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Candidate must use NAME=COLUMN syntax.")
    name, column = (part.strip() for part in value.split("=", 1))
    if not name or not column:
        raise argparse.ArgumentTypeError("Candidate name and column cannot be empty.")
    return name, column


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the fixed EcoOOD signal-selection and screening-routing protocol "
            "to a chemical-level scored queue."
        )
    )
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--development", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--objective",
        choices=sorted(SUPPORTED_OBJECTIVES),
        default="false_negative_capture",
    )
    parser.add_argument("--review-fraction", type=float, default=0.25)
    parser.add_argument("--high-error-quantile", type=float, default=0.90)
    parser.add_argument("--review-policy", choices=sorted(REVIEW_POLICIES),
                        help="Default: low_concern_first for omission control; all_queue otherwise.")
    parser.add_argument(
        "--default-signal",
        help="Default: threshold_proximity for low_concern_first; block_normalized_knn otherwise.",
    )
    parser.add_argument("--chemical-id-column", default="chemical_id")
    parser.add_argument("--pred-high-concern-column", default="pred_high_concern")
    parser.add_argument("--true-high-concern-column", default="true_high_concern")
    parser.add_argument("--y-true-column", default="y_true")
    parser.add_argument("--y-pred-column", default="y_pred")
    parser.add_argument("--eligible-column", default="input_eligible")
    parser.add_argument(
        "--candidate",
        action="append",
        type=_candidate,
        metavar="NAME=COLUMN",
        help=(
            "Override the fixed candidate library. Repeat for each candidate; "
            "higher values must indicate greater review priority."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queue = pd.read_csv(args.queue)
    development = pd.read_csv(args.development) if args.development else None
    review_policy = args.review_policy or (
        "low_concern_first" if args.objective == "false_negative_capture" else "all_queue")
    default_signal = args.default_signal or (
        "threshold_proximity" if review_policy == "low_concern_first" else "block_normalized_knn")
    candidate_columns = (
        dict(args.candidate)
        if args.candidate
        else (TASK_ALIGNED_CANDIDATE_COLUMNS if review_policy == "low_concern_first"
              else DEFAULT_CANDIDATE_COLUMNS)
    )
    result = apply_ecoood_protocol(
        queue,
        development=development,
        candidate_columns=candidate_columns,
        objective=args.objective,
        review_fraction=args.review_fraction,
        default_signal=default_signal,
        review_policy=review_policy,
        chemical_id_column=args.chemical_id_column,
        pred_high_concern_column=args.pred_high_concern_column,
        true_high_concern_column=args.true_high_concern_column,
        y_true_column=args.y_true_column,
        y_pred_column=args.y_pred_column,
        eligible_column=args.eligible_column,
        high_error_quantile=args.high_error_quantile,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    routed_path = args.output_dir / "routed_queue.csv"
    metrics_path = args.output_dir / "signal_selection.csv"
    summary_path = args.output_dir / "application_summary.json"
    result.routed_queue.to_csv(routed_path, index=False)
    result.selection_metrics.to_csv(metrics_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "selected_signal": result.selected_signal,
                "selection_source": result.selection_source,
                "objective": args.objective,
                "review_policy": review_policy,
                "candidate_columns": candidate_columns,
                "review_fraction": args.review_fraction,
                "queue_chemicals": int(len(result.routed_queue)),
                "reviewed_scoreable_chemicals": int(
                    result.routed_queue["reviewed"].sum()
                ),
                "route_counts": {
                    str(key): int(value)
                    for key, value in result.routed_queue[
                        "screening_action"
                    ].value_counts().items()
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Selected signal: {result.selected_signal}")
    print(f"Selection source: {result.selection_source}")
    print(f"Routed queue: {routed_path}")


if __name__ == "__main__":
    main()
