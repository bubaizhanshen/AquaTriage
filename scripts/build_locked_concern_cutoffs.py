#!/usr/bin/env python3
"""Build calibration-derived endpoint concern cutoffs for a prediction run.

The cutoff table is deliberately tied to one source dataset, one split builder,
and one prediction root.  This prevents calibration thresholds from an older
benchmark or RDKit configuration being reused with a different prediction set.
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

from ecoood.schema import DEFAULT_SCHEMA  # noqa: E402
from ecoood.splits import build_split  # noqa: E402


DEFAULT_SEEDS = (40, 41, 42, 43, 44)
DEFAULT_SPLITS = (
    "random",
    "chemical_random",
    "scaffold",
    "temporal",
    "species",
    "chemical_class",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--quantile", type=float, default=0.25)
    return parser.parse_args()


def _prediction_path(root: Path, seed: int, split: str) -> Path:
    candidates = (
        root / f"seed_{seed}" / split / "lightgbm" / "predictions.csv",
        root / f"seed_{seed}" / "structured" / split / "lightgbm" / "predictions.csv",
    )
    for path in candidates:
        if path.exists():
            return path
    # Return the conventional path in the error message when neither layout
    # exists; this keeps missing-run diagnostics predictable.
    return candidates[0]


def _cutoffs(frame: pd.DataFrame, quantile: float) -> dict[str, float]:
    grouped = (
        frame.groupby(
            [DEFAULT_SCHEMA.chemical_id, DEFAULT_SCHEMA.endpoint],
            dropna=False,
            as_index=False,
        )
        .agg(endpoint_target=(DEFAULT_SCHEMA.target, "median"))
    )
    return {
        str(endpoint): float(group["endpoint_target"].quantile(quantile))
        for endpoint, group in grouped.groupby(DEFAULT_SCHEMA.endpoint, sort=True)
    }


def _validate_test_frame(source: pd.DataFrame, test_idx: np.ndarray, prediction: pd.DataFrame) -> None:
    expected = source.loc[test_idx, [DEFAULT_SCHEMA.chemical_id, DEFAULT_SCHEMA.endpoint, DEFAULT_SCHEMA.target]].reset_index(drop=True)
    observed = prediction[["chemical_id", "endpoint", "y_true"]].reset_index(drop=True)
    if len(expected) != len(observed):
        raise ValueError(f"Test row count mismatch: expected {len(expected)}, observed {len(observed)}")
    if not expected[DEFAULT_SCHEMA.chemical_id].astype(str).equals(observed["chemical_id"].astype(str)):
        raise ValueError("Prediction chemical_id order does not match build_split test order")
    if not expected[DEFAULT_SCHEMA.endpoint].astype(str).equals(observed["endpoint"].astype(str)):
        raise ValueError("Prediction endpoint order does not match build_split test order")
    if not np.allclose(expected[DEFAULT_SCHEMA.target].to_numpy(float), observed["y_true"].to_numpy(float), equal_nan=True):
        raise ValueError("Prediction y_true values do not match the source dataset")


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.data, low_memory=False)
    rows: list[dict[str, object]] = []
    validation: list[dict[str, object]] = []
    for seed in args.seeds:
        for split in args.splits:
            path = _prediction_path(args.prediction_root, seed, split)
            if not path.exists():
                raise FileNotFoundError(path)
            indices = build_split(data, split=split, schema=DEFAULT_SCHEMA, seed=seed)
            calibration = data.loc[indices.calib]
            cutoffs = _cutoffs(calibration, args.quantile)
            prediction = pd.read_csv(path, low_memory=False)
            _validate_test_frame(data, indices.test, prediction)
            for endpoint, cutoff in cutoffs.items():
                rows.append(
                    {
                        "seed": seed,
                        "split": split,
                        "endpoint": endpoint,
                        "cutoff": cutoff,
                        "quantile": args.quantile,
                        "calibration_rows": int(len(indices.calib)),
                        "calibration_chemicals": int(calibration[DEFAULT_SCHEMA.chemical_id].nunique()),
                        "test_rows": int(len(indices.test)),
                        "test_chemicals": int(data.loc[indices.test, DEFAULT_SCHEMA.chemical_id].nunique()),
                    }
                )
            validation.append(
                {
                    "seed": seed,
                    "split": split,
                    "prediction_path": str(path),
                    "calibration_rows": int(len(indices.calib)),
                    "test_rows": int(len(indices.test)),
                    "calibration_chemicals": int(calibration[DEFAULT_SCHEMA.chemical_id].nunique()),
                    "test_chemicals": int(data.loc[indices.test, DEFAULT_SCHEMA.chemical_id].nunique()),
                    "status": "validated",
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    validation_path = args.output.with_name(args.output.stem + "_validation.csv")
    pd.DataFrame(validation).to_csv(validation_path, index=False)
    print(f"Wrote {len(rows)} cutoffs to {args.output}")
    print(f"Wrote {len(validation)} validation rows to {validation_path}")


if __name__ == "__main__":
    main()
