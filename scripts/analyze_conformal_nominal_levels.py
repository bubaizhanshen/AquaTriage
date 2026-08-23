from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ecoood.conformal import ScaledConformalRegressor
from ecoood.features import EcoFeatureBuilder, attach_rdkit_descriptors
from ecoood.models import BootstrapEnsembleRegressor
from ecoood.schema import DEFAULT_SCHEMA
from ecoood.splits import build_split


DEFAULT_SPLITS = (
    "random",
    "chemical_random",
    "scaffold",
    "temporal",
    "species",
    "chemical_class",
)


def _select(frame: pd.DataFrame, indices: np.ndarray) -> pd.DataFrame:
    return frame.loc[indices].reset_index(drop=True)


def _fit_once(
    frame: pd.DataFrame,
    *,
    split: str,
    seed: int,
    members: int,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split_indices = build_split(frame, split=split, schema=DEFAULT_SCHEMA, seed=seed)
    train = _select(frame, split_indices.train)
    calibration = _select(frame, split_indices.calib)
    test = _select(frame, split_indices.test)

    builder = EcoFeatureBuilder(
        schema=DEFAULT_SCHEMA,
        include_study_year=False,
        recompute_rdkit_logp=True,
    )
    train_features = builder.fit_transform(train)
    calibration_features = builder.transform(calibration)
    test_features = builder.transform(test)

    model = BootstrapEnsembleRegressor(
        model_name="lightgbm",
        n_members=members,
        seed=seed,
        n_jobs=n_jobs,
    ).fit(train_features.full, train[DEFAULT_SCHEMA.target].to_numpy(dtype=float))
    calibration_prediction = model.predict(calibration_features.full)
    test_prediction = model.predict(test_features.full)
    return (
        calibration[DEFAULT_SCHEMA.target].to_numpy(dtype=float),
        calibration_prediction.mean,
        calibration_prediction.std,
        np.column_stack(
            [
                test[DEFAULT_SCHEMA.target].to_numpy(dtype=float),
                test_prediction.mean,
                test_prediction.std,
            ]
        ),
    )


def _summarize_levels(
    calibration_y: np.ndarray,
    calibration_mean: np.ndarray,
    calibration_std: np.ndarray,
    test_values: np.ndarray,
    *,
    split: str,
    seed: int,
    levels: tuple[float, ...],
) -> list[dict[str, float | int | str]]:
    test_y = test_values[:, 0]
    test_mean = test_values[:, 1]
    test_std = test_values[:, 2]
    calibration_scale = np.maximum(calibration_std, 1e-3)
    test_scale = np.maximum(test_std, 1e-3)

    results: list[dict[str, float | int | str]] = []
    for level in levels:
        conformal = ScaledConformalRegressor(alpha=1.0 - level).fit(
            calibration_y,
            calibration_mean,
            scale=calibration_scale,
        )
        interval = conformal.predict(test_mean, scale=test_scale)
        covered = (test_y >= interval.lower) & (test_y <= interval.upper)
        results.append(
            {
                "split": split,
                "seed": seed,
                "nominal_coverage": level,
                "alpha": 1.0 - level,
                "calibration_n": conformal.n_calibration_,
                "quantile_rank": conformal.quantile_rank_,
                "qhat": float(conformal.qhat),
                "test_n": len(test_y),
                "empirical_coverage": float(covered.mean()),
                "coverage_gap": float(covered.mean() - level),
                "mean_interval_width": float(interval.width.mean()),
                "median_interval_width": float(np.median(interval.width)),
            }
        )
    return results


def _aggregate(detailed: pd.DataFrame) -> pd.DataFrame:
    summary = (
        detailed.groupby(["split", "nominal_coverage"], sort=False, as_index=False)
        .agg(
            calibration_n_mean=("calibration_n", "mean"),
            qhat_mean=("qhat", "mean"),
            qhat_sd=("qhat", "std"),
            empirical_coverage_mean=("empirical_coverage", "mean"),
            empirical_coverage_sd=("empirical_coverage", "std"),
            coverage_gap_mean=("coverage_gap", "mean"),
            coverage_gap_sd=("coverage_gap", "std"),
            mean_interval_width_mean=("mean_interval_width", "mean"),
            mean_interval_width_sd=("mean_interval_width", "std"),
        )
    )
    width_90 = summary.loc[
        np.isclose(summary["nominal_coverage"], 0.90),
        ["split", "mean_interval_width_mean"],
    ].rename(columns={"mean_interval_width_mean": "width_at_90"})
    summary = summary.merge(width_90, on="split", how="left")
    summary["width_ratio_to_90"] = (
        summary["mean_interval_width_mean"] / summary["width_at_90"]
    )
    return summary.drop(columns="width_at_90")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate scaled split-conformal intervals at several nominal coverage levels."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 42, 43, 44])
    parser.add_argument("--levels", nargs="+", type=float, default=[0.80, 0.90, 0.95])
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=5)
    args = parser.parse_args()

    if any(level <= 0 or level >= 1 for level in args.levels):
        raise ValueError("Every nominal coverage level must be between 0 and 1.")
    frame = pd.read_parquet(args.data) if args.data.suffix == ".parquet" else pd.read_csv(args.data)
    frame = attach_rdkit_descriptors(
        frame,
        DEFAULT_SCHEMA,
        recompute_logp=True,
    )
    frame = frame[frame[DEFAULT_SCHEMA.target].notna()].reset_index(drop=True)

    rows: list[dict[str, float | int | str]] = []
    for seed in args.seeds:
        for split in args.splits:
            calibration_y, calibration_mean, calibration_std, test_values = _fit_once(
                frame,
                split=split,
                seed=seed,
                members=args.members,
                n_jobs=args.n_jobs,
            )
            rows.extend(
                _summarize_levels(
                    calibration_y,
                    calibration_mean,
                    calibration_std,
                    test_values,
                    split=split,
                    seed=seed,
                    levels=tuple(args.levels),
                )
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detailed = pd.DataFrame(rows)
    detailed.to_csv(args.output_dir / "conformal_nominal_level_detailed.csv", index=False)
    _aggregate(detailed).to_csv(
        args.output_dir / "conformal_nominal_level_summary.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
