from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 42, 43, 44])
    args = parser.parse_args()

    frames = []
    for seed in args.seeds:
        path = (
            args.input_root
            / f"seed_{seed}"
            / "structured"
            / "random"
            / "lightgbm"
            / "calibration_meta_bootstrap.csv"
        )
        frame = pd.read_csv(path)
        frame["seed"] = seed
        frames.append(frame)
    detailed = pd.concat(frames, ignore_index=True)

    coefficient_columns = [column for column in detailed if column.startswith("coef_")]
    score_summary = pd.DataFrame(
        {
            "metric": ["sampled_positive_n", "score_spearman", "converged"],
            "mean": [
                detailed["sampled_positive_n"].mean(),
                detailed["score_spearman"].mean(),
                detailed["converged"].astype(float).mean(),
            ],
            "q025": [
                detailed["sampled_positive_n"].quantile(0.025),
                detailed["score_spearman"].quantile(0.025),
                detailed["converged"].astype(float).quantile(0.025),
            ],
            "median": [
                detailed["sampled_positive_n"].median(),
                detailed["score_spearman"].median(),
                detailed["converged"].astype(float).median(),
            ],
            "q975": [
                detailed["sampled_positive_n"].quantile(0.975),
                detailed["score_spearman"].quantile(0.975),
                detailed["converged"].astype(float).quantile(0.975),
            ],
            "n_replicates": [len(detailed)] * 3,
        }
    )
    coefficient_summary = (
        detailed.melt(
            id_vars=["seed", "replicate"],
            value_vars=coefficient_columns,
            var_name="component",
            value_name="coefficient",
        )
        .assign(component=lambda frame: frame["component"].str.removeprefix("coef_"))
        .groupby("component", as_index=False)
        .agg(
            coefficient_mean=("coefficient", "mean"),
            coefficient_sd=("coefficient", "std"),
            positive_fraction=("coefficient", lambda values: float((values > 0).mean())),
            n_replicates=("coefficient", "count"),
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(args.output_dir / "calibration_meta_bootstrap_all.csv", index=False)
    score_summary.to_csv(args.output_dir / "calibration_meta_bootstrap_summary.csv", index=False)
    coefficient_summary.to_csv(
        args.output_dir / "calibration_meta_bootstrap_coefficients.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
