from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ecoood.pipeline import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EcoOOD benchmark experiments.")
    parser.add_argument("--data", type=Path, required=True, help="CSV or Parquet dataset path.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["random", "scaffold", "chemical_class", "species", "temporal"],
    )
    parser.add_argument("--models", nargs="+", default=["lightgbm", "random_forest"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--ensemble-n-jobs", type=int, default=-1)
    parser.add_argument("--meta-bootstrap-replicates", type=int, default=0)
    parser.add_argument(
        "--reliability-component-mode",
        choices=["revised", "legacy"],
        default="revised",
    )
    parser.add_argument("--reliability-neighbors", type=int, default=5)
    parser.add_argument(
        "--reliability-fingerprint-metric",
        choices=["tanimoto", "cosine"],
        default="tanimoto",
    )
    parser.add_argument("--run-distance-sensitivity", action="store_true")
    parser.add_argument("--include-study-year", action="store_true")
    parser.add_argument("--keep-linked-logp", action="store_true")
    parser.add_argument("--allow-legacy-structure-placeholder", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmark"))
    args = parser.parse_args()

    if args.data.suffix == ".parquet":
        df = pd.read_parquet(args.data)
    else:
        df = pd.read_csv(args.data)
    summary = run_benchmark(
        df=df,
        splits=args.splits,
        models=args.models,
        output_dir=str(args.output_dir),
        alpha=args.alpha,
        seed=args.seed,
        n_members=args.members,
        ensemble_n_jobs=args.ensemble_n_jobs,
        meta_bootstrap_replicates=args.meta_bootstrap_replicates,
        reliability_component_mode=args.reliability_component_mode,
        reliability_n_neighbors=args.reliability_neighbors,
        reliability_fingerprint_metric=args.reliability_fingerprint_metric,
        run_distance_sensitivity=args.run_distance_sensitivity,
        include_study_year=args.include_study_year,
        recompute_rdkit_logp=not args.keep_linked_logp,
        allow_legacy_structure_placeholder=args.allow_legacy_structure_placeholder,
    )
    print(summary)


if __name__ == "__main__":
    main()
