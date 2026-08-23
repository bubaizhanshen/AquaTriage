#!/usr/bin/env python3
"""Run multiseed deployment benchmarks with distance-definition sensitivities."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ecoood.pipeline import run_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 42, 43, 44])
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["chemical_random", "scaffold", "temporal", "species", "chemical_class"],
    )
    parser.add_argument("--model", default="lightgbm")
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--ensemble-n-jobs", type=int, default=5)
    parser.add_argument("--include-study-year", action="store_true")
    parser.add_argument(
        "--keep-linked-logp",
        action="store_true",
        help="Retain linked logP instead of recomputing RDKit MolLogP.",
    )
    parser.add_argument(
        "--allow-legacy-structure-placeholder",
        action="store_true",
        help="Reproduce the frozen historical benchmark only; current deployment rejects these inputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.data, low_memory=False)
    for seed in args.seeds:
        run_benchmark(
            df=data,
            splits=args.splits,
            models=[args.model],
            output_dir=str(args.output_root / f"seed_{seed}"),
            seed=seed,
            n_members=args.members,
            ensemble_n_jobs=args.ensemble_n_jobs,
            include_study_year=args.include_study_year,
            recompute_rdkit_logp=not args.keep_linked_logp,
            allow_legacy_structure_placeholder=args.allow_legacy_structure_placeholder,
            run_distance_sensitivity=True,
        )


if __name__ == "__main__":
    main()
