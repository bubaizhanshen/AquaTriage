from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from ecoood.pipeline import run_benchmark
from ecoood.splits import named_class_for_seed


DEFAULT_SPLITS = [
    "random",
    "chemical_random",
    "scaffold",
    "temporal",
    "species",
    "chemical_class",
]
DEFAULT_MODELS = ["lightgbm", "random_forest", "xgboost"]
DEFAULT_SEEDS = [40, 41, 42, 43, 44]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the audited structured EcoOOD benchmark for a set of seeds."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--ensemble-n-jobs", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "data_path": str(args.data),
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "n_rows": int(len(df)),
        "n_chemicals": int(df["chemical_id"].nunique()),
        "splits": args.splits,
        "models": args.models,
        "seeds": args.seeds,
        "ensemble_members": args.members,
        "ensemble_n_jobs": args.ensemble_n_jobs,
        "conformal_alpha": args.alpha,
        "ecoood_primary_high_error_quantile": 0.9,
        "ecoood_high_error_quantile_sensitivity": [0.8, 0.9, 0.95],
        "endpoint_conditional_conformal_min_group_size": 20,
        "prediction_error_risk_components": (
            "distinct-chemical Tanimoto fingerprint kNN; shrinkage Mahalanobis descriptors; "
            "taxonomic lineage; endpoint-restricted missing-aware Gower context and missingness; "
            "missing-aware bioactivity distance and missingness; ensemble SD"
        ),
        "distance_sensitivity": (
            "legacy definitions; cosine fingerprint distance; Tanimoto k=1,3,5,10; "
            "case-level and distinct-chemical block-normalized kNN"
        ),
        "named_class_holdout_by_seed": {
            str(seed): named_class_for_seed(seed) for seed in args.seeds
        },
        "deterministic_rejection_policy": "excluded from scoreable benchmarks and assigned to withhold_review",
        "feature_policy": "explicit feature blocks exclude target, target-scale, identifier, source, and chemical-class fields",
        "publication_year_policy": "used for temporal splitting and audit only; excluded from predictor and reliability inputs",
        "logp_policy": "RDKit MolLogP recomputed for every parseable molecular input",
        "default_unlabeled_triage": (
            "five-neighbor case-level block-normalized kNN; a distinct-chemical version is reported as a sensitivity analysis"
        ),
    }
    (args.output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    for seed in args.seeds:
        output_dir = args.output_root / f"seed_{seed}" / "structured"
        summary = run_benchmark(
            df=df,
            splits=args.splits,
            models=args.models,
            output_dir=str(output_dir),
            alpha=args.alpha,
            seed=seed,
            n_members=args.members,
            ensemble_n_jobs=args.ensemble_n_jobs,
        )
        print(f"completed seed={seed} rows={len(summary)} output={output_dir}", flush=True)


if __name__ == "__main__":
    main()
