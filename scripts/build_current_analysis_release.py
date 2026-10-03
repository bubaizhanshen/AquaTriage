"""Package the fixed inputs and compact results used by the current paper."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIXED_TABLES = (
    "macro_summary.csv",
    "metrics_by_fit.csv",
    "paired_intervals.csv",
    "event_counts.csv",
    "endpoint_eligibility.csv",
    "relative_control.csv",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--internal-predictions", type=Path, required=True)
    parser.add_argument("--cutoffs", type=Path, required=True)
    parser.add_argument("--external-predictions", type=Path, required=True)
    parser.add_argument(
        "--external-panel",
        type=Path,
        default=ROOT / "data/processed/ecoood_external_expanded_v1.csv",
    )
    parser.add_argument("--fixed-results", type=Path, required=True)
    parser.add_argument("--review-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", default="v0.4.0")
    parser.add_argument("--analysis-root", type=Path)
    parser.add_argument("--bioactivity-manifest", type=Path)
    args = parser.parse_args()

    benchmark = pd.read_csv(args.benchmark, usecols=["chemical_id"])
    external = pd.read_csv(args.external_panel, usecols=["chemical_id", "case_id"])
    if (len(benchmark), benchmark.chemical_id.nunique()) != (4519, 775):
        raise ValueError("Identity-corrected primary benchmark must contain 4519 cases from 775 chemicals")
    if (len(external), external.chemical_id.nunique()) != (1027, 441):
        raise ValueError("External panel must contain 1027 cases from 441 chemicals")
    if external.case_id.duplicated().any():
        raise ValueError("External case identifiers must be unique")

    entries = {
        "data/strict_input_benchmark.csv": args.benchmark,
        "data/external_panel.csv": args.external_panel,
        "data/calibration_cutoffs.csv": args.cutoffs,
        "predictions/internal_predictions.csv": args.internal_predictions,
        "predictions/external_predictions.csv": args.external_predictions,
        **{
            f"tables/fixed_hazard/{name}": args.fixed_results / name
            for name in FIXED_TABLES
        },
        "tables/review/queue_macro_estimates.csv": args.review_results / "queue/macro_estimates.csv",
        "tables/review/queue_paired_intervals.csv": args.review_results / "queue/paired_macro_intervals.csv",
        "tables/review/queue_events.csv": args.review_results / "queue_events.csv",
        "tables/review/workload_areas.csv": args.review_results / "workload_areas.csv",
    }
    if args.analysis_root:
        entries.update({
            "data/split_assignments.csv":args.analysis_root/"identity_refit_split_assignments.csv",
            "data/feature_manifest.csv":args.analysis_root/"feature_manifest.csv",
            "audit/excluded_identity_records.csv":args.analysis_root/"excluded_identity_records.csv",
            "audit/pubchem_identity_adjudication.csv":args.analysis_root/"pubchem_identity_adjudication.csv",
            "audit/source_reference_groups.csv":args.analysis_root/"reference_holdout/row_reference_annotations.csv",
            "tables/reference_holdout_metrics.csv":args.analysis_root/"reference_holdout/benchmark_summary_agg.csv",
            "tables/predictor_family_and_controls.csv":args.analysis_root/"results/dependent_metrics.csv",
            "tables/predictor_family_review.csv":args.analysis_root/"results/dependent_review.csv",
            "tables/primary_prediction_metrics.csv":args.analysis_root/"identity_refit_metrics.csv",
            "tables/nominal_coverage.csv":args.analysis_root/"results/nominal_detailed.csv",
            "tables/risk_component_sensitivity.csv":args.analysis_root/"results/score_sensitivities.csv",
            "tables/local_interval_recalibration.csv":args.analysis_root/"results/fewshot_detailed.csv",
            "tables/bioactivity_feature_manifest.csv":args.bioactivity_manifest or ROOT/"data/bioactivity_feature_manifest.csv",
        })
        table_names = (
            "core_summary", "endpoint_summary", "ablation_summary",
            "score_detailed", "species_overlap", "partition_audit",
            "primary_quantile_sensitivity", "primary_endpoint_breadth",
            "component_association", "component_correlation",
            "expanded_external_aurc_summary", "expanded_external_capture_summary",
            "expanded_external_bootstrap_summary", "illustrative_disagreements",
        )
        entries.update({
            f"tables/{name}.csv": args.analysis_root / "results" / f"{name}.csv"
            for name in table_names
        })
        entries.update({
            "audit/duplicate_audit.csv": args.analysis_root / "duplicate_audit.csv",
            "audit/curation_counts.json": args.analysis_root / "curation_counts.json",
            "tables/conditional_margin_summary.json":
                args.analysis_root / "results/conditional_margin_summary.json",
            "tables/external_chemical_level_summary.csv":
                args.analysis_root / "external_refits/external_chemical_level_burden_summary.csv",
        })
        for size in (20, 50):
            for name in (
                "comparisons_summary", "paired_summary", "selection_audit",
                "selection_frequency", "comparisons_all",
            ):
                filename = f"external_deployment_{name}.csv"
                entries[f"tables/development_{size}/{filename}"] = (
                    args.analysis_root / "results" / f"development_{size}" / filename
                )
    for name, path in entries.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")

    manifest = {
        "release": args.version,
        "primary_benchmark": {"cases": 4519, "chemicals": 775},
        "external_panel": {"cases": 1027, "chemicals": 441},
        "files": [
            {"path": name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for name, path in entries.items()
        ],
    }
    readme = """# AquaTriage analysis data v0.4.0

This archive contains the identity-corrected 4519-case benchmark, the 1027-case
external panel, frozen calibration cutoffs, case-level predictions, and compact
tables for the review-allocation and fixed acute-hazard analyses.
Unconfirmed PubChem test-material mappings and generalized structures with
dummy atoms were removed before refitting. Split assignments, feature
definitions, source-reference groups, predictor-family comparisons and
conditional sensitivity results are included in the audit and tables folders.

Clone https://github.com/bubaizhanshen/AquaTriage, install its environment, and
run the fixed 1 mg/L analysis with:

```bash
python scripts/analyze_absolute_hazard_thresholds.py \\
  --benchmark data/strict_input_benchmark.csv \\
  --internal-predictions predictions/internal_predictions.csv \\
  --cutoffs data/calibration_cutoffs.csv \\
  --external-predictions predictions/external_predictions.csv \\
  --external-panel data/external_panel.csv \\
  --output-dir outputs/fixed_hazard
```

Run this command from the repository root after extracting the archive there,
or replace the five input paths with their extracted locations. The script
reports endpoint matching, 1/10/100 mg/L thresholds, chemical-level review
counts, same-queue random references, and paired chemical bootstrap intervals.
All outputs are derived from already-fitted predictions.
See `manifest.json` for file sizes and SHA-256 checksums.
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.output, "x", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in entries.items():
            archive.write(path, name)
        archive.writestr("README.md", readme)
        archive.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
