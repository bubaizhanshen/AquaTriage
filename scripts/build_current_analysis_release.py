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
    args = parser.parse_args()

    benchmark = pd.read_csv(args.benchmark, usecols=["chemical_id"])
    external = pd.read_csv(args.external_panel, usecols=["chemical_id", "case_id"])
    if (len(benchmark), benchmark.chemical_id.nunique()) != (4611, 801):
        raise ValueError("Primary benchmark must contain 4611 cases from 801 chemicals")
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
    for name, path in entries.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")

    manifest = {
        "release": "v0.3.0",
        "primary_benchmark": {"cases": 4611, "chemicals": 801},
        "external_panel": {"cases": 1027, "chemicals": 441},
        "files": [
            {"path": name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for name, path in entries.items()
        ],
    }
    readme = """# AquaTriage analysis data v0.3.0

This archive contains the strict-input 4611-case benchmark, the 1027-case
external panel, frozen calibration cutoffs, case-level predictions, and compact
tables for the review-allocation and fixed acute-hazard analyses.

Clone https://github.com/bubaizhanshen/EcoOOD, install its environment, and
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
