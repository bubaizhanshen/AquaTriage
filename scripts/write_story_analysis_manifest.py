#!/usr/bin/env python3
"""Write a provenance manifest for the story-plan analysis outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import rdkit


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal-data", type=Path, required=True)
    parser.add_argument("--internal-predictions", type=Path, required=True)
    parser.add_argument("--external-train", type=Path, required=True)
    parser.add_argument("--external-panel", type=Path, required=True)
    parser.add_argument("--external-predictions", type=Path, required=True)
    parser.add_argument("--cutoffs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    internal = pd.read_csv(args.internal_data, low_memory=False)
    external_train = pd.read_csv(args.external_train, low_memory=False)
    external_panel = pd.read_csv(args.external_panel, low_memory=False)
    payload = {
        "created": "2026-09-20",
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "rdkit_version": rdkit.__version__,
        "internal": {
            "data_path": str(args.internal_data),
            "data_sha256": sha256(args.internal_data),
            "n_rows": int(len(internal)),
            "n_chemicals": int(internal["chemical_id"].nunique()),
            "prediction_path": str(args.internal_predictions),
            "cutoff_path": str(args.cutoffs),
            "cutoff_sha256": sha256(args.cutoffs),
            "prediction_sha256": sha256(args.internal_predictions),
            "configuration_label": "4611_primary_strict_rdkit2024032",
        },
        "external": {
            "training_path": str(args.external_train),
            "training_sha256": sha256(args.external_train),
            "training_rows": int(len(external_train)),
            "training_chemicals": int(external_train["chemical_id"].nunique()),
            "panel_path": str(args.external_panel),
            "panel_sha256": sha256(args.external_panel),
            "panel_rows": int(len(external_panel)),
            "panel_chemicals": int(external_panel["chemical_id"].nunique()),
            "prediction_path": str(args.external_predictions),
            "prediction_sha256": sha256(args.external_predictions),
            "configuration_label": "4611_strict_external_run",
        },
        "interpretation": (
            "The 4611-case strict-input configuration is the sole primary benchmark for internal "
            "and external analyses. The 4942-case structured count is retained only as a historical "
            "record-level curation audit before the strict molecular-input gate; it is not a second "
            "prediction benchmark. Internal and external predictions remain separate evaluation "
            "sets, and no external case enters predictor or reliability fitting."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
