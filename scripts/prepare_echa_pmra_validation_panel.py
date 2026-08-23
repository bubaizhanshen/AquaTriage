from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.build_ecotox_dataset import concentration_to_molar, enrich_structures


DEFAULT_INPUT = ROOT / "results" / "external_regulatory_prep" / "echa_pmra_external_full" / "echa_pmra_exact_rows.csv"
DEFAULT_OUTPUT = ROOT / "results" / "external_regulatory_prep" / "echa_pmra_external_full" / "echa_pmra_exact_rows_enriched.csv"
DSSTOX_SOURCE = ROOT / "data" / "raw" / "DSSTox_CCD_dump_12092025_CSVs.zip"
STRUCTURE_CACHE = ROOT / "data" / "raw" / "pubchem_cache_external_opp.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve structures and molar values for the ECHA PMRA exact-row external panel."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dsstox-source", type=Path, default=DSSTOX_SOURCE)
    parser.add_argument("--structure-cache", type=Path, default=STRUCTURE_CACHE)
    parser.add_argument("--max-workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panel = pd.read_csv(args.input)
    args.structure_cache.parent.mkdir(parents=True, exist_ok=True)
    chemical_index = panel[["chemical_name", "casrn"]].drop_duplicates().copy()
    chemical_index = chemical_index.rename(columns={"casrn": "cas_number"})
    chemical_index["dtxsid"] = ""

    resolved = enrich_structures(
        chemical_index[["cas_number", "chemical_name", "dtxsid"]],
        cache_path=args.structure_cache,
        dsstox_sources=[args.dsstox_source],
        max_workers=args.max_workers,
    ).rename(columns={"cas_number": "casrn"})

    enriched = panel.merge(
        resolved,
        on=["chemical_name", "casrn"],
        how="left",
        suffixes=("", "_resolved"),
    )
    enriched["molecular_weight"] = pd.to_numeric(enriched["molecular_weight"], errors="coerce")
    enriched["source_value_molar"] = [
        concentration_to_molar(value, unit, mw)
        if pd.notna(value) and pd.notna(mw)
        else np.nan
        for value, unit, mw in zip(
            enriched["source_value"],
            enriched["source_unit"],
            enriched["molecular_weight"],
        )
    ]
    enriched["source_log_molar"] = [
        np.log10(value) if pd.notna(value) and value > 0 else np.nan
        for value in enriched["source_value_molar"]
    ]
    enriched["chemical_id"] = enriched["dtxsid"].fillna("").astype(str).str.strip()
    empty = enriched["chemical_id"].eq("")
    enriched.loc[empty, "chemical_id"] = enriched.loc[empty, "casrn"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(args.output, index=False)
    print(f"Rows written: {len(enriched)}")
    print(f"Unique chemicals: {enriched['casrn'].nunique()}")
    print(f"Resolved SMILES: {int(enriched['smiles'].notna().sum())}")


if __name__ == "__main__":
    main()
