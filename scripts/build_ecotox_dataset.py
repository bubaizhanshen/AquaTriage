from __future__ import annotations

import argparse
import csv
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable
from urllib.parse import quote
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests
from rdkit import Chem

from ecoood.dsstox import expand_source_paths, parse_clowder_zip_listing, resolve_chemical_index_from_sources
from ecoood.invitrodb import INVITRODB_V43_BLOB_URL, attach_mechanistic_features, load_or_build_mechanistic_features
from ecoood.molecular_identity import fixed_composition_molecule


ECOTOX_MEMBERS = {
    "chemicals": "ecotox_ascii_03_12_2026/validation/chemicals.txt",
    "species": "ecotox_ascii_03_12_2026/validation/species.txt",
    "references": "ecotox_ascii_03_12_2026/validation/references.txt",
    "tests": "ecotox_ascii_03_12_2026/tests.txt",
    "results": "ecotox_ascii_03_12_2026/results.txt",
    "media": "ecotox_ascii_03_12_2026/media_characteristics.txt",
}

PUBCHEM_FIELDS = ["ConnectivitySMILES", "MolecularWeight", "XLogP", "InChIKey"]
DETERMINISTIC_REJECTION_GROUP_KEYWORDS = {
    "arsenic",
    "cadmium",
    "chromium",
    "copper",
    "lead",
    "major ions",
    "mercury",
    "metals",
    "nickel",
    "silver",
    "zinc",
}
CACHE_COLUMNS = [
    "cas_number",
    "dtxsid",
    "chemical_name",
    "query",
    "smiles",
    "molecular_weight",
    "logp",
    "inchikey",
    "resolved",
    "resolution_source",
    "identity_verified",
    "identity_evidence_url",
]


def read_zip_member(zip_path: Path, member: str, usecols: list[str]) -> pd.DataFrame:
    with ZipFile(zip_path) as zf, zf.open(member) as fh:
        return pd.read_csv(fh, sep="|", usecols=usecols, dtype=str)


def exposure_hours(value: object, unit: object) -> float | None:
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return None
    unit_text = "" if pd.isna(unit) else str(unit).strip().lower()
    if unit_text in {"h", "hr", "hrs"}:
        return duration
    if unit_text in {"d", "day", "days"}:
        return duration * 24.0
    if unit_text in {"wk", "wks", "week", "weeks"}:
        return duration * 24.0 * 7.0
    if unit_text == "min":
        return duration / 60.0
    return None


def clean_concentration_unit(unit: object) -> str:
    if pd.isna(unit):
        return ""
    text = str(unit).strip().lower().replace("µ", "u").replace("μ", "u").replace("³", "3")
    text = re.sub(r"^ai\s+", "", text)
    return re.sub(r"\s+", "", text)


def concentration_to_molar(value: object, unit: object, molecular_weight: object) -> float | None:
    try:
        concentration = float(value)
    except (TypeError, ValueError):
        return None
    unit_text = clean_concentration_unit(unit)
    if unit_text in {"m", "mol/l", "mol/dm3"}:
        return concentration if concentration > 0 else None
    if unit_text in {"mm", "mmol/l", "mmol/dm3"}:
        return concentration * 1e-3 if concentration > 0 else None
    if unit_text in {"um", "umol/l", "umol/dm3"}:
        return concentration * 1e-6 if concentration > 0 else None
    if unit_text in {"nm", "nmol/l", "nmol/dm3"}:
        return concentration * 1e-9 if concentration > 0 else None
    if unit_text in {"pm", "pmol/l", "pmol/dm3"}:
        return concentration * 1e-12 if concentration > 0 else None
    try:
        mw = float(molecular_weight)
    except (TypeError, ValueError):
        return None
    if mw <= 0:
        return None
    grams_per_liter: dict[str, float] = {
        "g/l": 1.0,
        "mg/ml": 1.0,
        "mg/l": 1e-3,
        "mg/dm3": 1e-3,
        "ug/ml": 1e-3,
        "ug/l": 1e-6,
        "ug/dm3": 1e-6,
        "ng/ml": 1e-6,
        "ng/l": 1e-9,
        "pg/ml": 1e-9,
        "pg/l": 1e-12,
        "ppm": 1e-3,
        "ppb": 1e-6,
        "ppt": 1e-9,
    }
    factor = grams_per_liter.get(unit_text)
    if factor is None:
        return None
    if concentration <= 0:
        return None
    return concentration * factor / mw


def endpoint_group(row: pd.Series) -> str | None:
    duration_h = row.get("duration_h")
    endpoint = row.get("endpoint", "")
    tax_group = str(row.get("species_group", "")).lower()
    family = str(row.get("family", ""))
    if endpoint == "LC50" and "fish" in tax_group and duration_h is not None and 90 <= duration_h <= 102:
        return "fish_96h_lc50"
    if endpoint == "EC50" and family == "Daphniidae" and duration_h is not None and 42 <= duration_h <= 54:
        return "daphnia_48h_ec50"
    if endpoint == "EC50" and "algae" in tax_group and duration_h is not None and 66 <= duration_h <= 102:
        return "algae_72_96h_ec50"
    return None


def quality_filter(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()
    filtered = filtered[filtered["endpoint"].isin(["LC50", "EC50"])].copy()
    filtered["duration_h"] = [exposure_hours(v, u) for v, u in zip(filtered["exposure_duration_mean"], filtered["exposure_duration_unit"])]
    filtered["conc1_mean_op"] = filtered["conc1_mean_op"].fillna("")
    filtered = filtered[filtered["conc1_mean_op"].isin(["", "=", "NR"])].copy()
    filtered["species_group"] = filtered["species_group"].fillna("")
    filtered["endpoint"] = filtered["endpoint"].fillna("")
    filtered["endpoint_group"] = filtered.apply(endpoint_group, axis=1)
    filtered = filtered[filtered["endpoint_group"].notna()].copy()
    freshwater_codes = {"FW", "FW/", "AQU", "AQU/", "CUL", "CUL/"}
    filtered = filtered[filtered["media_type"].fillna("").isin(freshwater_codes)].copy()
    return filtered


def pubchem_url(identifier: str) -> str:
    digits = str(identifier).strip().replace("-", "")
    if digits.isdigit() and 5 <= len(digits) <= 10:
        identifier = f"{digits[:-3]}-{digits[-3:-1]}-{digits[-1]}"
    fields = ",".join(PUBCHEM_FIELDS)
    return f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(identifier)}/property/{fields}/CSV"


def fetch_pubchem(identifier: str, chemical_name: str, timeout: int = 30) -> dict[str, object]:
    session = requests.Session()
    session.headers["User-Agent"] = "EcoOOD/0.1"
    candidates = [identifier] if identifier else []
    if chemical_name and chemical_name not in candidates:
        candidates.append(chemical_name)
    for candidate in candidates:
        if not candidate:
            continue
        try:
            response = session.get(pubchem_url(candidate), timeout=timeout)
            if response.status_code != 200:
                continue
            lines = list(csv.DictReader(response.text.splitlines()))
            if not lines:
                continue
            # A name or CAS lookup can include a parent ion, several salts, or
            # different isomers. Row order is not an identity-resolution rule.
            keys = {row.get("InChIKey", "") for row in lines}
            if len(keys) != 1 or "" in keys:
                continue
            payload = lines[0]
            return {
                "cas_number": identifier,
                "dtxsid": "",
                "chemical_name": chemical_name,
                "query": candidate,
                "smiles": payload.get("ConnectivitySMILES", ""),
                "molecular_weight": payload.get("MolecularWeight", ""),
                "logp": payload.get("XLogP", ""),
                "inchikey": payload.get("InChIKey", ""),
                "resolved": True,
                "resolution_source": "pubchem",
                "identity_verified": False,
                "identity_evidence_url": "",
            }
        except requests.RequestException:
            time.sleep(0.2)
            continue
    return {
        "cas_number": identifier,
        "dtxsid": "",
        "chemical_name": chemical_name,
        "query": identifier or chemical_name,
        "smiles": "",
        "molecular_weight": "",
        "logp": "",
        "inchikey": "",
        "resolved": False,
        "resolution_source": "",
        "identity_verified": False,
        "identity_evidence_url": "",
    }


def load_cache(cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        cache = pd.read_csv(cache_path, dtype=str)
        legacy_renames = {
            "pubchem_molecular_weight": "molecular_weight",
            "physchem_logp": "logp",
        }
        cache = cache.rename(columns={k: v for k, v in legacy_renames.items() if k in cache.columns})
        for column in CACHE_COLUMNS:
            if column not in cache.columns:
                cache[column] = ""
        resolved_mask = cache["resolved"].fillna("").astype(str).str.lower() == "true"
        blank_source = cache["resolution_source"].fillna("").astype(str).str.strip() == ""
        cache.loc[resolved_mask & blank_source, "resolution_source"] = "pubchem"
        return cache[CACHE_COLUMNS].copy()
    return pd.DataFrame(columns=CACHE_COLUMNS)


def enrich_structures(
    chemicals: pd.DataFrame,
    cache_path: Path,
    dsstox_sources: list[str | Path] | None = None,
    max_workers: int = 6,
) -> pd.DataFrame:
    cache = load_cache(cache_path)
    if dsstox_sources:
        resolved_paths = expand_source_paths(dsstox_sources)
        if resolved_paths:
            dsstox_matches = resolve_chemical_index_from_sources(chemicals, resolved_paths)
            if not dsstox_matches.empty:
                cache = pd.concat([cache, dsstox_matches.reindex(columns=CACHE_COLUMNS)], ignore_index=True)
                cache = cache.drop_duplicates("cas_number", keep="last")

    resolved_cache = cache[cache["resolved"].fillna("").astype(str).str.lower() == "true"].copy()
    todo = chemicals[~chemicals["cas_number"].isin(set(resolved_cache["cas_number"]))].copy()
    rows: list[dict[str, object]] = []
    if not todo.empty:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(fetch_pubchem, row["cas_number"], row["chemical_name"]): row["cas_number"]
                for _, row in todo.iterrows()
            }
            for idx, future in enumerate(as_completed(future_map), start=1):
                rows.append(future.result())
                if idx % 100 == 0:
                    print(f"Resolved {idx}/{len(todo)} chemical identifiers")
        cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True)
        cache = cache.drop_duplicates("cas_number", keep="last")
        cache.to_csv(cache_path, index=False)
    else:
        cache = cache.drop_duplicates("cas_number", keep="last")
        cache.to_csv(cache_path, index=False)
    return cache[CACHE_COLUMNS].drop_duplicates("cas_number", keep="last")


def _is_single_molecule_inner_salt(name: str, molecule: Chem.Mol) -> bool:
    if not re.search(r"\binner[\s-]+salt\b", name, flags=re.IGNORECASE):
        return False
    charges = [atom.GetFormalCharge() for atom in molecule.GetAtoms()]
    return (
        len(Chem.GetMolFrags(molecule)) == 1
        and sum(charges) == 0
        and any(charge > 0 for charge in charges)
        and any(charge < 0 for charge in charges)
    )


def deterministic_rejection_flag(row: pd.Series) -> bool:
    source = str(row.get("structure_source", row.get("resolution_source", ""))).lower()
    if source == "pubchem":
        verified = str(row.get("identity_verified", "")).strip().lower()
        evidence = str(row.get("identity_evidence_url", "")).strip()
        if verified not in {"true", "1"} or not evidence.startswith(("https://", "http://")):
            return True
    group = str(row.get("chemical_class", "")).strip().lower()
    name = str(row.get("chemical_name", "")).lower()
    if any(keyword in group for keyword in DETERMINISTIC_REJECTION_GROUP_KEYWORDS):
        return True
    if any(token in name for token in ["mixture", "unknown", "inorganic", "organomet"]):
        return True
    smiles = str(row.get("smiles", "")).strip()
    if not smiles:
        return True
    molecule = fixed_composition_molecule(smiles)
    if molecule is None:
        return True
    # An inner salt is one zwitterionic molecule, not a separate counterion salt.
    if "salt" in name and not _is_single_molecule_inner_salt(name, molecule):
        return True
    return False


def build_dataset(
    zip_path: Path,
    cache_path: Path,
    mechanism_cache_path: Path | None = None,
    invitrodb_summary_path: Path | None = None,
    invitrodb_blob_url: str = INVITRODB_V43_BLOB_URL,
    dsstox_sources: list[str | Path] | None = None,
    max_chemicals: int | None = None,
    max_workers: int = 6,
) -> pd.DataFrame:
    chemicals = read_zip_member(
        zip_path,
        ECOTOX_MEMBERS["chemicals"],
        ["cas_number", "chemical_name", "ecotox_group", "dtxsid"],
    ).rename(columns={"ecotox_group": "chemical_class"})
    species = read_zip_member(
        zip_path,
        ECOTOX_MEMBERS["species"],
        [
            "species_number",
            "latin_name",
            "common_name",
            "phylum_division",
            "class",
            "tax_order",
            "family",
            "genus",
            "species",
            "ecotox_group",
        ],
    ).rename(
        columns={
            "latin_name": "species_name",
            "ecotox_group": "species_group",
            "phylum_division": "phylum",
            "tax_order": "order",
        }
    )
    references = read_zip_member(
        zip_path,
        ECOTOX_MEMBERS["references"],
        ["reference_number", "publication_year", "doi"],
    ).rename(columns={"publication_year": "study_year"})
    tests = read_zip_member(
        zip_path,
        ECOTOX_MEMBERS["tests"],
        [
            "test_id",
            "reference_number",
            "test_cas",
            "species_number",
            "exposure_duration_mean",
            "exposure_duration_unit",
            "media_type",
            "study_type",
            "test_location",
            "exposure_type",
        ],
    )
    results = read_zip_member(
        zip_path,
        ECOTOX_MEMBERS["results"],
        [
            "result_id",
            "test_id",
            "endpoint",
            "effect",
            "measurement",
            "conc1_mean_op",
            "conc1_mean",
            "conc1_unit",
            "conc1_type",
            "endpoint_assigned",
        ],
    )
    media = read_zip_member(
        zip_path,
        ECOTOX_MEMBERS["media"],
        [
            "result_id",
            "media_ph_mean",
            "media_temperature_mean",
            "media_hardness_mean",
        ],
    )

    merged = (
        results.merge(tests, on="test_id", how="left")
        .merge(species, on="species_number", how="left")
        .merge(chemicals, left_on="test_cas", right_on="cas_number", how="left")
        .merge(references, on="reference_number", how="left")
        .merge(media, on="result_id", how="left")
    )
    filtered = quality_filter(merged)

    chemical_index = (
        filtered[["cas_number", "chemical_name", "dtxsid"]]
        .dropna(subset=["cas_number"])
        .drop_duplicates()
        .sort_values("cas_number")
    )
    if max_chemicals is not None:
        chemical_index = chemical_index.head(max_chemicals)
        filtered = filtered[filtered["cas_number"].isin(set(chemical_index["cas_number"]))].copy()
    structure_cache = enrich_structures(
        chemical_index,
        cache_path=cache_path,
        dsstox_sources=dsstox_sources,
        max_workers=max_workers,
    )
    structure_fields = [column for column in CACHE_COLUMNS if column != "chemical_name"]
    dataset = filtered.merge(structure_cache[structure_fields], on="cas_number", how="left")
    dtxsid_primary = dataset["dtxsid_x"] if "dtxsid_x" in dataset.columns else pd.Series("", index=dataset.index)
    dtxsid_resolved = dataset["dtxsid_y"] if "dtxsid_y" in dataset.columns else pd.Series("", index=dataset.index)
    dtxsid_primary = dtxsid_primary.fillna("").astype(str).str.strip()
    dtxsid_resolved = dtxsid_resolved.fillna("").astype(str).str.strip()
    dataset["dtxsid"] = dtxsid_primary.mask(dtxsid_primary.eq(""), dtxsid_resolved).replace("", np.nan)
    dataset = dataset.drop(columns=[column for column in ["dtxsid_x", "dtxsid_y"] if column in dataset.columns])

    mechanism_features = None
    if mechanism_cache_path is not None:
        mechanism_features = load_or_build_mechanistic_features(
            cache_path=mechanism_cache_path,
            local_archive=invitrodb_summary_path,
            remote_url=invitrodb_blob_url,
        )
        dataset = attach_mechanistic_features(dataset, mechanism_features, dtxsid_col="dtxsid", casrn_col="cas_number")

    dataset["physchem_mol_wt"] = pd.to_numeric(dataset["molecular_weight"], errors="coerce")
    dataset["physchem_logp"] = pd.to_numeric(dataset["logp"], errors="coerce")
    dataset["temperature_c"] = pd.to_numeric(dataset["media_temperature_mean"], errors="coerce")
    dataset["ph"] = pd.to_numeric(dataset["media_ph_mean"], errors="coerce")
    dataset["ctx_hardness"] = pd.to_numeric(dataset["media_hardness_mean"], errors="coerce")
    dataset["study_year"] = pd.to_numeric(dataset["study_year"], errors="coerce")
    dataset["toxicity_value"] = pd.to_numeric(dataset["conc1_mean"], errors="coerce")
    dataset["toxicity_unit"] = dataset["conc1_unit"]
    dataset["molar_concentration"] = pd.to_numeric([
        concentration_to_molar(v, u, mw)
        for v, u, mw in zip(dataset["toxicity_value"], dataset["toxicity_unit"], dataset["physchem_mol_wt"])
    ], errors="coerce")
    dataset["target_log_molar"] = np.where(dataset["molar_concentration"] > 0, np.log10(dataset["molar_concentration"]), np.nan)
    dataset["chemical_id"] = dataset["dtxsid"].fillna(dataset["cas_number"])
    dataset["source"] = "ecotox"
    dataset["structure_source"] = dataset["resolution_source"].fillna("").replace("", "unresolved")
    dataset["medium"] = dataset["media_type"]
    dataset["species"] = dataset["species_name"]
    dataset["class_name"] = dataset["class"]
    dataset["chemical_class"] = dataset["chemical_class"].fillna("unclassified")
    dataset["species_group"] = dataset["species_group"].fillna("unknown")
    # The persisted column name is retained for compatibility with the frozen
    # analysis table; it is an audit flag and is never a scoreable split.
    dataset["is_hard_ood"] = dataset.apply(deterministic_rejection_flag, axis=1)
    dataset["known_ood"] = dataset["is_hard_ood"]
    mech_feature_cols = sorted(column for column in dataset.columns if column.startswith("mech_"))

    keep = [
        "chemical_id",
        "chemical_name",
        "cas_number",
        "dtxsid",
        "smiles",
        "inchikey",
        "endpoint",
        "endpoint_code",
        "effect",
        "measurement",
        "species",
        "common_name",
        "phylum",
        "class_name",
        "order",
        "family",
        "genus",
        "species_group",
        "duration_h",
        "medium",
        "temperature_c",
        "ph",
        "ctx_hardness",
        "study_year",
        "doi",
        "source",
        "structure_source",
        "identity_verified",
        "identity_evidence_url",
        "chemical_class",
        "toxicity_value",
        "toxicity_unit",
        "molar_concentration",
        "target_log_molar",
        "physchem_mol_wt",
        "physchem_logp",
        *mech_feature_cols,
        "is_hard_ood",
        "known_ood",
    ]
    dataset = dataset.rename(
        columns={
            "cas_number": "casrn",
            "endpoint": "endpoint_code",
            "endpoint_group": "endpoint",
        }
    )
    keep[2] = "casrn"
    dataset = dataset[keep].copy()
    dataset = dataset.replace({np.inf: np.nan, -np.inf: np.nan})
    return dataset


def summarise(df: pd.DataFrame) -> None:
    print("Rows:", len(df))
    print("Unique chemicals:", df["chemical_id"].nunique())
    print("Rows with resolved structure:", int(df["smiles"].fillna("").ne("").sum()))
    print("Rows with molar target:", int(df["target_log_molar"].notna().sum()))
    print(df["endpoint"].value_counts(dropna=False).to_string())
    if "structure_source" in df.columns:
        print(df["structure_source"].value_counts(dropna=False).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an EcoOOD-ready ECOTOX acute ecotoxicity dataset.")
    parser.add_argument("--ecotox-zip", type=Path, default=Path("data/raw/ecotox_ascii_03_12_2026.zip"))
    parser.add_argument("--structure-cache", "--pubchem-cache", dest="structure_cache", type=Path, default=Path("data/raw/pubchem_cache.csv"))
    parser.add_argument(
        "--dsstox-source",
        action="append",
        default=[],
        help="Path, directory, or glob for extracted DSSTox CSV files. May be provided multiple times.",
    )
    parser.add_argument(
        "--dsstox-clowder-page",
        type=Path,
        default=None,
        help="Optional saved Clowder file page HTML. If provided, zip inventory is summarized before build.",
    )
    parser.add_argument(
        "--mechanism-cache",
        type=Path,
        default=Path("data/processed/invitrodb_mechanism_features.csv"),
        help="CSV cache for invitrodb-derived bioactivity-proxy features.",
    )
    parser.add_argument(
        "--invitrodb-summary",
        type=Path,
        default=Path("data/raw/INVITRODB_SUMMARY.zip"),
        help="Path to the official invitrodb summary zip or a prefix download that contains the local members.",
    )
    parser.add_argument(
        "--invitrodb-blob-url",
        type=str,
        default=INVITRODB_V43_BLOB_URL,
        help="Clowder blob URL for byte-range access to invitrodb summary members.",
    )
    parser.add_argument(
        "--skip-mechanism",
        action="store_true",
        help="Skip invitrodb-derived bioactivity-proxy features.",
    )
    parser.add_argument("--output", type=Path, default=Path("data/processed/ecotox_acute_ecoood.csv"))
    parser.add_argument("--structured-output", type=Path, default=Path("data/processed/ecotox_acute_ecoood_structured.csv"))
    parser.add_argument("--max-chemicals", type=int, default=None, help="Optional cap for quick pilot runs.")
    parser.add_argument("--max-workers", type=int, default=6)
    args = parser.parse_args()

    if args.dsstox_clowder_page is not None and args.dsstox_clowder_page.exists():
        inventory = parse_clowder_zip_listing(args.dsstox_clowder_page.read_text())
        if not inventory.empty:
            print("DSSTox zip inventory preview:")
            print(inventory.head(20).to_string(index=False))
            print(f"Total listed members: {len(inventory)}")

    dataset = build_dataset(
        args.ecotox_zip,
        cache_path=args.structure_cache,
        mechanism_cache_path=None if args.skip_mechanism else args.mechanism_cache,
        invitrodb_summary_path=args.invitrodb_summary,
        invitrodb_blob_url=args.invitrodb_blob_url,
        dsstox_sources=args.dsstox_source,
        max_chemicals=args.max_chemicals,
        max_workers=args.max_workers,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)
    structured = dataset[dataset["target_log_molar"].notna() & ~dataset["is_hard_ood"]].copy()
    structured.to_csv(args.structured_output, index=False)
    summarise(dataset)
    print(f"Wrote full dataset to {args.output}")
    print(f"Wrote structured training subset to {args.structured_output}")


if __name__ == "__main__":
    main()
