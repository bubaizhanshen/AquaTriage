from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from ecoood.features import attach_rdkit_descriptors
from ecoood.invitrodb import attach_mechanistic_features, load_or_build_mechanistic_features
from ecoood.schema import DEFAULT_SCHEMA

RDLogger.DisableLog("rdApp.*")


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "processed" / "ecotox_acute_ecoood_1000chem_dsstox_mech_structured.csv"
INPUT_PATH = (
    ROOT
    / "results"
    / "external_regulatory_prep"
    / "echa_pmra_external_full"
    / "echa_pmra_exact_rows_enriched.csv"
)
OUT_DIR = ROOT / "results" / "external_regulatory_prep" / "echa_pmra_external_clean"
MECHANISM_CACHE = ROOT / "data" / "processed" / "invitrodb_mechanism_features.csv"
INVITRODB_SUMMARY = ROOT / "data" / "raw" / "INVITRODB_SUMMARY.zip"

STANDARD_SPECIES = {
    "Oncorhynchus mykiss",
    "Pimephales promelas",
    "Lepomis macrochirus",
    "Cyprinus carpio",
    "Daphnia magna",
    "Raphidocelis subcapitata",
    "Desmodesmus subspicatus",
}
MARINE_SPECIES = {
    "Skeletonema costatum",
    "Cyprinodon variegatus",
    "Mysidopsis bahia",
    "Americamysis bahia",
    "Menidia beryllina",
}
MARINE_PATTERN = re.compile(
    r"seawater|marine|salinity|estuar|skeletonema|mysid|sheepshead minnow|saltwater|brackish|cyprinodon variegatus",
    flags=re.IGNORECASE,
)
FRESHWATER_PATTERN = re.compile(
    r"freshwater|de-chlorinated freshwater|tap water|drinking water",
    flags=re.IGNORECASE,
)
VARIABLE_COMPOSITION_PATTERN = re.compile(
    r"\breaction (?:mass|products?)\b|\bmixtures?\b|\buvcb\b|\bmixed\b",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build clean case-level ECHA external panels for EcoOOD validation."
    )
    parser.add_argument("--input-path", type=Path, default=INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--source-prefix", default="echa_pmra")
    parser.add_argument("--train-path", type=Path, default=TRAIN_PATH)
    parser.add_argument("--mechanism-cache", type=Path, default=MECHANISM_CACHE)
    parser.add_argument("--invitrodb-summary", type=Path, default=INVITRODB_SUMMARY)
    return parser.parse_args()


def canonical_species(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.strip()
    text = re.sub(r"\s*\(.*?\)", "", text)
    text = re.sub(r"^other[^:]*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def mode_or_first(series: pd.Series) -> object:
    clean = series.dropna()
    if clean.empty:
        return np.nan
    mode = clean.mode()
    return mode.iloc[0] if not mode.empty else clean.iloc[0]


def join_unique(series: pd.Series) -> str:
    values = sorted(
        {
            str(value).strip()
            for value in series.dropna()
            if str(value).strip()
        }
    )
    return "; ".join(values)


def is_acceptable_study_record(value: object) -> bool:
    label = "" if pd.isna(value) else str(value).lower()
    return (
        "experimental study" in label
        and "disregarded" not in label
        and any(
            term in label
            for term in ("key", "supporting", "weight of evidence")
        )
    )


def has_usable_molecular_representation(value: object) -> bool:
    smiles = "" if pd.isna(value) else str(value).strip()
    if not smiles:
        return False
    molecule = Chem.MolFromSmiles(smiles)
    return molecule is not None and any(
        atom.GetAtomicNum() == 6 for atom in molecule.GetAtoms()
    )


def has_resolved_molecular_identity(row: pd.Series) -> bool:
    """Reject variable-composition substances represented by a surrogate structure."""
    name = "" if pd.isna(row.get("chemical_name")) else str(row["chemical_name"]).strip()
    smiles = "" if pd.isna(row.get("smiles")) else str(row["smiles"]).strip()
    inchikey = "" if pd.isna(row.get("inchikey")) else str(row["inchikey"]).strip()
    return bool(inchikey) and "|" not in smiles and not VARIABLE_COMPOSITION_PATTERN.search(name)


def build_species_meta(train_df: pd.DataFrame) -> pd.DataFrame:
    train = train_df.copy()
    train["species_canon"] = train["species"].map(canonical_species)
    keep_cols = [
        "endpoint",
        "species_canon",
        "species",
        "common_name",
        "phylum",
        "class_name",
        "order",
        "family",
        "genus",
        "species_group",
        "medium",
        "effect",
        "measurement",
        "endpoint_code",
        "temperature_c",
        "ph",
        "ctx_hardness",
    ]
    meta = (
        train[keep_cols]
        .groupby(["endpoint", "species_canon"], dropna=False, as_index=False)
        .agg(
            species=("species", mode_or_first),
            common_name=("common_name", mode_or_first),
            phylum=("phylum", mode_or_first),
            class_name=("class_name", mode_or_first),
            order=("order", mode_or_first),
            family=("family", mode_or_first),
            genus=("genus", mode_or_first),
            species_group=("species_group", mode_or_first),
            medium=("medium", mode_or_first),
            effect=("effect", mode_or_first),
            measurement=("measurement", mode_or_first),
            endpoint_code=("endpoint_code", mode_or_first),
            temperature_c=("temperature_c", "median"),
            ph=("ph", "median"),
            ctx_hardness=("ctx_hardness", "median"),
        )
    )
    return meta


def attach_row_flags(rows: pd.DataFrame, species_meta: pd.DataFrame) -> pd.DataFrame:
    flagged = rows.copy()
    flagged["species_canon"] = flagged["regulatory_species"].map(canonical_species)
    flagged["source_log_molar"] = pd.to_numeric(flagged["source_log_molar"], errors="coerce")
    flagged["source_value_molar"] = pd.to_numeric(flagged["source_value_molar"], errors="coerce")
    flagged["molecular_weight"] = pd.to_numeric(flagged["molecular_weight"], errors="coerce")
    text_cols = [
        "details_on_test_conditions",
        "remarks_on_result",
        "nominal_measured_context",
        "regulatory_taxon",
        "regulatory_species",
    ]
    flagged["filter_text"] = flagged[text_cols].fillna("").agg(" ".join, axis=1)
    flagged["marine_keyword_flag"] = flagged["filter_text"].str.contains(MARINE_PATTERN, na=False)
    flagged["freshwater_keyword_flag"] = flagged["filter_text"].str.contains(FRESHWATER_PATTERN, na=False)
    flagged["marine_species_flag"] = flagged["species_canon"].isin(MARINE_SPECIES)
    flagged["standard_species_flag"] = flagged["species_canon"].isin(STANDARD_SPECIES)
    flagged["usable_molecular_representation_flag"] = flagged["smiles"].map(
        has_usable_molecular_representation
    )
    flagged["resolved_molecular_identity_flag"] = flagged.apply(
        has_resolved_molecular_identity,
        axis=1,
    )
    flagged["acceptable_study_record_flag"] = flagged["document_label"].map(
        is_acceptable_study_record
    )

    meta_keys = set(zip(species_meta["endpoint"], species_meta["species_canon"]))
    flagged["has_train_species_meta"] = flagged.apply(
        lambda rec: (rec["target_endpoint"], rec["species_canon"]) in meta_keys,
        axis=1,
    )
    flagged["include_main_row"] = (
        flagged["source_log_molar"].notna()
        & flagged["usable_molecular_representation_flag"]
        & flagged["resolved_molecular_identity_flag"]
        & flagged["acceptable_study_record_flag"]
        & flagged["has_train_species_meta"]
        & ~flagged["marine_species_flag"]
        & ~flagged["marine_keyword_flag"]
    )
    flagged["include_strict_row"] = flagged["include_main_row"] & flagged["standard_species_flag"]
    return flagged


def aggregate_case_panel(
    rows: pd.DataFrame,
    species_meta: pd.DataFrame,
    *,
    panel_name: str,
    source_prefix: str,
) -> pd.DataFrame:
    group_cols = [
        "chemical_id",
        "chemical_name",
        "casrn",
        "dtxsid",
        "smiles",
        "inchikey",
        "resolution_source",
        "molecular_weight",
        "logp",
        "target_endpoint",
        "species_canon",
    ]
    grouped = (
        rows.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            case_row_count=("target_endpoint", "size"),
            document_count=("document_key", "nunique"),
            asset_ids=("asset_id", join_unique),
            document_keys=("document_key", join_unique),
            document_urls=("document_url", join_unique),
            source_log_molar=("source_log_molar", "median"),
            source_log_molar_min=("source_log_molar", "min"),
            source_log_molar_max=("source_log_molar", "max"),
            source_value_molar=("source_value_molar", "median"),
            duration_h=("duration_h_used", "median"),
            regulatory_species=("regulatory_species", mode_or_first),
            regulatory_taxon=("regulatory_taxon", mode_or_first),
            study_type=("study_type", mode_or_first),
            nominal_or_measured=("nominal_or_measured", mode_or_first),
            conc_based_on=("conc_based_on", mode_or_first),
            basis_for_effect=("basis_for_effect", mode_or_first),
            freshwater_keyword_flag=("freshwater_keyword_flag", "max"),
            standard_species_flag=("standard_species_flag", "max"),
        )
    )
    grouped["case_spread_log_molar"] = grouped["source_log_molar_max"] - grouped["source_log_molar_min"]

    meta = species_meta.rename(columns={"endpoint": "target_endpoint"})
    cases = grouped.merge(meta, on=["target_endpoint", "species_canon"], how="left")
    cases["endpoint"] = cases["target_endpoint"]
    cases["chemical_class"] = "unclassified"
    cases["structure_source"] = cases["resolution_source"].fillna("").replace("", "unresolved")
    cases["physchem_mol_wt"] = pd.to_numeric(cases["molecular_weight"], errors="coerce")
    cases["physchem_logp"] = pd.to_numeric(cases["logp"], errors="coerce")
    cases["target_log_molar"] = cases["source_log_molar"]
    # The modeled case target is the median on the log10 molar scale. Convert
    # that aggregate back to molar concentration so all released target columns
    # describe the same value, including groups with an even number of records.
    cases["molar_concentration"] = np.power(10.0, cases["target_log_molar"])
    cases["toxicity_value"] = cases["molar_concentration"]
    cases["toxicity_unit"] = "M"
    cases["study_year"] = np.nan
    cases["source"] = f"{source_prefix}_{panel_name}"
    cases["known_ood"] = False
    cases["is_hard_ood"] = False
    cases["case_id"] = (
        cases["chemical_id"].astype(str)
        + "__"
        + cases["endpoint"].astype(str)
        + "__"
        + cases["species"].astype(str)
    )
    return cases


def attach_mechanism(
    cases: pd.DataFrame,
    *,
    mechanism_cache: Path,
    invitrodb_summary: Path,
) -> pd.DataFrame:
    mechanism = load_or_build_mechanistic_features(
        cache_path=mechanism_cache,
        local_archive=invitrodb_summary,
    )
    working = cases.rename(columns={"casrn": "cas_number"}).copy()
    working["dtxsid"] = working["dtxsid"].fillna("").astype(str).str.strip()
    working["cas_number"] = working["cas_number"].fillna("").astype(str).str.strip()
    working = attach_mechanistic_features(
        working,
        mechanism,
        dtxsid_col="dtxsid",
        casrn_col="cas_number",
    )
    working = working.rename(columns={"cas_number": "casrn"})
    return attach_rdkit_descriptors(working, DEFAULT_SCHEMA)


def finalize_columns(cases: pd.DataFrame) -> pd.DataFrame:
    ordered = [
        "case_id",
        "chemical_id",
        "chemical_name",
        "casrn",
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
        "source",
        "structure_source",
        "chemical_class",
        "toxicity_value",
        "toxicity_unit",
        "molar_concentration",
        "target_log_molar",
        "physchem_mol_wt",
        "physchem_logp",
        "known_ood",
        "is_hard_ood",
        "case_row_count",
        "document_count",
        "asset_ids",
        "document_keys",
        "document_urls",
        "case_spread_log_molar",
        "species_canon",
        "regulatory_species",
        "regulatory_taxon",
        "study_type",
        "nominal_or_measured",
        "conc_based_on",
        "basis_for_effect",
        "freshwater_keyword_flag",
        "standard_species_flag",
    ]
    mech_cols = sorted(col for col in cases.columns if col.startswith("mech_"))
    keep_cols = [col for col in ordered if col in cases.columns] + mech_cols
    return cases[keep_cols].copy()


def write_summary(
    out_dir: Path,
    flagged_rows: pd.DataFrame,
    main_cases: pd.DataFrame,
    strict_cases: pd.DataFrame,
) -> None:
    lines = [
        "ECHA PMRA clean panel summary",
        f"Input exact rows: {len(flagged_rows)}",
        f"Rows with train-species metadata: {int(flagged_rows['has_train_species_meta'].sum())}",
        f"Rows with non-null target_log_molar: {int(flagged_rows['source_log_molar'].notna().sum())}",
        f"Rows with a usable carbon-containing molecular representation: {int(flagged_rows['usable_molecular_representation_flag'].sum())}",
        f"Rows with a resolved fixed-composition molecular identity: {int(flagged_rows['resolved_molecular_identity_flag'].sum())}",
        f"Rows with an accepted experimental-study label: {int(flagged_rows['acceptable_study_record_flag'].sum())}",
        f"Rows excluded by marine keyword/species flag: {int((flagged_rows['marine_keyword_flag'] | flagged_rows['marine_species_flag']).sum())}",
        f"Main panel rows retained: {int(flagged_rows['include_main_row'].sum())}",
        f"Strict panel rows retained: {int(flagged_rows['include_strict_row'].sum())}",
        "",
        f"Main panel cases: {len(main_cases)}",
        f"Main panel chemicals: {main_cases['chemical_name'].nunique()}",
        "Main panel endpoint counts:",
        main_cases["endpoint"].value_counts().to_string(),
        "",
        f"Strict panel cases: {len(strict_cases)}",
        f"Strict panel chemicals: {strict_cases['chemical_name'].nunique()}",
        "Strict panel endpoint counts:",
        strict_cases["endpoint"].value_counts().to_string(),
    ]
    (out_dir / "echa_pmra_clean_panel_summary.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_path)
    source_rows = pd.read_csv(args.input_path)
    species_meta = build_species_meta(train_df)
    flagged_rows = attach_row_flags(source_rows, species_meta)
    flagged_rows.to_csv(out_dir / "echa_pmra_row_filter_audit.csv", index=False)
    species_meta.to_csv(out_dir / "echa_pmra_species_meta_map.csv", index=False)

    main_rows = flagged_rows.loc[flagged_rows["include_main_row"]].copy()
    strict_rows = flagged_rows.loc[flagged_rows["include_strict_row"]].copy()

    main_cases = aggregate_case_panel(
        main_rows,
        species_meta,
        panel_name="main",
        source_prefix=args.source_prefix,
    )
    strict_cases = aggregate_case_panel(
        strict_rows,
        species_meta,
        panel_name="strict",
        source_prefix=args.source_prefix,
    )

    main_cases = finalize_columns(
        attach_mechanism(
            main_cases,
            mechanism_cache=args.mechanism_cache,
            invitrodb_summary=args.invitrodb_summary,
        )
    )
    strict_cases = finalize_columns(
        attach_mechanism(
            strict_cases,
            mechanism_cache=args.mechanism_cache,
            invitrodb_summary=args.invitrodb_summary,
        )
    )

    main_cases.to_csv(out_dir / "echa_pmra_case_panel_main.csv", index=False)
    strict_cases.to_csv(out_dir / "echa_pmra_case_panel_strict.csv", index=False)
    write_summary(out_dir, flagged_rows, main_cases, strict_cases)


if __name__ == "__main__":
    main()
