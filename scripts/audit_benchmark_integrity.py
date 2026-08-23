from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ecoood.features import EcoFeatureBuilder, attach_rdkit_descriptors, smiles_to_mol
from ecoood.schema import DEFAULT_SCHEMA


STRICT_REJECTION_GROUP_KEYWORDS = {
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
STRICT_REJECTION_NAME_TOKENS = ("mixture", "unknown", "inorganic", "organomet", "salt")


EXACT_RECORD_KEY = [
    DEFAULT_SCHEMA.chemical_id,
    DEFAULT_SCHEMA.species,
    DEFAULT_SCHEMA.endpoint,
    DEFAULT_SCHEMA.effect,
    "measurement",
    DEFAULT_SCHEMA.duration_h,
    DEFAULT_SCHEMA.medium,
    DEFAULT_SCHEMA.temperature_c,
    DEFAULT_SCHEMA.ph,
    "ctx_hardness",
    DEFAULT_SCHEMA.study_year,
    DEFAULT_SCHEMA.target,
]


def _feature_block(builder: EcoFeatureBuilder, column: str) -> tuple[str, bool]:
    if column == DEFAULT_SCHEMA.smiles:
        return "chemical fingerprint", True
    if column in builder.descriptor_cols:
        return "physicochemical descriptor", True
    if column in builder.mechanism_cols:
        return "bioactivity proxy", True
    if column in builder.context_cols or column in builder.context_categorical_cols:
        return "experimental context", True
    if column in builder.species_cols or column in builder.species_categorical_cols:
        return "species/taxonomy", True
    if column in builder.categorical_cols:
        return "categorical model input", True
    return "identifier, target, or audit-only field", False


def _source_or_derivation(column: str) -> str:
    """Describe the field's provenance in the public predictor manifest."""
    schema = DEFAULT_SCHEMA
    if column.startswith("mech_"):
        return "invitrodb/ToxCast summary linkage"
    if column in {"smiles", "dtxsid", "inchikey", "structure_source"}:
        return "DSSTox/CompTox structure linkage"
    if column == "physchem_logp":
        return "RDKit MolLogP structure derivation"
    if column == "physchem_mol_wt":
        return "DSSTox/CompTox or PubChem physicochemical linkage"
    if column.startswith("physchem_"):
        return "RDKit structure derivation"
    if column in {schema.hard_ood, schema.known_ood, "chemical_class"}:
        return "curation and audit derivation"
    if column in {
        schema.target,
        "molar_concentration",
        schema.value,
        schema.unit,
        "endpoint_code",
        "effect",
        "measurement",
    }:
        return "ECOTOX value/end-point curation"
    if column in {
        schema.chemical_id,
        schema.chemical_name,
        schema.casrn,
        "doi",
        "source",
        "common_name",
    }:
        return "ECOTOX record metadata"
    return "ECOTOX study/taxon context curation"


def build_feature_manifest(df: pd.DataFrame) -> pd.DataFrame:
    working = attach_rdkit_descriptors(df, DEFAULT_SCHEMA)
    builder = EcoFeatureBuilder(schema=DEFAULT_SCHEMA).fit(working)
    rows: list[dict[str, object]] = []
    for column in working.columns:
        block, included = _feature_block(builder, column)
        rows.append(
            {
                "field": column,
                "raw_dtype": str(working[column].dtype),
                "n_unique": int(working[column].nunique(dropna=True)),
                "missing_fraction": float(working[column].isna().mean()),
                "feature_block": block,
                "source_or_derivation": _source_or_derivation(column),
                "predictive_input": included,
            }
        )
    manifest = pd.DataFrame(rows)
    required_exclusions = {
        DEFAULT_SCHEMA.target,
        "molar_concentration",
        DEFAULT_SCHEMA.value,
        DEFAULT_SCHEMA.unit,
        DEFAULT_SCHEMA.chemical_id,
        DEFAULT_SCHEMA.casrn,
        DEFAULT_SCHEMA.chemical_name,
        "dtxsid",
        "inchikey",
        "doi",
    }
    leaked = manifest.loc[
        manifest["field"].isin(required_exclusions) & manifest["predictive_input"], "field"
    ].tolist()
    if leaked:
        raise RuntimeError(f"Target-derived or identifier fields entered the predictor matrix: {leaked}")
    return manifest


def duplicate_audit(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    exact_keys = EXACT_RECORD_KEY.copy()
    exact_keys = [column for column in exact_keys if column in df.columns]
    checks = {
        "exact_record_key": exact_keys,
        "same_chemical_species_endpoint_target": [
            column
            for column in [
                DEFAULT_SCHEMA.chemical_id,
                DEFAULT_SCHEMA.species,
                DEFAULT_SCHEMA.endpoint,
                DEFAULT_SCHEMA.target,
            ]
            if column in df.columns
        ],
        "same_structure_species_endpoint": [
            column
            for column in ["inchikey", DEFAULT_SCHEMA.species, DEFAULT_SCHEMA.endpoint]
            if column in df.columns
        ],
    }
    rows: list[dict[str, object]] = []
    summary: dict[str, int] = {}
    for label, keys in checks.items():
        group_sizes = df.groupby(keys, dropna=False).size()
        duplicate_groups = group_sizes[group_sizes > 1]
        duplicate_rows = int(duplicate_groups.sum())
        rows.append(
            {
                "check": label,
                "key_fields": "; ".join(keys),
                "total_rows": int(len(df)),
                "duplicate_groups": int(len(duplicate_groups)),
                "rows_in_duplicate_groups": duplicate_rows,
                "rows_beyond_first_record": int(duplicate_rows - len(duplicate_groups)),
            }
        )
        summary[label] = duplicate_rows
    return pd.DataFrame(rows), summary


def deduplicate_exact_records(df: pd.DataFrame) -> pd.DataFrame:
    keys = [column for column in EXACT_RECORD_KEY if column in df.columns]
    return df.drop_duplicates(keys, keep="first").reset_index(drop=True)


def structure_parse_audit(df: pd.DataFrame) -> pd.DataFrame:
    """List records whose resolved structure does not parse in reference RDKit."""
    parseable = df[DEFAULT_SCHEMA.smiles].map(lambda value: smiles_to_mol(value) is not None)
    columns = [
        column
        for column in (
            DEFAULT_SCHEMA.chemical_id,
            DEFAULT_SCHEMA.chemical_name,
            DEFAULT_SCHEMA.casrn,
            DEFAULT_SCHEMA.smiles,
            "structure_source",
        )
        if column in df.columns
    ]
    return df.loc[~parseable, columns].drop_duplicates().reset_index(drop=True)


def strict_input_eligibility_audit(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the finalized molecular-input gate and return retained and rejected rows."""

    def rejection_reason(row: pd.Series) -> str:
        chemical_class = str(row.get("chemical_class", "")).strip().lower()
        chemical_name = str(row.get(DEFAULT_SCHEMA.chemical_name, "")).strip().lower()
        if any(token in chemical_class for token in STRICT_REJECTION_GROUP_KEYWORDS):
            return "listed metal or inorganic class"
        if any(token in chemical_name for token in STRICT_REJECTION_NAME_TOKENS):
            return "predefined name token"
        molecule = smiles_to_mol(row.get(DEFAULT_SCHEMA.smiles))
        if molecule is None:
            return "missing or unparseable structure"
        if not any(atom.GetAtomicNum() == 6 for atom in molecule.GetAtoms()):
            return "carbon-free structure"
        return ""

    reasons = df.apply(rejection_reason, axis=1)
    rejected = df.loc[reasons.ne("")].copy()
    rejected.insert(0, "input_eligibility_reason", reasons[reasons.ne("")].to_numpy())
    retained = df.loc[reasons.eq("")].copy().reset_index(drop=True)
    return retained, rejected.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit feature inclusion, target leakage exclusions, and duplicate records."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--deduplicated-output",
        type=Path,
        help="Optional path for a strict sensitivity dataset with exact duplicate records collapsed.",
    )
    parser.add_argument(
        "--strict-eligible-output",
        type=Path,
        help="Optional path for records retained by the finalized molecular-input gate.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_feature_manifest(df)
    duplicate_frame, duplicate_summary = duplicate_audit(df)
    deduplicated = deduplicate_exact_records(df)
    parse_failures = structure_parse_audit(df)
    strict_eligible, strict_rejected = strict_input_eligibility_audit(df)

    manifest.to_csv(args.output_dir / "feature_manifest.csv", index=False)
    duplicate_frame.to_csv(args.output_dir / "duplicate_audit.csv", index=False)
    parse_failures.to_csv(args.output_dir / "structure_parse_audit.csv", index=False)
    strict_rejected.to_csv(args.output_dir / "strict_input_eligibility_rejections.csv", index=False)
    summary = {
        "data": str(args.data),
        "rows": int(len(df)),
        "chemicals": int(df[DEFAULT_SCHEMA.chemical_id].nunique()),
        "predictive_fields": int(manifest["predictive_input"].sum()),
        "excluded_fields": int((~manifest["predictive_input"]).sum()),
        "unparseable_structure_rows": int(
            df[DEFAULT_SCHEMA.smiles].map(lambda value: smiles_to_mol(value) is None).sum()
        ),
        "unparseable_structure_chemicals": int(len(parse_failures)),
        "strict_eligible_rows": int(len(strict_eligible)),
        "strict_eligible_chemicals": int(strict_eligible[DEFAULT_SCHEMA.chemical_id].nunique()),
        "strict_rejected_rows": int(len(strict_rejected)),
        "strict_rejected_chemicals": int(
            strict_rejected[DEFAULT_SCHEMA.chemical_id].nunique()
        ),
        "strict_rejection_reasons": {
            str(key): int(value)
            for key, value in strict_rejected["input_eligibility_reason"]
            .value_counts()
            .sort_index()
            .items()
        },
        "duplicate_checks": duplicate_summary,
        "deduplicated_rows": int(len(deduplicated)),
    }
    (args.output_dir / "integrity_audit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if args.deduplicated_output is not None:
        args.deduplicated_output.parent.mkdir(parents=True, exist_ok=True)
        deduplicated.to_csv(args.deduplicated_output, index=False)
    if args.strict_eligible_output is not None:
        args.strict_eligible_output.parent.mkdir(parents=True, exist_ok=True)
        strict_eligible.to_csv(args.strict_eligible_output, index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
