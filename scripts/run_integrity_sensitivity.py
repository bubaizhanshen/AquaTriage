from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ecoood.pipeline import ExperimentConfig, run_single_experiment
from ecoood.schema import DEFAULT_SCHEMA


def make_missingness_only_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Retain split metadata but replace predictive content with missingness indicators."""
    schema = DEFAULT_SCHEMA
    result = pd.DataFrame(index=df.index)
    for column in [
        schema.target,
        schema.chemical_id,
        schema.chemical_name,
        schema.casrn,
        schema.chemical_class,
        schema.hard_ood,
        schema.known_ood,
    ]:
        if column in df.columns:
            result[column] = df[column]

    # The current pipeline rejects missing or unparseable structures before
    # feature construction.  Use one fixed, parseable placeholder so this
    # negative control tests whether missingness indicators alone carry signal
    # without bypassing the same molecular-input gate.
    result[schema.smiles] = "C"
    for column in [
        schema.endpoint,
        schema.species,
        schema.genus,
        schema.family,
        schema.order,
        schema.clazz,
        schema.phylum,
        schema.trophic_group,
        schema.medium,
    ]:
        if column in df.columns:
            result[column] = "missing"

    predictive_fields = [
        column
        for column in df.columns
        if column.startswith(("physchem_", "mech_", "tax_", "ctx_"))
        or column
        in {
            schema.duration_h,
            schema.temperature_c,
            schema.ph,
            schema.study_year,
        }
    ]
    for column in predictive_fields:
        result[f"ctx_missing_{column}"] = df[column].isna().astype(float)
    return result.reset_index(drop=True)


def drop_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return a copy without an explicitly audited predictor block."""
    return df.drop(columns=[column for column in columns if column in df.columns]).copy()


def bioactivity_observed_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict the benchmark to cases with at least one bioactivity-proxy result."""
    if "mech_feature_count" not in df.columns:
        raise KeyError("The structured benchmark does not contain mech_feature_count.")
    coverage = pd.to_numeric(df["mech_feature_count"], errors="coerce").fillna(0) > 0
    return df.loc[coverage].reset_index(drop=True).copy()


def without_pubchem_fallback_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude chemicals whose molecular representation came from PubChem fallback."""
    if "structure_source" not in df.columns:
        raise KeyError("The structured benchmark does not contain structure_source.")
    keep = df["structure_source"].fillna("").astype(str).str.casefold().ne("pubchem")
    return df.loc[keep].reset_index(drop=True).copy()


def build_profiles(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Define leakage and field-ablation checks using the same random split protocol."""
    schema = DEFAULT_SCHEMA
    record_context = [
        schema.medium,
        schema.duration_h,
        schema.temperature_c,
        schema.ph,
        schema.study_year,
        "ctx_hardness",
    ]
    profiles: dict[str, pd.DataFrame] = {
        "full_input": data,
        "permuted_training_target": data,
        "missingness_only": make_missingness_only_frame(data),
        "without_bioactivity_proxy": drop_columns(
            data,
            [column for column in data.columns if column.startswith("mech_")],
        ),
        # Dropping the linked field lets the shared feature builder recompute
        # MolLogP from each parseable structure before fitting.
        "with_rdkit_logp_recomputed": drop_columns(data, ["physchem_logp"]),
        "bioactivity_proxy_observed_cases": bioactivity_observed_frame(data),
        "without_pubchem_fallback": without_pubchem_fallback_frame(data),
        "without_all_record_context": drop_columns(data, record_context),
    }
    field_labels = {
        "medium": schema.medium,
        "duration": schema.duration_h,
        "temperature": schema.temperature_c,
        "ph": schema.ph,
        "study_year": schema.study_year,
        "hardness": "ctx_hardness",
    }
    for label, column in field_labels.items():
        profiles[f"without_{label}"] = drop_columns(data, [column])
    return profiles


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    # Keep the profile/sample identifiers out of the aggregate and select
    # numeric metric columns explicitly.  Some model wrappers return string
    # metadata (for example, a model label or a revision tag) alongside the
    # numeric metrics; those fields are not audit measurements.
    excluded = {"profile", "seed", "split", "model", "permuted_training_targets"}
    value_cols = [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]
    summary = frame.groupby("profile", as_index=False)[value_cols].agg(["mean", "std"]).reset_index()
    summary.columns = [
        column
        if isinstance(column, str)
        else "_".join(part for part in column if part).rstrip("_")
        for column in summary.columns.to_flat_index()
    ]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run leakage, missingness, bioactivity, and context-field integrity checks for the EcoOOD benchmark."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 42, 43, 44])
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--ensemble-n-jobs", type=int, default=5)
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=None,
        help="Optional subset of profile names. Defaults to the complete audit set.",
    )
    args = parser.parse_args()

    schema = DEFAULT_SCHEMA
    data = pd.read_csv(args.data)
    profiles = build_profiles(data)
    if args.profiles is not None:
        unknown = sorted(set(args.profiles) - set(profiles))
        if unknown:
            raise ValueError(f"Unknown audit profile(s): {', '.join(unknown)}")
        profiles = {name: profiles[name] for name in args.profiles}
    rows: list[dict[str, object]] = []
    for profile, frame in profiles.items():
        for seed in args.seeds:
            config = ExperimentConfig(
                split="random",
                model_name="lightgbm",
                seed=seed,
                n_members=args.members,
                ensemble_n_jobs=args.ensemble_n_jobs,
                permute_training_targets=profile == "permuted_training_target",
            )
            metrics, _, _ = run_single_experiment(frame, config=config)
            rows.append(
                {
                    "profile": profile,
                    "seed": seed,
                    "n_rows_input": len(frame),
                    "n_chemicals_input": frame[schema.chemical_id].nunique(),
                    **metrics,
                }
            )
            print(f"completed profile={profile} seed={seed}", flush=True)

    all_results = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results.to_csv(args.output_dir / "integrity_sensitivity_all.csv", index=False)
    summary = summarize(all_results)
    summary.to_csv(args.output_dir / "integrity_sensitivity_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
