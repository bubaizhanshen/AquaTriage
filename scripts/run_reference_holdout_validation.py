from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rdkit import RDLogger

from ecoood.ad import ApplicabilityDomainScorer
from ecoood.conformal import ScaledConformalRegressor, decision_labels
from ecoood.evaluation import (
    interval_metrics,
    ood_metrics,
    reference_ood_metrics,
    regression_metrics,
)
from ecoood.features import EcoFeatureBuilder, attach_rdkit_descriptors
from ecoood.models import BootstrapEnsembleRegressor
from ecoood.ood import PredictionErrorRiskScorer
from ecoood.schema import DEFAULT_SCHEMA
from ecoood.splits import balanced_group_fold_split

if __package__:
    from scripts.build_ecotox_dataset import ECOTOX_MEMBERS, quality_filter, read_zip_member
else:
    from build_ecotox_dataset import ECOTOX_MEMBERS, quality_filter, read_zip_member

RDLogger.DisableLog("rdApp.*")


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "ecotox_acute_ecoood_1000chem_dsstox_mech_structured.csv"
RAW_ZIP = ROOT / "data" / "raw" / "ecotox_ascii_03_12_2026.zip"
DEFAULT_OUT_DIR = ROOT / "outputs" / "reference_holdout"

MODEL_NAME = "lightgbm"
DEFAULT_SEEDS = [40, 41, 42, 43, 44]


@dataclass(frozen=True)
class ReferenceAnnotations:
    frame: pd.DataFrame
    dataset_metadata: dict[str, object]


def _load_reference_annotations() -> ReferenceAnnotations:
    chemicals = read_zip_member(
        RAW_ZIP,
        ECOTOX_MEMBERS["chemicals"],
        ["cas_number", "chemical_name", "ecotox_group", "dtxsid"],
    ).rename(columns={"ecotox_group": "chemical_class"})
    species = read_zip_member(
        RAW_ZIP,
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
        RAW_ZIP,
        ECOTOX_MEMBERS["references"],
        ["reference_number", "publication_year", "doi"],
    ).rename(columns={"publication_year": "study_year"})
    tests = read_zip_member(
        RAW_ZIP,
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
        RAW_ZIP,
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

    merged = (
        results.merge(tests, on="test_id", how="left")
        .merge(species, on="species_number", how="left")
        .merge(chemicals, left_on="test_cas", right_on="cas_number", how="left")
        .merge(references, on="reference_number", how="left")
    )
    filtered = quality_filter(merged).copy()
    filtered["duration_h"] = pd.to_numeric(filtered["exposure_duration_mean"], errors="coerce")
    unit = filtered["exposure_duration_unit"].fillna("").astype(str).str.lower()
    filtered.loc[unit.eq("d"), "duration_h"] *= 24
    filtered.loc[unit.eq("wk"), "duration_h"] *= 24 * 7
    filtered.loc[unit.eq("h"), "duration_h"] *= 1
    filtered["medium"] = filtered["media_type"]
    filtered["species"] = filtered["species_name"]
    filtered["study_year"] = pd.to_numeric(filtered["study_year"], errors="coerce")
    filtered["toxicity_value"] = pd.to_numeric(filtered["conc1_mean"], errors="coerce")
    filtered["toxicity_unit"] = filtered["conc1_unit"]
    raw = filtered[
        [
            "reference_number",
            "test_id",
            "study_type",
            "test_location",
            "exposure_type",
            "cas_number",
            "species",
            "endpoint",
            "effect",
            "measurement",
            "duration_h",
            "medium",
            "study_year",
            "doi",
            "toxicity_value",
            "toxicity_unit",
        ]
    ].copy()
    raw = raw.rename(columns={"cas_number": "casrn", "endpoint": "endpoint_code"})
    raw["casrn_key"] = raw["casrn"].fillna("").astype(str).str.strip()
    for col in ["species", "endpoint_code", "effect", "measurement", "medium", "doi", "toxicity_unit"]:
        raw[col] = raw[col].fillna("").astype(str).str.strip()

    dataset_metadata = {
        "n_reference_numbers": int(raw["reference_number"].nunique(dropna=True)),
        "n_test_ids": int(raw["test_id"].nunique(dropna=True)),
        "test_location_counts": raw["test_location"].fillna("NA").value_counts().to_dict(),
        "study_type_counts": raw["study_type"].fillna("NA").value_counts().to_dict(),
    }
    return ReferenceAnnotations(frame=raw, dataset_metadata=dataset_metadata)


def _attach_reference_metadata(df: pd.DataFrame, annotations: ReferenceAnnotations) -> tuple[pd.DataFrame, pd.DataFrame]:
    proc = df.copy().reset_index(drop=True)
    proc["proc_row"] = proc.index
    proc["casrn_key"] = proc["casrn"].fillna("").astype(str).str.strip()
    for col in ["species", "endpoint_code", "effect", "measurement", "medium", "doi", "toxicity_unit"]:
        proc[col] = proc[col].fillna("").astype(str).str.strip()

    raw = annotations.frame.copy()
    raw = raw[raw["casrn_key"].isin(set(proc["casrn_key"]))].copy()
    keys = [
        "casrn_key",
        "species",
        "endpoint_code",
        "effect",
        "measurement",
        "duration_h",
        "medium",
        "study_year",
        "doi",
        "toxicity_value",
        "toxicity_unit",
    ]
    joined = proc.merge(raw, on=keys, how="left")
    if joined["reference_number"].isna().any():
        missing = int(joined["reference_number"].isna().sum())
        raise ValueError(f"Failed to recover reference_number for {missing} processed rows.")

    def _collapse(series: pd.Series) -> str:
        vals = sorted({str(v).strip() for v in series if pd.notna(v) and str(v).strip()})
        return ";".join(vals)

    row_meta = (
        joined.groupby("proc_row", as_index=False)
        .agg(
            reference_number=("reference_number", "first"),
            test_id=("test_id", _collapse),
            study_type=("study_type", _collapse),
            test_location=("test_location", _collapse),
            exposure_type=("exposure_type", _collapse),
        )
    )

    out = proc.merge(row_meta, on="proc_row", how="left")
    return out.drop(columns=["proc_row", "casrn_key"]), row_meta


def compute_seed_metrics(
    train_df: pd.DataFrame,
    calib_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
    ensemble_n_jobs: int = 5,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    feature_builder = EcoFeatureBuilder(schema=DEFAULT_SCHEMA)
    train_bundle = feature_builder.fit_transform(train_df)
    calib_bundle = feature_builder.transform(calib_df)
    test_bundle = feature_builder.transform(test_df)

    model = BootstrapEnsembleRegressor(
        model_name=MODEL_NAME,
        n_members=5,
        seed=seed,
        n_jobs=ensemble_n_jobs,
    ).fit(train_bundle.full, train_df[DEFAULT_SCHEMA.target].to_numpy())
    calib_pred = model.predict(calib_bundle.full)
    test_pred = model.predict(test_bundle.full)

    conformal = ScaledConformalRegressor(alpha=0.1).fit(
        calib_df[DEFAULT_SCHEMA.target].to_numpy(),
        calib_pred.mean,
        scale=calib_pred.std.clip(min=1e-3),
    )
    calib_interval = conformal.predict(calib_pred.mean, scale=calib_pred.std.clip(min=1e-3))
    test_interval = conformal.predict(test_pred.mean, scale=test_pred.std.clip(min=1e-3))

    scorer = PredictionErrorRiskScorer(schema=DEFAULT_SCHEMA).fit(train_df, train_bundle)
    calib_components = scorer.component_frame(
        calib_df,
        calib_bundle,
        model_std=calib_pred.std,
    )
    scorer.fit_meta(
        calib_components,
        residuals=(calib_df[DEFAULT_SCHEMA.target].to_numpy() - calib_pred.mean).astype(float).__abs__(),
    )
    test_components = scorer.predict(
        test_df,
        test_bundle,
        model_std=test_pred.std,
    )
    calib_components_pred = scorer.predict(
        calib_df,
        calib_bundle,
        model_std=calib_pred.std,
    )

    ad_scorer = ApplicabilityDomainScorer().fit(train_bundle)
    test_ad = ad_scorer.predict(
        test_bundle,
        model_std=test_pred.std,
        interval_width=test_interval.width,
    )

    score_warn = float(pd.Series(calib_components_pred.prediction_error_risk_score).quantile(0.50))
    score_abstain = float(pd.Series(calib_components_pred.prediction_error_risk_score).quantile(0.85))
    decisions = decision_labels(
        test_components.prediction_error_risk_score,
        None,
        score_warn_threshold=score_warn,
        score_abstain_threshold=score_abstain,
    )

    y_test = test_df[DEFAULT_SCHEMA.target].to_numpy()
    metrics = {
        "split": "reference_holdout",
        "model": MODEL_NAME,
        "seed": seed,
        **regression_metrics(y_test, test_pred.mean),
        **interval_metrics(
            y_test,
            test_interval.lower,
            test_interval.upper,
            uncertainty=test_pred.std,
            novelty=test_components.prediction_error_risk_score,
        ),
        **ood_metrics(
            y_test,
            test_pred.mean,
            test_components.prediction_error_risk_score,
            pd.Series([True] * len(test_df)).to_numpy(),
        ),
        **reference_ood_metrics(
            id_scores=calib_components_pred.prediction_error_risk_score,
            ood_scores=test_components.prediction_error_risk_score,
        ),
        "predict_fraction": float((decisions == "predict").mean()),
        "warn_fraction": float((decisions == "warn").mean()),
        "abstain_fraction": float((decisions == "abstain").mean()),
    }

    predictions = test_df[
        [
            DEFAULT_SCHEMA.chemical_id,
            DEFAULT_SCHEMA.chemical_name,
            DEFAULT_SCHEMA.casrn,
            DEFAULT_SCHEMA.chemical_class,
            DEFAULT_SCHEMA.species,
            DEFAULT_SCHEMA.study_year,
            "reference_number",
            "study_type",
            "test_location",
            "exposure_type",
        ]
    ].copy()
    predictions["seed"] = seed
    predictions["y_true"] = y_test
    predictions["y_pred"] = test_pred.mean
    predictions["interval_width"] = test_interval.width
    predictions["prediction_error_risk_score"] = test_components.prediction_error_risk_score
    predictions["decision"] = decisions
    predictions["ad_similarity"] = test_ad.similarity
    return metrics, predictions


def aggregate_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "rmse",
        "mae",
        "spearman",
        "bias",
        "coverage",
        "mean_interval_width",
        "uncertainty_error_corr",
        "uncertainty_novelty_corr",
        "aurc",
        "top_decile_error_capture_rate",
        "auroc_id_vs_ood",
        "aupr_id_vs_ood",
        "fpr95",
        "predict_fraction",
        "warn_fraction",
        "abstain_fraction",
    ]
    row: dict[str, float | str] = {"group": "all", "split": "reference_holdout", "model": MODEL_NAME}
    for col in metric_columns:
        row[f"{col}_mean"] = float(summary[col].mean())
        std = summary[col].std(ddof=1)
        row[f"{col}_std"] = float(0.0 if pd.isna(std) else std)
    return pd.DataFrame([row])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run grouped reference-holdout validation on the structured EcoOOD benchmark.")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Random seeds to evaluate. Defaults to the five-seed benchmark sweep.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_PATH,
        help="Structured benchmark CSV used for the validation.",
    )
    parser.add_argument(
        "--ensemble-n-jobs",
        type=int,
        default=5,
        help="Parallel workers used to fit each bootstrap ensemble.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for validation outputs.",
    )
    return parser.parse_args()


def write_progress_outputs(
    out_dir: Path,
    metric_rows: list[dict[str, float | int | str]],
    split_rows: list[dict[str, object]],
    row_meta: pd.DataFrame,
    annotations: ReferenceAnnotations,
) -> None:
    summary_all = pd.DataFrame(metric_rows)
    split_meta = pd.DataFrame(split_rows)
    summary_all.to_csv(out_dir / "benchmark_summary_all_seeds.csv", index=False)
    if not summary_all.empty:
        summary_agg = aggregate_metrics(summary_all)
        summary_agg.to_csv(out_dir / "benchmark_summary_agg.csv", index=False)
    if not split_meta.empty:
        split_meta.to_csv(out_dir / "split_metadata.csv", index=False)
    row_meta.to_csv(out_dir / "row_reference_annotations.csv", index=False)
    pd.DataFrame([annotations.dataset_metadata]).to_csv(out_dir / "dataset_reference_metadata.csv", index=False)


def write_status(out_dir: Path, message: str) -> None:
    text = message.rstrip()
    (out_dir / "run_status.txt").write_text(text + "\n")
    print(text, flush=True)


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    write_status(out_dir, f"Starting reference-holdout validation for seeds: {', '.join(str(seed) for seed in args.seeds)}")

    df = pd.read_csv(args.data_path)
    df = attach_rdkit_descriptors(df, DEFAULT_SCHEMA)
    df = df[df[DEFAULT_SCHEMA.target].notna()].reset_index(drop=True)

    annotations = _load_reference_annotations()
    df, row_meta = _attach_reference_metadata(df, annotations)
    df["reference_number"] = df["reference_number"].astype(str).str.strip()

    metric_rows: list[dict[str, float | int | str]] = []
    prediction_rows: list[pd.DataFrame] = []
    split_rows: list[dict[str, object]] = []

    for index, seed in enumerate(args.seeds, start=1):
        write_status(out_dir, f"Running seed {seed} ({index}/{len(args.seeds)})")
        split = balanced_group_fold_split(
            df,
            group_col="reference_number",
            split_name="reference_holdout",
            fold_index=index - 1,
            n_splits=len(args.seeds),
            fold_seed=20260724,
            calib_seed=seed,
        )
        train_df = df.loc[split.train].reset_index(drop=True)
        calib_df = df.loc[split.calib].reset_index(drop=True)
        test_df = df.loc[split.test].reset_index(drop=True)

        metrics, predictions = compute_seed_metrics(
            train_df,
            calib_df,
            test_df,
            seed=seed,
            ensemble_n_jobs=args.ensemble_n_jobs,
        )
        metric_rows.append(metrics)
        prediction_rows.append(predictions)
        predictions.to_csv(out_dir / f"predictions_seed_{seed}.csv", index=False)

        split_rows.append(
            {
                "seed": seed,
                "train_rows": int(len(train_df)),
                "calib_rows": int(len(calib_df)),
                "test_rows": int(len(test_df)),
                "train_references": int(train_df["reference_number"].nunique()),
                "calib_references": int(calib_df["reference_number"].nunique()),
                "test_references": int(test_df["reference_number"].nunique()),
                "train_calib_reference_overlap": int(
                    len(
                        set(train_df["reference_number"])
                        & set(calib_df["reference_number"])
                    )
                ),
                "train_test_reference_overlap": int(
                    len(
                        set(train_df["reference_number"])
                        & set(test_df["reference_number"])
                    )
                ),
                "calib_test_reference_overlap": int(
                    len(
                        set(calib_df["reference_number"])
                        & set(test_df["reference_number"])
                    )
                ),
                "train_chemicals": int(train_df[DEFAULT_SCHEMA.chemical_id].nunique()),
                "calib_chemicals": int(calib_df[DEFAULT_SCHEMA.chemical_id].nunique()),
                "test_chemicals": int(test_df[DEFAULT_SCHEMA.chemical_id].nunique()),
                "train_species": int(train_df[DEFAULT_SCHEMA.species].nunique()),
                "calib_species": int(calib_df[DEFAULT_SCHEMA.species].nunique()),
                "test_species": int(test_df[DEFAULT_SCHEMA.species].nunique()),
                "test_location_top": test_df["test_location"].fillna("NA").astype(str).str.split(";").explode().value_counts().head(5).to_dict(),
                "study_type_top": test_df["study_type"].fillna("NA").astype(str).str.split(";").explode().value_counts().head(5).to_dict(),
            }
        )
        write_progress_outputs(out_dir, metric_rows, split_rows, row_meta, annotations)
        write_status(out_dir, f"Completed seed {seed} ({index}/{len(args.seeds)})")

    summary_all = pd.DataFrame(metric_rows)
    summary_agg = aggregate_metrics(summary_all)
    predictions_all = pd.concat(prediction_rows, ignore_index=True)
    split_meta = pd.DataFrame(split_rows)

    summary_all.to_csv(out_dir / "benchmark_summary_all_seeds.csv", index=False)
    summary_agg.to_csv(out_dir / "benchmark_summary_agg.csv", index=False)
    predictions_all.to_csv(out_dir / "predictions_all_seeds.csv", index=False)
    split_meta.to_csv(out_dir / "split_metadata.csv", index=False)
    row_meta.to_csv(out_dir / "row_reference_annotations.csv", index=False)
    pd.DataFrame([annotations.dataset_metadata]).to_csv(out_dir / "dataset_reference_metadata.csv", index=False)

    notes = [
        "Reference-holdout validation groups rows by ECOTOX reference_number after matching processed benchmark rows back to raw ECOTOX records.",
        f"Structured benchmark path: {args.data_path}.",
        f"Structured rows matched to reference metadata: {len(df)}.",
        f"Unique reference_numbers in structured benchmark: {df['reference_number'].nunique()}.",
        f"Unique test_ids represented: {row_meta['test_id'].replace('', pd.NA).nunique(dropna=True)}.",
        f"Seeds evaluated in this run: {', '.join(str(seed) for seed in args.seeds)}.",
        "This validation approximates leave-study-out transfer within ECOTOX, not an independent external dataset.",
    ]
    (out_dir / "reference_holdout_notes.txt").write_text("\n".join(notes) + "\n")
    write_status(out_dir, "Completed all requested seeds.")


if __name__ == "__main__":
    main()
