from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

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


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "ecotox_acute_ecoood_1000chem_dsstox_mech_structured.csv"
DEFAULT_OUT_DIR = ROOT / "outputs" / "temporal_block"

MODEL_NAME = "lightgbm"
SEEDS = [40, 41, 42, 43, 44]
TRAIN_END_YEAR = 2012
CALIB_START_YEAR = 2013
CALIB_END_YEAR = 2014
TEST_START_YEAR = 2015

@dataclass(frozen=True)
class TemporalBlock:
    train_idx: pd.Index
    calib_idx: pd.Index
    test_idx: pd.Index
    metadata: dict[str, object]


def build_temporal_block(df: pd.DataFrame) -> TemporalBlock:
    years = pd.to_numeric(df[DEFAULT_SCHEMA.study_year], errors="coerce")
    if years.isna().any():
        raise ValueError("Temporal block validation requires non-null study_year values.")

    train_mask = years <= TRAIN_END_YEAR
    calib_mask = (years >= CALIB_START_YEAR) & (years <= CALIB_END_YEAR)
    test_mask = years >= TEST_START_YEAR

    if not train_mask.any() or not calib_mask.any() or not test_mask.any():
        raise ValueError("Temporal block split is empty for at least one partition.")

    train_idx = df.index[train_mask]
    calib_idx = df.index[calib_mask]
    test_idx = df.index[test_mask]

    metadata = {
        "train_years": f"{int(years[train_mask].min())}-{int(years[train_mask].max())}",
        "calib_years": f"{int(years[calib_mask].min())}-{int(years[calib_mask].max())}",
        "test_years": f"{int(years[test_mask].min())}-{int(years[test_mask].max())}",
        "train_rows": int(train_mask.sum()),
        "calib_rows": int(calib_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "train_chemicals": int(df.loc[train_mask, DEFAULT_SCHEMA.chemical_id].nunique()),
        "calib_chemicals": int(df.loc[calib_mask, DEFAULT_SCHEMA.chemical_id].nunique()),
        "test_chemicals": int(df.loc[test_mask, DEFAULT_SCHEMA.chemical_id].nunique()),
    }
    return TemporalBlock(train_idx=train_idx, calib_idx=calib_idx, test_idx=test_idx, metadata=metadata)


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
    calib_ad = ad_scorer.predict(
        calib_bundle,
        model_std=calib_pred.std,
        interval_width=calib_interval.width,
    )

    score_warn = float(
        pd.Series(calib_components_pred.prediction_error_risk_score).quantile(0.50)
    )
    score_abstain = float(
        pd.Series(calib_components_pred.prediction_error_risk_score).quantile(0.85)
    )
    decisions = decision_labels(
        test_components.prediction_error_risk_score,
        None,
        score_warn_threshold=score_warn,
        score_abstain_threshold=score_abstain,
    )

    y_test = test_df[DEFAULT_SCHEMA.target].to_numpy()
    metrics = {
        "split": "temporal_block",
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
        ]
    ].copy()
    predictions["seed"] = seed
    predictions["y_true"] = y_test
    predictions["y_pred"] = test_pred.mean
    predictions["interval_width"] = test_interval.width
    predictions["prediction_error_risk_score"] = test_components.prediction_error_risk_score
    predictions["decision"] = decisions
    predictions["ad_similarity"] = ad_scorer.predict(
        test_bundle,
        model_std=test_pred.std,
        interval_width=test_interval.width,
    ).similarity
    predictions["calib_similarity_mean"] = float(calib_ad.similarity.mean())
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
    row: dict[str, float | str] = {"group": "all", "split": "temporal_block", "model": MODEL_NAME}
    for col in metric_columns:
        row[f"{col}_mean"] = float(summary[col].mean())
        row[f"{col}_std"] = float(summary[col].std(ddof=1))
    return pd.DataFrame([row])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a fixed-year temporal block validation on the structured EcoOOD benchmark."
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--ensemble-n-jobs", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.data_path)
    working = attach_rdkit_descriptors(raw, DEFAULT_SCHEMA)
    working = working[working[DEFAULT_SCHEMA.target].notna()].reset_index(drop=True)
    block = build_temporal_block(working)

    train_df = working.loc[block.train_idx].reset_index(drop=True)
    calib_df = working.loc[block.calib_idx].reset_index(drop=True)
    test_df = working.loc[block.test_idx].reset_index(drop=True)

    metric_rows: list[dict[str, float | int | str]] = []
    prediction_frames: list[pd.DataFrame] = []
    for seed in args.seeds:
        metrics, predictions = compute_seed_metrics(
            train_df,
            calib_df,
            test_df,
            seed,
            ensemble_n_jobs=args.ensemble_n_jobs,
        )
        metric_rows.append(metrics)
        prediction_frames.append(predictions)

    summary_all = pd.DataFrame(metric_rows)
    summary_agg = aggregate_metrics(summary_all)
    metadata = pd.DataFrame([block.metadata])

    summary_all.to_csv(args.output_dir / "benchmark_summary_all_seeds.csv", index=False)
    summary_agg.to_csv(args.output_dir / "benchmark_summary_agg.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(args.output_dir / "predictions_all_seeds.csv", index=False)
    metadata.to_csv(args.output_dir / "temporal_block_metadata.csv", index=False)

    note = [
        "Prospective temporal block validation",
        f"Structured benchmark path: {args.data_path}",
        f"Train years: {block.metadata['train_years']} ({block.metadata['train_rows']} rows; {block.metadata['train_chemicals']} chemicals)",
        f"Calibration years: {block.metadata['calib_years']} ({block.metadata['calib_rows']} rows; {block.metadata['calib_chemicals']} chemicals)",
        f"Test years: {block.metadata['test_years']} ({block.metadata['test_rows']} rows; {block.metadata['test_chemicals']} chemicals)",
        "",
    ]
    row = summary_agg.iloc[0]
    note.extend(
        [
            f"RMSE = {row['rmse_mean']:.4f} ± {row['rmse_std']:.4f}",
            f"Coverage = {row['coverage_mean']:.4f} ± {row['coverage_std']:.4f}",
            f"AURC = {row['aurc_mean']:.4f} ± {row['aurc_std']:.4f}",
            f"Predict / warn / abstain = {row['predict_fraction_mean']:.4f} / {row['warn_fraction_mean']:.4f} / {row['abstain_fraction_mean']:.4f}",
        ]
    )
    (args.output_dir / "temporal_block_notes.txt").write_text("\n".join(note), encoding="utf-8")

    print("\n".join(note))


if __name__ == "__main__":
    main()
