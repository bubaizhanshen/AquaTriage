from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .ad import ApplicabilityDomainScorer
from .conformal import (
    GroupConditionalScaledConformalRegressor,
    ScaledConformalRegressor,
    decision_labels,
)
from .evaluation import interval_metrics, ood_metrics, reference_ood_metrics, regression_metrics, save_metrics, save_predictions, score_method_metrics
from .features import EcoFeatureBuilder, attach_rdkit_descriptors
from .models import BootstrapEnsembleRegressor
from .ood import CalibrationRiskScorer, EcoOODScorer, calibration_meta_bootstrap
from .schema import DEFAULT_SCHEMA, EcoOODSchema
from .splits import SplitIndices, build_split


@dataclass
class ExperimentConfig:
    split: str
    model_name: str = "lightgbm"
    alpha: float = 0.1
    seed: int = 42
    n_members: int = 5
    output_dir: str | None = None
    estimator_params: dict | None = None
    ensemble_n_jobs: int = -1
    permute_training_targets: bool = False
    high_error_quantile: float = 0.9
    high_error_quantile_sensitivity: tuple[float, ...] = (0.8, 0.9, 0.95)
    endpoint_conformal_min_group_size: int = 20
    include_study_year: bool = False
    recompute_rdkit_logp: bool = True
    allow_legacy_structure_placeholder: bool = False
    meta_bootstrap_replicates: int = 0
    reliability_component_mode: str = "revised"
    reliability_n_neighbors: int = 5
    reliability_fingerprint_metric: str = "tanimoto"
    run_distance_sensitivity: bool = False


def _select(df: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
    return df.loc[idx].reset_index(drop=True)


def _known_ood_labels(df_test: pd.DataFrame, split_indices: SplitIndices, schema: EcoOODSchema) -> np.ndarray:
    del df_test, schema
    return split_indices.test_is_ood


def _metric_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")


def _endpoint_concern_cutoffs(
    frame: pd.DataFrame,
    *,
    schema: EcoOODSchema,
    quantile: float = 0.25,
) -> dict[str, float]:
    chemical_endpoint = (
        frame.groupby([schema.chemical_id, schema.endpoint], dropna=False, as_index=False)
        .agg(endpoint_target=(schema.target, "median"))
    )
    return {
        str(endpoint): float(group["endpoint_target"].quantile(quantile))
        for endpoint, group in chemical_endpoint.groupby(schema.endpoint, sort=True)
    }


def run_single_experiment(
    df: pd.DataFrame,
    config: ExperimentConfig,
    schema: EcoOODSchema = DEFAULT_SCHEMA,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    working = attach_rdkit_descriptors(
        df,
        schema,
        recompute_logp=config.recompute_rdkit_logp,
    )
    working = working[working[schema.target].notna()].reset_index(drop=True)
    split_indices = build_split(working, split=config.split, schema=schema, seed=config.seed)
    train_df = _select(working, split_indices.train)
    calib_df = _select(working, split_indices.calib)
    test_df = _select(working, split_indices.test)

    feature_builder = EcoFeatureBuilder(
        schema=schema,
        include_study_year=config.include_study_year,
        recompute_rdkit_logp=config.recompute_rdkit_logp,
        allow_legacy_structure_placeholder=config.allow_legacy_structure_placeholder,
    )
    train_bundle = feature_builder.fit_transform(train_df)
    calib_bundle = feature_builder.transform(calib_df)
    test_bundle = feature_builder.transform(test_df)

    train_target = train_df[schema.target].to_numpy(copy=True)
    if config.permute_training_targets:
        train_target = np.random.default_rng(config.seed + 9173).permutation(train_target)

    model = BootstrapEnsembleRegressor(
        model_name=config.model_name,
        n_members=config.n_members,
        seed=config.seed,
        n_jobs=config.ensemble_n_jobs,
        estimator_params=config.estimator_params,
    ).fit(train_bundle.full, train_target)
    calib_pred = model.predict(calib_bundle.full)
    test_pred = model.predict(test_bundle.full)

    conformal = ScaledConformalRegressor(alpha=config.alpha).fit(
        calib_df[schema.target].to_numpy(),
        calib_pred.mean,
        scale=np.maximum(calib_pred.std, 1e-3),
    )
    calib_interval = conformal.predict(calib_pred.mean, scale=np.maximum(calib_pred.std, 1e-3))
    test_interval = conformal.predict(test_pred.mean, scale=np.maximum(test_pred.std, 1e-3))
    endpoint_conformal = GroupConditionalScaledConformalRegressor(
        alpha=config.alpha,
        min_group_size=config.endpoint_conformal_min_group_size,
    ).fit(
        calib_df[schema.target].to_numpy(),
        calib_pred.mean,
        groups=calib_df[schema.endpoint],
        scale=np.maximum(calib_pred.std, 1e-3),
    )
    endpoint_test_interval = endpoint_conformal.predict(
        test_pred.mean,
        groups=test_df[schema.endpoint],
        scale=np.maximum(test_pred.std, 1e-3),
    )

    scorer = EcoOODScorer(
        schema=schema,
        component_mode=config.reliability_component_mode,
        n_neighbors=config.reliability_n_neighbors,
        fingerprint_metric=config.reliability_fingerprint_metric,
        include_study_year=config.include_study_year,
    ).fit(train_df, train_bundle)
    calib_components = scorer.component_frame(
        calib_df,
        calib_bundle,
        model_std=calib_pred.std,
        interval_width=calib_interval.width,
    )
    scorer.fit_meta(
        calib_components,
        residuals=np.abs(calib_df[schema.target].to_numpy() - calib_pred.mean),
        high_error_quantile=config.high_error_quantile,
    )
    meta_bootstrap = (
        calibration_meta_bootstrap(
            calib_components,
            np.abs(calib_df[schema.target].to_numpy() - calib_pred.mean),
            calib_df[schema.chemical_id],
            n_replicates=config.meta_bootstrap_replicates,
            high_error_quantile=config.high_error_quantile,
            seed=config.seed + 6103,
        )
        if config.meta_bootstrap_replicates > 0
        else pd.DataFrame()
    )
    test_component_frame = scorer.component_frame(
        test_df,
        test_bundle,
        model_std=test_pred.std,
        interval_width=test_interval.width,
    )
    test_components = scorer.predict(
        test_df,
        test_bundle,
        model_std=test_pred.std,
        interval_width=test_interval.width,
    )
    calib_components_pred = scorer.predict(
        calib_df,
        calib_bundle,
        model_std=calib_pred.std,
        interval_width=calib_interval.width,
    )
    calib_ecoood_score = scorer.score_components(calib_components)

    ad_scorer = ApplicabilityDomainScorer().fit(
        train_bundle,
        chemical_ids=train_df[schema.chemical_id],
    )
    calib_ad = ad_scorer.predict(
        calib_bundle,
        model_std=calib_pred.std,
        interval_width=calib_interval.width,
    )
    test_ad = ad_scorer.predict(
        test_bundle,
        model_std=test_pred.std,
        interval_width=test_interval.width,
    )

    calib_residuals = np.abs(calib_df[schema.target].to_numpy() - calib_pred.mean)
    calibrated_score_specs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    comparator_specs = {
        "ensemble_sd_risk": (
            pd.DataFrame({"ensemble_sd": calib_pred.std}),
            pd.DataFrame({"ensemble_sd": test_pred.std}),
        ),
        "input_space_knn_plus_sd_risk": (
            pd.DataFrame(
                {
                    "input_space_knn": calib_ad.distance_to_model,
                    "ensemble_sd": calib_pred.std,
                }
            ),
            pd.DataFrame(
                {
                    "input_space_knn": test_ad.distance_to_model,
                    "ensemble_sd": test_pred.std,
                }
            ),
        ),
        "equal_block_knn_plus_sd_risk": (
            pd.DataFrame(
                {
                    "equal_block_knn": calib_ad.equal_block_distance,
                    "ensemble_sd": calib_pred.std,
                }
            ),
            pd.DataFrame(
                {
                    "equal_block_knn": test_ad.equal_block_distance,
                    "ensemble_sd": test_pred.std,
                }
            ),
        ),
        "generic_support_plus_sd_risk": (
            pd.DataFrame(
                {
                    "similarity_novelty": calib_ad.similarity,
                    "equal_block_knn": calib_ad.equal_block_distance,
                    "ensemble_sd": calib_pred.std,
                }
            ),
            pd.DataFrame(
                {
                    "similarity_novelty": test_ad.similarity,
                    "equal_block_knn": test_ad.equal_block_distance,
                    "ensemble_sd": test_pred.std,
                }
            ),
        ),
    }
    for name, (calib_features, test_features) in comparator_specs.items():
        risk_scorer = CalibrationRiskScorer().fit(calib_features, calib_residuals)
        calibrated_score_specs[name] = (
            risk_scorer.predict(calib_features),
            risk_scorer.predict(test_features),
        )

    if config.run_distance_sensitivity:
        distance_variants = {
            "prediction_error_risk_legacy": ("legacy", 5, "cosine"),
            "prediction_error_risk_cosine_k5": ("revised", 5, "cosine"),
            "prediction_error_risk_tanimoto_k1": ("revised", 1, "tanimoto"),
            "prediction_error_risk_tanimoto_k3": ("revised", 3, "tanimoto"),
            "prediction_error_risk_tanimoto_k10": ("revised", 10, "tanimoto"),
        }
        for name, (mode, neighbors, fingerprint_metric) in distance_variants.items():
            variant_scorer = EcoOODScorer(
                schema=schema,
                component_mode=mode,
                n_neighbors=neighbors,
                fingerprint_metric=fingerprint_metric,
                include_study_year=config.include_study_year,
            ).fit(train_df, train_bundle)
            variant_calibration = variant_scorer.component_frame(
                calib_df,
                calib_bundle,
                model_std=calib_pred.std,
                interval_width=calib_interval.width,
            )
            variant_test = variant_scorer.component_frame(
                test_df,
                test_bundle,
                model_std=test_pred.std,
                interval_width=test_interval.width,
            )
            risk_scorer = CalibrationRiskScorer().fit(
                variant_calibration,
                calib_residuals,
                high_error_quantile=config.high_error_quantile,
            )
            calibrated_score_specs[name] = (
                risk_scorer.predict(variant_calibration),
                risk_scorer.predict(variant_test),
            )

    concern_cutoffs = _endpoint_concern_cutoffs(calib_df, schema=schema)
    calibration_cutoff = calib_df[schema.endpoint].astype(str).map(concern_cutoffs)
    calibration_directional_miss = (
        calib_df[schema.target].to_numpy(dtype=float)
        <= calibration_cutoff.to_numpy(dtype=float)
    ) & (calib_pred.mean > calibration_cutoff.to_numpy(dtype=float))
    directional_scorer = CalibrationRiskScorer().fit_labels(
        calib_components,
        calibration_directional_miss,
    )
    calibrated_score_specs["ecoood_directional_miss_risk"] = (
        directional_scorer.predict(calib_components),
        directional_scorer.predict(test_component_frame),
    )

    component_axes = scorer.axis_components
    for axis, columns in component_axes.items():
        columns = list(columns)
        risk_scorer = CalibrationRiskScorer().fit(
            calib_components.drop(columns=columns),
            calib_residuals,
        )
        calibrated_score_specs[f"ecoood_minus_{axis}"] = (
            risk_scorer.predict(calib_components.drop(columns=columns)),
            risk_scorer.predict(test_component_frame.drop(columns=columns)),
        )

    # Preserve the axis-level analysis while also testing whether either signal
    # within the chemical and biological axes contributes independently.
    for component in calib_components.columns:
        retained_columns = [
            column for column in calib_components.columns if column != component
        ]
        risk_scorer = CalibrationRiskScorer().fit(
            calib_components.loc[:, retained_columns],
            calib_residuals,
        )
        calibrated_score_specs[f"ecoood_minus_component_{component}"] = (
            risk_scorer.predict(calib_components.loc[:, retained_columns]),
            risk_scorer.predict(test_component_frame.loc[:, retained_columns]),
        )

    for quantile in config.high_error_quantile_sensitivity:
        if np.isclose(quantile, config.high_error_quantile):
            continue
        sensitivity_scorer = EcoOODScorer(
            schema=schema,
            component_mode=config.reliability_component_mode,
            n_neighbors=config.reliability_n_neighbors,
            fingerprint_metric=config.reliability_fingerprint_metric,
            include_study_year=config.include_study_year,
        ).fit(train_df, train_bundle)
        sensitivity_scorer.fit_meta(
            calib_components,
            residuals=calib_residuals,
            high_error_quantile=quantile,
        )
        calibrated_score_specs[f"ecoood_q{int(round(quantile * 100)):02d}"] = (
            sensitivity_scorer.score_components(calib_components),
            sensitivity_scorer.score_components(test_component_frame),
        )

    endpoint_balanced_scorer = EcoOODScorer(
        schema=schema,
        component_mode=config.reliability_component_mode,
        n_neighbors=config.reliability_n_neighbors,
        fingerprint_metric=config.reliability_fingerprint_metric,
        include_study_year=config.include_study_year,
    ).fit(train_df, train_bundle)
    endpoint_balanced_scorer.fit_meta(
        calib_components,
        residuals=calib_residuals,
        high_error_quantile=config.high_error_quantile,
        groups=calib_df[schema.endpoint],
        groupwise_labels=True,
        balance_groups=True,
    )
    calibrated_score_specs["ecoood_endpoint_balanced"] = (
        endpoint_balanced_scorer.score_components(calib_components),
        endpoint_balanced_scorer.score_components(test_component_frame),
    )

    score_warn = float(np.quantile(calib_ecoood_score, 0.5))
    score_abstain = float(np.quantile(calib_ecoood_score, 0.85))
    decisions = decision_labels(
        test_components.ecoood_score,
        None,
        score_warn_threshold=score_warn,
        score_abstain_threshold=score_abstain,
    )

    y_test = test_df[schema.target].to_numpy()
    known_ood = _known_ood_labels(test_df, split_indices, schema)
    calibration_high_error = (
        calib_residuals >= float(scorer.high_error_threshold_)
    )
    calibration_endpoint_metrics: dict[str, int] = {}
    for endpoint in sorted(calib_df[schema.endpoint].dropna().astype(str).unique()):
        endpoint_mask = calib_df[schema.endpoint].astype(str).eq(endpoint).to_numpy()
        token = _metric_token(endpoint)
        calibration_endpoint_metrics[f"calibration_{token}_n"] = int(endpoint_mask.sum())
        calibration_endpoint_metrics[f"calibration_{token}_high_error_n"] = int(
            np.sum(calibration_high_error & endpoint_mask)
        )

    metrics = {
        "split": config.split,
        "model": config.model_name,
        "permuted_training_targets": bool(config.permute_training_targets),
        **regression_metrics(y_test, test_pred.mean),
        **interval_metrics(
            y_test,
            test_interval.lower,
            test_interval.upper,
            uncertainty=test_pred.std,
            novelty=test_components.ecoood_score,
        ),
        **{
            f"endpoint_conditional_{key}": value
            for key, value in interval_metrics(
                y_test,
                endpoint_test_interval.lower,
                endpoint_test_interval.upper,
                uncertainty=test_pred.std,
                novelty=test_components.ecoood_score,
            ).items()
        },
        **ood_metrics(
            y_test,
            test_pred.mean,
            test_components.ecoood_score,
            known_ood,
        ),
        **reference_ood_metrics(
            id_scores=calib_components_pred.ecoood_score,
            ood_scores=test_components.ecoood_score if np.any(split_indices.test_is_ood) else np.array([]),
        ),
        "predict_fraction": float(np.mean(decisions == "predict")),
        "warn_fraction": float(np.mean(decisions == "warn")),
        "abstain_fraction": float(np.mean(decisions == "abstain")),
        "conformal_qhat": float(conformal.qhat),
        "conformal_calibration_n": int(conformal.n_calibration_),
        "conformal_quantile_rank": int(conformal.quantile_rank_),
        "endpoint_conformal_supported_groups": int(len(endpoint_conformal.group_qhat_)),
        "directional_calibration_positive_n": int(directional_scorer.positive_count_),
        "directional_calibration_positive_rate": float(directional_scorer.positive_rate_),
        "directional_meta_converged": bool(directional_scorer.converged_),
        "n_model_features": int(train_bundle.full.shape[1]),
        "n_fingerprint_features": int(train_bundle.fingerprint.shape[1]),
        "n_descriptor_features": int(train_bundle.descriptor.shape[1]),
        "n_species_features": int(train_bundle.species.shape[1]),
        "n_context_features": int(train_bundle.context.shape[1]),
        "n_bioactivity_proxy_features": int(train_bundle.mechanism.shape[1]),
        "reliability_component_mode": config.reliability_component_mode,
        "reliability_n_neighbors": int(config.reliability_n_neighbors),
        "reliability_fingerprint_metric": config.reliability_fingerprint_metric,
        "distance_sensitivity_run": bool(config.run_distance_sensitivity),
        **{
            f"equal_block_rms_{name}": float(value)
            for name, value in ad_scorer.equal_block_scale_by_name.items()
        },
        **calibration_endpoint_metrics,
        **{f"ecoood_{key}": value for key, value in scorer.diagnostics().items()},
    }

    predictions = test_df.copy()
    predictions["y_true"] = y_test
    predictions["y_pred"] = test_pred.mean
    predictions["model_std"] = test_pred.std
    predictions["interval_lower"] = test_interval.lower
    predictions["interval_upper"] = test_interval.upper
    predictions["interval_width"] = test_interval.width
    predictions["endpoint_interval_lower"] = endpoint_test_interval.lower
    predictions["endpoint_interval_upper"] = endpoint_test_interval.upper
    predictions["endpoint_interval_width"] = endpoint_test_interval.width
    predictions["prediction_error_risk_score"] = test_components.ecoood_score
    # Compatibility alias for analysis tables produced before the public name
    # distinguished this candidate score from the EcoOOD framework.
    predictions["ecoood_score"] = test_components.ecoood_score
    predictions["d_chem"] = test_components.chemical
    predictions["d_species"] = test_components.species
    predictions["d_context"] = test_components.context
    predictions["d_mech"] = test_components.mechanism
    for column in test_component_frame.columns:
        predictions[column] = test_component_frame[column].to_numpy()
    predictions["decision"] = decisions
    predictions["known_ood"] = known_ood
    predictions["ad_similarity"] = test_ad.similarity
    predictions["ad_leverage"] = test_ad.leverage
    predictions["ad_range"] = test_ad.descriptor_range
    predictions["ad_distance_to_model"] = test_ad.distance_to_model
    predictions["ad_equal_block_distance"] = test_ad.equal_block_distance
    predictions["ad_equal_block_distance_distinct_chemical"] = (
        test_ad.equal_block_distance_distinct_chemical
    )
    predictions["uncertainty_interval_width_score"] = test_ad.interval_width
    predictions["ood_mahalanobis"] = test_ad.mahalanobis
    predictions["ood_isolation_forest"] = test_ad.isolation_forest
    predictions["ood_lof"] = test_ad.lof
    for name, (_, test_score) in calibrated_score_specs.items():
        predictions[name] = test_score
    predictions["direction_aligned_risk_score"] = predictions[
        "ecoood_directional_miss_risk"
    ]

    score_rows = []
    score_specs = {
        "ecoood": (calib_ecoood_score, test_components.ecoood_score),
        "ad_similarity": (calib_ad.similarity, test_ad.similarity),
        "ad_leverage": (calib_ad.leverage, test_ad.leverage),
        "ad_range": (calib_ad.descriptor_range, test_ad.descriptor_range),
        "ad_distance_to_model": (calib_ad.distance_to_model, test_ad.distance_to_model),
        "ad_equal_block_distance": (
            calib_ad.equal_block_distance,
            test_ad.equal_block_distance,
        ),
        "ad_equal_block_distance_distinct_chemical": (
            calib_ad.equal_block_distance_distinct_chemical,
            test_ad.equal_block_distance_distinct_chemical,
        ),
        "uncertainty_interval_width": (calib_ad.interval_width, test_ad.interval_width),
        "ood_mahalanobis": (calib_ad.mahalanobis, test_ad.mahalanobis),
        "ood_isolation_forest": (calib_ad.isolation_forest, test_ad.isolation_forest),
        "ood_lof": (calib_ad.lof, test_ad.lof),
    }
    score_specs.update(calibrated_score_specs)
    for method, (calib_score, test_score) in score_specs.items():
        row = {
            "split": config.split,
            "model": config.model_name,
            **score_method_metrics(
                method=method,
                y_true=y_test,
                y_pred=test_pred.mean,
                score=test_score,
                known_ood=known_ood,
                id_scores=calib_score,
                ood_scores=test_score if np.any(split_indices.test_is_ood) else np.array([]),
            ),
        }
        if method == "ecoood":
            row.update({f"meta_{key}": value for key, value in scorer.diagnostics().items()})
        score_rows.append(row)
    score_summary = pd.DataFrame(score_rows)

    if config.output_dir:
        base = Path(config.output_dir) / config.split / config.model_name
        save_metrics(metrics, base / "metrics.json")
        save_predictions(predictions, base / "predictions.csv")
        save_predictions(score_summary, base / "ood_score_summary.csv")
        if not meta_bootstrap.empty:
            save_predictions(meta_bootstrap, base / "calibration_meta_bootstrap.csv")
    return metrics, predictions, score_summary


def run_benchmark(
    df: pd.DataFrame,
    splits: list[str],
    models: list[str],
    output_dir: str,
    schema: EcoOODSchema = DEFAULT_SCHEMA,
    alpha: float = 0.1,
    seed: int = 42,
    n_members: int = 5,
    ensemble_n_jobs: int = -1,
    include_study_year: bool = False,
    recompute_rdkit_logp: bool = True,
    allow_legacy_structure_placeholder: bool = False,
    meta_bootstrap_replicates: int = 0,
    reliability_component_mode: str = "revised",
    reliability_n_neighbors: int = 5,
    reliability_fingerprint_metric: str = "tanimoto",
    run_distance_sensitivity: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    score_rows: list[pd.DataFrame] = []
    for split in splits:
        for model_name in models:
            config = ExperimentConfig(
                split=split,
                model_name=model_name,
                alpha=alpha,
                seed=seed,
                n_members=n_members,
                output_dir=output_dir,
                ensemble_n_jobs=ensemble_n_jobs,
                include_study_year=include_study_year,
                recompute_rdkit_logp=recompute_rdkit_logp,
                allow_legacy_structure_placeholder=allow_legacy_structure_placeholder,
                meta_bootstrap_replicates=meta_bootstrap_replicates,
                reliability_component_mode=reliability_component_mode,
                reliability_n_neighbors=reliability_n_neighbors,
                reliability_fingerprint_metric=reliability_fingerprint_metric,
                run_distance_sensitivity=run_distance_sensitivity,
            )
            metrics, _, score_summary = run_single_experiment(df, config=config, schema=schema)
            rows.append(metrics)
            score_rows.append(score_summary)
    summary = pd.DataFrame(rows)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "benchmark_summary.csv", index=False)
    if score_rows:
        pd.concat(score_rows, ignore_index=True).to_csv(output / "ood_score_summary.csv", index=False)
    return summary
