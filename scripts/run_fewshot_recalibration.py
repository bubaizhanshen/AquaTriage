from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ecoood.conformal import ScaledConformalRegressor
from ecoood.evaluation import interval_metrics, ood_metrics
from ecoood.features import EcoFeatureBuilder, FeatureBundle, attach_rdkit_descriptors
from ecoood.models import BootstrapEnsembleRegressor
from ecoood.ood import PredictionErrorRiskScorer
from ecoood.schema import DEFAULT_SCHEMA, EcoOODSchema
from ecoood.splits import SplitIndices, build_split


@dataclass
class SplitFit:
    train_df: pd.DataFrame
    calib_df: pd.DataFrame
    test_df: pd.DataFrame
    train_bundle: FeatureBundle
    calib_bundle: FeatureBundle
    test_bundle: FeatureBundle
    split_indices: SplitIndices
    y_test: np.ndarray
    y_pred_test: np.ndarray
    model_std_test: np.ndarray
    base_interval_lower: np.ndarray
    base_interval_upper: np.ndarray
    base_interval_width: np.ndarray
    base_prediction_error_risk_score: np.ndarray


def _select(df: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
    return df.loc[idx].reset_index(drop=True)


def _slice_bundle(bundle: FeatureBundle, idx: np.ndarray) -> FeatureBundle:
    return FeatureBundle(
        full=bundle.full[idx],
        fingerprint=bundle.fingerprint[idx],
        descriptor=bundle.descriptor[idx],
        species=bundle.species[idx],
        context=bundle.context[idx],
        mechanism=bundle.mechanism[idx],
        feature_names=bundle.feature_names,
    )


def _sample_adaptation_indices(
    frame: pd.DataFrame,
    n_adapt: int,
    seed: int,
    *,
    group_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    n_total = len(frame)
    if n_adapt <= 0:
        return np.array([], dtype=int), np.arange(n_total, dtype=int)
    if n_adapt >= n_total:
        raise ValueError("n_adapt must be smaller than n_total.")
    rng = np.random.default_rng(seed)
    groups = frame[group_column].fillna("missing").astype(str)
    group_values = groups.drop_duplicates().to_numpy()
    rng.shuffle(group_values)
    selected_groups: list[str] = []
    selected_rows = 0
    for group in group_values:
        group_n = int((groups == group).sum())
        if selected_rows + group_n >= n_total:
            continue
        selected_groups.append(group)
        selected_rows += group_n
        if selected_rows >= n_adapt:
            break
    chosen = np.flatnonzero(groups.isin(selected_groups).to_numpy())
    if len(chosen) == 0:
        raise RuntimeError("No complete adaptation group could be sampled.")
    mask = np.ones(n_total, dtype=bool)
    mask[chosen] = False
    remaining = np.arange(n_total, dtype=int)[mask]
    return chosen, remaining


def _known_ood_labels(df_test: pd.DataFrame, split_indices: SplitIndices, schema: EcoOODSchema) -> np.ndarray:
    del df_test, schema
    return split_indices.test_is_ood


def _fit_split(
    df: pd.DataFrame,
    split: str,
    model_name: str,
    seed: int,
    alpha: float,
    members: int,
    ensemble_n_jobs: int,
    schema: EcoOODSchema,
) -> SplitFit:
    working = attach_rdkit_descriptors(df, schema)
    working = working[working[schema.target].notna()].reset_index(drop=True)
    split_indices = build_split(working, split=split, schema=schema, seed=seed)
    train_df = _select(working, split_indices.train)
    calib_df = _select(working, split_indices.calib)
    test_df = _select(working, split_indices.test)

    feature_builder = EcoFeatureBuilder(schema=schema)
    train_bundle = feature_builder.fit_transform(train_df)
    calib_bundle = feature_builder.transform(calib_df)
    test_bundle = feature_builder.transform(test_df)

    model = BootstrapEnsembleRegressor(
        model_name=model_name,
        n_members=members,
        seed=seed,
        n_jobs=ensemble_n_jobs,
    ).fit(train_bundle.full, train_df[schema.target].to_numpy())
    calib_pred = model.predict(calib_bundle.full)
    test_pred = model.predict(test_bundle.full)

    conformal = ScaledConformalRegressor(alpha=alpha).fit(
        calib_df[schema.target].to_numpy(),
        calib_pred.mean,
        scale=np.maximum(calib_pred.std, 1e-3),
    )
    base_interval = conformal.predict(test_pred.mean, scale=np.maximum(test_pred.std, 1e-3))

    scorer = PredictionErrorRiskScorer(schema=schema).fit(train_df, train_bundle)
    calib_components = scorer.component_frame(
        calib_df,
        calib_bundle,
        model_std=calib_pred.std,
        interval_width=conformal.predict(calib_pred.mean, scale=np.maximum(calib_pred.std, 1e-3)).width,
    )
    scorer.fit_meta(
        calib_components,
        residuals=np.abs(calib_df[schema.target].to_numpy() - calib_pred.mean),
    )
    test_components = scorer.predict(
        test_df,
        test_bundle,
        model_std=test_pred.std,
        interval_width=base_interval.width,
    )

    return SplitFit(
        train_df=train_df,
        calib_df=calib_df,
        test_df=test_df,
        train_bundle=train_bundle,
        calib_bundle=calib_bundle,
        test_bundle=test_bundle,
        split_indices=split_indices,
        y_test=test_df[schema.target].to_numpy(),
        y_pred_test=test_pred.mean,
        model_std_test=test_pred.std,
        base_interval_lower=base_interval.lower,
        base_interval_upper=base_interval.upper,
        base_interval_width=base_interval.width,
        base_prediction_error_risk_score=test_components.prediction_error_risk_score,
    )


def _evaluate_subset(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_std: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    score: np.ndarray,
    known_ood: np.ndarray,
) -> dict[str, float]:
    return {
        **interval_metrics(
            y_true,
            lower,
            upper,
            uncertainty=model_std,
            novelty=score,
        ),
        **ood_metrics(
            y_true,
            y_pred,
            score,
            known_ood,
        ),
    }


def _recalibrated_score(
    fit: SplitFit,
    adapt_idx: np.ndarray,
    eval_idx: np.ndarray,
    adapt_interval_width: np.ndarray,
    eval_interval_width: np.ndarray,
    schema: EcoOODSchema,
    high_error_quantile: float,
    min_score_refit_records: int,
) -> np.ndarray:
    if len(adapt_idx) < min_score_refit_records:
        return fit.base_prediction_error_risk_score[eval_idx]
    local_scorer = PredictionErrorRiskScorer(schema=schema).fit(fit.train_df, fit.train_bundle)
    adapt_df = fit.test_df.iloc[adapt_idx].reset_index(drop=True)
    adapt_bundle = _slice_bundle(fit.test_bundle, adapt_idx)
    adapt_components = local_scorer.component_frame(
        adapt_df,
        adapt_bundle,
        model_std=fit.model_std_test[adapt_idx],
        interval_width=adapt_interval_width,
    )
    local_scorer.fit_meta(
        adapt_components,
        residuals=np.abs(fit.y_test[adapt_idx] - fit.y_pred_test[adapt_idx]),
        high_error_quantile=high_error_quantile,
    )
    eval_df = fit.test_df.iloc[eval_idx].reset_index(drop=True)
    eval_bundle = _slice_bundle(fit.test_bundle, eval_idx)
    eval_components = local_scorer.predict(
        eval_df,
        eval_bundle,
        model_std=fit.model_std_test[eval_idx],
        interval_width=eval_interval_width,
    )
    return eval_components.prediction_error_risk_score


def _aggregate_results(frame: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        col
        for col in frame.columns
        if col not in {"split", "model", "seed", "adapt_seed", "shots"}
    ]
    aggregated = (
        frame.groupby(["split", "model", "shots"], dropna=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregated.columns = [
        "_".join(str(part) for part in col if part).rstrip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in aggregated.columns.to_flat_index()
    ]
    return aggregated


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local interval recalibration experiments.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["temporal", "species", "chemical_class"])
    parser.add_argument("--models", nargs="+", default=["lightgbm"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 42, 43, 44])
    parser.add_argument(
        "--adapt-seeds",
        nargs="+",
        type=int,
        default=list(range(100, 130)),
    )
    parser.add_argument("--shots", nargs="+", type=int, default=[0, 20, 50, 100, 200])
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--ensemble-n-jobs", type=int, default=5)
    parser.add_argument("--high-error-quantile", type=float, default=0.9)
    parser.add_argument("--min-score-refit-records", type=int, default=100)
    parser.add_argument(
        "--refit-score",
        action="store_true",
        help=(
            "Opt in to refitting the high-error score with local labels. "
            "This analysis recalibrates conformal intervals only by default."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/fewshot_recalibration"))
    args = parser.parse_args()

    if args.data.suffix == ".parquet":
        df = pd.read_parquet(args.data)
    else:
        df = pd.read_csv(args.data)

    rows: list[dict[str, float | int | str]] = []
    schema = DEFAULT_SCHEMA
    for split in args.splits:
        for model_name in args.models:
            for seed in args.seeds:
                fit = _fit_split(
                    df=df,
                    split=split,
                    model_name=model_name,
                    seed=seed,
                    alpha=args.alpha,
                    members=args.members,
                    ensemble_n_jobs=args.ensemble_n_jobs,
                    schema=schema,
                )
                known_ood = _known_ood_labels(fit.test_df, fit.split_indices, schema)
                n_test = len(fit.test_df)
                valid_shots = [shot for shot in args.shots if shot < n_test]
                for shot in valid_shots:
                    active_adapt_seeds = args.adapt_seeds if shot > 0 else [0]
                    for adapt_seed in active_adapt_seeds:
                        adapt_idx, eval_idx = (
                            _sample_adaptation_indices(
                                fit.test_df,
                                shot,
                                seed=adapt_seed,
                                group_column=schema.chemical_id,
                            )
                            if shot > 0
                            else (np.array([], dtype=int), np.arange(n_test, dtype=int))
                        )
                        base_metrics = _evaluate_subset(
                            y_true=fit.y_test[eval_idx],
                            y_pred=fit.y_pred_test[eval_idx],
                            model_std=fit.model_std_test[eval_idx],
                            lower=fit.base_interval_lower[eval_idx],
                            upper=fit.base_interval_upper[eval_idx],
                            score=fit.base_prediction_error_risk_score[eval_idx],
                            known_ood=known_ood[eval_idx],
                        )

                        if shot == 0:
                            recal_lower = fit.base_interval_lower[eval_idx]
                            recal_upper = fit.base_interval_upper[eval_idx]
                            recal_width = fit.base_interval_width[eval_idx]
                            recal_score = fit.base_prediction_error_risk_score[eval_idx]
                        else:
                            conformal = ScaledConformalRegressor(alpha=args.alpha).fit(
                                fit.y_test[adapt_idx],
                                fit.y_pred_test[adapt_idx],
                                scale=np.maximum(fit.model_std_test[adapt_idx], 1e-3),
                            )
                            adapt_interval = conformal.predict(
                                fit.y_pred_test[adapt_idx],
                                scale=np.maximum(fit.model_std_test[adapt_idx], 1e-3),
                            )
                            eval_interval = conformal.predict(
                                fit.y_pred_test[eval_idx],
                                scale=np.maximum(fit.model_std_test[eval_idx], 1e-3),
                            )
                            recal_lower = eval_interval.lower
                            recal_upper = eval_interval.upper
                            recal_width = eval_interval.width
                            recal_score = (
                                _recalibrated_score(
                                    fit=fit,
                                    adapt_idx=adapt_idx,
                                    eval_idx=eval_idx,
                                    adapt_interval_width=adapt_interval.width,
                                    eval_interval_width=recal_width,
                                    schema=schema,
                                    high_error_quantile=args.high_error_quantile,
                                    min_score_refit_records=args.min_score_refit_records,
                                )
                                if args.refit_score
                                else fit.base_prediction_error_risk_score[eval_idx]
                            )

                        recal_metrics = _evaluate_subset(
                            y_true=fit.y_test[eval_idx],
                            y_pred=fit.y_pred_test[eval_idx],
                            model_std=fit.model_std_test[eval_idx],
                            lower=recal_lower,
                            upper=recal_upper,
                            score=recal_score,
                            known_ood=known_ood[eval_idx],
                        )

                        rows.append(
                            {
                                "split": split,
                                "model": model_name,
                                "seed": seed,
                                "adapt_seed": adapt_seed,
                                "shots": shot,
                                "n_adapt": int(len(adapt_idx)),
                                "n_adapt_chemicals": int(
                                    fit.test_df.iloc[adapt_idx][schema.chemical_id].nunique()
                                ),
                                "n_eval": int(len(eval_idx)),
                                "score_refit": bool(
                                    args.refit_score
                                    and len(adapt_idx) >= args.min_score_refit_records
                                ),
                                "score_high_error_positive_count": (
                                    int(
                                        np.sum(
                                            np.abs(
                                                fit.y_test[adapt_idx]
                                                - fit.y_pred_test[adapt_idx]
                                            )
                                            >= np.quantile(
                                                np.abs(
                                                    fit.y_test[adapt_idx]
                                                    - fit.y_pred_test[adapt_idx]
                                                ),
                                                args.high_error_quantile,
                                            )
                                        )
                                    )
                                    if len(adapt_idx)
                                    else 0
                                ),
                                "coverage_before": base_metrics["coverage"],
                                "coverage_after": recal_metrics["coverage"],
                                "coverage_recovery": recal_metrics["coverage"] - base_metrics["coverage"],
                                "mean_interval_width_before": base_metrics["mean_interval_width"],
                                "mean_interval_width_after": recal_metrics["mean_interval_width"],
                                "aurc_before": base_metrics["aurc"],
                                "aurc_after": recal_metrics["aurc"],
                                "aurc_improvement": base_metrics["aurc"] - recal_metrics["aurc"],
                                "top_decile_error_capture_before": base_metrics[
                                    "top_decile_error_capture_rate"
                                ],
                                "top_decile_error_capture_after": recal_metrics[
                                    "top_decile_error_capture_rate"
                                ],
                                "target_coverage_gap_before": (1 - args.alpha) - base_metrics["coverage"],
                                "target_coverage_gap_after": (1 - args.alpha) - recal_metrics["coverage"],
                            }
                        )

    all_results = pd.DataFrame(rows)
    summary = _aggregate_results(all_results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results.to_csv(args.output_dir / "fewshot_recalibration_all.csv", index=False)
    summary.to_csv(args.output_dir / "fewshot_recalibration_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
