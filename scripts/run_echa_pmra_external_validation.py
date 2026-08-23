from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger

from ecoood.ad import ApplicabilityDomainScorer
from ecoood.conformal import ScaledConformalRegressor
from ecoood.evaluation import ood_metrics, regression_metrics
from ecoood.features import EcoFeatureBuilder, attach_rdkit_descriptors
from ecoood.models import BootstrapEnsembleRegressor
from ecoood.ood import CalibrationRiskScorer, EcoOODScorer
from ecoood.schema import DEFAULT_SCHEMA
from scripts.audit_benchmark_integrity import strict_input_eligibility_audit

RDLogger.DisableLog("rdApp.*")


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "EcoOOD_benchmark_snapshot_structured.csv"
DEFAULT_PANEL_PATH = (
    ROOT
    / "results"
    / "external_regulatory_prep"
    / "echa_external_main.csv"
)
PUBLIC_PANEL_PATH = ROOT / "data" / "processed" / "echa_external_main.csv"
DEFAULT_OUT_DIR = ROOT / "results" / "echa_pmra_external_validation_main"
MODEL_NAME = "lightgbm"
DEFAULT_SEEDS = [40, 41, 42, 43, 44]
REVIEW_FRACTION = 0.25
RELIABILITY_METHODS = {
    "ecoood": "ecoood_score",
    "ecoood_endpoint_balanced": "ecoood_endpoint_balanced",
    "ensemble_sd_risk": "ensemble_sd_risk",
    "input_space_knn_plus_sd_risk": "input_space_knn_plus_sd_risk",
    "equal_block_knn_plus_sd_risk": "equal_block_knn_plus_sd_risk",
    "generic_support_plus_sd_risk": "generic_support_plus_sd_risk",
    "distance_to_model": "ad_distance_to_model",
    "equal_block_distance": "ad_equal_block_distance",
    "distinct_chemical_equal_block_distance": (
        "ad_equal_block_distance_distinct_chemical"
    ),
    "similarity_ad_risk": "ad_similarity_risk",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run case-level external validation on a cleaned ECHA evaluation set."
    )
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument(
        "--panel-path",
        type=Path,
        default=PUBLIC_PANEL_PATH if PUBLIC_PANEL_PATH.exists() else DEFAULT_PANEL_PATH,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--ensemble-n-jobs", type=int, default=5)
    parser.add_argument(
        "--use-original-benchmark",
        action="store_true",
        help=(
            "Use the deduplicated original benchmark without the current "
            "molecular-input gate. The default uses the current application inputs."
        ),
    )
    parser.add_argument("--include-study-year", action="store_true")
    parser.add_argument("--keep-linked-logp", action="store_true")
    parser.add_argument("--allow-legacy-structure-placeholder", action="store_true")
    return parser.parse_args()


def write_status(out_dir: Path, message: str) -> None:
    text = message.rstrip()
    (out_dir / "run_status.txt").write_text(text + "\n")
    print(text, flush=True)


def calibration_split_by_chemical(
    df: pd.DataFrame,
    seed: int,
    calib_fraction: float = 0.125,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    group_sizes = df.groupby(DEFAULT_SCHEMA.chemical_id, dropna=False).size().sort_values(ascending=False)
    groups = group_sizes.index.to_list()
    rng.shuffle(groups)
    target_rows = max(1, int(round(len(df) * calib_fraction)))
    calib_groups: list[object] = []
    running = 0
    for group in groups:
        calib_groups.append(group)
        running += int(group_sizes.loc[group])
        if running >= target_rows:
            break
    calib_mask = df[DEFAULT_SCHEMA.chemical_id].isin(calib_groups)
    calib_df = df.loc[calib_mask].reset_index(drop=True)
    train_df = df.loc[~calib_mask].reset_index(drop=True)
    return train_df, calib_df


def fixed_burden_metrics(
    frame: pd.DataFrame,
    score_col: str,
    *,
    review_fraction: float = REVIEW_FRACTION,
) -> dict[str, float]:
    scores = frame[score_col].to_numpy(dtype=float)
    order = np.argsort(scores)[::-1]
    review_n = max(1, int(round(len(frame) * review_fraction)))
    review_idx = order[:review_n]
    review_mask = np.zeros(len(frame), dtype=bool)
    review_mask[review_idx] = True

    errors = frame["abs_error"].to_numpy(dtype=float)
    high_error_threshold = float(np.quantile(errors, 0.9))
    high_error_mask = errors >= high_error_threshold
    return {
        "review_fraction": review_fraction,
        "review_n": int(review_mask.sum()),
        "mean_abs_error_review": float(errors[review_mask].mean()) if review_mask.any() else float("nan"),
        "mean_abs_error_propagate": float(errors[~review_mask].mean()) if (~review_mask).any() else float("nan"),
        "high_error_capture_rate": (
            float(review_mask[high_error_mask].mean())
            if high_error_mask.any()
            else float("nan")
        ),
        "propagate_high_error_rate": (
            float(high_error_mask[~review_mask].mean())
            if (~review_mask).any()
            else float("nan")
        ),
    }


def summarize_seed(seed: int, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    burden_rows: list[dict[str, object]] = []
    grouped = list(frame.groupby("endpoint", sort=False)) + [("pooled", frame)]
    for endpoint_name, subset in grouped:
        if subset.empty:
            continue
        y_true = subset["target_log_molar"].to_numpy(dtype=float)
        y_pred = subset["y_pred"].to_numpy(dtype=float)
        base = {
            "seed": seed,
            "endpoint": endpoint_name,
            "n_cases": int(len(subset)),
            "n_chemicals": int(subset["chemical_id"].nunique()),
            **regression_metrics(y_true, y_pred),
        }
        for method_name, score_col in RELIABILITY_METHODS.items():
            score = subset[score_col].to_numpy(dtype=float)
            direct = ood_metrics(y_true, y_pred, score, None)
            metric_rows.append(
                {
                    **base,
                    "method": method_name,
                    "score_col": score_col,
                    "aurc": direct["aurc"],
                    "top_decile_error_capture_rate": direct[
                        "top_decile_error_capture_rate"
                    ],
                }
            )
            burden = fixed_burden_metrics(
                subset,
                score_col,
                review_fraction=REVIEW_FRACTION,
            )
            burden_rows.append(
                {
                    "seed": seed,
                    "endpoint": endpoint_name,
                    "method": method_name,
                    "score_col": score_col,
                    **burden,
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(burden_rows)


def chemical_level_fixed_workload(seed: int, frame: pd.DataFrame) -> pd.DataFrame:
    """Evaluate external error capture with chemicals as the decision unit."""
    endpoint = frame.groupby(["chemical_id", "endpoint"], as_index=False).agg(
        abs_error=("abs_error", "median"),
        **{
            method_name: (score_col, "median")
            for method_name, score_col in RELIABILITY_METHODS.items()
        },
    )
    chemicals = endpoint.groupby("chemical_id", as_index=False).agg(
        abs_error=("abs_error", "max"),
        **{
            method_name: (method_name, "max")
            for method_name in RELIABILITY_METHODS
        },
    )
    n_chemicals = len(chemicals)
    review_n = max(1, int(round(REVIEW_FRACTION * n_chemicals)))
    high_error_n = max(1, int(np.ceil(0.10 * n_chemicals)))
    high_error_ids = set(
        chemicals.sort_values(
            ["abs_error", "chemical_id"],
            ascending=[False, True],
            kind="mergesort",
        )
        .head(high_error_n)["chemical_id"]
    )

    rows: list[dict[str, object]] = []
    for method_name in RELIABILITY_METHODS:
        reviewed_ids = set(
            chemicals.sort_values(
                [method_name, "chemical_id"],
                ascending=[False, True],
                kind="mergesort",
            )
            .head(review_n)["chemical_id"]
        )
        rows.append(
            {
                "seed": seed,
                "method": method_name,
                "n_chemicals": n_chemicals,
                "review_n": review_n,
                "high_error_n": high_error_n,
                "high_error_capture_rate": (
                    len(reviewed_ids & high_error_ids) / high_error_n
                ),
            }
        )
    return pd.DataFrame(rows)


def aggregate_summary(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    value_cols = [col for col in frame.columns if col not in group_cols]
    summary = (
        frame.groupby(group_cols, as_index=False)[value_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        col if isinstance(col, str) else "_".join([piece for piece in col if piece]).rstrip("_")
        for col in summary.columns.to_flat_index()
    ]
    return summary


def chemical_identity_overlap_audit(train_df: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for field in ["casrn", "dtxsid", "inchikey"]:
        if field not in train_df or field not in panel:
            continue
        train_values = set(train_df[field].dropna().astype(str).str.strip()) - {""}
        panel_values = set(panel[field].dropna().astype(str).str.strip()) - {""}
        rows.append(
            {
                "identity_field": field,
                "n_train_values": len(train_values),
                "n_panel_values": len(panel_values),
                "n_overlapping_values": len(train_values & panel_values),
            }
        )
    return pd.DataFrame(rows)


def chemical_cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    seed: int,
    n_replicates: int,
) -> pd.DataFrame:
    """Cluster bootstrap case-level regression metrics by chemical identity."""
    chemical_groups = list(frame.groupby("chemical_id", sort=False))
    if len(chemical_groups) < 2:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    metric_rows: list[dict[str, float | int]] = []
    chemical_ids = np.array([chemical_id for chemical_id, _ in chemical_groups], dtype=object)
    group_map = {chemical_id: group for chemical_id, group in chemical_groups}
    for replicate in range(n_replicates):
        sampled = rng.choice(chemical_ids, size=len(chemical_ids), replace=True)
        sample = pd.concat([group_map[chemical_id] for chemical_id in sampled], ignore_index=True)
        metrics = regression_metrics(
            sample["target_log_molar"].to_numpy(dtype=float),
            sample["y_pred"].to_numpy(dtype=float),
        )
        metric_rows.append({"seed": seed, "replicate": replicate, **metrics})
    return pd.DataFrame(metric_rows)


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    write_status(out_dir, "Loading train benchmark and cleaned external panel.")
    recompute_logp = not args.keep_linked_logp
    source_train_df = pd.read_csv(args.data_path)
    if args.use_original_benchmark:
        analysis_train_df = source_train_df
    else:
        analysis_train_df, _ = strict_input_eligibility_audit(source_train_df)
    train_df = attach_rdkit_descriptors(
        analysis_train_df,
        DEFAULT_SCHEMA,
        recompute_logp=recompute_logp,
    )
    panel = attach_rdkit_descriptors(
        pd.read_csv(args.panel_path),
        DEFAULT_SCHEMA,
        recompute_logp=recompute_logp,
    )
    chemical_identity_overlap_audit(train_df, panel).to_csv(
        out_dir / "external_train_panel_identity_overlap_audit.csv", index=False
    )

    seed_predictions: list[pd.DataFrame] = []
    metrics_frames: list[pd.DataFrame] = []
    burden_frames: list[pd.DataFrame] = []
    chemical_burden_frames: list[pd.DataFrame] = []
    cluster_bootstrap_frames: list[pd.DataFrame] = []

    for idx, seed in enumerate(args.seeds, start=1):
        write_status(out_dir, f"Running seed {seed} ({idx}/{len(args.seeds)})")
        train_seed_df, calib_df = calibration_split_by_chemical(train_df, seed=seed)

        feature_builder = EcoFeatureBuilder(
            schema=DEFAULT_SCHEMA,
            include_study_year=args.include_study_year,
            recompute_rdkit_logp=recompute_logp,
            allow_legacy_structure_placeholder=args.allow_legacy_structure_placeholder,
        )
        train_bundle = feature_builder.fit_transform(train_seed_df)
        calib_bundle = feature_builder.transform(calib_df)
        external_bundle = feature_builder.transform(panel)

        model = BootstrapEnsembleRegressor(
            model_name=MODEL_NAME,
            n_members=5,
            seed=seed,
            n_jobs=args.ensemble_n_jobs,
        ).fit(train_bundle.full, train_seed_df[DEFAULT_SCHEMA.target].to_numpy())
        calib_pred = model.predict(calib_bundle.full)
        external_pred = model.predict(external_bundle.full)

        conformal = ScaledConformalRegressor(alpha=0.1).fit(
            calib_df[DEFAULT_SCHEMA.target].to_numpy(),
            calib_pred.mean,
            scale=np.maximum(calib_pred.std, 1e-3),
        )
        calib_interval = conformal.predict(calib_pred.mean, scale=np.maximum(calib_pred.std, 1e-3))
        external_interval = conformal.predict(external_pred.mean, scale=np.maximum(external_pred.std, 1e-3))

        scorer = EcoOODScorer(
            schema=DEFAULT_SCHEMA,
            include_study_year=args.include_study_year,
        ).fit(train_seed_df, train_bundle)
        calib_components = scorer.component_frame(
            calib_df,
            calib_bundle,
            model_std=calib_pred.std,
        )
        scorer.fit_meta(
            calib_components,
            residuals=np.abs(calib_df[DEFAULT_SCHEMA.target].to_numpy() - calib_pred.mean),
        )
        external_components = scorer.predict(
            panel,
            external_bundle,
            model_std=external_pred.std,
        )
        endpoint_balanced_scorer = EcoOODScorer(
            schema=DEFAULT_SCHEMA,
            include_study_year=args.include_study_year,
        ).fit(
            train_seed_df,
            train_bundle,
        )
        endpoint_balanced_scorer.fit_meta(
            calib_components,
            residuals=np.abs(
                calib_df[DEFAULT_SCHEMA.target].to_numpy() - calib_pred.mean
            ),
            groups=calib_df[DEFAULT_SCHEMA.endpoint],
            groupwise_labels=True,
            balance_groups=True,
        )

        ad_scorer = ApplicabilityDomainScorer().fit(
            train_bundle,
            chemical_ids=train_seed_df[DEFAULT_SCHEMA.chemical_id],
        )
        calib_ad = ad_scorer.predict(
            calib_bundle,
            model_std=calib_pred.std,
            interval_width=calib_interval.width,
        )
        external_ad = ad_scorer.predict(
            external_bundle,
            model_std=external_pred.std,
            interval_width=external_interval.width,
        )

        calib_residuals = np.abs(calib_df[DEFAULT_SCHEMA.target].to_numpy() - calib_pred.mean)
        ensemble_sd_risk = CalibrationRiskScorer().fit(
            pd.DataFrame({"ensemble_sd": calib_pred.std}),
            calib_residuals,
        )
        input_space_knn_plus_sd_risk = CalibrationRiskScorer().fit(
            pd.DataFrame(
                {
                    "input_space_knn": calib_ad.distance_to_model,
                    "ensemble_sd": calib_pred.std,
                }
            ),
            calib_residuals,
        )
        equal_block_knn_plus_sd_risk = CalibrationRiskScorer().fit(
            pd.DataFrame(
                {
                    "equal_block_knn": calib_ad.equal_block_distance,
                    "ensemble_sd": calib_pred.std,
                }
            ),
            calib_residuals,
        )
        generic_support_plus_sd_risk = CalibrationRiskScorer().fit(
            pd.DataFrame(
                {
                    "similarity_novelty": calib_ad.similarity,
                    "equal_block_knn": calib_ad.equal_block_distance,
                    "ensemble_sd": calib_pred.std,
                }
            ),
            calib_residuals,
        )

        pred = panel[
            [
                "case_id",
                "chemical_id",
                "chemical_name",
                "casrn",
                "endpoint",
                "species",
                "target_log_molar",
                "case_row_count",
                "document_count",
                "case_spread_log_molar",
                "standard_species_flag",
                "freshwater_keyword_flag",
            ]
        ].copy()
        pred["seed"] = seed
        pred["y_pred"] = external_pred.mean
        pred["abs_error"] = np.abs(pred["y_pred"] - pred["target_log_molar"])
        pred["model_std"] = external_pred.std
        pred["interval_width"] = external_interval.width
        pred["ecoood_score"] = external_components.ecoood_score
        external_component_frame = scorer.component_frame(
            panel,
            external_bundle,
            model_std=external_pred.std,
        )
        pred["ecoood_endpoint_balanced"] = (
            endpoint_balanced_scorer.score_components(external_component_frame)
        )
        pred["ensemble_sd_risk"] = ensemble_sd_risk.predict(
            pd.DataFrame({"ensemble_sd": external_pred.std})
        )
        pred["input_space_knn_plus_sd_risk"] = input_space_knn_plus_sd_risk.predict(
            pd.DataFrame(
                {
                    "input_space_knn": external_ad.distance_to_model,
                    "ensemble_sd": external_pred.std,
                }
            )
        )
        pred["equal_block_knn_plus_sd_risk"] = equal_block_knn_plus_sd_risk.predict(
            pd.DataFrame(
                {
                    "equal_block_knn": external_ad.equal_block_distance,
                    "ensemble_sd": external_pred.std,
                }
            )
        )
        pred["generic_support_plus_sd_risk"] = generic_support_plus_sd_risk.predict(
            pd.DataFrame(
                {
                    "similarity_novelty": external_ad.similarity,
                    "equal_block_knn": external_ad.equal_block_distance,
                    "ensemble_sd": external_pred.std,
                }
            )
        )
        pred["ad_similarity"] = external_ad.similarity
        pred["ad_distance_to_model"] = external_ad.distance_to_model
        pred["ad_equal_block_distance"] = external_ad.equal_block_distance
        pred["ad_equal_block_distance_distinct_chemical"] = (
            external_ad.equal_block_distance_distinct_chemical
        )
        pred["ad_similarity_risk"] = pred["ad_similarity"]
        seed_predictions.append(pred)
        pred.to_csv(out_dir / f"external_predictions_seed_{seed}.csv", index=False)

        metric_frame, burden_frame = summarize_seed(seed, pred)
        metrics_frames.append(metric_frame)
        burden_frames.append(burden_frame)
        chemical_burden_frames.append(chemical_level_fixed_workload(seed, pred))
        cluster_bootstrap_frames.append(
            chemical_cluster_bootstrap(
                pred,
                seed=seed,
                n_replicates=args.bootstrap_replicates,
            )
        )

    predictions = pd.concat(seed_predictions, ignore_index=True)
    metrics_all = pd.concat(metrics_frames, ignore_index=True)
    burden_all = pd.concat(burden_frames, ignore_index=True)
    chemical_burden_all = pd.concat(chemical_burden_frames, ignore_index=True)

    predictions.to_csv(out_dir / "external_predictions_all.csv", index=False)
    metrics_all.to_csv(out_dir / "external_metrics_all.csv", index=False)
    burden_all.to_csv(out_dir / "external_burden_all.csv", index=False)
    chemical_burden_all.to_csv(
        out_dir / "external_chemical_level_burden_all.csv", index=False
    )
    cluster_bootstrap = pd.concat(cluster_bootstrap_frames, ignore_index=True)
    cluster_bootstrap.to_csv(out_dir / "external_cluster_bootstrap_all.csv", index=False)
    cluster_quantiles = (
        cluster_bootstrap[["rmse", "mae", "spearman", "bias"]]
        .quantile([0.025, 0.5, 0.975])
        .transpose()
        .reset_index()
        .rename(columns={"index": "metric", 0.025: "ci_2_5", 0.5: "median", 0.975: "ci_97_5"})
    )
    cluster_quantiles.to_csv(out_dir / "external_cluster_bootstrap_summary.csv", index=False)

    metrics_summary = aggregate_summary(metrics_all, ["endpoint", "method", "score_col"])
    burden_summary = aggregate_summary(burden_all, ["endpoint", "method", "score_col"])
    chemical_burden_summary = aggregate_summary(chemical_burden_all, ["method"])
    metrics_summary.to_csv(out_dir / "external_metrics_summary.csv", index=False)
    burden_summary.to_csv(out_dir / "external_burden_summary.csv", index=False)
    chemical_burden_summary.to_csv(
        out_dir / "external_chemical_level_burden_summary.csv", index=False
    )

    case_summary = (
        predictions.groupby(
            ["case_id", "chemical_id", "chemical_name", "casrn", "endpoint", "species"],
            as_index=False,
        )
        .agg(
            target_log_molar=("target_log_molar", "first"),
            case_row_count=("case_row_count", "first"),
            document_count=("document_count", "first"),
            case_spread_log_molar=("case_spread_log_molar", "first"),
            y_pred_mean=("y_pred", "mean"),
            y_pred_std=("y_pred", "std"),
            abs_error_mean=("abs_error", "mean"),
            abs_error_std=("abs_error", "std"),
            ecoood_score_mean=("ecoood_score", "mean"),
            distance_to_model_mean=("ad_distance_to_model", "mean"),
            similarity_risk_mean=("ad_similarity_risk", "mean"),
        )
    )
    case_summary.to_csv(out_dir / "external_case_summary.csv", index=False)

    lines = [
        f"Train benchmark path: {args.data_path}",
        f"Benchmark rows loaded: {len(source_train_df)}",
        f"Analysis training rows before external calibration split: {len(train_df)}",
        f"Application input gate used: {not args.use_original_benchmark}",
        f"Panel path: {args.panel_path}",
        f"Cases: {panel.shape[0]}",
        f"Chemicals: {panel['chemical_name'].nunique()}",
        "Endpoint counts:",
        panel["endpoint"].value_counts().to_string(),
    ]
    (out_dir / "external_validation_notes.txt").write_text("\n".join(lines) + "\n")
    write_status(out_dir, "Completed ECHA PMRA external validation.")


if __name__ == "__main__":
    main()
