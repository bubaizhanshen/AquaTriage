from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


DEPLOYMENT_SPLITS = [
    "chemical_random",
    "scaffold",
    "temporal",
    "species",
    "chemical_class",
]
DECISION_MAP_SPLITS = ["temporal", "species", "chemical_class"]
MODELS = ["lightgbm", "random_forest", "xgboost"]
METHOD_COLUMNS = {
    "Prediction-error risk score": "max_ecoood",
    "Ensemble SD risk": "max_ensemble_sd_risk",
    "Input-space kNN + SD risk": "max_input_space_knn_plus_sd_risk",
    "Block-normalized kNN + SD risk": "max_equal_block_knn_plus_sd_risk",
    "Generic support + SD risk": "max_generic_support_plus_sd_risk",
    "Input-space kNN distance": "max_input_distance",
    "Block-normalized kNN distance": "max_equal_block_distance",
    "Distinct-chemical block-normalized kNN distance": (
        "max_equal_block_distance_distinct_chemical"
    ),
    "Similarity AD": "max_similarity_risk",
    "Prediction-error risk score, endpoint-balanced": "max_ecoood_endpoint_balanced",
    "Prediction-error risk score, top 20%": "max_ecoood_q80",
    "Prediction-error risk score, top 5%": "max_ecoood_q95",
}
ACTION_ORDER = ["screen_now", "lower_priority", "withhold_review", "prioritize_testing"]
DEFAULT_TRIAGE_COLUMN = "ad_equal_block_distance"
DEFAULT_TRIAGE_AGGREGATE = "max_equal_block_distance"
CLASS_ORDER = [
    "Conazoles",
    "Per- and Polyfluoroalkyl Substances (PFAS)",
    "Neonicotinoids",
    "Endocrine Disrupting Chemicals (EDCs)",
    "Pharmaceutical Personal Care Products (PPCPs)",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate multi-seed EcoOOD benchmark and fixed-workload screening results."
    )
    parser.add_argument("--input-root", type=Path, default=Path("outputs/corrected"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/corrected/aggregate"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 42, 43, 44])
    parser.add_argument("--models", nargs="+", default=MODELS)
    return parser.parse_args()


def _group_summary(frame: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    summary = frame.groupby(group_cols, dropna=False)[value_cols].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in col if part).rstrip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in summary.columns.to_flat_index()
    ]
    return summary


def _run_dir(root: Path, seed: int, split: str) -> Path:
    del split
    structured = root / f"seed_{seed}" / "structured"
    return structured if structured.exists() else root / f"seed_{seed}"


def load_core_outputs(
    root: Path,
    seeds: list[int],
    models: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_frames: list[pd.DataFrame] = []
    score_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    for seed in seeds:
        run_dir = _run_dir(root, seed, "")
        metrics = pd.read_csv(run_dir / "benchmark_summary.csv")
        metrics = metrics[metrics["model"].isin(models)].copy()
        metrics["seed"] = seed
        metric_frames.append(metrics)

        scores = pd.read_csv(run_dir / "ood_score_summary.csv")
        scores = scores[scores["model"].isin(models)].copy()
        scores["seed"] = seed
        score_frames.append(scores)

        for split in DEPLOYMENT_SPLITS:
            run_dir = _run_dir(root, seed, split)
            for model in models:
                path = run_dir / split / model / "predictions.csv"
                frame = pd.read_csv(path)
                frame["seed"] = seed
                frame["split"] = split
                frame["model"] = model
                prediction_frames.append(frame)
    return (
        pd.concat(metric_frames, ignore_index=True),
        pd.concat(score_frames, ignore_index=True),
        pd.concat(prediction_frames, ignore_index=True),
    )


PREDICTION_SCORE_COLUMNS = {
    "ecoood_score": "max_ecoood",
    "ensemble_sd_risk": "max_ensemble_sd_risk",
    "input_space_knn_plus_sd_risk": "max_input_space_knn_plus_sd_risk",
    "equal_block_knn_plus_sd_risk": "max_equal_block_knn_plus_sd_risk",
    "generic_support_plus_sd_risk": "max_generic_support_plus_sd_risk",
    "ad_distance_to_model": "max_input_distance",
    "ad_equal_block_distance": "max_equal_block_distance",
    "ad_equal_block_distance_distinct_chemical": (
        "max_equal_block_distance_distinct_chemical"
    ),
    "ad_similarity": "max_similarity_risk",
    "ecoood_endpoint_balanced": "max_ecoood_endpoint_balanced",
    "ecoood_q80": "max_ecoood_q80",
    "ecoood_q95": "max_ecoood_q95",
}


def aggregate_chemical_predictions(
    predictions: pd.DataFrame,
    *,
    mode: str = "extreme",
) -> pd.DataFrame:
    identifier_cols = [
        "seed",
        "model",
        "split",
        "chemical_id",
        "chemical_name",
        "casrn",
        "chemical_class",
    ]
    available_scores = {
        source: output
        for source, output in PREDICTION_SCORE_COLUMNS.items()
        if source in predictions.columns
    }
    if mode == "extreme":
        aggregations: dict[str, tuple[str, str]] = {
            "min_true_tox": ("y_true", "min"),
            "min_pred_tox": ("y_pred", "min"),
            "endpoint_breadth": ("endpoint", "nunique"),
            "row_count": ("chemical_id", "size"),
        }
        aggregations.update(
            {output: (source, "max") for source, output in available_scores.items()}
        )
        result = predictions.groupby(
            identifier_cols,
            dropna=False,
            as_index=False,
        ).agg(**aggregations)
    elif mode == "endpoint_median":
        endpoint_cols = [*identifier_cols, "endpoint"]
        endpoint_aggregations: dict[str, tuple[str, str]] = {
            "endpoint_true_tox": ("y_true", "median"),
            "endpoint_pred_tox": ("y_pred", "median"),
            "endpoint_row_count": ("chemical_id", "size"),
        }
        endpoint_aggregations.update(
            {
                f"endpoint_{output}": (source, "median")
                for source, output in available_scores.items()
            }
        )
        endpoint_frame = predictions.groupby(
            endpoint_cols,
            dropna=False,
            as_index=False,
        ).agg(**endpoint_aggregations)
        chemical_aggregations: dict[str, tuple[str, str]] = {
            "min_true_tox": ("endpoint_true_tox", "min"),
            "min_pred_tox": ("endpoint_pred_tox", "min"),
            "endpoint_breadth": ("endpoint", "nunique"),
            "row_count": ("endpoint_row_count", "sum"),
        }
        chemical_aggregations.update(
            {
                output: (f"endpoint_{output}", "max")
                for output in available_scores.values()
            }
        )
        result = endpoint_frame.groupby(
            identifier_cols,
            dropna=False,
            as_index=False,
        ).agg(**chemical_aggregations)
    else:
        raise ValueError("mode must be 'extreme' or 'endpoint_median'.")
    result["aggregation_method"] = mode
    return result


def aggregate_endpoint_relative_chemical_predictions(
    predictions: pd.DataFrame,
    *,
    high_concern_quantile: float = 0.25,
) -> pd.DataFrame:
    """Build chemical-level labels from endpoint-specific relative thresholds.

    Records are first summarized within chemical-endpoint groups. The measured
    endpoint median is compared with the corresponding endpoint distribution,
    and a chemical is high concern if any represented endpoint falls below its
    endpoint-specific cutoff. Reliability signals are median-aggregated within
    endpoint and then maximized across a chemical's represented endpoints.
    """
    identifier_cols = [
        "seed",
        "model",
        "split",
        "chemical_id",
        "chemical_name",
        "casrn",
        "chemical_class",
    ]
    endpoint_cols = [*identifier_cols, "endpoint"]
    available_scores = {
        source: output
        for source, output in PREDICTION_SCORE_COLUMNS.items()
        if source in predictions.columns
    }
    endpoint_aggregations: dict[str, tuple[str, str]] = {
        "endpoint_true_tox": ("y_true", "median"),
        "endpoint_pred_tox": ("y_pred", "median"),
        "endpoint_row_count": ("chemical_id", "size"),
    }
    endpoint_aggregations.update(
        {
            f"endpoint_{output}": (source, "median")
            for source, output in available_scores.items()
        }
    )
    endpoint_frame = predictions.groupby(
        endpoint_cols,
        dropna=False,
        as_index=False,
    ).agg(**endpoint_aggregations)
    endpoint_frame["endpoint_toxicity_cutoff"] = endpoint_frame.groupby(
        ["seed", "model", "split", "endpoint"],
        dropna=False,
    )["endpoint_true_tox"].transform(
        lambda values: float(values.quantile(high_concern_quantile))
    )
    endpoint_frame["endpoint_true_high_concern"] = (
        endpoint_frame["endpoint_true_tox"]
        <= endpoint_frame["endpoint_toxicity_cutoff"]
    )
    endpoint_frame["endpoint_pred_high_concern"] = (
        endpoint_frame["endpoint_pred_tox"]
        <= endpoint_frame["endpoint_toxicity_cutoff"]
    )

    chemical_aggregations: dict[str, tuple[str, str]] = {
        "min_true_tox": ("endpoint_true_tox", "min"),
        "min_pred_tox": ("endpoint_pred_tox", "min"),
        "endpoint_breadth": ("endpoint", "nunique"),
        "row_count": ("endpoint_row_count", "sum"),
        "true_high_concern_endpoint_relative": (
            "endpoint_true_high_concern",
            "max",
        ),
        "pred_high_concern_endpoint_relative": (
            "endpoint_pred_high_concern",
            "max",
        ),
    }
    chemical_aggregations.update(
        {
            output: (f"endpoint_{output}", "max")
            for output in available_scores.values()
        }
    )
    result = endpoint_frame.groupby(
        identifier_cols,
        dropna=False,
        as_index=False,
    ).agg(**chemical_aggregations)
    result["aggregation_method"] = "endpoint_relative"
    result["high_concern_quantile"] = high_concern_quantile
    return result


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _assign_actions(frame: pd.DataFrame, review_mask: pd.Series) -> pd.Series:
    action = pd.Series("lower_priority", index=frame.index, dtype=object)
    action.loc[frame["pred_high_concern"] & ~review_mask] = "screen_now"
    action.loc[~frame["pred_high_concern"] & review_mask] = "withhold_review"
    action.loc[frame["pred_high_concern"] & review_mask] = "prioritize_testing"
    return action


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def _workload_metrics(
    classified: pd.DataFrame,
    *,
    seed: int,
    model: str,
    split: str,
    method: str,
    burden: float,
    toxicity_cutoff: float,
    review_n: int,
) -> dict[str, object]:
    lower = classified["screening_action"].eq("lower_priority")
    baseline_fn = classified["baseline_false_negative"]
    rescued = classified["rescued_false_negative"]
    high_concern_left_lower = classified["true_high_concern"] & lower
    false_omission_rate = _safe_rate(
        int(high_concern_left_lower.sum()),
        int(lower.sum()),
    )
    return {
        "seed": seed,
        "model": model,
        "split": split,
        "method": method,
        "review_burden": burden,
        "toxicity_cutoff": toxicity_cutoff,
        "review_count": int(classified["reviewed"].sum()),
        "lower_priority_false_omission_rate": false_omission_rate,
        # Backward-compatible alias retained for frozen v0.2.0 output readers.
        "lower_priority_false_reassurance": false_omission_rate,
        "lower_priority_queue_size": int(lower.sum()),
        "high_concern_left_lower_priority_count": int(
            high_concern_left_lower.sum()
        ),
        "high_concern_left_lower_priority_fraction": _safe_rate(
            int(high_concern_left_lower.sum()),
            int(classified["true_high_concern"].sum()),
        ),
        "baseline_false_negatives": int(baseline_fn.sum()),
        "rescued_false_negatives": int(rescued.sum()),
        "rescued_baseline_misses": int(rescued.sum()),
        "rescued_false_negative_fraction": _safe_rate(
            int(rescued.sum()),
            int(baseline_fn.sum()),
        ),
        "rescued_baseline_miss_fraction": _safe_rate(
            int(rescued.sum()),
            int(baseline_fn.sum()),
        ),
        "rescued_per_100_reviews": _safe_rate(int(rescued.sum()) * 100, review_n),
        "prioritize_testing_count": int(
            classified["screening_action"].eq("prioritize_testing").sum()
        ),
        "withhold_review_count": int(
            classified["screening_action"].eq("withhold_review").sum()
        ),
    }


def summarize_direct_rule(chemical_panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    classified_frames: list[pd.DataFrame] = []
    rows: list[dict[str, object]] = []
    for (seed, model, split), group in chemical_panel.groupby(["seed", "model", "split"], sort=False):
        frame = group.copy()
        toxicity_cutoff = float(frame["min_true_tox"].quantile(0.25))
        score_cutoff = float(frame[DEFAULT_TRIAGE_AGGREGATE].quantile(0.75))
        frame["toxicity_cutoff"] = toxicity_cutoff
        frame["score_cutoff"] = score_cutoff
        frame["true_high_concern"] = frame["min_true_tox"] <= toxicity_cutoff
        frame["pred_high_concern"] = frame["min_pred_tox"] <= toxicity_cutoff
        frame["reviewed"] = frame[DEFAULT_TRIAGE_AGGREGATE] >= score_cutoff
        frame["baseline_action"] = np.where(frame["pred_high_concern"], "screen_now", "lower_priority")
        frame["screening_action"] = _assign_actions(frame, frame["reviewed"])
        frame["baseline_false_negative"] = frame["true_high_concern"] & ~frame["pred_high_concern"]
        frame["rescued_false_negative"] = frame["baseline_false_negative"] & frame["reviewed"]
        classified_frames.append(frame)

        baseline_low = ~frame["pred_high_concern"]
        routed_low = frame["screening_action"].eq("lower_priority")
        baseline_fn = frame["baseline_false_negative"]
        rescued = frame["rescued_false_negative"]
        baseline_high_concern_left_lower = (
            frame["true_high_concern"] & baseline_low
        )
        routed_high_concern_left_lower = (
            frame["true_high_concern"] & routed_low
        )
        baseline_false_omission_rate = _safe_rate(
            int(baseline_high_concern_left_lower.sum()),
            int(baseline_low.sum()),
        )
        routed_false_omission_rate = _safe_rate(
            int(routed_high_concern_left_lower.sum()),
            int(routed_low.sum()),
        )
        row: dict[str, object] = {
            "seed": seed,
            "model": model,
            "split": split,
            "n_chemical_split_cases": int(len(frame)),
            "toxicity_cutoff": toxicity_cutoff,
            "reliability_cutoff": score_cutoff,
            "baseline_false_omission_rate": baseline_false_omission_rate,
            "routed_false_omission_rate": routed_false_omission_rate,
            "baseline_false_reassurance": baseline_false_omission_rate,
            "routed_false_reassurance": routed_false_omission_rate,
            "baseline_lower_priority_queue_size": int(baseline_low.sum()),
            "routed_lower_priority_queue_size": int(routed_low.sum()),
            "routed_high_concern_left_lower_priority_fraction": _safe_rate(
                int(routed_high_concern_left_lower.sum()),
                int(frame["true_high_concern"].sum()),
            ),
            "baseline_false_negatives": int(baseline_fn.sum()),
            "rescued_false_negatives": int(rescued.sum()),
            "rescued_baseline_misses": int(rescued.sum()),
            "rescued_false_negative_fraction": _safe_rate(int(rescued.sum()), int(baseline_fn.sum())),
            "rescued_baseline_miss_fraction": _safe_rate(
                int(rescued.sum()),
                int(baseline_fn.sum()),
            ),
        }
        for action in ACTION_ORDER:
            row[f"{action}_count"] = int(frame["screening_action"].eq(action).sum())
        rows.append(row)
    return pd.concat(classified_frames, ignore_index=True), pd.DataFrame(rows)


def summarize_fixed_workload(
    chemical_panel: pd.DataFrame,
    burdens: tuple[float, ...] = (0.15, 0.20, 0.25, 0.30, 0.35),
    random_replicates: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    classification_frames: list[pd.DataFrame] = []
    rows: list[dict[str, object]] = []
    for (seed, model, split), group in chemical_panel.groupby(["seed", "model", "split"], sort=False):
        frame = group.copy()
        endpoint_relative = {
            "true_high_concern_endpoint_relative",
            "pred_high_concern_endpoint_relative",
        }.issubset(frame.columns)
        if endpoint_relative:
            toxicity_cutoff = float("nan")
            frame["true_high_concern"] = frame[
                "true_high_concern_endpoint_relative"
            ].astype(bool)
            frame["pred_high_concern"] = frame[
                "pred_high_concern_endpoint_relative"
            ].astype(bool)
        else:
            toxicity_cutoff = float(frame["min_true_tox"].quantile(0.25))
            frame["true_high_concern"] = frame["min_true_tox"] <= toxicity_cutoff
            frame["pred_high_concern"] = frame["min_pred_tox"] <= toxicity_cutoff
        frame["baseline_false_negative"] = frame["true_high_concern"] & ~frame["pred_high_concern"]
        available_methods = {
            method: score_col
            for method, score_col in METHOD_COLUMNS.items()
            if score_col in frame.columns
        }
        for burden in burdens:
            review_n = max(1, int(round(len(frame) * burden)))
            for method, score_col in available_methods.items():
                ordered = frame.sort_values([score_col, "split", "chemical_id"], ascending=[False, True, True])
                reviewed_idx = ordered.head(review_n).index
                reviewed = frame.index.to_series().isin(reviewed_idx)
                classified = frame.copy()
                classified["method"] = method
                classified["review_burden"] = burden
                classified["reviewed"] = reviewed.to_numpy()
                classified["screening_action"] = _assign_actions(classified, classified["reviewed"])
                classified["rescued_false_negative"] = (
                    classified["baseline_false_negative"] & classified["reviewed"]
                )
                classification_frames.append(classified)
                rows.append(
                    _workload_metrics(
                        classified,
                        seed=seed,
                        model=model,
                        split=split,
                        method=method,
                        burden=burden,
                        toxicity_cutoff=toxicity_cutoff,
                        review_n=review_n,
                    )
                )
            rng = np.random.default_rng(
                _stable_seed(seed, model, split, burden, "random_review")
            )
            n_cases = len(frame)
            random_order = np.argsort(
                rng.random((random_replicates, n_cases)),
                axis=1,
            )[:, :review_n]
            reviewed_matrix = np.zeros(
                (random_replicates, n_cases),
                dtype=bool,
            )
            reviewed_matrix[
                np.arange(random_replicates)[:, None],
                random_order,
            ] = True

            true_high = frame["true_high_concern"].to_numpy(dtype=bool)
            pred_high = frame["pred_high_concern"].to_numpy(dtype=bool)
            baseline_fn = frame["baseline_false_negative"].to_numpy(dtype=bool)
            lower_priority = (~pred_high)[None, :] & ~reviewed_matrix
            rescued = baseline_fn[None, :] & reviewed_matrix
            lower_counts = lower_priority.sum(axis=1)
            false_omission_counts = (
                true_high[None, :] & lower_priority
            ).sum(axis=1)
            false_omission_rate = np.divide(
                false_omission_counts,
                lower_counts,
                out=np.full(random_replicates, np.nan, dtype=float),
                where=lower_counts > 0,
            )
            high_concern_count = int(true_high.sum())
            high_concern_left_lower_fraction = np.divide(
                false_omission_counts,
                high_concern_count,
                out=np.full(random_replicates, np.nan, dtype=float),
                where=high_concern_count > 0,
            )
            baseline_fn_count = int(baseline_fn.sum())
            rescued_counts = rescued.sum(axis=1)
            rescued_fraction = (
                rescued_counts / baseline_fn_count
                if baseline_fn_count
                else np.full(random_replicates, np.nan, dtype=float)
            )

            first_random = frame.copy()
            first_random["method"] = "Random review"
            first_random["review_burden"] = burden
            first_random["reviewed"] = reviewed_matrix[0]
            first_random["screening_action"] = _assign_actions(
                first_random,
                first_random["reviewed"],
            )
            first_random["rescued_false_negative"] = (
                first_random["baseline_false_negative"]
                & first_random["reviewed"]
            )
            classification_frames.append(first_random)
            random_row: dict[str, object] = {
                "seed": seed,
                "model": model,
                "split": split,
                "method": "Random review",
                "review_burden": burden,
                "toxicity_cutoff": toxicity_cutoff,
                "review_count": review_n,
                "random_replicates": random_replicates,
                "lower_priority_false_omission_rate": float(
                    np.nanmean(false_omission_rate)
                    if np.isfinite(false_omission_rate).any()
                    else np.nan
                ),
                # Backward-compatible alias retained for frozen output readers.
                "lower_priority_false_reassurance": float(
                    np.nanmean(false_omission_rate)
                    if np.isfinite(false_omission_rate).any()
                    else np.nan
                ),
                "lower_priority_queue_size": float(lower_counts.mean()),
                "high_concern_left_lower_priority_count": float(
                    false_omission_counts.mean()
                ),
                "high_concern_left_lower_priority_fraction": float(
                    np.nanmean(high_concern_left_lower_fraction)
                    if np.isfinite(high_concern_left_lower_fraction).any()
                    else np.nan
                ),
                "baseline_false_negatives": baseline_fn_count,
                "rescued_false_negatives": float(rescued_counts.mean()),
                "rescued_baseline_misses": float(rescued_counts.mean()),
                "rescued_false_negative_fraction": float(
                    np.nanmean(rescued_fraction)
                    if np.isfinite(rescued_fraction).any()
                    else np.nan
                ),
                "rescued_baseline_miss_fraction": float(
                    np.nanmean(rescued_fraction)
                    if np.isfinite(rescued_fraction).any()
                    else np.nan
                ),
                "rescued_per_100_reviews": float(
                    100 * rescued_counts.mean() / review_n
                ),
                "prioritize_testing_count": float(
                    (reviewed_matrix & pred_high[None, :]).sum(axis=1).mean()
                ),
                "withhold_review_count": float(
                    (reviewed_matrix & ~pred_high[None, :]).sum(axis=1).mean()
                ),
            }
            rows.append(random_row)
    return pd.concat(classification_frames, ignore_index=True), pd.DataFrame(rows)


def _primary_class(value: object) -> str | None:
    text = "" if pd.isna(value) else str(value)
    for class_name in CLASS_ORDER:
        if class_name in text:
            return class_name
    return None


def build_class_focused_panel(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = predictions[
        predictions["model"].eq("lightgbm") & predictions["split"].isin(DECISION_MAP_SPLITS)
    ].copy()
    frame["primary_class"] = frame["chemical_class"].map(_primary_class)
    frame = frame[frame["primary_class"].notna()].copy()
    panel = (
        frame.groupby(["chemical_id", "chemical_name", "casrn"], dropna=False, as_index=False)
        .agg(
            primary_class=("primary_class", lambda values: values.mode().iat[0]),
            class_labels=("chemical_class", lambda values: "; ".join(sorted(set(values.astype(str))))),
            n_seeds=("seed", "nunique"),
            split_breadth=("split", "nunique"),
            endpoint_breadth=("endpoint", "nunique"),
            min_pred_tox=("y_pred", "min"),
            min_true_tox=("y_true", "min"),
            max_reliability=(DEFAULT_TRIAGE_COLUMN, "max"),
            median_reliability=(DEFAULT_TRIAGE_COLUMN, "median"),
        )
    )
    toxicity_cutoff = float(panel["min_pred_tox"].quantile(0.25))
    score_cutoff = float(panel["max_reliability"].quantile(0.75))
    panel["toxicity_cutoff"] = toxicity_cutoff
    panel["reliability_cutoff"] = score_cutoff
    panel["pred_high_concern"] = panel["min_pred_tox"] <= toxicity_cutoff
    panel["reviewed"] = panel["max_reliability"] >= score_cutoff
    panel["screening_action"] = _assign_actions(panel, panel["reviewed"])
    counts = (
        panel.groupby(["primary_class", "screening_action"], as_index=False)
        .size()
        .rename(columns={"size": "n_chemicals"})
    )
    return panel, counts


def build_decision_map_summary(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = predictions[
        predictions["model"].eq("lightgbm") & predictions["split"].isin(DECISION_MAP_SPLITS)
    ].copy()
    toxicity_cutoff = float(frame["y_pred"].quantile(0.25))
    score_cutoff = float(frame[DEFAULT_TRIAGE_COLUMN].quantile(0.75))
    frame["pred_high_concern"] = frame["y_pred"] <= toxicity_cutoff
    frame["reviewed"] = frame[DEFAULT_TRIAGE_COLUMN] >= score_cutoff
    frame["screening_action"] = _assign_actions(frame, frame["reviewed"])
    frame["abs_error"] = (frame["y_true"] - frame["y_pred"]).abs()
    severe_cutoff = float(frame["abs_error"].quantile(0.90))
    frame["top_decile_error"] = frame["abs_error"] >= severe_cutoff
    frame["true_high_concern"] = frame["y_true"] <= toxicity_cutoff
    frame["covered"] = (frame["y_true"] >= frame["interval_lower"]) & (
        frame["y_true"] <= frame["interval_upper"]
    )
    summary = (
        frame.groupby("screening_action", as_index=False)
        .agg(
            n_rows=("screening_action", "size"),
            rmse=("abs_error", lambda values: float(np.sqrt(np.mean(np.square(values))))),
            coverage=("covered", "mean"),
            top_decile_error_rate=("top_decile_error", "mean"),
            measured_high_concern_fraction=("true_high_concern", "mean"),
        )
    )
    summary["toxicity_cutoff"] = toxicity_cutoff
    summary["reliability_cutoff"] = score_cutoff
    return frame, summary


def build_threshold_sensitivity(decision_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for toxicity_q in [0.20, 0.25, 0.30]:
        for reliability_q in [0.70, 0.75, 0.80]:
            frame = decision_rows.copy()
            toxicity_cutoff = float(frame["y_pred"].quantile(toxicity_q))
            score_cutoff = float(frame[DEFAULT_TRIAGE_COLUMN].quantile(reliability_q))
            frame["pred_high_concern"] = frame["y_pred"] <= toxicity_cutoff
            frame["reviewed"] = frame[DEFAULT_TRIAGE_COLUMN] >= score_cutoff
            frame["action_tmp"] = _assign_actions(frame, frame["reviewed"])
            row: dict[str, object] = {
                "toxicity_quantile": toxicity_q,
                "reliability_quantile": reliability_q,
                "toxicity_cutoff": toxicity_cutoff,
                "reliability_cutoff": score_cutoff,
            }
            for action in ACTION_ORDER:
                row[f"{action}_fraction"] = float(frame["action_tmp"].eq(action).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def write_outputs(
    output_dir: Path,
    metrics: pd.DataFrame,
    scores: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_values = [
        col for col in metrics.columns if col not in {"seed", "split", "model"} and pd.api.types.is_numeric_dtype(metrics[col])
    ]
    score_values = [
        col
        for col in scores.columns
        if col not in {"seed", "split", "model", "method"} and pd.api.types.is_numeric_dtype(scores[col])
    ]
    metrics.to_csv(output_dir / "benchmark_summary_all_seeds.csv", index=False)
    _group_summary(metrics, ["split", "model"], metric_values).to_csv(
        output_dir / "benchmark_summary_agg.csv", index=False
    )
    scores.to_csv(output_dir / "ood_score_summary_all_seeds.csv", index=False)
    _group_summary(scores, ["split", "model", "method"], score_values).to_csv(
        output_dir / "ood_score_summary_agg.csv", index=False
    )
    predictions.to_csv(output_dir / "predictions_all_seeds.csv", index=False)

    chemical_panel = aggregate_chemical_predictions(predictions)
    chemical_panel.to_csv(output_dir / "chemical_split_panel.csv", index=False)
    direct_rows, direct_summary = summarize_direct_rule(chemical_panel)
    direct_rows.to_csv(output_dir / "screening_rule_classified.csv", index=False)
    direct_summary.to_csv(output_dir / "screening_rule_summary_all_seeds.csv", index=False)
    direct_values = [
        col
        for col in direct_summary.columns
        if col not in {"seed", "model", "split"} and pd.api.types.is_numeric_dtype(direct_summary[col])
    ]
    _group_summary(direct_summary, ["model", "split"], direct_values).to_csv(
        output_dir / "screening_rule_summary_agg.csv", index=False
    )

    workload_rows, workload_summary = summarize_fixed_workload(chemical_panel)
    workload_rows.to_csv(output_dir / "review_workload_classified.csv", index=False)
    workload_summary.to_csv(output_dir / "review_workload_summary_all_seeds.csv", index=False)
    workload_values = [
        col
        for col in workload_summary.columns
        if col not in {"seed", "model", "split", "method"} and pd.api.types.is_numeric_dtype(workload_summary[col])
    ]
    _group_summary(
        workload_summary,
        ["model", "split", "method", "review_burden"],
        workload_values,
    ).to_csv(output_dir / "review_workload_summary_agg.csv", index=False)

    endpoint_median_panel = aggregate_chemical_predictions(
        predictions,
        mode="endpoint_median",
    )
    endpoint_median_panel.to_csv(
        output_dir / "chemical_split_panel_endpoint_median.csv",
        index=False,
    )
    _, endpoint_median_summary = summarize_fixed_workload(endpoint_median_panel)
    endpoint_median_summary.to_csv(
        output_dir / "review_workload_endpoint_median_all_seeds.csv",
        index=False,
    )
    endpoint_median_values = [
        col
        for col in endpoint_median_summary.columns
        if col not in {"seed", "model", "split", "method"}
        and pd.api.types.is_numeric_dtype(endpoint_median_summary[col])
    ]
    _group_summary(
        endpoint_median_summary,
        ["model", "split", "method", "review_burden"],
        endpoint_median_values,
    ).to_csv(
        output_dir / "review_workload_endpoint_median_agg.csv",
        index=False,
    )

    endpoint_relative_panel = aggregate_endpoint_relative_chemical_predictions(
        predictions,
        high_concern_quantile=0.25,
    )
    endpoint_relative_panel.to_csv(
        output_dir / "chemical_split_panel_endpoint_relative.csv",
        index=False,
    )
    endpoint_relative_rows, endpoint_relative_summary = summarize_fixed_workload(
        endpoint_relative_panel
    )
    endpoint_relative_rows.to_csv(
        output_dir / "review_workload_endpoint_relative_classified.csv",
        index=False,
    )
    endpoint_relative_summary.to_csv(
        output_dir / "review_workload_endpoint_relative_all_seeds.csv",
        index=False,
    )
    endpoint_relative_values = [
        col
        for col in endpoint_relative_summary.columns
        if col not in {"seed", "model", "split", "method"}
        and pd.api.types.is_numeric_dtype(endpoint_relative_summary[col])
    ]
    _group_summary(
        endpoint_relative_summary,
        ["model", "split", "method", "review_burden"],
        endpoint_relative_values,
    ).to_csv(
        output_dir / "review_workload_endpoint_relative_agg.csv",
        index=False,
    )

    class_panel, class_counts = build_class_focused_panel(predictions)
    class_panel.to_csv(output_dir / "class_focused_screening_panel.csv", index=False)
    class_counts.to_csv(output_dir / "class_focused_screening_counts.csv", index=False)

    decision_rows, decision_summary = build_decision_map_summary(predictions)
    decision_rows.to_csv(output_dir / "decision_map_rows.csv", index=False)
    decision_summary.to_csv(output_dir / "decision_map_summary.csv", index=False)
    build_threshold_sensitivity(decision_rows).to_csv(
        output_dir / "decision_map_threshold_sensitivity.csv", index=False
    )


def main() -> None:
    args = parse_args()
    metrics, scores, predictions = load_core_outputs(args.input_root, args.seeds, args.models)
    write_outputs(args.output_dir, metrics, scores, predictions)
    print(f"Wrote multi-seed summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
