"""Build compact statistical audit tables from completed benchmark runs.

The script deliberately keeps split/seed instances separate.  It does not pool
the same chemical across deployment splits when producing fixed-workload
summaries.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from ecoood.splits import named_class_for_seed


CORE_SPLITS = [
    "random",
    "chemical_random",
    "scaffold",
    "temporal",
    "species",
    "chemical_class",
]
DEPLOYMENT_SPLITS = [
    "chemical_random",
    "scaffold",
    "temporal",
    "species",
    "chemical_class",
]
LARGE_DEPLOYMENT_SPLITS = [
    "chemical_random",
    "scaffold",
    "temporal",
    "species",
]
PRIMARY_WORKLOAD_METHODS = [
    "Random review",
    "Prediction-error risk score",
    "Ensemble SD risk",
    "Block-normalized kNN + SD risk",
    "Block-normalized kNN distance",
    "Distinct-chemical block-normalized kNN distance",
    "Similarity AD",
]
COMPONENT_COLUMNS = [
    "d_chem_knn",
    "d_chem_mahal",
    "d_species_tax",
    "d_context",
    "context_missing_fraction",
    "d_mech",
    "bioactivity_missing_fraction",
    "u_model",
    "interval_width",
]


def _safe_nanmean(values: list[float] | np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if len(finite) else float("nan")


def _mean_sd(frame: pd.DataFrame, keys: list[str], values: list[str]) -> pd.DataFrame:
    result = frame.groupby(keys, dropna=False)[values].agg(["mean", "std"]).reset_index()
    result.columns = [
        item if isinstance(item, str) else "_".join(part for part in item if part).rstrip("_")
        for item in result.columns.to_flat_index()
    ]
    return result


def _normalize_legacy_score_names(frame: pd.DataFrame) -> pd.DataFrame:
    """Read frozen pre-terminology-change tables without emitting old names."""
    normalized = frame.copy()
    column_map = {
        "ecoood_score": "prediction_error_risk_score",
        "ecoood_endpoint_balanced": "prediction_error_risk_endpoint_balanced",
        "ecoood_q80": "prediction_error_risk_q80",
        "ecoood_q95": "prediction_error_risk_q95",
        "ecoood_minus_chemical": "prediction_error_risk_minus_chemical",
        "ecoood_minus_biological": "prediction_error_risk_minus_biological",
        "ecoood_minus_contextual": "prediction_error_risk_minus_contextual",
        "ecoood_minus_bioactivity": "prediction_error_risk_minus_bioactivity",
        "ecoood_minus_uncertainty": "prediction_error_risk_minus_uncertainty",
    }
    rename = {
        old: new
        for old, new in column_map.items()
        if old in normalized.columns and new not in normalized.columns
    }
    if rename:
        normalized = normalized.rename(columns=rename)
    if "method" in normalized:
        normalized["method"] = normalized["method"].replace(
            {
                "ecoood": "prediction_error_risk",
                "ecoood_endpoint_balanced": (
                    "prediction_error_risk_endpoint_balanced"
                ),
                "ecoood_q80": "prediction_error_risk_q80",
                "ecoood_q95": "prediction_error_risk_q95",
            }
        )
    return normalized


def aggregate_deduplicated(root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob("seed_*/structured/benchmark_summary.csv")):
        frame = _normalize_legacy_score_names(pd.read_csv(path))
        frame["seed"] = int(path.parts[-3].split("_")[-1])
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No deduplicated benchmark summaries found under {root}.")
    combined = pd.concat(frames, ignore_index=True)
    metrics = ["rmse", "mae", "spearman", "coverage", "aurc"]
    return _mean_sd(combined, ["split", "model"], metrics)


def component_correlations(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    component_columns = [
        column for column in COMPONENT_COLUMNS if column in predictions.columns
    ]
    rows: list[dict[str, object]] = []
    ratios: list[dict[str, object]] = []
    subset = predictions.loc[
        (predictions["model"] == "lightgbm")
        & predictions["split"].isin(CORE_SPLITS),
        ["seed", "split", *component_columns],
    ].copy()
    for (seed, split), group in subset.groupby(["seed", "split"], sort=True):
        corr = group[component_columns].corr(method="spearman")
        for i, left in enumerate(component_columns):
            for right in component_columns[i + 1 :]:
                rows.append(
                    {
                        "seed": seed,
                        "split": split,
                        "component_left": left,
                        "component_right": right,
                        "spearman_rho": corr.loc[left, right],
                    }
                )
        valid = group["u_model"].abs() > 1e-12
        ratios.append(
            {
                "seed": seed,
                "split": split,
                "interval_width_to_u_model_min": (
                    group.loc[valid, "interval_width"] / group.loc[valid, "u_model"]
                ).min(),
                "interval_width_to_u_model_max": (
                    group.loc[valid, "interval_width"] / group.loc[valid, "u_model"]
                ).max(),
            }
        )
    correlation = _mean_sd(
        pd.DataFrame(rows),
        ["split", "component_left", "component_right"],
        ["spearman_rho"],
    )
    return correlation, pd.DataFrame(ratios)


def axis_ablation(ood_scores: pd.DataFrame) -> pd.DataFrame:
    methods = [
        "prediction_error_risk",
        "prediction_error_risk_minus_chemical",
        "prediction_error_risk_minus_biological",
        "prediction_error_risk_minus_contextual",
        "prediction_error_risk_minus_bioactivity",
        "prediction_error_risk_minus_uncertainty",
    ]
    subset = ood_scores.loc[
        (ood_scores["model"] == "lightgbm")
        & ood_scores["method"].isin(methods)
        & ood_scores["split"].isin(CORE_SPLITS),
    ].copy()
    complete = subset.pivot_table(
        index=["seed", "split"], columns="method", values="aurc", aggfunc="first"
    ).reset_index()
    rows = []
    for method in methods:
        if method == "prediction_error_risk" or method not in complete:
            continue
        axis = method.removeprefix("prediction_error_risk_minus_")
        frame = complete[
            ["seed", "split", "prediction_error_risk", method]
        ].dropna().copy()
        frame["axis_removed"] = axis
        frame["aurc_delta_vs_full"] = (
            frame[method] - frame["prediction_error_risk"]
        )
        rows.append(frame[["seed", "split", "axis_removed", "aurc_delta_vs_full"]])
    all_rows = pd.concat(rows, ignore_index=True)
    return _mean_sd(all_rows, ["split", "axis_removed"], ["aurc_delta_vs_full"])


def hierarchical_workload_ci(
    classified: pd.DataFrame,
    *,
    n_bootstrap: int,
    random_state: int,
) -> pd.DataFrame:
    """Hierarchical bootstrap over seeds then unique chemicals within seed/split.

    The fixed-workload rows are already chemical-level.  This calculation avoids
    treating replicate endpoint rows or chemicals recurring in other shifts as
    independent observations.
    """
    subset = classified.loc[
        (classified["model"] == "lightgbm")
        & (classified["review_burden"] == 0.25)
        & classified["split"].isin(DEPLOYMENT_SPLITS),
    ].copy()
    subset = subset.loc[
        subset["method"].isin(PRIMARY_WORKLOAD_METHODS)
    ].copy()
    metrics = [
        "lower_priority_false_omission_rate",
        "lower_priority_false_reassurance",
        "rescued_false_negative_fraction",
        "rescued_per_100_reviews",
    ]
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(random_state)
    methods = sorted(subset["method"].unique())
    for split in DEPLOYMENT_SPLITS:
        split_df = subset.loc[subset["split"] == split]
        for method in methods:
            method_df = split_df.loc[split_df["method"] == method]
            if method_df.empty:
                continue
            seeds = np.array(sorted(method_df["seed"].unique()))
            seed_arrays: dict[int, np.ndarray] = {}
            for seed in seeds:
                frame = method_df.loc[method_df["seed"] == seed]
                if frame["chemical_id"].duplicated().any():
                    raise ValueError(
                        f"Duplicate chemical-level workload rows for {split}, "
                        f"{method}, seed {seed}."
                    )
                seed_arrays[int(seed)] = _classified_workload_array(frame)
            replicates: list[dict[str, float]] = []
            for _ in range(n_bootstrap):
                sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
                fractions = []
                rescued_fraction = []
                rescued_per_100 = []
                for seed in sampled_seeds:
                    values = seed_arrays[int(seed)]
                    sampled = values[
                        rng.integers(0, len(values), size=len(values))
                    ]
                    if method == "Random review":
                        review_n = max(1, int(round(0.25 * len(sampled))))
                        reviewed = np.zeros(len(sampled), dtype=bool)
                        reviewed[
                            rng.choice(len(sampled), size=review_n, replace=False)
                        ] = True
                        sampled[:, 0] = (~sampled[:, 5]) & ~reviewed
                        sampled[:, 2] = sampled[:, 1] & reviewed
                        sampled[:, 3] = reviewed
                    sample_metrics = _classified_workload_array_values(sampled)
                    fractions.append(
                        sample_metrics["lower_priority_false_omission_rate"]
                    )
                    rescued_fraction.append(
                        sample_metrics["rescued_false_negative_fraction"]
                    )
                    rescued_per_100.append(
                        sample_metrics["rescued_per_100_reviews"]
                    )
                replicates.append(
                    {
                        "lower_priority_false_omission_rate": _safe_nanmean(
                            fractions
                        ),
                        "lower_priority_false_reassurance": _safe_nanmean(fractions),
                        "rescued_false_negative_fraction": _safe_nanmean(rescued_fraction),
                        "rescued_per_100_reviews": _safe_nanmean(rescued_per_100),
                    }
                )
            result = pd.DataFrame(replicates)
            for metric in metrics:
                values = result[metric].dropna().to_numpy()
                rows.append(
                    {
                        "split": split,
                        "method": method,
                        "metric": metric,
                        "ci_2_5": np.quantile(values, 0.025),
                        "median": np.quantile(values, 0.5),
                        "ci_97_5": np.quantile(values, 0.975),
                    }
                )
    return pd.DataFrame(rows)


def _classified_workload_array(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            frame["screening_action"].eq("lower_priority").to_numpy(dtype=bool),
            frame["baseline_false_negative"].to_numpy(dtype=bool),
            frame["rescued_false_negative"].to_numpy(dtype=bool),
            frame["reviewed"].to_numpy(dtype=bool),
            frame["true_high_concern"].to_numpy(dtype=bool),
            frame["pred_high_concern"].to_numpy(dtype=bool),
        ]
    )


def _classified_workload_array_values(values: np.ndarray) -> dict[str, float]:
    lower_priority = values[:, 0]
    baseline_false_negative = values[:, 1]
    rescued = values[:, 2]
    reviewed = values[:, 3]
    true_high_concern = values[:, 4]
    high_concern_left_lower_priority = true_high_concern & lower_priority
    n_baseline = int(baseline_false_negative.sum())
    n_review = int(reviewed.sum())
    false_omission_rate = (
        float(high_concern_left_lower_priority.sum() / lower_priority.sum())
        if lower_priority.sum()
        else np.nan
    )
    return {
        "lower_priority_false_omission_rate": false_omission_rate,
        # Backward-compatible alias for the frozen v0.2.0 tables.
        "lower_priority_false_reassurance": false_omission_rate,
        "lower_priority_queue_size": float(lower_priority.sum()),
        "high_concern_left_lower_priority_fraction": (
            float(
                high_concern_left_lower_priority.sum()
                / true_high_concern.sum()
            )
            if true_high_concern.sum()
            else np.nan
        ),
        "rescued_false_negative_fraction": (
            float(rescued.sum() / n_baseline) if n_baseline else np.nan
        ),
        "rescued_per_100_reviews": (
            float(100 * rescued.sum() / n_review) if n_review else np.nan
        ),
    }


def _classified_workload_values(frame: pd.DataFrame) -> dict[str, float]:
    return _classified_workload_array_values(_classified_workload_array(frame))


def _redraw_random_review(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    review_burden: float,
) -> np.ndarray:
    """Redraw random-review assignments within one bootstrap sample."""
    randomized = values.copy()
    review_n = max(1, int(round(review_burden * len(randomized))))
    reviewed = np.zeros(len(randomized), dtype=bool)
    reviewed[
        rng.choice(len(randomized), size=review_n, replace=False)
    ] = True
    randomized[:, 0] = (~randomized[:, 5]) & ~reviewed
    randomized[:, 2] = randomized[:, 1] & reviewed
    randomized[:, 3] = reviewed
    return randomized


def paired_hierarchical_workload_delta_ci(
    classified: pd.DataFrame,
    *,
    n_bootstrap: int,
    random_state: int,
    reference_method: str = "Prediction-error risk score",
) -> pd.DataFrame:
    """Cluster-aware paired deltas using identical sampled chemicals per method."""
    subset = classified.loc[
        (classified["model"] == "lightgbm")
        & (classified["review_burden"] == 0.25)
        & classified["split"].isin(DEPLOYMENT_SPLITS),
    ].copy()
    subset = subset.loc[subset["method"].isin(PRIMARY_WORKLOAD_METHODS)].copy()
    rng = np.random.default_rng(random_state)
    metrics = [
        "lower_priority_false_omission_rate",
        "lower_priority_false_reassurance",
        "rescued_false_negative_fraction",
        "rescued_per_100_reviews",
    ]
    rows: list[dict[str, object]] = []
    for split in DEPLOYMENT_SPLITS:
        split_df = subset.loc[subset["split"] == split]
        methods = sorted(
            set(split_df["method"]) - {reference_method, "Random review"}
        )
        seeds = np.array(sorted(split_df["seed"].unique()))
        for method in methods:
            aligned: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            for seed in seeds:
                seed_df = split_df.loc[split_df["seed"] == seed]
                if not {reference_method, method}.issubset(set(seed_df["method"])):
                    continue
                reference = seed_df.loc[
                    seed_df["method"] == reference_method
                ].set_index("chemical_id")
                comparator = seed_df.loc[
                    seed_df["method"] == method
                ].set_index("chemical_id")
                if not reference.index.is_unique or not comparator.index.is_unique:
                    raise ValueError(
                        f"Duplicate chemical-level workload rows for {split}, "
                        f"{method}, seed {seed}."
                    )
                common = reference.index.intersection(comparator.index)
                if common.empty:
                    continue
                aligned[int(seed)] = (
                    _classified_workload_array(reference.loc[common]),
                    _classified_workload_array(comparator.loc[common]),
                )
            available_seeds = np.array(sorted(aligned))
            if len(available_seeds) == 0:
                continue
            replicates = {metric: [] for metric in metrics}
            for _ in range(n_bootstrap):
                sampled_seeds = rng.choice(
                    available_seeds,
                    size=len(available_seeds),
                    replace=True,
                )
                seed_deltas = {metric: [] for metric in metrics}
                for seed in sampled_seeds:
                    reference, comparator = aligned[int(seed)]
                    sampled = rng.integers(
                        0,
                        len(reference),
                        size=len(reference),
                    )
                    reference_sample = reference[sampled]
                    comparator_sample = comparator[sampled]
                    if reference_method == "Random review":
                        reference_sample = _redraw_random_review(
                            reference_sample,
                            rng=rng,
                            review_burden=0.25,
                        )
                    if method == "Random review":
                        comparator_sample = _redraw_random_review(
                            comparator_sample,
                            rng=rng,
                            review_burden=0.25,
                        )
                    reference_values = _classified_workload_array_values(
                        reference_sample
                    )
                    comparator_values = _classified_workload_array_values(
                        comparator_sample
                    )
                    for metric in metrics:
                        seed_deltas[metric].append(
                            comparator_values[metric] - reference_values[metric]
                        )
                for metric in metrics:
                    replicates[metric].append(_safe_nanmean(seed_deltas[metric]))
            for metric in metrics:
                values = np.asarray(replicates[metric], dtype=float)
                values = values[np.isfinite(values)]
                rows.append(
                    {
                        "split": split,
                        "reference_method": reference_method,
                        "comparator_method": method,
                        "metric": metric,
                        "delta_definition": f"comparator minus {reference_method}",
                        "ci_2_5": float(np.quantile(values, 0.025)),
                        "median": float(np.quantile(values, 0.5)),
                        "ci_97_5": float(np.quantile(values, 0.975)),
                    }
                )
    return pd.DataFrame(rows)


def review_burden_macro_sensitivity(review: pd.DataFrame) -> pd.DataFrame:
    """Macro-average workload outcomes across the four larger shift settings."""
    subset = review.loc[
        (review["model"] == "lightgbm")
        & review["split"].isin(LARGE_DEPLOYMENT_SPLITS)
        & review["method"].isin(PRIMARY_WORKLOAD_METHODS)
        & review["review_burden"].isin([0.15, 0.20, 0.25, 0.30, 0.35]),
    ].copy()
    false_omission_column = (
        "lower_priority_false_omission_rate"
        if "lower_priority_false_omission_rate" in subset.columns
        else "lower_priority_false_reassurance"
    )
    split_means = (
        subset.groupby(
            ["split", "method", "review_burden"],
            as_index=False,
        )[
            [
                false_omission_column,
                "rescued_per_100_reviews",
            ]
        ]
        .mean()
        .rename(
            columns={
                false_omission_column: "lower_priority_false_omission_rate",
            }
        )
    )
    return (
        split_means.groupby(
            ["method", "review_burden"],
            as_index=False,
        )
        .agg(
            lower_priority_false_omission_rate_mean=(
                "lower_priority_false_omission_rate",
                "mean",
            ),
            lower_priority_false_omission_rate_sd_across_settings=(
                "lower_priority_false_omission_rate",
                "std",
            ),
            rescued_per_100_reviews_mean=(
                "rescued_per_100_reviews",
                "mean",
            ),
            rescued_per_100_reviews_sd_across_settings=(
                "rescued_per_100_reviews",
                "std",
            ),
            n_settings=("split", "nunique"),
        )
    )


def high_concern_threshold_sensitivity(classified: pd.DataFrame) -> pd.DataFrame:
    """Recalculate fixed-workload outcomes across relative prioritization cutoffs."""
    subset = classified.loc[
        (classified["model"] == "lightgbm")
        & (classified["review_burden"] == 0.25)
        & classified["split"].isin(DEPLOYMENT_SPLITS),
    ].copy()
    rows: list[dict[str, object]] = []
    for (seed, split, method), frame in subset.groupby(["seed", "split", "method"], sort=True):
        for quantile in [0.10, 0.20, 0.25, 0.30]:
            cutoff = float(frame["min_true_tox"].quantile(quantile))
            true_high = frame["min_true_tox"] <= cutoff
            pred_high = frame["min_pred_tox"] <= cutoff
            baseline_fn = true_high & ~pred_high
            lower = frame["screening_action"].eq("lower_priority")
            rescued = baseline_fn & frame["reviewed"]
            false_omission_rate = (
                float((true_high & lower).sum() / lower.sum())
                if lower.any()
                else np.nan
            )
            rows.append(
                {
                    "seed": seed,
                    "split": split,
                    "method": method,
                    "high_concern_quantile": quantile,
                    "toxicity_cutoff": cutoff,
                    "lower_priority_false_omission_rate": false_omission_rate,
                    "lower_priority_false_reassurance": false_omission_rate,
                    "lower_priority_queue_size": int(lower.sum()),
                    "high_concern_left_lower_priority_fraction": (
                        float((true_high & lower).sum() / true_high.sum())
                        if true_high.any()
                        else np.nan
                    ),
                    "rescued_false_negative_fraction": (
                        float(rescued.sum() / baseline_fn.sum()) if baseline_fn.any() else np.nan
                    ),
                }
            )
    return _mean_sd(
        pd.DataFrame(rows),
        ["split", "method", "high_concern_quantile"],
        [
            "toxicity_cutoff",
            "lower_priority_false_omission_rate",
            "lower_priority_false_reassurance",
            "lower_priority_queue_size",
            "high_concern_left_lower_priority_fraction",
            "rescued_false_negative_fraction",
        ],
    )


def endpoint_resolved_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Report endpoint-specific accuracy without treating pooled metrics as endpoint-neutral."""
    rows: list[dict[str, object]] = []
    subset = predictions.loc[
        (predictions["model"] == "lightgbm") & predictions["split"].isin(CORE_SPLITS)
    ].copy()
    for (seed, split, endpoint), frame in subset.groupby(["seed", "split", "endpoint"], sort=True):
        y_true = frame["y_true"].to_numpy(dtype=float)
        y_pred = frame["y_pred"].to_numpy(dtype=float)
        covered = (frame["y_true"] >= frame["interval_lower"]) & (
            frame["y_true"] <= frame["interval_upper"]
        )
        endpoint_covered = (
            (frame["y_true"] >= frame["endpoint_interval_lower"])
            & (frame["y_true"] <= frame["endpoint_interval_upper"])
            if {"endpoint_interval_lower", "endpoint_interval_upper"}.issubset(frame.columns)
            else pd.Series(np.nan, index=frame.index)
        )
        rows.append(
            {
                "seed": seed,
                "split": split,
                "endpoint": endpoint,
                "n_cases": len(frame),
                "n_chemicals": frame["chemical_id"].nunique(),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "coverage": float(covered.mean()),
                "endpoint_conditional_coverage": float(endpoint_covered.mean()),
                "mean_interval_width": float(frame["interval_width"].mean()),
                "endpoint_conditional_mean_interval_width": (
                    float(frame["endpoint_interval_width"].mean())
                    if "endpoint_interval_width" in frame
                    else np.nan
                ),
            }
        )
    return _mean_sd(
        pd.DataFrame(rows),
        ["split", "endpoint"],
        [
            "n_cases",
            "n_chemicals",
            "rmse",
            "coverage",
            "endpoint_conditional_coverage",
            "mean_interval_width",
            "endpoint_conditional_mean_interval_width",
        ],
    )


def record_count_associations(chemical_panel: pd.DataFrame) -> pd.DataFrame:
    score_columns = [
        column
        for column in chemical_panel.columns
        if column.startswith("max_")
    ]
    rows: list[dict[str, object]] = []
    for (seed, model, split), frame in chemical_panel.groupby(
        ["seed", "model", "split"],
        sort=True,
    ):
        for column in score_columns:
            rows.append(
                {
                    "seed": seed,
                    "model": model,
                    "split": split,
                    "score": column,
                    "spearman_rho_with_row_count": frame["row_count"].corr(
                        frame[column],
                        method="spearman",
                    ),
                }
            )
    return _mean_sd(
        pd.DataFrame(rows),
        ["model", "split", "score"],
        ["spearman_rho_with_row_count"],
    )


def named_class_fold_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    subset = predictions.loc[
        (predictions["model"] == "lightgbm")
        & (predictions["split"] == "chemical_class")
    ].copy()
    rows: list[dict[str, object]] = []
    for seed, frame in subset.groupby("seed", sort=True):
        held_out = named_class_for_seed(int(seed))
        contains_held_out = frame["chemical_class"].fillna("").map(
            lambda value: held_out
            in {token.strip() for token in str(value).split(";")}
        )
        if not contains_held_out.all():
            raise ValueError(
                f"Named-class test fold for seed {seed} contains rows outside "
                f"the held-out class {held_out!r}."
            )
        rows.append(
            {
                "seed": int(seed),
                "held_out_class": held_out,
                "n_cases": int(len(frame)),
                "n_chemicals": int(frame["chemical_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def high_error_label_sensitivity(ood_scores: pd.DataFrame) -> pd.DataFrame:
    methods = [
        "prediction_error_risk_q80",
        "prediction_error_risk",
        "prediction_error_risk_q95",
        "prediction_error_risk_endpoint_balanced",
    ]
    subset = ood_scores.loc[
        (ood_scores["model"] == "lightgbm")
        & ood_scores["split"].isin(CORE_SPLITS)
        & ood_scores["method"].isin(methods),
    ].copy()
    return _mean_sd(
        subset,
        ["split", "method"],
        ["aurc", "top_decile_error_capture_rate"],
    )


def coefficient_stability(ood_scores: pd.DataFrame) -> pd.DataFrame:
    coefficient_columns = [
        column for column in ood_scores.columns if column.startswith("meta_coef_")
    ]
    subset = ood_scores.loc[
        (ood_scores["model"] == "lightgbm")
        & (ood_scores["method"] == "prediction_error_risk")
        & ood_scores["split"].isin(CORE_SPLITS),
        ["seed", "split", *coefficient_columns],
    ].copy()
    if not coefficient_columns:
        return pd.DataFrame()
    long = subset.melt(
        id_vars=["seed", "split"],
        value_vars=coefficient_columns,
        var_name="subcomponent",
        value_name="coefficient",
    )
    long["subcomponent"] = long["subcomponent"].str.removeprefix("meta_coef_")
    summary = (
        long.groupby(["split", "subcomponent"], as_index=False)
        .agg(
            coefficient_mean=("coefficient", "mean"),
            coefficient_std=("coefficient", "std"),
            positive_seed_fraction=("coefficient", lambda values: float((values > 0).mean())),
            n_seeds=("coefficient", "count"),
        )
    )
    return summary


def external_panel_flow(
    filter_audit_path: Path,
    main_panel_path: Path,
    strict_panel_path: Path,
) -> pd.DataFrame:
    """Summarize the external-panel construction as a sequential data flow."""
    audit = pd.read_csv(filter_audit_path)
    main = pd.read_csv(main_panel_path)
    strict = pd.read_csv(strict_panel_path)
    has_target = audit["source_log_molar"].notna()
    has_species_metadata = audit["has_train_species_meta"].fillna(False).astype(bool)

    def row(stage: str, frame: pd.DataFrame, definition: str) -> dict[str, object]:
        chemical_col = "chemical_id" if "chemical_id" in frame else "casrn"
        return {
            "stage": stage,
            "n_rows_or_cases": int(len(frame)),
            "n_chemicals": int(frame[chemical_col].nunique()),
            "definition": definition,
        }

    return pd.DataFrame(
        [
            row(
                "Exact dossier endpoint rows",
                audit,
                "Exact fish, daphnid, or algal acute endpoint rows extracted from ECHA dossiers",
            ),
            row(
                "Quantitative rows with species metadata",
                audit.loc[has_target & has_species_metadata],
                "Rows with a harmonized log-molar target and species metadata available to the feature builder",
            ),
            row(
                "Freshwater-aligned source rows",
                audit.loc[audit["include_main_row"].fillna(False).astype(bool)],
                "Rows retained after freshwater and endpoint/species alignment",
            ),
            row(
                "Main aggregated case panel",
                main,
                "Chemical-species-endpoint cases used for the main external analysis",
            ),
            row(
                "Strict standard-species case panel",
                strict,
                "Main-panel cases restricted to the predefined standard-species subset",
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize statistical audits from a completed EcoOOD benchmark."
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path("outputs/integrity_benchmark"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/integrity_benchmark/analysis_tables"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument(
        "--external-filter-audit",
        type=Path,
        default=Path(
            "results/external_regulatory_prep/echa_pmra_external_clean/"
            "echa_pmra_row_filter_audit.csv"
        ),
    )
    parser.add_argument(
        "--external-main-panel",
        type=Path,
        default=Path(
            "results/external_regulatory_prep/echa_pmra_external_clean/"
            "echa_pmra_case_panel_main.csv"
        ),
    )
    parser.add_argument(
        "--external-strict-panel",
        type=Path,
        default=Path(
            "results/external_regulatory_prep/echa_pmra_external_clean/"
            "echa_pmra_case_panel_strict.csv"
        ),
    )
    args = parser.parse_args()

    root = args.benchmark_root
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    benchmark = pd.read_csv(root / "aggregate" / "benchmark_summary_agg.csv")
    benchmark.loc[benchmark["split"].isin(CORE_SPLITS)].to_csv(
        output / "core_benchmark_summary.csv", index=False
    )

    review_path = root / "aggregate" / "review_workload_endpoint_relative_agg.csv"
    if not review_path.exists():
        review_path = root / "aggregate" / "review_workload_summary_agg.csv"
    review = pd.read_csv(review_path)
    review.loc[
        (review["model"] == "lightgbm")
        & (review["review_burden"] == 0.25)
        & review["split"].isin(DEPLOYMENT_SPLITS)
    ].to_csv(output / "fixed_workload_split_summary.csv", index=False)

    classified_path = (
        root / "aggregate" / "review_workload_endpoint_relative_classified.csv"
    )
    if not classified_path.exists():
        classified_path = root / "aggregate" / "review_workload_classified.csv"
    classified = pd.read_csv(classified_path)
    hierarchical_workload_ci(
        classified,
        n_bootstrap=args.bootstrap_replicates,
        random_state=20260723,
    ).to_csv(output / "fixed_workload_hierarchical_bootstrap_ci.csv", index=False)
    paired_hierarchical_workload_delta_ci(
        classified.loc[
            classified["method"].isin(
                [*PRIMARY_WORKLOAD_METHODS, "Random review"]
            )
        ],
        n_bootstrap=args.bootstrap_replicates,
        random_state=20260724,
        reference_method="Prediction-error risk score",
    ).to_csv(
        output / "fixed_workload_paired_delta_ci.csv",
        index=False,
    )
    paired_hierarchical_workload_delta_ci(
        classified.loc[
            classified["method"].isin(
                [*PRIMARY_WORKLOAD_METHODS, "Random review"]
            )
        ],
        n_bootstrap=args.bootstrap_replicates,
        random_state=20260731,
        reference_method="Random review",
    ).to_csv(
        output / "fixed_workload_paired_delta_vs_random_ci.csv",
        index=False,
    )
    review_all_seed_path = (
        root / "aggregate" / "review_workload_endpoint_relative_all_seeds.csv"
    )
    if not review_all_seed_path.exists():
        review_all_seed_path = (
            root / "aggregate" / "review_workload_summary_all_seeds.csv"
        )
    review_burden_macro_sensitivity(
        pd.read_csv(review_all_seed_path)
    ).to_csv(
        output / "fixed_workload_review_burden_sensitivity.csv",
        index=False,
    )
    pooled_classified = pd.read_csv(
        root / "aggregate" / "review_workload_classified.csv"
    )
    high_concern_threshold_sensitivity(pooled_classified).to_csv(
        output / "fixed_workload_high_concern_threshold_sensitivity.csv", index=False
    )
    pooled_review = pd.read_csv(
        root / "aggregate" / "review_workload_summary_agg.csv"
    )
    pooled_review.loc[
        (pooled_review["model"] == "lightgbm")
        & (pooled_review["review_burden"] == 0.25)
        & pooled_review["split"].isin(DEPLOYMENT_SPLITS)
    ].to_csv(
        output / "fixed_workload_pooled_threshold_sensitivity.csv",
        index=False,
    )

    predictions = _normalize_legacy_score_names(
        pd.read_csv(root / "aggregate" / "predictions_all_seeds.csv")
    )
    named_class_fold_summary(predictions).to_csv(
        output / "named_class_fold_summary.csv",
        index=False,
    )
    correlations, interval_ratio = component_correlations(predictions)
    correlations.to_csv(
        output / "prediction_error_risk_component_correlations.csv", index=False
    )
    interval_ratio.to_csv(output / "conformal_scale_ratio_audit.csv", index=False)
    endpoint_resolved_metrics(predictions).to_csv(
        output / "endpoint_resolved_benchmark_metrics.csv", index=False
    )
    chemical_panel = pd.read_csv(root / "aggregate" / "chemical_split_panel.csv")
    record_count_associations(chemical_panel).to_csv(
        output / "chemical_record_count_score_associations.csv",
        index=False,
    )

    ood_scores = _normalize_legacy_score_names(
        pd.read_csv(root / "aggregate" / "ood_score_summary_all_seeds.csv")
    )
    axis_ablation(ood_scores).to_csv(
        output / "prediction_error_risk_component_ablation.csv", index=False
    )
    high_error_label_sensitivity(ood_scores).to_csv(
        output / "prediction_error_risk_high_error_label_sensitivity.csv",
        index=False,
    )
    coefficient_stability(ood_scores).to_csv(
        output / "prediction_error_risk_logistic_coefficient_stability.csv",
        index=False,
    )
    endpoint_median = root / "aggregate" / "review_workload_endpoint_median_agg.csv"
    if endpoint_median.exists():
        pd.read_csv(endpoint_median).to_csv(
            output / "fixed_workload_endpoint_median_sensitivity.csv",
            index=False,
        )
    deduplicated_root = root / "dedup_sensitivity"
    if any(deduplicated_root.glob("seed_*/structured/benchmark_summary.csv")):
        aggregate_deduplicated(deduplicated_root).to_csv(
            output / "exact_duplicate_sensitivity.csv", index=False
        )

    for panel in ["external_main", "external_strict"]:
        source = root / panel
        if source.exists():
            pd.read_csv(source / "external_metrics_summary.csv").to_csv(
                output / f"{panel}_metrics.csv", index=False
            )
            pd.read_csv(source / "external_burden_summary.csv").to_csv(
                output / f"{panel}_fixed_workload.csv", index=False
            )
            pd.read_csv(source / "external_cluster_bootstrap_summary.csv").to_csv(
                output / f"{panel}_chemical_cluster_ci.csv", index=False
            )
            pd.read_csv(source / "external_train_panel_identity_overlap_audit.csv").to_csv(
                output / f"{panel}_identity_overlap.csv", index=False
            )

    external_paths = [
        args.external_filter_audit,
        args.external_main_panel,
        args.external_strict_panel,
    ]
    if all(path.exists() for path in external_paths):
        external_panel_flow(*external_paths).to_csv(
            output / "external_panel_data_flow.csv",
            index=False,
        )

    print(f"Wrote benchmark audit tables to {output}")


if __name__ == "__main__":
    main()
