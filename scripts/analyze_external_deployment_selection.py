from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecoood.application import (  # noqa: E402
    route_screening_queue,
    select_reliability_signal,
)
from ecoood.schema import DEFAULT_SCHEMA  # noqa: E402
from scripts.audit_benchmark_integrity import (  # noqa: E402
    strict_input_eligibility_audit,
)
from scripts.run_echa_pmra_external_validation import (  # noqa: E402
    calibration_split_by_chemical,
)


CANDIDATE_COLUMNS: dict[str, str] = {
    "block_normalized_knn": "ad_equal_block_distance",
    "block_normalized_knn_plus_sd": "equal_block_knn_plus_sd_risk",
    "prediction_error_risk": "prediction_error_risk_score",
    "three_signal_risk": "generic_support_plus_sd_risk",
    "ensemble_sd_risk": "ensemble_sd_risk",
    "similarity_ad": "ad_similarity",
}

COMPARATORS: dict[str, str] = {
    "fixed_block_normalized_knn": "block_normalized_knn",
    "fixed_similarity_ad": "similarity_ad",
    "fixed_prediction_error_risk": "prediction_error_risk",
}

ROUTES = (
    "screen_now",
    "lower_priority",
    "withhold_review",
    "prioritize_testing",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate EcoOOD measure selection and routing in a chemical-identity-"
            "disjoint external screening queue."
        )
    )
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--development-size", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--review-fraction", type=float, default=0.25)
    parser.add_argument("--random-draws", type=int, default=1000)
    return parser.parse_args()


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def calibration_cutoffs(frame: pd.DataFrame) -> dict[str, float]:
    medians = frame.groupby(
        [DEFAULT_SCHEMA.chemical_id, DEFAULT_SCHEMA.endpoint],
        as_index=False,
    )[DEFAULT_SCHEMA.target].median()
    return (
        medians.groupby(DEFAULT_SCHEMA.endpoint)[DEFAULT_SCHEMA.target]
        .quantile(0.25)
        .to_dict()
    )


def normalize_prediction_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    """Support frozen releases that predate the descriptive risk-score name."""
    normalized = predictions.copy()
    if (
        "prediction_error_risk_score" not in normalized
        and "ecoood_score" in normalized
    ):
        normalized["prediction_error_risk_score"] = normalized[
            "ecoood_score"
        ]
    return normalized


def build_chemical_panel(
    predictions: pd.DataFrame,
    cutoffs: Mapping[str, float],
    *,
    candidate_columns: Mapping[str, str] = CANDIDATE_COLUMNS,
) -> pd.DataFrame:
    missing = sorted(
        set(candidate_columns.values())
        - set(predictions.columns)
    )
    if missing:
        raise ValueError("Missing candidate columns: " + ", ".join(missing))

    endpoint = predictions.groupby(
        ["chemical_id", "endpoint"],
        as_index=False,
    ).agg(
        y_true=("target_log_molar", "median"),
        y_pred=("y_pred", "median"),
        n_cases=("case_id", "nunique"),
        **{
            score_column: (score_column, "median")
            for score_column in candidate_columns.values()
        },
    )
    endpoint["concern_cutoff"] = endpoint["endpoint"].map(cutoffs)
    if endpoint["concern_cutoff"].isna().any():
        missing_endpoints = sorted(
            endpoint.loc[
                endpoint["concern_cutoff"].isna(),
                "endpoint",
            ].unique()
        )
        raise ValueError(
            "Missing calibration concern cutoffs for: "
            + ", ".join(missing_endpoints)
        )
    endpoint["true_high_concern"] = (
        endpoint["y_true"] <= endpoint["concern_cutoff"]
    )
    endpoint["pred_high_concern"] = (
        endpoint["y_pred"] <= endpoint["concern_cutoff"]
    )
    endpoint["abs_error"] = np.abs(endpoint["y_true"] - endpoint["y_pred"])
    endpoint["directional_underprediction_loss"] = np.maximum(
        endpoint["y_pred"] - endpoint["y_true"],
        0.0,
    )

    panel = endpoint.groupby("chemical_id", as_index=False).agg(
        true_high_concern=("true_high_concern", "max"),
        pred_high_concern=("pred_high_concern", "max"),
        n_endpoints=("endpoint", "nunique"),
        n_cases=("n_cases", "sum"),
        max_abs_error=("abs_error", "max"),
        directional_underprediction_loss=(
            "directional_underprediction_loss",
            "max",
        ),
        **{
            score_column: (score_column, "max")
            for score_column in candidate_columns.values()
        },
    )
    panel["input_eligible"] = True
    return panel


def routed_metrics(routed: pd.DataFrame) -> dict[str, float]:
    true_high = routed["true_high_concern"].to_numpy(dtype=bool)
    pred_high = routed["pred_high_concern"].to_numpy(dtype=bool)
    reviewed = routed["reviewed"].to_numpy(dtype=bool)
    lower_priority = routed["route"].eq("lower_priority").to_numpy()
    false_negative = true_high & ~pred_high
    omitted = true_high & lower_priority
    return {
        "n_chemicals": int(len(routed)),
        "review_n": int(reviewed.sum()),
        "n_measured_high_concern": int(true_high.sum()),
        "n_false_negative": int(false_negative.sum()),
        "n_lower_priority": int(lower_priority.sum()),
        "n_high_concern_omitted": int(omitted.sum()),
        "lower_priority_false_omission_rate": (
            float(omitted.sum() / lower_priority.sum())
            if lower_priority.any()
            else float("nan")
        ),
        "high_concern_left_low_priority_fraction": (
            float(omitted.sum() / true_high.sum())
            if true_high.any()
            else float("nan")
        ),
        "false_negative_capture": (
            float((false_negative & reviewed).sum() / false_negative.sum())
            if false_negative.any()
            else float("nan")
        ),
        "reviewed_high_concern_fraction": (
            float((true_high & reviewed).sum() / reviewed.sum())
            if reviewed.any()
            else float("nan")
        ),
    }


def route_with_signal(
    evaluation: pd.DataFrame,
    signal: str,
    *,
    review_fraction: float,
) -> pd.DataFrame:
    return route_screening_queue(
        evaluation,
        selected_signal=signal,
        candidate_columns=CANDIDATE_COLUMNS,
        review_fraction=review_fraction,
    )


def random_reference(
    evaluation: pd.DataFrame,
    *,
    review_fraction: float,
    draws: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    if not 0 <= review_fraction <= 1:
        raise ValueError("review_fraction must be in [0, 1].")
    review_n = int(round(review_fraction * len(evaluation)))
    n_chemicals = len(evaluation)
    sampled_order = np.argsort(
        rng.random((draws, n_chemicals)),
        axis=1,
    )[:, :review_n]
    reviewed = np.zeros((draws, n_chemicals), dtype=bool)
    reviewed[
        np.arange(draws)[:, None],
        sampled_order,
    ] = True

    true_high = evaluation["true_high_concern"].to_numpy(dtype=bool)
    pred_high = evaluation["pred_high_concern"].to_numpy(dtype=bool)
    false_negative = true_high & ~pred_high
    lower_priority = (~pred_high)[None, :] & ~reviewed
    omitted = true_high[None, :] & lower_priority
    lower_n = lower_priority.sum(axis=1)
    omitted_n = omitted.sum(axis=1)
    captured_n = (false_negative[None, :] & reviewed).sum(axis=1)
    reviewed_high_n = (true_high[None, :] & reviewed).sum(axis=1)

    false_omission = np.divide(
        omitted_n,
        lower_n,
        out=np.full(draws, np.nan, dtype=float),
        where=lower_n > 0,
    )
    high_concern_left = (
        omitted_n / true_high.sum() if true_high.any() else None
    )
    false_negative_capture = (
        captured_n / false_negative.sum() if false_negative.any() else None
    )
    return {
        "n_chemicals": float(n_chemicals),
        "review_n": float(review_n),
        "n_measured_high_concern": float(true_high.sum()),
        "n_false_negative": float(false_negative.sum()),
        "n_lower_priority": float(lower_n.mean()),
        "n_high_concern_omitted": float(omitted_n.mean()),
        "lower_priority_false_omission_rate": float(
            np.nanmean(false_omission)
        ) if np.isfinite(false_omission).any() else float("nan"),
        "high_concern_left_low_priority_fraction": float(
            np.mean(high_concern_left)
        ) if high_concern_left is not None else float("nan"),
        "false_negative_capture": (
            float(np.mean(false_negative_capture))
            if false_negative_capture is not None
            else float("nan")
        ),
        "reviewed_high_concern_fraction": float(
            (reviewed_high_n / review_n).mean()
        ) if review_n else float("nan"),
    }


def route_summary_rows(
    routed: pd.DataFrame,
    *,
    seed: int,
    repeat: int,
    selected_signal: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    total = len(routed)
    for route in ROUTES:
        subset = routed.loc[routed["route"].eq(route)]
        rows.append(
            {
                "seed": seed,
                "repeat": repeat,
                "selected_signal": selected_signal,
                "route": route,
                "route_n": int(len(subset)),
                "route_fraction": float(len(subset) / total),
                "measured_high_concern_n": int(
                    subset["true_high_concern"].sum()
                ),
                "measured_high_concern_fraction": (
                    float(subset["true_high_concern"].mean())
                    if len(subset)
                    else float("nan")
                ),
            }
        )
    return rows


def summarize_across_seeds(
    frame: pd.DataFrame,
    *,
    group_column: str,
    value_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_summary = frame.groupby(
        ["seed", group_column],
        as_index=False,
    )[value_columns].mean()
    summary = (
        seed_summary.groupby(group_column)[value_columns]
        .agg(["mean", "std"])
    )
    summary.columns = [
        "_".join(part for part in column if part)
        for column in summary.columns
    ]
    summary = summary.reset_index()
    return seed_summary, summary


def main() -> None:
    args = parse_args()
    if args.development_size < 1:
        raise ValueError("development-size must be positive.")
    if args.repeats < 1 or args.random_draws < 1:
        raise ValueError("repeats and random-draws must be positive.")

    source_train = pd.read_csv(args.train_path)
    train, _ = strict_input_eligibility_audit(source_train)
    predictions = normalize_prediction_columns(pd.read_csv(args.predictions))
    chemical_ids = sorted(predictions["chemical_id"].astype(str).unique())
    if args.development_size >= len(chemical_ids):
        raise ValueError(
            "development-size must leave at least one evaluation chemical."
        )

    splits: list[tuple[int, set[str]]] = []
    chemical_array = np.asarray(chemical_ids, dtype=object)
    for repeat in range(args.repeats):
        rng = np.random.default_rng(stable_seed("external_development", repeat))
        selected = rng.choice(
            chemical_array,
            size=args.development_size,
            replace=False,
        )
        splits.append((repeat, set(selected.astype(str))))

    comparison_rows: list[dict[str, object]] = []
    selection_rows: list[pd.DataFrame] = []
    selection_audit_rows: list[dict[str, object]] = []
    selected_route_rows: list[dict[str, object]] = []
    selected_queue_rows: list[pd.DataFrame] = []

    for seed, seed_predictions in predictions.groupby("seed", sort=True):
        _, calibration = calibration_split_by_chemical(train, seed=int(seed))
        cutoffs = calibration_cutoffs(calibration)
        panel = build_chemical_panel(seed_predictions, cutoffs)
        if set(panel["chemical_id"].astype(str)) != set(chemical_ids):
            raise ValueError(f"Chemical identities changed in seed {seed}.")

        for repeat, development_ids in splits:
            development = panel.loc[
                panel["chemical_id"].astype(str).isin(development_ids)
            ].reset_index(drop=True)
            evaluation = panel.loc[
                ~panel["chemical_id"].astype(str).isin(development_ids)
            ].reset_index(drop=True)
            if len(development) != args.development_size:
                raise AssertionError("Development chemical count changed.")

            selected_signal, selection_metrics, selection_source = (
                select_reliability_signal(
                    development,
                    candidate_columns=CANDIDATE_COLUMNS,
                    objective="directional_underprediction_area",
                    review_fraction=args.review_fraction,
                )
            )
            selection_metrics = selection_metrics.assign(
                seed=int(seed),
                repeat=repeat,
                n_development_chemicals=len(development),
                n_evaluation_chemicals=len(evaluation),
                selection_source=selection_source,
            )
            selection_rows.append(selection_metrics)

            finite_capture = selection_metrics[
                "directional_underprediction_capture_area"
            ].dropna()
            primary_tie_n = (
                int(
                    np.isclose(
                        finite_capture,
                        finite_capture.max(),
                    ).sum()
                )
                if len(finite_capture)
                else len(selection_metrics)
            )
            ranking = selection_metrics.assign(
                _primary=selection_metrics[
                    "directional_underprediction_capture_area"
                ].fillna(-np.inf),
                _secondary=selection_metrics[
                    "directional_underprediction_capture"
                ].fillna(-np.inf),
            )
            best = ranking.sort_values(
                ["_primary", "_secondary", "candidate_order"],
                ascending=[False, False, True],
                kind="mergesort",
            ).iloc[0]
            full_tie_n = int(
                (
                    np.isclose(ranking["_primary"], best["_primary"])
                    & np.isclose(ranking["_secondary"], best["_secondary"])
                ).sum()
            )
            selection_audit_rows.append(
                {
                    "seed": int(seed),
                    "repeat": repeat,
                    "selected_signal": selected_signal,
                    "selection_source": selection_source,
                    "development_false_negative_n": int(
                        selection_metrics["false_negative_count"].iloc[0]
                    ),
                    "development_directional_underprediction_loss": float(
                        development["directional_underprediction_loss"].sum()
                    ),
                    "primary_tie_n": primary_tie_n,
                    "full_tie_n": full_tie_n,
                }
            )

            candidate_evaluation: dict[str, tuple[pd.DataFrame, dict[str, float]]] = {}
            for signal in CANDIDATE_COLUMNS:
                routed = route_with_signal(
                    evaluation,
                    signal,
                    review_fraction=args.review_fraction,
                )
                candidate_evaluation[signal] = (routed, routed_metrics(routed))

            selected_routed, selected_metrics = candidate_evaluation[selected_signal]
            comparison_rows.append(
                {
                    "seed": int(seed),
                    "repeat": repeat,
                    "comparison": "selected_measure",
                    "selected_signal": selected_signal,
                    "selection_source": selection_source,
                    **selected_metrics,
                }
            )
            selected_route_rows.extend(
                route_summary_rows(
                    selected_routed,
                    seed=int(seed),
                    repeat=repeat,
                    selected_signal=selected_signal,
                )
            )
            selected_queue_rows.append(
                selected_routed.assign(
                    seed=int(seed),
                    repeat=repeat,
                    selection_source=selection_source,
                )
            )

            for comparison, signal in COMPARATORS.items():
                _, metrics = candidate_evaluation[signal]
                comparison_rows.append(
                    {
                        "seed": int(seed),
                        "repeat": repeat,
                        "comparison": comparison,
                        "selected_signal": signal,
                        "selection_source": "fixed_comparator",
                        **metrics,
                    }
                )

            oracle_signal, _, _ = select_reliability_signal(
                evaluation,
                candidate_columns=CANDIDATE_COLUMNS,
                objective="false_negative_capture",
                review_fraction=args.review_fraction,
            )
            _, oracle_metrics = candidate_evaluation[oracle_signal]
            comparison_rows.append(
                {
                    "seed": int(seed),
                    "repeat": repeat,
                    "comparison": "post_hoc_best",
                    "selected_signal": oracle_signal,
                    "selection_source": "evaluation_outcomes",
                    **oracle_metrics,
                }
            )

            comparison_rows.append(
                {
                    "seed": int(seed),
                    "repeat": repeat,
                    "comparison": "random_review",
                    "selected_signal": "random_review",
                    "selection_source": "random_reference",
                    **random_reference(
                        evaluation,
                        review_fraction=args.review_fraction,
                        draws=args.random_draws,
                        seed=stable_seed("external_random", seed, repeat),
                    ),
                }
            )

    comparisons = pd.DataFrame(comparison_rows)
    selection_details = pd.concat(selection_rows, ignore_index=True)
    selection_audit = pd.DataFrame(selection_audit_rows)
    route_details = pd.DataFrame(selected_route_rows)
    selected_queues = pd.concat(selected_queue_rows, ignore_index=True)

    metric_columns = [
        "n_chemicals",
        "review_n",
        "n_measured_high_concern",
        "n_false_negative",
        "n_lower_priority",
        "n_high_concern_omitted",
        "lower_priority_false_omission_rate",
        "high_concern_left_low_priority_fraction",
        "false_negative_capture",
        "reviewed_high_concern_fraction",
    ]
    comparison_seed_summary, comparison_summary = summarize_across_seeds(
        comparisons,
        group_column="comparison",
        value_columns=metric_columns,
    )
    route_seed_summary, route_summary = summarize_across_seeds(
        route_details,
        group_column="route",
        value_columns=[
            "route_n",
            "route_fraction",
            "measured_high_concern_n",
            "measured_high_concern_fraction",
        ],
    )

    selection_frequency = (
        selection_audit.groupby("selected_signal", as_index=False)
        .size()
        .rename(columns={"size": "n_selected"})
    )
    selection_frequency["selection_fraction"] = (
        selection_frequency["n_selected"] / len(selection_audit)
    )

    pivot = comparisons.pivot_table(
        index=["seed", "repeat"],
        columns="comparison",
        values="lower_priority_false_omission_rate",
    ).reset_index()
    paired_rows: list[pd.DataFrame] = []
    for comparison in (
        "random_review",
        "fixed_similarity_ad",
        "fixed_block_normalized_knn",
        "fixed_prediction_error_risk",
    ):
        paired = pivot[["seed", "repeat"]].copy()
        paired["comparison"] = comparison
        paired["selected_minus_comparator_for"] = (
            pivot["selected_measure"] - pivot[comparison]
        )
        paired_rows.append(paired)
    paired_differences = pd.concat(paired_rows, ignore_index=True)
    paired_seed_summary, paired_summary = summarize_across_seeds(
        paired_differences,
        group_column="comparison",
        value_columns=["selected_minus_comparator_for"],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparisons.to_csv(args.output_dir / "external_deployment_comparisons_all.csv", index=False)
    comparison_seed_summary.to_csv(
        args.output_dir / "external_deployment_comparisons_seed_summary.csv",
        index=False,
    )
    comparison_summary.to_csv(
        args.output_dir / "external_deployment_comparisons_summary.csv",
        index=False,
    )
    selection_details.to_csv(
        args.output_dir / "external_deployment_selection_details.csv",
        index=False,
    )
    selection_audit.to_csv(
        args.output_dir / "external_deployment_selection_audit.csv",
        index=False,
    )
    selection_frequency.to_csv(
        args.output_dir / "external_deployment_selection_frequency.csv",
        index=False,
    )
    route_details.to_csv(
        args.output_dir / "external_deployment_route_details.csv",
        index=False,
    )
    route_seed_summary.to_csv(
        args.output_dir / "external_deployment_route_seed_summary.csv",
        index=False,
    )
    route_summary.to_csv(
        args.output_dir / "external_deployment_route_summary.csv",
        index=False,
    )
    selected_queues.to_csv(
        args.output_dir / "external_deployment_selected_queues.csv",
        index=False,
    )
    paired_differences.to_csv(
        args.output_dir / "external_deployment_paired_differences.csv",
        index=False,
    )
    paired_seed_summary.to_csv(
        args.output_dir / "external_deployment_paired_seed_summary.csv",
        index=False,
    )
    paired_summary.to_csv(
        args.output_dir / "external_deployment_paired_summary.csv",
        index=False,
    )

    notes = [
        f"External chemicals: {len(chemical_ids)}",
        f"Development chemicals per repetition: {args.development_size}",
        f"Evaluation chemicals per repetition: {len(chemical_ids) - args.development_size}",
        f"Model fits: {predictions['seed'].nunique()}",
        f"Repeated chemical-level splits: {args.repeats}",
        f"Review fraction: {args.review_fraction:.3f}",
        (
            "Selection objective: area under the directional-underprediction "
            "capture curve, with capture at the fixed review fraction as a "
            "tie-break"
        ),
        (
            "Development repetitions with no false negatives: "
            f"{(selection_audit['development_false_negative_n'] == 0).mean():.3f}"
        ),
        (
            "Development repetitions with zero directional-underprediction loss: "
            f"{(selection_audit['development_directional_underprediction_loss'] == 0).mean():.3f}"
        ),
        (
            "Development repetitions with a primary tie: "
            f"{(selection_audit['primary_tie_n'] > 1).mean():.3f}"
        ),
        (
            "Development repetitions requiring the fixed candidate order: "
            f"{(selection_audit['full_tie_n'] > 1).mean():.3f}"
        ),
    ]
    (args.output_dir / "external_deployment_notes.txt").write_text(
        "\n".join(notes) + "\n"
    )
    print(comparison_summary.to_string(index=False))
    print("\nSelection frequency")
    print(
        selection_frequency.sort_values(
            "selection_fraction",
            ascending=False,
        ).to_string(index=False)
    )
    print("\nRoute summary")
    print(route_summary.to_string(index=False))


if __name__ == "__main__":
    main()
