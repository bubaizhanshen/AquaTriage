from __future__ import annotations

import pandas as pd

from scripts.export_release_bundle import normalize_release_columns


def test_release_export_renames_compatibility_score_columns() -> None:
    source = pd.DataFrame(
        {
            "ecoood_score": [0.4],
            "ecoood_q80": [0.3],
            "ecoood_endpoint_balanced": [0.2],
            "catastrophic_error_capture_rate": [0.1],
        }
    )

    released = normalize_release_columns(source)

    assert released.columns.tolist() == [
        "prediction_error_risk_score",
        "prediction_error_risk_q80",
        "prediction_error_risk_endpoint_balanced",
        "top_decile_error_capture_rate",
    ]


def test_release_export_prefers_current_column_when_legacy_alias_is_also_present() -> None:
    source = pd.DataFrame(
        {
            "prediction_error_risk_score": [0.7],
            "ecoood_score": [0.4],
        }
    )

    released = normalize_release_columns(source)

    assert released.columns.tolist() == ["prediction_error_risk_score"]
    assert released.iloc[0, 0] == 0.7


def test_release_export_renames_internal_score_values() -> None:
    source = pd.DataFrame(
        {
            "method": ["ecoood", "ecoood_endpoint_balanced", "similarity_ad_risk"],
            "score_col": ["ecoood_score", "ecoood_q80", "ad_similarity_risk"],
            "score": ["max_ecoood", "max_ecoood_q95", "max_similarity_ad"],
            "chemical_name": ["EcoOOD control", "Example", "Reference"],
        }
    )

    released = normalize_release_columns(source)

    assert released["method"].tolist() == [
        "prediction_error_risk_score",
        "prediction_error_risk_endpoint_balanced",
        "similarity_ad_risk",
    ]
    assert released["score_col"].tolist() == [
        "prediction_error_risk_score",
        "prediction_error_risk_q80",
        "ad_similarity_risk",
    ]
    assert released["score"].tolist() == [
        "max_prediction_error_risk_score",
        "max_prediction_error_risk_q95",
        "max_similarity_ad",
    ]
    assert released["chemical_name"].tolist() == [
        "EcoOOD control",
        "Example",
        "Reference",
    ]


def test_release_export_renames_display_score_values() -> None:
    source = pd.DataFrame(
        {
            "method": ["EcoOOD", "EcoOOD endpoint-balanced", "EcoOOD q80"],
            "delta_definition": [
                "comparator minus EcoOOD",
                "unchanged",
                "unchanged",
            ],
        }
    )

    released = normalize_release_columns(source)

    assert released["method"].tolist() == [
        "Prediction-error risk score",
        "Prediction-error risk score, endpoint-balanced",
        "Prediction-error risk score, top 20%",
    ]
    assert released["delta_definition"].tolist()[0] == (
        "comparator minus prediction-error risk score"
    )
