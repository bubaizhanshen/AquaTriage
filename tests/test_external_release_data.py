from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_external_evaluation_tables_match_reported_units() -> None:
    main = pd.read_csv(ROOT / "data" / "processed" / "echa_external_main.csv")
    strict = pd.read_csv(
        ROOT / "data" / "processed" / "echa_external_seven_species.csv"
    )

    assert len(main) == 100
    assert main["chemical_id"].nunique() == 48
    assert len(strict) == 93
    assert strict["chemical_id"].nunique() == 47
    assert set(strict["case_id"]).issubset(set(main["case_id"]))
    assert main["case_id"].is_unique
    assert strict["case_id"].is_unique
    assert main["document_urls"].fillna("").str.len().gt(0).all()
    assert strict["document_urls"].fillna("").str.len().gt(0).all()
    for panel in (main, strict):
        assert panel["target_log_molar"].notna().all()
        assert (panel["toxicity_unit"] == "M").all()
        assert np.allclose(panel["toxicity_value"], panel["molar_concentration"])
        assert np.allclose(
            panel["target_log_molar"],
            np.log10(panel["molar_concentration"]),
        )


def test_external_extension_sample_is_fixed_before_outcome_analysis() -> None:
    candidates = pd.read_csv(
        ROOT / "data" / "processed" / "echa_extension_candidates.csv"
    )

    assert len(candidates) == 150
    assert candidates["representative_casrn"].nunique() == 150
    assert (candidates["endpoint_count"] == 3).all()
    assert not {"toxicity_value", "target_log_molar", "prediction"}.intersection(
        candidates.columns
    )


def test_expanded_external_panel_matches_current_public_snapshot() -> None:
    panel = pd.read_csv(
        ROOT / "data" / "processed" / "ecoood_external_expanded_v1.csv"
    )

    assert len(panel) == 1027
    assert panel["chemical_id"].nunique() == 441
    assert panel["case_id"].is_unique
    assert set(panel["source"]) == {
        "echa_extension_main",
        "echa_pmra_main",
        "japan_moe_official_summary",
    }
    echa_mask = panel["source"].isin({"echa_extension_main", "echa_pmra_main"})
    assert panel.loc[echa_mask, "document_urls"].fillna("").str.len().gt(0).all()
    assert panel["target_log_molar"].notna().all()
    assert set(panel["toxicity_unit"]) == {"M", "mg/L"}
    assert panel["molar_concentration"].notna().all()
    assert np.allclose(
        panel["target_log_molar"],
        np.log10(panel["molar_concentration"]),
    )
