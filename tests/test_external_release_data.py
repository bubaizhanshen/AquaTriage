from pathlib import Path

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
    assert main["target_log_molar"].notna().all()
    assert strict["target_log_molar"].notna().all()
    assert main["document_urls"].fillna("").str.len().gt(0).all()


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
