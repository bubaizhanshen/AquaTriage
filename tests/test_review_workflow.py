import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from ecoood.review_workflow import (
    BIOACTIVITY, aggregate_cases, export_run, fit_reference, load_reference,
    measured_target, predict_cases, prepare_inputs, rank_chemicals, read_input_csv,
    run_review, save_reference, template_csv,
)


def raw_cases():
    rows = []
    for index in range(12):
        for repeat in range(4):
            rows.append({
                "chemical_id": f"c{index:02}", "smiles": "C" * (index + 1) + "O",
                "species": "Pimephales promelas", "genus": "Pimephales",
                "endpoint": "fish_96h_lc50", "duration_h": 96,
                "identity_verified": True, "target_log_molar": -2.0 - index * 0.2 + repeat * 0.05,
                "medium": "FW", "temperature_c": 20 + repeat,
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def reference():
    return fit_reference(raw_cases(), trees=8)


def test_raw_contract_and_duration():
    data = raw_cases().head(5).copy()
    data.loc[0, "duration_h"] = 89
    data.loc[1, "smiles"] = "[Na+].[Cl-]"
    data.loc[2, "identity_verified"] = False
    data.loc[3, "ph"] = "not a number"
    accepted, rejected = prepare_inputs(data)
    assert len(accepted) == 1
    assert len(rejected) == 4
    assert "Duration" in rejected.iloc[0].rejection_reason
    assert len(BIOACTIVITY) == 34


def test_identity_and_alias_handling():
    data = raw_cases().iloc[[0, 4]].copy()
    data.loc[data.index[1], "chemical_id"] = "c00"
    a, r = prepare_inputs(data)
    assert a.empty and len(r) == 2
    data = raw_cases().iloc[[0, 4]].copy()
    data.loc[data.index[1], "smiles"] = data.iloc[0].smiles
    a, r = prepare_inputs(data)
    assert r.empty and a.ids_sharing_representation.eq(2).all()


@pytest.mark.parametrize("value,unit,mw,expected", [
    (100, "mg/L", 200, np.log10(0.1 / 200)),
    (2, "uM", 200, np.log10(2e-6)),
    (1, "AI ppb", 100, -8),
])
def test_unit_conversion(value, unit, mw, expected):
    assert measured_target(value, unit, mw) == pytest.approx(expected)


@pytest.mark.parametrize("value,unit", [(">100", "mg/L"), (0, "mg/L"), (1, "bogus"), (np.inf, "M")])
def test_bad_target(value, unit):
    with pytest.raises((ValueError, TypeError)):
        measured_target(value, unit, 200)


def test_csv_identifiers_and_template():
    data = read_input_csv(template_csv())
    assert data.identity_verified.iloc[0] == ""
    assert read_input_csv(b"chemical_id,value\n001,2\n").chemical_id.iloc[0] == "001"
    with pytest.raises(ValueError, match="unique"):
        read_input_csv(b"chemical_id,chemical_id\nA,B\n")


def test_hosted_csv_limits():
    with pytest.raises(ValueError, match="MB limit"):
        read_input_csv(b"column\nvalue\n", max_bytes=5)
    with pytest.raises(ValueError, match="at most 1 records"):
        read_input_csv(b"column\none\ntwo\n", max_records=1)


def test_prediction_ignores_targets_and_unapproved_inputs(reference):
    query = raw_cases().iloc[[0, 8, 16]].copy()
    one, rejected, neighbors = predict_cases(reference, query)
    query["target_log_molar"] = 1000
    query["ctx_target_leak"] = -1000
    query["study_year"] = 3000
    query["physchem_mol_wt"] = 1e8
    two, _, _ = predict_cases(reference, query)
    assert rejected.empty
    np.testing.assert_allclose(one.prediction_log_molar, two.prediction_log_molar)
    assert len(neighbors) == len(one) * 5
    np.testing.assert_allclose(one.block_knn, neighbors.groupby("case_id", sort=False).distance.mean())


def test_training_preprocessing_is_frozen(reference):
    frame = raw_cases().iloc[[0, 4]].copy()
    one, _, _ = predict_cases(reference, frame)
    extra = frame.iloc[[1]].copy()
    extra["chemical_id"] = "new"
    extra["smiles"] = "c1ccccc1"
    extra["temperature_c"] = 1000
    extra["species"] = "Unknown species"
    both, _, _ = predict_cases(reference, pd.concat([frame, extra], ignore_index=True))
    np.testing.assert_allclose(one.block_knn, both.block_knn.iloc[:2])
    assert both.iloc[-1].unseen_species


def test_endpoint_aggregation(reference):
    cases, _, _ = predict_cases(reference, raw_cases().head(4))
    cases["prediction_log_mg_l"] = [0, 1, 2, -2]
    cases["endpoint"] = ["fish_96h_lc50"] * 3 + ["daphnia_48h_ec50"]
    cases["block_knn"] = [1, 2, 100, 4]
    endpoints, chemicals = aggregate_cases(cases, 1.0)
    assert endpoints.set_index("endpoint").loc["fish_96h_lc50", "prediction_log_mg_l"] == 1
    assert chemicals.iloc[0].prediction_log_mg_l == -2
    assert chemicals.iloc[0].block_knn == 4
    assert chemicals.iloc[0].predicted_high_concern
    assert len(chemicals) == 1


def chemical_table():
    return pd.DataFrame({
        "chemical_id": ["c", "b", "a", "d"], "margin_log10": [2.0, 1.0, 1.0, -2.0],
        "block_knn": [9.0, 3.0, 1.0, 10.0], "similarity_ad": [.9, .8, .7, .6],
        "ensemble_sd": [1.0, 2.0, 3.0, 4.0], "predicted_high_concern": [False, False, False, True],
    })


def test_task_populations_and_ties():
    data = chemical_table()
    omission = rank_chemicals(data, task="hazard_omission", review_count=2)
    assert omission.chemical_id.tolist() == ["a", "b", "c", "d"]
    general = rank_chemicals(data, task="general_error", review_count=1)
    assert general.chemical_id.tolist() == ["d", "c", "b", "a"]
    full = rank_chemicals(data, task="hazard_omission", review_count=4)
    assert full.selected_for_review.all()
    none = rank_chemicals(data, task="general_error", review_count=0)
    assert not none.selected_for_review.any()
    with pytest.raises(ValueError):
        rank_chemicals(data, task="general_error", review_count=5)
    with pytest.raises(ValueError):
        rank_chemicals(data, task="general_error", review_count=1, rule="threshold_proximity")


def test_rounded_quota_and_result_archive(reference):
    frame = raw_cases().iloc[[0, 4]].copy()
    frame["chemical_name"] = "=1+1"
    run = run_review(reference, frame, review_fraction=0.25)
    assert run.manifest["review_count"] == 0  # round(0.5), matching benchmark
    run = run_review(reference, frame, review_fraction=1)
    with zipfile.ZipFile(io.BytesIO(export_run(run))) as archive:
        assert set(archive.namelist()) == {
            "cases.csv", "endpoints.csv", "chemicals.csv", "rejected.csv",
            "neighbors.csv", "run_manifest.json",
        }
        assert "'=1+1" in archive.read("chemicals.csv").decode()
        manifest = json.loads(archive.read("run_manifest.json"))
        assert manifest["selection_uses_measured_queue_outcomes"] is False


def test_all_rejected_and_partial_chemical(reference):
    frame = raw_cases().head(2).copy()
    frame["identity_verified"] = False
    run = run_review(reference, frame)
    assert run.chemicals.empty and len(run.rejected) == 2
    assert run.manifest["review_count"] == 0
    with zipfile.ZipFile(io.BytesIO(export_run(run))) as archive:
        assert pd.read_csv(archive.open("chemicals.csv")).empty
        assert "prediction_mg_l" in pd.read_csv(archive.open("cases.csv"))
    frame.loc[0, "identity_verified"] = True
    run = run_review(reference, frame, review_count=1)
    assert run.chemicals.iloc[0].rejected_case_count == 1


def test_reference_roundtrip(reference, tmp_path):
    path = tmp_path / "reference.joblib"
    save_reference(reference, path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    restored = load_reference(path, expected_sha256=metadata["artifact_sha256"])
    queue = raw_cases().iloc[[0, 4]]
    a = run_review(reference, queue)
    b = run_review(restored, queue)
    pd.testing.assert_frame_equal(a.chemicals, b.chemicals)
    path.write_bytes(b"not a pickle")
    with pytest.raises(ValueError, match="checksum"):
        load_reference(path, expected_sha256=metadata["artifact_sha256"])


def test_bad_training_not_silently_dropped():
    data = raw_cases()
    data.loc[0, "identity_verified"] = False
    with pytest.raises(ValueError, match="Resolve 1 invalid"):
        fit_reference(data, trees=1)
