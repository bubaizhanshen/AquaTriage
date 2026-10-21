"""Hosted-entrypoint checks; run with app/requirements.txt installed."""
import io
from pathlib import Path
import tempfile

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_rdkit_molecular_drawing_runtime():
    from PIL import Image
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    drawer = rdMolDraw2D.MolDraw2DCairo(440, 330)
    drawer.drawOptions().useBWAtomPalette()
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"))
    drawer.FinishDrawing()
    image = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB")
    assert image.size == (440, 330)
    assert any(low < high for low, high in image.getextrema())


@pytest.fixture
def cloud(monkeypatch, tmp_path):
    for name in (
        "AQUATRIAGE_CLOUD", "AQUATRIAGE_WORK_DIR", "AQUATRIAGE_REFERENCE",
        "PYTHONDONTWRITEBYTECODE", "TMPDIR", "TMP", "TEMP",
        "XDG_CACHE_HOME", "MPLCONFIGDIR", "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "PYTHONPATH",
    ):
        monkeypatch.setenv(name, "")
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    app = AppTest.from_file(str(ROOT / "app" / "cloud.py"), default_timeout=90)
    app.run()
    assert not app.exception
    assert not app.error
    return app


def run_button(app):
    return next(button for button in app.button if button.label == "Run review")


def test_cloud_recovers_presentation_cached_before_ui_update(cloud, monkeypatch):
    from ecoood import review_presentation

    # A running Cloud process can retain the previous module after a Git update.
    monkeypatch.delattr(review_presentation, "TASK_HELP")
    monkeypatch.delattr(review_presentation, "chemical_review_details")
    cloud.run()
    assert not cloud.exception
    run_button(cloud).click().run()
    assert not cloud.exception
    assert cloud.session_state["run"].manifest["review_count"] == 3
    assert "priority_reason" in review_presentation.review_queue_view(cloud.session_state["run"])


def test_cloud_is_inference_only_and_loads_published_reference(cloud):
    from streamlit import file_util

    assert file_util.file_in_pythonpath(str(ROOT / "src/ecoood/review_presentation.py"))
    assert not any(r.label == "Training reference" for r in cloud.radio)
    assert not any(b.label == "Fit reference" for b in cloud.button)
    run_button(cloud).click().run()
    assert not cloud.exception
    result = cloud.session_state["run"]
    assert result.manifest["reference"]["records"] == 4519
    assert result.manifest["reference"]["chemicals"] == 775
    assert result.manifest["eligible_chemicals"] == 12
    assert result.manifest["review_count"] == 3
    assert result.manifest["rule"] == "block_knn"
    assert cloud.session_state["review_tab"] == "Review queue"
    assert cloud.session_state["queue_selected_only"] is True


def test_hosted_config_watches_source_updates():
    import tomllib

    config = tomllib.loads((ROOT / ".streamlit/config.toml").read_text())
    assert config["server"]["fileWatcherType"] == "auto"
    assert "app" in config["server"]["folderWatchList"]


def test_cloud_objective_switch_and_quota_endpoints(cloud):
    next(s for s in cloud.selectbox if s.label == "Objective").select(
        "Find potentially missed hazards",
    ).run()
    slider = next(s for s in cloud.slider if s.label == "Eligible chemicals to review (%)")
    slider.set_value(0).run()
    run_button(cloud).click().run()
    assert not cloud.exception
    assert cloud.session_state["run"].manifest["rule"] == "threshold_proximity"
    assert cloud.session_state["run"].manifest["review_count"] == 0
    next(s for s in cloud.slider if s.label == slider.label).set_value(100).run()
    run_button(cloud).click().run()
    assert not cloud.exception
    assert cloud.session_state["run"].manifest["review_count"] == 12


def test_manual_entry_validation_and_count_quota(cloud):
    next(w for w in cloud.button_group if w.label == "Input source").set_value("Enter cases").run()
    assert not cloud.exception
    assert run_button(cloud).disabled
    next(w for w in cloud.button_group if w.label == "Review workload").set_value("Chemical count").run()
    assert not cloud.exception
    next(w for w in cloud.text_input if w.label == "Chemical identifier").set_value("example-001")
    next(w for w in cloud.text_input if w.label == "Confirmed SMILES").set_value("CCO")
    next(w for w in cloud.selectbox if w.label == "Species").select("Pimephales promelas")
    next(w for w in cloud.button if w.label == "Add case").click().run()
    assert not cloud.exception
    assert cloud.error
    assert run_button(cloud).disabled
    assert next(w for w in cloud.text_input if w.label == "Confirmed SMILES").value == "CCO"
    next(w for w in cloud.checkbox if "confirmed" in w.label).check()
    next(w for w in cloud.button if w.label == "Add case").click().run()
    assert not cloud.exception
    assert not cloud.error
    assert len(cloud.session_state["entered_cases"]) == 1
    next(w for w in cloud.number_input if w.label == "Chemicals to review").set_value(1).run()
    run_button(cloud).click().run()
    assert not cloud.exception
    result = cloud.session_state["run"]
    assert result.manifest["eligible_chemicals"] == 1
    assert result.manifest["review_count"] == 1
    assert result.endpoints.endpoint.tolist() == ["fish_96h_lc50"]
    next(w for w in cloud.button if w.label == "Clear").click().run()
    assert not cloud.exception
    assert not cloud.session_state["entered_cases"]
    assert run_button(cloud).disabled


def test_changed_settings_do_not_relabel_completed_run(cloud):
    run_button(cloud).click().run()
    next(s for s in cloud.selectbox if s.label == "Objective").select("Find potentially missed hazards").run()
    assert not cloud.exception
    assert cloud.session_state["run"].manifest["task"] == "general_error"
    assert any("previous run" in w.value for w in cloud.warning)
    summary = next(
        table.value for table in cloud.dataframe
        if list(table.value.columns) == ["Setting", "Value"]
    )
    assert summary.set_index("Setting").loc["Review objective", "Value"] == "Check prediction accuracy"


def test_preset_and_custom_ranking_are_explicit(cloud):
    ranking = next(s for s in cloud.selectbox if s.label == "Ranking")
    assert ranking.disabled
    assert ranking.value == "block_knn"
    next(t for t in cloud.toggle if t.label == "Use objective preset").set_value(False).run()
    next(s for s in cloud.selectbox if s.label == "Ranking").select("ensemble_sd").run()
    run_button(cloud).click().run()
    assert not cloud.exception
    assert cloud.session_state["run"].manifest["rule"] == "ensemble_sd"
    next(t for t in cloud.toggle if t.label == "Use objective preset").set_value(True).run()
    ranking = next(s for s in cloud.selectbox if s.label == "Ranking")
    assert ranking.disabled
    assert ranking.value == "block_knn"
    assert any("previous run" in w.value for w in cloud.warning)
    run_button(cloud).click().run()
    assert cloud.session_state["run"].manifest["rule"] == "block_knn"


def test_chemical_navigation_preserves_review_order(cloud):
    run_button(cloud).click().run()
    ids = cloud.session_state["run"].chemicals.chemical_id.tolist()
    assert cloud.session_state["record_chemical"] == ids[0]
    assert cloud.button(key="previous_chemical").disabled
    cloud.button(key="next_chemical").click().run()
    assert not cloud.exception
    assert cloud.session_state["record_chemical"] == ids[1]
    cloud.button(key="previous_chemical").click().run()
    assert cloud.session_state["record_chemical"] == ids[0]
    next(b for b in cloud.button if b.label == "Queue").click().run()
    assert cloud.session_state["review_tab"] == "Review queue"
