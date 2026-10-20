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


def test_cloud_is_inference_only_and_loads_published_reference(cloud):
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


def test_cloud_objective_switch_and_quota_endpoints(cloud):
    next(s for s in cloud.selectbox if s.label == "Objective").select(
        "Recover hazard omissions",
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
