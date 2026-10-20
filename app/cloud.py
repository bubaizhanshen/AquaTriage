"""Inference-only entrypoint for Streamlit Community Cloud."""
from pathlib import Path
import os
import runpy
import tempfile

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "tmp" / "cloud"
WORK.mkdir(parents=True, exist_ok=True, mode=0o700)
os.environ["AQUATRIAGE_CLOUD"] = "1"
os.environ["AQUATRIAGE_WORK_DIR"] = str(WORK)
os.environ["AQUATRIAGE_REFERENCE"] = str(ROOT / "models" / "review-v1" / "reference.joblib")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for key in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "MPLCONFIGDIR"):
    os.environ[key] = str(WORK)
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
tempfile.tempdir = str(WORK)
runpy.run_path(str(ROOT / "app" / "review.py"), run_name="__main__")
