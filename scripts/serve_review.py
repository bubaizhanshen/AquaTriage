"""Launch the local review workspace with all runtime files inside the project."""
from pathlib import Path
import argparse
import os
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--reference", type=Path, default=ROOT / "models/review-v1/reference.joblib")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "tmp/platform")
    args = parser.parse_args()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.environ["AQUATRIAGE_WORK_DIR"] = str(work)
    os.environ["AQUATRIAGE_REFERENCE"] = str(args.reference.resolve())
    for key in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "MPLCONFIGDIR"):
        os.environ[key] = str(work)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.execv(sys.executable, [
        sys.executable, "-m", "streamlit", "run", str(ROOT / "app/review.py"),
        "--server.address=127.0.0.1", f"--server.port={args.port}",
        "--server.headless=true", "--browser.gatherUsageStats=false",
        "--server.maxUploadSize=20", "--server.fileWatcherType=none",
        "--theme.base=light", "--theme.primaryColor=#127f83",
    ])


if __name__ == "__main__":
    main()
