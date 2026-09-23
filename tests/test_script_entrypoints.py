from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_reference_holdout_script_supports_direct_cli_execution() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_reference_holdout_validation.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "reference" in result.stdout.lower()


def test_application_script_supports_direct_cli_execution() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/apply_ecoood.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "screening-routing protocol" in result.stdout.lower()


def test_echa_candidate_selection_supports_direct_cli_execution() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/select_echa_external_candidates.py",
            "--help",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "endpoint availability" in result.stdout.lower()


def test_external_deployment_selection_supports_direct_cli_execution() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/analyze_external_deployment_selection.py",
            "--help",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "external screening queue" in result.stdout.lower()
