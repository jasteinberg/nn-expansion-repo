"""Type annotations (mypy) and undefined / unused names (ruff F) are checked.

Both tools come with the dev extra: pip install -e .[dev]
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", *args], cwd=REPO, capture_output=True, text=True)


def test_mypy() -> None:
    r = _run("mypy")
    assert r.returncode == 0, r.stdout + r.stderr


def test_ruff() -> None:
    r = _run("ruff", "check", "nn_expansion", "experiments", "tests", "scripts")
    assert r.returncode == 0, r.stdout + r.stderr
