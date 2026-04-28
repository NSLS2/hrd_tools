"""Smoke-test that every FDR design script responds to ``--help``.

This catches the common breakages introduced by the shared-parameter
refactor: import-time errors, missing CLI plumbing, and mis-paths to the
blessed YAML / ROI files.

Each script is run via ``subprocess`` so import-time side effects in one
script don't poison another.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

FDR_DIR = Path(__file__).resolve().parents[1] / "design_scripts" / "FDR"

# Skip helper module + anything intentionally not a script.
_SKIP = {"_fdr_params.py"}


def _scripts() -> list[Path]:
    return sorted(p for p in FDR_DIR.glob("*.py") if p.name not in _SKIP)


@pytest.mark.parametrize("script", _scripts(), ids=lambda p: p.name)
def test_script_help_exits_zero(script: Path):
    # Make the helper module importable for the subprocess.
    env = dict(os.environ)
    env["PYTHONPATH"] = str(FDR_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    # Force a non-interactive matplotlib backend so ``import matplotlib.pyplot``
    # at module top-level can't accidentally try to open a display.
    env.setdefault("MPLBACKEND", "Agg")
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"{script.name} --help failed (rc={result.returncode})\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    # All FDR scripts share the standard CLI; --outdir should appear.
    assert "--outdir" in result.stdout, (
        f"{script.name} --help is missing --outdir flag:\n{result.stdout}"
    )
