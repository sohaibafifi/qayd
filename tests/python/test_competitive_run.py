"""Regression checks for the measured subprocess runner."""

import importlib.util
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[2] / "bench" / "common" / "competitive.py"


def load_competitive_module():
    spec = importlib.util.spec_from_file_location("qayd_competitive", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_measured_drains_large_stdout_and_stderr(tmp_path):
    competitive = load_competitive_module()
    size = 2 * 1024 * 1024
    script = (
        "import sys; "
        f"sys.stdout.write('o' * {size}); sys.stdout.flush(); "
        f"sys.stderr.write('e' * {size}); sys.stderr.flush()"
    )

    result = competitive.run_measured(
        [sys.executable, "-c", script], timeout=5, cwd=tmp_path,
    )

    assert not result["timed_out"]
    assert result["return_code"] == 0
    assert len(result["stdout"]) == size
    assert len(result["stderr"]) == size
