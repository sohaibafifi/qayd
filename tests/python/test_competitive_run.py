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


def test_run_measured_bounds_process_table_sampling(monkeypatch, tmp_path):
    competitive = load_competitive_module()
    assert 0.25 <= competitive._RSS_SAMPLE_INTERVAL_SECONDS <= 0.5
    sampled_pids = []
    monkeypatch.setattr(
        competitive,
        "process_tree_rss_kib",
        lambda pid: sampled_pids.append(pid) or 0,
    )
    child_duration = 0.65

    result = competitive.run_measured(
        [
            sys.executable,
            "-c",
            f"import time; time.sleep({child_duration!r})",
        ],
        timeout=2,
        cwd=tmp_path,
    )

    assert not result["timed_out"]
    assert result["return_code"] == 0
    assert 2 <= len(sampled_pids) <= 4
    assert len(set(sampled_pids)) == 1


def test_run_measured_enforces_timeout_between_rss_samples(monkeypatch, tmp_path):
    competitive = load_competitive_module()
    sampled_pids = []
    monkeypatch.setattr(
        competitive,
        "process_tree_rss_kib",
        lambda pid: sampled_pids.append(pid) or 0,
    )
    timeout = 0.2

    result = competitive.run_measured(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        timeout=timeout,
        cwd=tmp_path,
    )

    assert result["timed_out"]
    assert result["return_code"] != 0
    assert timeout * 0.8 <= result["wall_seconds"] < timeout + 0.25
    assert len(sampled_pids) == 2
