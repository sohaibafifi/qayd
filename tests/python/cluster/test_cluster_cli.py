"""Public entry points and the boundary between observers and frozen execution code."""
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "bench"))
from cluster.lib import campaign


@pytest.mark.parametrize("arguments", [[], ["prepare"], ["prepare", "tools"], ["prepare", "build"],
                                        ["prepare", "corpus"], ["prepare", "inventory"], ["prepare", "storage"],
                                        ["run"], ["run", "--add-cosoco"], ["status"], ["collect"]])
def test_public_help_uses_the_reorganized_modules_without_running_jobs(arguments):
    result = subprocess.run([sys.executable, str(ROOT / "bench/cluster"), *arguments, "--help"],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert not any(word in result.stdout for word in ("Contrôleur", "témoins", "préparation"))


@pytest.mark.parametrize("script", ["fetch_collections.py", "routing.py", "solvers.py"])
def test_downloaders_start_without_site_imports_from_another_directory(script, tmp_path):
    # A fresh Linux interpreter may not have imported collections before the
    # script starts. -S exposes sibling modules that shadow the standard library.
    path = ROOT / "bench/cluster/lib/downloads" / script
    result = subprocess.run([sys.executable, "-S", str(path), "--help"],
                            cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_observer_changes_do_not_change_frozen_source_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    directory = tmp_path / "bench/cluster"
    (directory / "lib").mkdir(parents=True)
    for name in ("status.py", "collect.py", "lib/runners.py"):
        (directory / name).write_text("# original\n")
    (tmp_path / "Cargo.toml").write_text("[workspace]\n")
    original = campaign.source_identity()["sha256"]
    (directory / "status.py").write_text("# display updated\n")
    (directory / "collect.py").write_text("# export updated\n")
    (directory / "runs").mkdir()
    (directory / "runs/result.json").write_text("{}")
    assert campaign.source_identity()["sha256"] == original
    (directory / "lib/runners.py").write_text("# solver invocation changed\n")
    assert campaign.source_identity()["sha256"] != original


def test_runs_are_ignored_and_small_fixtures_remain_versionable():
    for relative, ignored in (("bench/cluster/runs/new-result.json", True),
                              ("bench/cluster/fixtures/maximize.xml", False),
                              ("bench/cluster/README.md", False)):
        result = subprocess.run(["git", "check-ignore", "--no-index", "-q", relative], cwd=ROOT)
        assert (result.returncode == 0) == ignored
        assert result.returncode in (0, 1)
