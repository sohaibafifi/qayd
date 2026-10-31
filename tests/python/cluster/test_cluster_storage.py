"""Shared storage must preserve corpus, results and stable repository aliases."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "bench"))
from cluster.lib import storage


def test_storage_preserves_data_and_results_and_is_resumable(tmp_path, monkeypatch):
    root, data_root = tmp_path / "source", tmp_path / "shared data"
    root.mkdir()
    monkeypatch.setattr(storage, "ROOT", root)
    for name in storage.DIRECTORIES:
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "retained.txt").write_text(name)
    mapping = storage.configure(data_root)
    assert storage.configure(data_root) == mapping
    for name in storage.DIRECTORIES:
        assert (root/name).is_symlink()
        assert (root/name/"retained.txt").read_text() == name
        assert (data_root/name/"retained.txt").read_text() == name
    assert list(root.glob("data/**/*.txt")) == [root/"data/retained.txt"]
    assert (root/"bench/cluster/runs/storage.json").is_file()


def test_storage_refuses_conflicts_before_moving_anything(tmp_path, monkeypatch):
    root, data_root = tmp_path / "source", tmp_path / "shared"
    (root/"data").mkdir(parents=True)
    (root/"data/keep").write_text("input")
    (root/"bench/cluster/runs").mkdir(parents=True)
    (data_root/"bench/cluster/runs").mkdir(parents=True)
    monkeypatch.setattr(storage, "ROOT", root)
    with pytest.raises(ValueError, match="refusing to merge"):
        storage.configure(data_root)
    assert not (root/"data").is_symlink()
    assert (root/"data/keep").read_text() == "input"
    assert not (data_root/"data").exists()


def test_storage_rejects_home_and_source_descendants(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    monkeypatch.setattr(storage, "ROOT", root)
    with pytest.raises(ValueError, match="outside the source checkout"):
        storage.configure(root/"data-store")
    with pytest.raises(ValueError, match="outside HOME"):
        storage.configure(Path.home()/"test-qayd-data-store")
