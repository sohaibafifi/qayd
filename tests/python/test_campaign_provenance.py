"""Exact source-tree provenance for benchmark campaigns."""

import importlib.util
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve().parents[2] / "bench" / "common" / "competitive.py"


def load_module():
    spec = importlib.util.spec_from_file_location("qayd_competitive", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def test_source_fingerprint_covers_tracked_staged_and_untracked_edits(tmp_path):
    competitive = load_module()
    git(tmp_path, "init")
    git(tmp_path, "config", "user.email", "test@example.invalid")
    git(tmp_path, "config", "user.name", "Qayd Test")
    (tmp_path / ".gitignore").write_text("results/\n", encoding="utf-8")
    source = tmp_path / "solver.rs"
    source.write_text("one\n", encoding="utf-8")
    git(tmp_path, "add", ".gitignore", "solver.rs")
    git(tmp_path, "commit", "-m", "base")

    clean = competitive.source_tree_sha256(tmp_path)
    assert clean is not None and len(clean) == 64
    assert competitive.source_tree_sha256(tmp_path) == clean

    source.write_text("two\n", encoding="utf-8")
    modified = competitive.source_tree_sha256(tmp_path)
    assert modified != clean
    git(tmp_path, "add", "solver.rs")
    assert competitive.source_tree_sha256(tmp_path) == modified

    extra = tmp_path / "new.rs"
    extra.write_text("new\n", encoding="utf-8")
    untracked = competitive.source_tree_sha256(tmp_path)
    assert untracked not in {clean, modified}

    ignored = tmp_path / "results" / "run.jsonl"
    ignored.parent.mkdir()
    ignored.write_text("result\n", encoding="utf-8")
    assert competitive.source_tree_sha256(tmp_path) == untracked
