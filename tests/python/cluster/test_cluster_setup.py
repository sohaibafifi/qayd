"""Check the simple setup launcher without contacting an OAR scheduler."""
from pathlib import Path
import shlex
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))
from cluster.lib import setup


PREDICATE = "network_address NOT IN (SELECT network_address FROM resources WHERE gpumodel IS NOT NULL)"


def arguments(*extra):
    return setup.parser().parse_args(["--cores", "4", "--walltime", "01:00:00",
                                     "--property", PREDICATE, *extra])


def prepared_root(tmp_path, monkeypatch):
    root = tmp_path / "repo with spaces"
    monkeypatch.setattr(setup, "ROOT", root)
    for name in (".venv/bin/python", "bench/cluster/runs/tools/cargo/bin/rustup"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("prepared")
    return root


def test_setup_submits_one_lp_build_with_both_types_and_cpu_filter(tmp_path, monkeypatch):
    root = prepared_root(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(setup.subprocess, "run", lambda argv, **kw: calls.append((argv, kw)))
    command = setup.submit(arguments("--lp", "--execute"))
    assert len(calls) == 1
    assert calls[0][1]["cwd"] == root
    assert command[command.index("-l") + 1] == "/core=4,walltime=01:00:00"
    assert command[command.index("-p") + 1] == PREDICATE
    assert command.count("-t") == 2
    assert "idempotent" in command and "besteffort" in command
    payload = shlex.split(command[-1])
    assert payload == ["bash", str(root / "bench/cluster/scripts/bootstrap.oar"), "--repo", str(root),
                       "--jobs", "4", "--lp"]
    assert "--stage" not in payload


def test_setup_dry_run_keeps_resources_and_queue_configurable(tmp_path, monkeypatch):
    prepared_root(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(setup.subprocess, "run", lambda *args, **kw: calls.append(args))
    command = setup.submit(arguments("--resource", "/nodes=1/core=4", "--queue", "chosen-queue"))
    assert calls == []
    assert command[command.index("-l") + 1] == "/nodes=1/core=4,walltime=01:00:00"
    assert command[command.index("-q") + 1] == "chosen-queue"
    assert "--lp" not in shlex.split(command[-1])


def test_setup_passes_the_compute_node_scratch_path_without_resolving_it_locally(tmp_path, monkeypatch):
    prepared_root(tmp_path, monkeypatch)
    command = setup.submit(arguments("--scratch-dir", "/scratch/personal directory", "--lp"))
    payload = shlex.split(command[-1])
    assert payload[payload.index("--scratch-dir")+1] == "/scratch/personal directory"
    assert "--lp" in payload


def test_setup_refuses_missing_preparation_or_empty_cpu_filter(tmp_path, monkeypatch):
    monkeypatch.setattr(setup, "ROOT", tmp_path)
    calls = []
    monkeypatch.setattr(setup.subprocess, "run", lambda *args, **kw: calls.append(args))
    with pytest.raises(ValueError, match="preparation is incomplete"):
        setup.submit(arguments("--execute"))
    args = arguments("--execute")
    args.property = "  "
    with pytest.raises(ValueError, match="CPU-only"):
        setup.submit(args)
    assert not calls
