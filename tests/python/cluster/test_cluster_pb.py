"""Pinned PB baselines use ordinary paired tasks and independent OPB replay."""
import copy
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "bench"))
from cluster.lib import campaign, install_pb, runners
from cluster.lib.adapters import measurement


def plan(root):
    return {"root": str(root), "runtime": {"python": sys.executable, "env": "/usr/bin/env",
            "scip-naps": "/tools/scip-naps.py", "naps": "/tools/naps", "uwrmaxsat-scip": "/tools/uwrmaxsat"},
            "memory_mb": 1024, "grace_seconds": 1, "checker_seconds": 10, "checker_memory_mb": 256}


def test_pb_baselines_are_normal_unseeded_arms_in_all_stages():
    config = campaign.read(campaign.DEFAULT_CONFIG)
    campaign.validate_config(config)
    solvers = ["qayd-pb", "sat4j", "scip-naps", "uwrmaxsat-scip"]
    assert config["solvers"]["pb"] == solvers
    items = [{"id": "decision", "family": "decision", "track": "pb", "path": "a.opb", "sense": "satisfy"},
             {"id": "optimization", "family": "optimization", "track": "pb", "path": "b.opb", "sense": "min"}]
    for stage in ("smoke", "gate", "full"):
        tasks = campaign.tasks_for(items, config, stage)
        assert len(tasks) == 2
        assert all(task["solvers"] == solvers and task["seed"] is None for task in tasks)
        assert all(task.get("solver_seeds", {}).get(solver, task["seed"]) is None
                   for task in tasks for solver in solvers)
    assert {task["seconds"] for task in campaign.tasks_for(items, config, "full")} == {3600}
    previous = copy.deepcopy(config)
    previous["solvers"]["pb"] = ["qayd-pb", "sat4j"]
    campaign.validate_config(previous)
    assert all(task["solvers"] == ["qayd-pb", "sat4j"] for task in campaign.tasks_for(items, previous, "full"))


@pytest.mark.parametrize("sense,mode,scip_seconds", [("satisfy", "--dec", 500), ("min", "--opt", 1000)])
def test_scip_naps_command_preserves_official_modes_and_outer_budget(sense, mode, scip_seconds):
    argv = runners.command("scip-naps", {"track": "pb", "sense": sense}, {"seconds": 3600, "seed": None},
                           plan(ROOT), Path("/scratch/case.opb"))
    assert mode in argv and f"--sciptime={scip_seconds}" in argv
    assert not any(arg.startswith("--pbargs") for arg in argv)
    assert argv[argv.index("--pbsolver") + 1] == "/tools/naps"
    assert argv[argv.index("--output") + 1] == "/scratch/scip-naps.presolved.opb"
    assert argv[-1] == "/scratch/case.opb"
    assert "-u" in argv[:argv.index(sys.executable)] and argv[argv.index(sys.executable) + 1] == "-u"


def test_native_pb_commands_clear_environment_overrides_and_keep_scip_sequential(tmp_path, monkeypatch):
    monkeypatch.setenv("UWRFLAGS", "-no-scip -first")
    monkeypatch.setenv("PBSATISFIABILITYONLY", "1")
    fake = tmp_path / "solver"
    fake.write_text("#!/usr/bin/env python3\nimport json,os,sys\n"
                    "print(json.dumps({'argv':sys.argv[1:],'env':dict(os.environ)}))\n")
    fake.chmod(0o755)
    setup = plan(tmp_path)
    setup["runtime"]["uwrmaxsat-scip"] = str(fake)
    argv = runners.command("uwrmaxsat-scip", {"track": "pb", "sense": "satisfy"},
                           {"seconds": 30, "seed": None}, setup, Path("case.opb"))
    output = json.loads(subprocess.check_output(argv, text=True))
    assert output["argv"] == ["-no-par", "-cpu-lim=30", "case.opb"]
    assert "UWRFLAGS" not in output["env"] and "PBSATISFIABILITYONLY" not in output["env"]
    assert all(output["env"][key] == "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"))
    with pytest.raises(ValueError, match="unsupported PB solver"):
        runners.command("unknown", {"track": "pb"}, {"seconds": 30, "seed": None}, setup, Path("case.opb"))


@pytest.mark.parametrize("solver", ["scip-naps", "uwrmaxsat-scip"])
@pytest.mark.parametrize("witness,valid", [("v x1 -x2", True), ("v -x1 -x2", False)])
def test_new_pb_arms_replay_compressed_instances_before_accepting_an_optimum(tmp_path, monkeypatch, solver, witness, valid):
    source = "* #variable= 2 #constraint= 1\nmin: +1 x1 +2 x2 ;\n+1 x1 +1 x2 >= 1 ;\n"
    (tmp_path / "case.opb.gz").write_bytes(gzip.compress(source.encode()))

    def measure(argv, seconds, memory, grace, stem, sense):
        assert Path(argv[-1]).read_text() == source
        return {"stdout": f"o 1\ns OPTIMUM FOUND\n{witness}\n", "stderr": "", "claimed_status": "OPTIMUM",
                "best_incumbent": {"value": 1}, "has_solution": True, "returncode": 0, "timed_out": False,
                "elapsed_wall_seconds": 1, "cpu_seconds": 1, "peak_rss_kb": 1, "first_incumbent": None}

    monkeypatch.setattr(runners, "measure", measure)
    row = runners.run_arm(solver, {"track": "pb", "path": "case.opb.gz", "sense": "min"},
                          {"seconds": 30, "seed": None}, plan(tmp_path), tmp_path)
    assert row["validation"]["valid"] is valid
    assert row["solved"] is valid and row["has_incumbent"] is valid
    assert row["status"] == ("OPTIMUM" if valid else "INVALID")
    assert row["objective"] == (1 if valid else None)


def fake_archive(tmp_path, monkeypatch):
    directory = tmp_path / "PB tools"
    downloads = directory / "downloads"
    downloads.mkdir(parents=True)
    spec = dict(install_pb.SOURCES["uwrmaxsat-scip"])
    archive = downloads / spec["filename"]
    with zipfile.ZipFile(archive, "w") as stream:
        for name, content in (("bin/uwrmaxsat", b"#!/bin/sh\nexit 0\n"), ("code/uwrmaxsat/LICENSE", b"license"),
                              ("code/uwrmaxsat/README.md", b"upstream README")):
            stream.writestr("UWrMaxSAT-2026/" + name, content)
    spec["sha256"] = install_pb.sha(archive)
    monkeypatch.setattr(install_pb, "SOURCES", {**install_pb.SOURCES, "uwrmaxsat-scip": spec})
    monkeypatch.setattr(install_pb, "UWR_BINARY_SHA256", hashlib.sha256(b"#!/bin/sh\nexit 0\n").hexdigest())
    original_cached = install_pb.cached
    monkeypatch.setattr(install_pb, "cached", lambda directory, key: original_cached(directory, key)
                        if key == "uwrmaxsat-scip" else archive)
    # Mark unrelated source downloads as present; no network is needed.
    for key, entry in install_pb.SOURCES.items():
        if key != "uwrmaxsat-scip":
            (downloads / entry["filename"]).touch()
    monkeypatch.setattr(install_pb.urllib.request, "urlopen", lambda *a, **kw: pytest.fail("unexpected network"))
    install_pb.prepare(directory)
    return directory


def test_offline_pb_runtime_verification_detects_tampered_binary_license_and_receipt(tmp_path, monkeypatch):
    directory = fake_archive(tmp_path, monkeypatch)
    binary = install_pb.verify(directory, "uwrmaxsat-scip")["uwrmaxsat-scip"]
    files = install_pb.artifacts(directory, "uwrmaxsat-scip")
    assert binary in files and binary.parent / "LICENSE" in files and binary.parent.parent / "install.json" in files
    install_pb.prepare(directory)
    for path in (binary, binary.parent / "LICENSE", binary.parent.parent / "install.json"):
        original = path.read_bytes()
        path.write_bytes(b"changed")
        with pytest.raises(ValueError):
            install_pb.verify(directory, "uwrmaxsat-scip")
        path.write_bytes(original)
    assert install_pb.verify(directory, "uwrmaxsat-scip")["uwrmaxsat-scip"] == binary


def test_pb_prepare_rejects_changed_archive_without_replacing_an_installed_solver(tmp_path, monkeypatch):
    directory = fake_archive(tmp_path, monkeypatch)
    binary = install_pb.verify(directory, "uwrmaxsat-scip")["uwrmaxsat-scip"]
    original = binary.read_bytes()
    archive = directory / "downloads" / install_pb.SOURCES["uwrmaxsat-scip"]["filename"]
    archive.write_bytes(b"changed archive")
    with pytest.raises(ValueError, match="missing or changed PB source"):
        install_pb.prepare(directory)
    assert binary.read_bytes() == original


def test_login_node_build_is_rejected_before_running_compilers(tmp_path, monkeypatch):
    monkeypatch.delenv("OAR_JOB_ID", raising=False)
    with pytest.raises(ValueError, match="requires an OAR allocation"):
        install_pb.build(tmp_path, tmp_path / "scratch", 1)
    assert not (tmp_path / "scratch").exists()


def test_pb_runtime_freezes_scip_companions_and_checks_pyscipopt_version(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    paths = {}
    for key in ("naps", "scip-naps", "uwrmaxsat-scip"):
        paths[key] = tmp_path / key
        paths[key].write_text("executable")
    receipt = tmp_path / "install.json"
    receipt.write_text("receipt")
    monkeypatch.setattr(install_pb, "verify", lambda _, solver: {key: path for key, path in paths.items()
                        if key == solver or key == "naps" and solver == "scip-naps"})
    monkeypatch.setattr(install_pb, "artifacts", lambda *_: [receipt, *paths.values()])
    monkeypatch.setattr(importlib.metadata, "version", lambda _: install_pb.PYSCIPOPT_VERSION)
    libscip = tmp_path / "libscip.so.10.0"
    libscip.write_text("SCIP")
    monkeypatch.setattr(campaign, "package_files", lambda _: {str(libscip): "unused"})
    setup = campaign.runtime_identity({"solvers": {"pb": ["scip-naps", "uwrmaxsat-scip"]}},
                                      [{"track": "pb"}], tmp_path, SimpleNamespace())
    assert set(map(str, [receipt, libscip, *paths.values()])) <= setup["artifacts"].keys()
    assert setup["runtime"]["pyscipopt_version"] == install_pb.PYSCIPOPT_VERSION
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "other")
    with pytest.raises(ValueError, match="requires pyscipopt"):
        campaign.runtime_identity({"solvers": {"pb": ["scip-naps"]}}, [{"track": "pb"}], tmp_path, SimpleNamespace())


def test_versioned_shared_libraries_are_part_of_runtime_provenance(tmp_path, monkeypatch):
    for name in ("binding.py", "extension.so", "libscip.so.10.0", "libgfortran.so.5.0", "cache.pyc"):
        (tmp_path / name).touch()
    distribution = SimpleNamespace(files=list(tmp_path.iterdir()), locate_file=lambda path: path)
    monkeypatch.setattr(importlib.metadata, "distribution", lambda _: distribution)
    names = {Path(path).name for path in campaign.package_files("pyscipopt")}
    assert names == {"binding.py", "extension.so", "libscip.so.10.0", "libgfortran.so.5.0"}


def test_wrapper_and_backend_share_one_memory_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(measurement, "process_tree_rss_kb", lambda _: 1024 * 513)
    result = measurement.execute_streaming(
        [sys.executable, "-c", "import time; time.sleep(10)"], "min", 10, 0, 512, 0.1,
        tmp_path / "stdout", tmp_path / "stderr",
    )
    assert result["memory_exceeded"] and not result["timed_out"]
    assert result["peak_rss_kb"] == 1024 * 513
    assert result["elapsed_wall_seconds"] < 5


def test_memory_limit_cannot_leave_a_solved_pb_claim(tmp_path, monkeypatch):
    (tmp_path / "case.opb").write_text("* #variable= 1 #constraint= 1\nmin: +1 x1 ;\n+1 x1 >= 1 ;\n")
    monkeypatch.setattr(runners, "measure", lambda *_a, **_kw: {
        "stdout": "o 1\ns OPTIMUM FOUND\nv x1\n", "stderr": "", "claimed_status": "OPTIMUM",
        "best_incumbent": {"value": 1}, "returncode": 0, "memory_exceeded": True, "timed_out": False,
        "elapsed_wall_seconds": 1, "cpu_seconds": 1, "peak_rss_kb": 2000000, "first_incumbent": None,
    })
    row = runners.run_arm("scip-naps", {"track": "pb", "path": "case.opb", "sense": "min"},
                          {"seconds": 30, "seed": None}, plan(tmp_path), tmp_path)
    assert row["status"] == "ERROR" and not row["solved"]
    assert row["details"]["resource_limit"] == "memory"
