"""Coverage, budget, replay and bounded scheduler contracts for cluster runs."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import textwrap
from types import SimpleNamespace
import venv

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "bench"))
from cluster.lib import campaign, fetch, hexaly, minizinc, report, runners
from cluster.lib.hexaly import parse_rcp, psplib_text
from cluster.lib.hexaly_report import compare as compare_published


def private_workspace(root):
    (root / "integrations/amthal").mkdir(parents=True, exist_ok=True)
    (root / "integrations/amthal/Cargo.toml").write_text('[features]\ndefault = ["lp-relaxation"]\n')
    backend = root.parent / "amthal"
    backend.mkdir(exist_ok=True)
    (backend / "Cargo.toml").write_text('[package]\nname = "amthal"\n')
    (backend / "src").mkdir(exist_ok=True)
    (backend / "src/lib.rs").write_text("// authorized test backend\n")


@pytest.mark.parametrize("lp_feature", [False, True])
def test_build_selects_the_same_workspace_for_binaries_and_python(tmp_path, monkeypatch, lp_feature):
    monkeypatch.delenv("CARGO_TARGET_DIR", raising=False)
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    monkeypatch.setattr(campaign, "source_identity", lambda: {"sha256": "source"})
    if lp_feature:
        private_workspace(tmp_path)
    extension = tmp_path / "_core.so"
    extension.write_bytes(b"extension")
    monkeypatch.setattr(campaign, "installed_extension", lambda: extension)
    monkeypatch.setattr(campaign, "checked", lambda _: "rustc test")
    for name in ("qayd", "qayd-fzn", "qayd-sat", "qayd-pb"):
        path = tmp_path / "target/release" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    calls = []
    monkeypatch.setattr(campaign.subprocess, "run", lambda args, **kw: calls.append((args, kw)))
    receipt = tmp_path / "receipt.json"
    campaign.build(SimpleNamespace(receipt=receipt, lp_feature=lp_feature))
    assert len(calls) == 2
    binary, python = (call[0] for call in calls)
    assert "--workspace" in binary and "--release" in binary
    assert "--locked" in binary and "--locked" in python
    manifest = str(tmp_path / ("integrations/amthal/Cargo.toml" if lp_feature else "Cargo.toml"))
    assert binary[binary.index("--manifest-path") + 1] == manifest
    assert python[python.index("--manifest-path") + 1] == manifest
    assert "--features" not in binary
    assert python[python.index("--features") + 1] == "python"
    saved = campaign.read(receipt)
    assert saved["features"] == {"binaries": ["lp-relaxation"] if lp_feature else [],
                                 "python": ["python", "lp-relaxation"] if lp_feature else ["python"]}
    assert len(saved["artifacts"]) == 5
    assert saved["commands"] == {"binaries": binary, "python": python}
    assert saved["workspace_manifest"] == manifest
    assert bool(saved["lp_backend"]) == lp_feature
    if lp_feature:
        assert "src/lib.rs" in saved["lp_backend"]["files"]


def test_failed_build_does_not_write_a_success_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "source_identity", lambda: {"sha256": "source"})
    def failed(*_args, **_kwargs):
        raise subprocess.CalledProcessError(1, ["cargo", "build"])
    monkeypatch.setattr(campaign.subprocess, "run", failed)
    receipt = tmp_path / "receipt.json"
    with pytest.raises(subprocess.CalledProcessError):
        campaign.build(SimpleNamespace(receipt=receipt, lp_feature=False))
    assert not receipt.exists()


def test_lp_missing_private_workspace_fails_before_any_build(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "ROOT", tmp_path / "qayd")
    monkeypatch.setattr(campaign, "source_identity", lambda: {"sha256": "source"})
    monkeypatch.setattr(campaign.subprocess, "run", lambda *_a, **_k: pytest.fail("must fail before Cargo"))
    receipt = tmp_path / "receipt.json"
    with pytest.raises(ValueError, match="authorized sibling"):
        campaign.build(SimpleNamespace(receipt=receipt, lp_feature=True))
    assert not receipt.exists()


def test_backend_change_during_build_cannot_publish_a_success_receipt(tmp_path, monkeypatch):
    root = tmp_path / "qayd"
    private_workspace(root)
    monkeypatch.setattr(campaign, "ROOT", root)
    monkeypatch.setattr(campaign, "source_identity", lambda: {"sha256": "source"})
    def changed(argv, **_kwargs):
        if argv[0] == "cargo":
            (tmp_path / "amthal/src/lib.rs").write_text("// changed during build\n")
    monkeypatch.setattr(campaign.subprocess, "run", changed)
    receipt = root / "receipt.json"
    with pytest.raises(ValueError, match="Amthal sources changed"):
        campaign.build(SimpleNamespace(receipt=receipt, lp_feature=True))
    assert not receipt.exists()


@pytest.mark.parametrize("via_environment", [False, True])
def test_scratch_build_publishes_binaries_that_survive_scratch_cleanup(tmp_path, monkeypatch, via_environment):
    root, scratch = tmp_path / "source", tmp_path / "local scratch"
    root.mkdir()
    monkeypatch.setattr(campaign, "ROOT", root)
    monkeypatch.setattr(campaign, "source_identity", lambda: {"sha256": "source"})
    private_workspace(root)
    extension = root / "_core.so"
    extension.write_bytes(b"installed Python extension")
    monkeypatch.setattr(campaign, "installed_extension", lambda: extension)
    monkeypatch.setattr(campaign, "checked", lambda _: "rustc test")
    names = ("qayd", "qayd-fzn", "qayd-sat", "qayd-pb")
    for name in names:
        path = scratch / "release" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        path.chmod(0o755)
    calls = []
    monkeypatch.setattr(campaign.subprocess, "run", lambda argv, **kw: calls.append(argv))
    receipt = root / "build.json"
    if via_environment:
        monkeypatch.setenv("CARGO_TARGET_DIR", str(scratch))
    campaign.build(SimpleNamespace(receipt=receipt, lp_feature=True,
                                   target_dir=None if via_environment else scratch))
    assert all(argv[argv.index("--target-dir")+1] == str(scratch) for argv in calls)
    shutil.rmtree(scratch)
    saved = campaign.read(receipt)
    for name in names:
        path = root / "target/release" / name
        assert path.read_bytes() == name.encode()
        assert path.stat().st_mode & 0o111
        assert saved["artifacts"][str(path)] == campaign.sha256_file(path)


@pytest.mark.parametrize("previously_imported", [False, True])
def test_build_receipt_uses_the_newly_installed_package_in_a_fresh_interpreter(tmp_path, previously_imported):
    environment = tmp_path / "environment"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    workspace = tmp_path / "source"
    workspace.mkdir()
    packages = []
    for name in ("old", "installed"):
        directory = tmp_path / name
        package = directory / "qayd"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        (package / "_core.py").write_text(f"BUILD = {name!r}\n")
        packages.append(directory)
    program = textwrap.dedent('''
        import site
        import subprocess
        import sys
        from pathlib import Path
        from types import SimpleNamespace

        repository, workspace, old, installed = map(Path, sys.argv[1:5])
        sys.path.insert(0, str(repository / "bench"))
        from cluster.lib import campaign
        campaign.ROOT = workspace
        campaign.source_identity = lambda: {"sha256": "source"}
        if sys.argv[5] == "True":
            sys.path.insert(0, str(old))
            import qayd._core
            assert qayd._core.BUILD == "old"

        original_run, original_checked = campaign.subprocess.run, campaign.checked
        def run(command, **kwargs):
            if command[0] == "cargo":
                target = Path(command[command.index("--target-dir") + 1])
                for name in ("qayd", "qayd-fzn", "qayd-sat", "qayd-pb"):
                    binary = target / "release" / name
                    binary.parent.mkdir(parents=True, exist_ok=True)
                    binary.write_bytes(name.encode())
                return subprocess.CompletedProcess(command, 0)
            if command[1:3] == ["-m", "maturin"]:
                # Emulate the editable installation performed by maturin develop.
                # The running parent's sys.path and imported modules stay unchanged.
                destination = Path(site.getsitepackages()[0]) / "qayd-installed.pth"
                destination.write_text(str(installed) + "\\n")
                return subprocess.CompletedProcess(command, 0)
            return original_run(command, **kwargs)
        campaign.subprocess.run = run
        campaign.checked = lambda command, **kwargs: (
            "rustc test" if command[0] == "rustc" else original_checked(command, **kwargs))
        campaign.build(SimpleNamespace(receipt=workspace / "build.json", lp_feature=False,
                                       target_dir=workspace / "temporary target"))
    ''')
    env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    result = subprocess.run([str(environment / "bin/python"), "-c", program, str(ROOT), str(workspace),
                             *map(str, packages), str(previously_imported)],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    saved = campaign.read(workspace / "build.json")
    extension = packages[1] / "qayd/_core.py"
    assert saved["artifacts"][str(extension)] == campaign.sha256_file(extension)
    assert str(packages[0] / "qayd/_core.py") not in saved["artifacts"]
    assert len(saved["artifacts"]) == 5


def test_default_campaign_downloads_and_all_stages_defer_sat():
    config = campaign.read(campaign.DEFAULT_CONFIG)
    campaign.validate_config(config)
    assert "sat" in config["deferred_tracks"]
    assert all(dataset["track"] != "sat" for dataset in config["datasets"])
    assert all(dataset["track"] != "sat" for dataset in campaign.smoke_config(config)["datasets"])
    assert all(command[0] != "bench/sat/fetch.py" for command in fetch.corpus_commands(config))
    assert all(command[0].startswith("bench/cluster/") for command in fetch.corpus_commands(config))


def test_hexaly_keeps_corpus_paths_inside_the_repository_alias(tmp_path, monkeypatch):
    root, shared = tmp_path/"source", tmp_path/"shared-instances"
    shared.mkdir()
    alias = root/"data/benchmarks/scheduling"
    alias.parent.mkdir(parents=True)
    alias.symlink_to(shared, target_is_directory=True)
    monkeypatch.setattr(hexaly, "ROOT", root)
    seen = []
    def convert(directory, destination):
        seen.append(directory)
        return destination/"catalog.json"
    monkeypatch.setattr(hexaly, "convert_rg300", convert)
    campaign.write(root/"bench/cluster/config/hexaly-jssp-1000x1000.json", {
        "reference_protocol": {"hexaly": {"reference_incumbent_10m": {}}}})
    out = tmp_path/"profile"
    monkeypatch.setattr(sys, "argv", ["hexaly.py", "--out", str(out)])
    hexaly.main()
    assert seen == [alias/"rcplib/v2.4"]
    config = campaign.read(out/"config.json")
    assert next(d["catalog"] for d in config["datasets"] if d["track"] == "rcpsp").startswith("data/")


def test_no_partial_corpus_or_overlapping_selection_is_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    (tmp_path / "sat").mkdir()
    (tmp_path / "sat/a.cnf").write_text("p cnf 1 1\n1 0\n")
    dataset = {"track": "sat", "family": "main", "glob": "sat/*.cnf", "expected_count": 400}
    _, _, missing = campaign.discover({"datasets": [dataset]})
    assert "1/400" in str(missing)
    with pytest.raises(ValueError, match="overlapping"):
        campaign.discover({"datasets": [dataset, dataset]})


def test_frontends_without_seed_are_not_counted_as_independent_runs():
    config = {"seeds": [0, 1, 2, 3, 4], "solvers": {"pb": ["qayd-pb"], "minizinc": ["qayd-fzn"], "sat": ["qayd-sat"]},
              "protocols": {track: {"seconds": 10} for track in ("pb", "minizinc", "sat")}}
    items = [{"id": str(i), "track": track, "path": "unused", "family": track}
             for i, track in enumerate(("pb", "minizinc", "sat"))]
    tasks = campaign.tasks_for(items, config, "full")
    assert [t["seed"] for t in tasks if t["item"]["track"] == "pb"] == [None]
    assert [t["seed"] for t in tasks if t["item"]["track"] == "minizinc"] == [None]
    assert len([t for t in tasks if t["item"]["track"] == "sat"]) == 5
    assert len({t["key"] for t in tasks}) == len(tasks)


def test_cvrp_budget_counts_customers_and_not_the_depot(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    (tmp_path / "instance.vrp").write_text("DIMENSION : 101\n")
    config = {"protocols": {"cvrp": {"seconds_per_customer": 2.4}}}
    assert campaign.budget({"track": "cvrp", "path": "instance.vrp"}, config, "full") == 240


def test_large_campaign_uses_bounded_balanced_oar_groups():
    tasks = [{"seconds": 1 + i % 100, "solvers": ["a", "b", "c"]} for i in range(10000)]
    groups = campaign.partition_tasks(tasks, 4)
    assert len(groups) == 4
    assert sorted(i for group in groups for i in group) == list(range(10000))
    loads = [sum(tasks[i]["seconds"] * 3 for i in group) for group in groups]
    assert max(loads) - min(loads) <= 300


def test_oar_submits_only_the_requested_pool_and_dry_run_is_default(tmp_path, monkeypatch):
    tasks = [{"seconds": 10, "solvers": ["a", "b"]} for _ in range(100)]
    plan = {"tasks": tasks, "sha256": "p", "stage": "gate", "memory_mb": 1024, "threads": 1}
    monkeypatch.setattr(campaign, "load_plan", lambda _: plan)
    monkeypatch.setattr(campaign, "verify_snapshot", lambda _: None)
    calls = []
    monkeypatch.setattr(campaign.subprocess, "run", lambda args, **kw: calls.append(args))
    args = SimpleNamespace(plan=tmp_path / "plan.json", jobs=3, workers_per_job=4,
                           resource="/nodes=1/core=4", walltime="04:00:00", queue=None,
                           property="gpu_count = 0", execute=False)
    campaign.submit(args)
    assert calls == []
    submission = campaign.read(tmp_path / "oar/submission.json")
    assert len(submission["commands"]) == 3
    assert submission["required_cores_per_job"] == 4
    assert submission["types"] == ["idempotent", "besteffort"]
    for command in submission["commands"]:
        assert command[command.index("-p")+1] == "gpu_count = 0"
        assert command[1:5] == ["-t", "idempotent", "-t", "besteffort"]
    assert all("--array" not in command for command in submission["commands"])
    args.execute = True
    campaign.submit(args)
    assert len(calls) == 3


def test_oar_refuses_submission_without_explicit_cpu_only_filter(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(campaign.subprocess, "run", lambda *args, **kw: calls.append(args))
    for predicate in (None, "", "   "):
        with pytest.raises(ValueError, match="CPU-only"):
            campaign.submit(SimpleNamespace(property=predicate))
    assert not calls


def test_plan_tampering_and_artifact_change_are_rejected(tmp_path, monkeypatch):
    plan = {"root": str(campaign.ROOT), "tasks": []}
    plan["sha256"] = campaign.digest(plan)
    path = tmp_path / "plan.json"
    campaign.write(path, plan)
    assert campaign.load_plan(path) == plan
    plan["tasks"] = ["tampered"]
    campaign.write(path, plan)
    with pytest.raises(ValueError, match="hash mismatch"):
        campaign.load_plan(path)
    artifact = tmp_path / "solver"
    artifact.write_text("one")
    frozen = {"source": {"sha256": "source"}, "artifacts": {str(artifact): campaign.sha256_file(artifact)}}
    monkeypatch.setattr(campaign, "source_identity", lambda: {"sha256": "source"})
    campaign.verify_snapshot(frozen)
    artifact.write_text("two")
    with pytest.raises(ValueError, match="artifact changed"):
        campaign.verify_snapshot(frozen)


def test_millisecond_protocol_and_missing_seed_options():
    runtime = {"qayd_msc": "/tmp/current.msc", "minizinc": "minizinc", "qayd-pb": "qayd-pb"}
    plan = {"root": str(ROOT), "runtime": runtime, "memory_mb": 8192}
    args = runners.command("qayd-fzn", {"track": "minizinc"}, {"seconds": 1200, "seed": None}, plan, Path("a.mzn"))
    assert args[args.index("--time-limit") + 1] == "1200000"
    assert "--seed" not in args and "-r" not in args and "-p" not in args
    pb = runners.command("qayd-pb", {"track": "pb"}, {"seconds": 3600, "seed": None}, plan, Path("a.opb"))
    assert pb[pb.index("-t") + 1] == "3600" and "--seed" not in pb


def fake_plan():
    task = {"key": "key", "solvers": ["qayd", "reference"], "seed": 0, "seconds": 10,
            "item": {"track": "xcsp-cop", "family": "f", "id": "i", "sense": "min"}}
    return {"tasks": [task], "sha256": "plan", "environment": "env", "stage": "full"}


def arm(solver, status, objective=None):
    feasible = objective is not None
    return {"solver": solver, "status": status, "objective": objective,
            "has_incumbent": feasible, "solved": status in ("UNSAT", "OPTIMUM"),
            "validation": {"valid": True if feasible else None}, "wall_seconds": 1,
            "par2": 1 if status == "OPTIMUM" else 20, "details": {},
            "plan_sha256": "plan", "task_key": "key"}


def save_result(tmp_path, arms):
    campaign.write(tmp_path / "tasks/0000000/result.json", {
        "complete": True, "plan_sha256": "plan", "task_key": "key", "arms": arms})


def test_report_refuses_missing_duplicate_and_contradictory_results(tmp_path):
    plan = fake_plan()
    assert report.audit(plan, tmp_path)["missing_tasks"] == [0]
    save_result(tmp_path, [arm("qayd", "OPTIMUM", 7), arm("qayd", "OPTIMUM", 7)])
    corrupted = report.audit(plan, tmp_path)
    assert corrupted["blocking_errors"] and not corrupted["complete"]
    save_result(tmp_path, [arm("qayd", "OPTIMUM", 7), arm("reference", "SAT", 6)])
    audited = report.audit(plan, tmp_path)
    assert not audited["ready"] and audited["contradictions"]
    assert audited["complete"] and not audited["blocking_errors"]
    rejected = next(row for row in audited["records"] if row["solver"] == "qayd")
    assert rejected["status"] == "ERROR" and rejected["raw_status"] == "OPTIMUM"
    assert not rejected["solved"] and not audited["pairwise"]
    save_result(tmp_path, [arm("qayd", "UNSAT"), arm("reference", "SAT", 6)])
    assert not report.audit(plan, tmp_path)["ready"]


def test_wrapped_rcp_normalization_preserves_tasks_resources_and_precedence():
    from qayd.datasets.scheduling import parse_psplib
    capacities, rows = parse_rcp("3 2\n3 4\n0 0 0 1\n2\n5 2 3 1 3\n0 0 0 0\n")
    parsed = parse_psplib(psplib_text(capacities, rows))
    assert parsed.capacities == (3, 4)
    assert parsed.jobs[0].successors == (2,)
    assert parsed.jobs[1].modes[0].duration == 5
    assert parsed.jobs[1].modes[0].demands == (2, 3)
    assert parsed.jobs[1].successors == (3,)
    with pytest.raises(ValueError, match="truncated"):
        parse_rcp("3 2 3 4 0 0")
    with pytest.raises(ValueError, match="trailing"):
        parse_rcp("1 1 2 0 0 0 999")


def test_maximization_stream_keeps_the_largest_incumbent(tmp_path):
    measured = runners.measure([sys.executable, "-c", "print('o 1'); print('o 7'); print('s OPTIMUM FOUND')"],
                               2, 0, 0, tmp_path / "max", sense="max")
    assert measured["best_incumbent"]["value"] == 7


def test_published_comparison_keeps_missing_seeds_and_rejects_late_quality():
    item = {"track": "jssp", "path": "case.data"}
    tasks = [{"item": item, "seconds": 600, "seed": seed, "solvers": ["qayd-native"]} for seed in (0, 1)]
    plan = {"tasks": tasks, "config": {"published_reference": {"jssp_incumbents": {"case": 100}}}}
    late = {"task": 0, "solver": "qayd-native", "has_incumbent": True, "objective": [90],
            "budget": 600, "wall_seconds": 601, "seed": 0}
    compared = compare_published(plan, {"records": [late], "ready": False})
    assert compared["jssp"][0]["expected_runs"] == 2
    assert compared["jssp"][0]["verified_runs"] == 0
    assert compared["jssp"][0]["late_verified_runs"] == 1
    assert compared["jssp"][0]["median_gap_to_published_percent"] is None
    assert compared["coverage_ready"] is False


@pytest.mark.skipif(shutil.which("minizinc") is None, reason="MiniZinc not installed")
def test_minizinc_replay_multidimensional_and_boolean_arrays(tmp_path):
    (tmp_path / "array.mzn").write_text(
        "array[1..2,1..2] of var 0..2: x; array[1..2] of var bool: b;\n"
        "constraint sum(array1d(x))=3; constraint b[1] != b[2]; solve minimize x[1,1];\n")
    item = {"path": "array.mzn", "data": [], "outputs": ["x", "b"], "sense": "min"}
    def measure(argv, timeout, _stem):
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return {"stdout": result.stdout, "returncode": result.returncode, "timed_out": False}
    candidate = {"has_solution": True, "objective": 0,
                 "values": {"x": [[0, 1], [0, 2]], "b": [True, False]}}
    assert minizinc.replay("minizinc", item, candidate, tmp_path, tmp_path, 10, measure)["valid"] is True
    candidate["values"]["x"] = [[0, 0], [0, 0]]
    assert minizinc.replay("minizinc", item, candidate, tmp_path, tmp_path, 10, measure)["valid"] is False


@pytest.mark.skipif(shutil.which("minizinc") is None, reason="MiniZinc not installed")
def test_minizinc_replay_rejects_bad_witness_and_recomputes_objective(tmp_path):
    model = tmp_path / "model.mzn"
    model.write_text("var 0..10: x; var 0..10: y; var int: cost=2*x+y; constraint x+y>=7; solve minimize cost;\n")
    item = {"path": "model.mzn", "data": [], "outputs": ["x", "y"], "sense": "min"}
    def measure(argv, timeout, stem):
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return {"stdout": completed.stdout, "returncode": completed.returncode, "timed_out": False}
    observed = {"has_solution": True, "values": {"x": 0, "y": 7}, "objective": 7}
    assert minizinc.replay("minizinc", item, observed, tmp_path, tmp_path, 10, measure)["valid"] is True
    wrong_objective = {**observed, "objective": 6}
    assert minizinc.replay("minizinc", item, wrong_objective, tmp_path, tmp_path, 10, measure)["valid"] is False
    wrong_values = {**observed, "values": {"x": 0, "y": 0}}
    assert minizinc.replay("minizinc", item, wrong_values, tmp_path, tmp_path, 10, measure)["valid"] is False
    incomplete = {**observed, "values": {"x": 0}}
    assert minizinc.replay("minizinc", item, incomplete, tmp_path, tmp_path, 10, measure)["valid"] is None
