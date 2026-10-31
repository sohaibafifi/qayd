"""Fresh cluster policies, unseeded comparators and frozen command semantics."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "bench"))
from cluster.lib import campaign, report, runners


def default_config():
    return campaign.read(campaign.DEFAULT_CONFIG)


@pytest.mark.parametrize("solver", ["qayd-api", "qayd-native"])
@pytest.mark.parametrize("family", ["ft06", "taillard-1000x1000"])
def test_fresh_jssp_explicitly_selects_single_thread_adaptive_ls(solver, family):
    plan = {"root": str(ROOT), "runtime": {"python": sys.executable}, "threads": 1,
            "memory_mb": 16384, "config": default_config()}
    command = runners.command(solver, {"track": "jssp"}, {"seconds": 600, "seed": 3},
                              plan, Path(family))
    assert command[command.index("--engine") + 1] == "ls"
    assert command[command.index("--schedule-jssp-search") + 1] == "tsab-adaptive"
    assert command[command.index("--threads") + 1] == "1"
    assert command[command.index("--seed") + 1] == "3"
    assert "--schedule-cdcl" not in command


def test_prior_frozen_plans_keep_legacy_auto_and_other_tracks_unchanged():
    plan = {"threads": 1, "memory_mb": 1024, "config": default_config()}
    previous = {**plan, "config": {}}
    assert runners.domain_arguments({}, previous, "jssp").qayd_engine == "auto"
    assert runners.domain_arguments({}, previous, "jssp").qayd_schedule_jssp_search == "legacy"
    for track in ("rcpsp", "cvrp", "cvrptw"):
        args = runners.domain_arguments({}, plan, track)
        assert args.qayd_engine == "auto"
        assert args.qayd_schedule_jssp_search == "legacy"


def test_adaptive_policy_is_frozen_and_cannot_silently_use_other_threads():
    config = default_config()
    campaign.validate_config(config)
    changed = copy.deepcopy(config)
    changed["model_options"]["jssp"]["schedule_jssp_search"] = "legacy"
    assert campaign.digest(changed) != campaign.digest(config)
    for field, value in (("engine", "auto"), ("threads", 2), ("threads", True)):
        invalid = copy.deepcopy(config)
        invalid["model_options"]["jssp"][field] = value
        with pytest.raises(ValueError):
            campaign.validate_config(invalid)
    with pytest.raises(ValueError, match="threads differ"):
        runners.domain_arguments({}, {"threads": 2, "memory_mb": 1, "config": config}, "jssp")


@pytest.mark.parametrize("stage,repetitions", [("smoke", 1), ("gate", 1), ("full", 5)])
def test_cosoco_is_one_unseeded_normal_arm_per_instance(stage, repetitions):
    config = default_config()
    items = [{"id": str(i), "track": "xcsp-fastcop", "path": f"instance-{i}.xml",
              "family": f"family-{i}", "sense": "min"} for i in range(2)]
    tasks = campaign.tasks_for(items, config, stage)
    assert len(tasks) == 2 * repetitions
    assert sum("cosoco" in task["solvers"] for task in tasks) == 2
    for item in items:
        grouped = [task for task in tasks if task["item"] == item]
        cosoco = next(task for task in grouped if "cosoco" in task["solvers"])
        assert cosoco["seed"] == 0
        assert cosoco["solver_seeds"]["cosoco"] is None
        assert sum(len(task["solvers"]) for task in grouped) == 3 * repetitions + 1
    assert len({task["key"] for task in tasks}) == len(tasks)


def test_cosoco_command_never_invents_a_seed():
    plan = {"root": str(ROOT), "runtime": {"cosoco": "/tools/cosoco"}, "memory_mb": 16384}
    task = {"seconds": 180, "seed": 0, "solver_seeds": {"cosoco": None}}
    argv = runners.command("cosoco", {"track": "xcsp-fastcop"}, task, plan, Path("case.xml"))
    assert argv == ["/tools/cosoco", "case.xml", "-cpu_lim=180", "-nbcores=1", "-model=2", "-colors=0"]
    assert not any("seed" in arg for arg in argv)
    with pytest.raises(ValueError, match="unseeded"):
        runners.command("cosoco", {"track": "xcsp-fastcop"}, {"seconds": 180, "seed": 0}, plan, Path("case.xml"))


@pytest.mark.parametrize("track", ["xcsp-csp", "xcsp-cop", "xcsp-fastcop"])
def test_cosoco_is_a_regular_baseline_in_every_xcsp_category(track):
    config = default_config()
    assert "cosoco" in config["solvers"][track]
    config["datasets"] = [{"track": track, "family": "test", "glob": "case.xml"}]
    campaign.validate_config(config)
    item = {"id": "case", "track": track, "path": "case.xml", "family": "test", "sense": "min"}
    tasks = campaign.tasks_for([item], config, "full")
    references = [task for task in tasks if "cosoco" in task["solvers"]]
    assert len(references) == 1
    plan = {"root": str(ROOT), "runtime": {"cosoco": "/tools/cosoco"}, "memory_mb": 16384}
    argv = runners.command("cosoco", item, references[0], plan, Path("case.xml"))
    assert argv[:2] == ["/tools/cosoco", "case.xml"]
    assert f"-cpu_lim={config['protocols'][track]['seconds']}" in argv
    assert "-nbcores=1" in argv and not any("seed" in arg for arg in argv)


def test_official_cosoco_freezes_runtime_companions_and_download_receipt(tmp_path):
    installation = tmp_path / "cosoco-2.6.1"
    binary = installation / "runtime/cosoco_bin/cosoco"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"solver")
    license_file = binary.with_name("LICENCE")
    license_file.write_text("license")
    receipt = installation / "download.json"
    receipt.write_text("{}")
    assert set(campaign.cosoco_artifacts(binary)) == {str(path.resolve()) for path in (binary, license_file, receipt)}


def test_cosoco_runtime_requires_java_checker_and_hashes_the_installed_binary(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    checker = tmp_path / "bench/cluster/runs/tools/solvers/ACE-2.5.jar"
    checker.parent.mkdir(parents=True)
    checker.write_bytes(b"checker")
    installation = tmp_path / f"bench/cluster/runs/tools/cosoco-{campaign.install_cosoco.VERSION}"
    binary = installation / "runtime/cosoco_bin/cosoco"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    receipt = installation / "download.json"
    receipt.write_text("{}")
    checked = []
    def verify(destination):
        checked.append(destination)
        return binary
    monkeypatch.setattr(campaign.install_cosoco, "verify", verify)
    runtime = campaign.runtime_identity(
        {"solvers": {"xcsp-fastcop": ["cosoco"]}}, [{"track": "xcsp-fastcop"}], tmp_path,
        SimpleNamespace(java=sys.executable))
    assert checked == [installation]
    assert runtime["runtime"]["cosoco"] == str(binary)
    assert runtime["runtime"]["ace"] == str(checker)
    assert {str(binary), str(checker), str(receipt)} <= runtime["artifacts"].keys()


def test_default_cosoco_requires_a_verified_installation(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    checker = tmp_path / "bench/cluster/runs/tools/solvers/ACE-2.5.jar"
    checker.parent.mkdir(parents=True)
    checker.write_bytes(b"checker")
    binary = tmp_path / "bench/cluster/runs/tools/cosoco-2.6.1/runtime/cosoco_bin/cosoco"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    with pytest.raises(ValueError, match="not prepared"):
        campaign.runtime_identity({"solvers": {"xcsp-fastcop": ["cosoco"]}},
                                  [{"track": "xcsp-fastcop"}], tmp_path,
                                  SimpleNamespace(java=sys.executable))


def measured(stdout=""):
    return {"stdout": stdout, "stderr": "", "claimed_status": "SAT", "best_incumbent": {"value": 8},
            "has_solution": True, "returncode": 0, "timed_out": False,
            "elapsed_wall_seconds": 1, "cpu_seconds": 1, "peak_rss_kb": 1, "first_incumbent": None}


def test_cosoco_uses_the_normal_ace_witness_validation(tmp_path, monkeypatch):
    source = tmp_path / "case.xml"
    source.write_text("<instance/>")
    monkeypatch.setattr(runners, "measure", lambda *_args, **_kw: measured())
    checked = []
    def validate(argv, output, log, seconds, **kwargs):
        checked.append((argv, output, seconds, kwargs))
        return {"valid": True}
    monkeypatch.setattr(runners, "validate_solution", validate)
    plan = {"root": str(tmp_path), "runtime": {"cosoco": "/tools/cosoco", "java": "java", "ace": "ACE.jar"},
            "memory_mb": 1024, "grace_seconds": 2, "checker_memory_mb": 256, "checker_seconds": 120}
    row = runners.run_arm("cosoco", {"track": "xcsp-fastcop", "path": "case.xml", "sense": "min"},
                          {"seconds": 180, "seed": 0, "solver_seeds": {"cosoco": None}}, plan, tmp_path)
    assert row["status"] == "SAT" and row["has_incumbent"]
    assert row["seed"] is None
    assert len(checked) == 1
    assert "org.xcsp.parser.callbacks.SolutionChecker" in checked[0][0]
    assert checked[0][3]["expected_objective"] == 8


def test_adaptive_result_selectors_must_match_the_frozen_plan(tmp_path, monkeypatch):
    source = tmp_path / "case"
    source.write_text("1 1\n0 1\n")
    def measure(*_args, **_kw):
        return measured('{"status":"SATISFIABLE","objectives":[1],"verified":true,"engine":"auto",'
                        '"schedule_jssp_search":"legacy"}')
    monkeypatch.setattr(runners, "measure", measure)
    plan = {"root": str(tmp_path), "runtime": {"python": sys.executable}, "threads": 1,
            "memory_mb": 1024, "grace_seconds": 2, "config": default_config()}
    row = runners.run_arm("qayd-native", {"track": "jssp", "path": "case", "sense": "min"},
                          {"seconds": 2, "seed": 0}, plan, tmp_path)
    assert row["status"] == "ERROR"
    assert not row["has_incumbent"] and row["objective"] is None
    assert "selectors" in row["details"]["parse_error"]


def test_graceful_preparation_timeout_does_not_invent_a_selector_mismatch(tmp_path, monkeypatch):
    source = tmp_path / "case"
    source.write_text("1 1\n0 1\n")
    monkeypatch.setattr(runners, "measure", lambda *_a, **_k: measured(
        '{"status":"UNKNOWN","objectives":[],"verified":false,'
        '"diagnostic":"preparation exhausted the end-to-end deadline"}'))
    plan = {"root": str(tmp_path), "runtime": {"python": sys.executable}, "threads": 1,
            "memory_mb": 1024, "grace_seconds": 2, "config": default_config()}
    row = runners.run_arm("qayd-native", {"track": "jssp", "path": "case", "sense": "min"},
                          {"seconds": 2, "seed": 0}, plan, tmp_path)
    assert row["status"] == "UNKNOWN" and not row["has_incumbent"]
    assert row["details"]["diagnostic"] == "preparation exhausted the end-to-end deadline"
    assert "parse_error" not in row["details"]


def seeded_fastcop_plan():
    item = {"id": "instance", "track": "xcsp-fastcop", "family": "family", "sense": "min"}
    return {"sha256": "plan", "environment": "env", "stage": "full", "tasks": [
        {"key": "first", "item": item, "seconds": 180, "seed": 0,
         "solver_seeds": {"qayd": 0, "cosoco": None}, "solvers": ["qayd", "cosoco"]},
        {"key": "second", "item": item, "seconds": 180, "seed": 1,
         "solver_seeds": {"qayd": 1}, "solvers": ["qayd"]}]}


def save_task(directory, index, task, arms):
    rows = [{"solver": solver, "status": status, "objective": value,
             "seed": task["solver_seeds"][solver],
             "has_incumbent": True, "solved": status == "OPTIMUM", "validation": {"valid": True},
             "wall_seconds": 1, "par2": 1, "details": {}, "plan_sha256": "plan", "task_key": task["key"]}
            for solver, status, value in arms]
    campaign.write(directory / f"tasks/{index:07d}/result.json", {
        "complete": True, "plan_sha256": "plan", "task_key": task["key"], "arms": rows})


def test_report_compares_cosoco_once_and_retains_its_null_seed(tmp_path):
    plan = seeded_fastcop_plan()
    save_task(tmp_path, 0, plan["tasks"][0], [("qayd", "SAT", 9), ("cosoco", "SAT", 8)])
    save_task(tmp_path, 1, plan["tasks"][1], [("qayd", "SAT", 7)])
    audit = report.audit(plan, tmp_path)
    assert audit["ready"]
    assert len(audit["pairwise"]) == 1
    assert audit["pairwise"][0]["baseline_seed"] is None
    assert audit["pairwise"][0]["seed_paired"] is False
    assert audit["pairwise"][0]["outcome"] == "loss"
    cosoco = next(row for row in audit["records"] if row["solver"] == "cosoco")
    assert cosoco["seed"] is None and cosoco["reference_seed"] == 0
    assert not audit["timing_comparable"] and audit["pairing_warnings"]
    result_path = tmp_path / "tasks/0000000/result.json"
    result = campaign.read(result_path)
    result["arms"][1]["seed"] = 0
    campaign.write(result_path, result)
    assert report.audit(plan, tmp_path)["errors"]


def test_unseeded_optimum_is_checked_against_every_seeded_repetition(tmp_path):
    plan = seeded_fastcop_plan()
    save_task(tmp_path, 0, plan["tasks"][0], [("qayd", "SAT", 9), ("cosoco", "OPTIMUM", 8)])
    save_task(tmp_path, 1, plan["tasks"][1], [("qayd", "SAT", 7)])
    audit = report.audit(plan, tmp_path)
    assert not audit["ready"]
    assert any("across-repetitions" in row["reason"] for row in audit["contradictions"])
