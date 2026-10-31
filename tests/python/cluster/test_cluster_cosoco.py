"""A new solver arm must not mutate or replay an immutable campaign."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "bench"))
from cluster.lib import campaign, cosoco, report
from cluster.lib import controller as launch


def fixture(tmp_path):
    tasks = []
    for instance in range(2):
        for seed in range(5):
            task = {"item": {"id": str(instance), "path": f"i{instance}.xml", "data": [],
                             "track": "xcsp-fastcop", "family": "test", "sense": "min"},
                    "seconds": 180, "seed": seed, "solvers": ["qayd", "choco"]}
            task["key"] = campaign.digest(task)
            tasks.append(task)
    parent = {"root": str(campaign.ROOT), "stage": "full", "tasks": tasks,
              "runtime": {}, "source": {"sha256": "source"}, "environment": "original",
              "artifacts": {}, "assets": {"i0.xml": "hash0", "i1.xml": "hash1"},
              "memory_mb": 16384, "threads": 1, "checker_seconds": 120,
              "config": {"solvers": {"xcsp-fastcop": ["qayd", "choco"]}, "seeds": list(range(5))},
              "overview": {"paired_tasks": 10, "solver_runs": 20}}
    parent["sha256"] = campaign.digest(parent)
    extension = {"environment": "new", "binary": "/cosoco", "driver": "/extra/driver.py", "artifacts": {},
                 "parent_plans": {"full": str(tmp_path / "original/full/plan.json")}}
    return parent, extension


def test_addition_keeps_parent_and_runs_one_unseeded_arm_per_instance(tmp_path):
    parent, extension = fixture(tmp_path)
    before = copy.deepcopy(parent)
    added = cosoco.derive_plan(parent, parent, extension, "full", campaign)
    assert parent == before
    assert len(added["tasks"]) == 2
    assert added["overview"]["solver_runs"] == 2
    assert added["overview"]["maximum_solver_hours"] == .1
    assert added["source"] == parent["source"]
    assert added["memory_mb"] == parent["memory_mb"] and added["threads"] == 1
    for task in added["tasks"]:
        assert task["solvers"] == ["cosoco"] and task["seed"] is None
        assert task["parent_seed"] == 0 and len(task["parent_task_keys"]) == 5
        original = next(t for t in parent["tasks"] if t["key"] == task["parent_task_key"])
        assert original["item"] == task["item"] and original["seconds"] == task["seconds"]
    command = cosoco.cosoco_command("/cosoco", added["tasks"][0], "/i.xml")
    assert command == ["/cosoco", "/i.xml", "-cpu_lim=180", "-nbcores=1", "-model=2", "-colors=0"]
    with pytest.raises(ValueError, match="cannot accept"):
        cosoco.cosoco_command("/cosoco", parent["tasks"][0], "/i.xml")


def test_inconsistent_parent_budget_is_rejected(tmp_path):
    parent, extension = fixture(tmp_path)
    parent["tasks"][1]["seconds"] = 30
    with pytest.raises(ValueError, match="different budgets"):
        cosoco.derive_plan(parent, parent, extension, "full", campaign)


def test_full_requires_its_own_matching_and_reaudited_gate(tmp_path, monkeypatch):
    gate = {"sha256": "gate", "environment": "new"}
    qualification = {"ready": True, "stage": "gate", "plan_sha256": "gate", "environment": "new"}
    fake = SimpleNamespace(load_plan=lambda _: gate, read=lambda _: qualification)
    audited = SimpleNamespace(audit=lambda *_: {"ready": True})
    cosoco.require_gate(tmp_path, {"environment": "new"}, fake, audited)
    qualification["environment"] = "old"
    with pytest.raises(ValueError, match="own complete, clean gate"):
        cosoco.require_gate(tmp_path, {"environment": "new"}, fake, audited)
    qualification["environment"] = "new"
    audited.audit = lambda *_: {"ready": False}
    with pytest.raises(ValueError, match="own complete, clean gate"):
        cosoco.require_gate(tmp_path, {"environment": "new"}, fake, audited)


def test_hooks_keep_original_profiles_and_oar_pool_constraints(tmp_path):
    parent, extension = fixture(tmp_path)
    settings = {"out": str(tmp_path), "root": str(tmp_path), "python": "/python", "driver": "/old.py",
                "prefix": "qayd-same-", "resource": "/nodes=1", "walltime": "12:00:00",
                "property": "network_address NOT IN (SELECT network_address FROM resources WHERE gpumodel IS NOT NULL)",
                "jobs": 2}
    driver = SimpleNamespace(next_stage=lambda *_: "original", command_for=launch.command_for)
    old = driver.command_for(settings, tmp_path / "xcsp-fastcop/full/plan.json", tmp_path / "tasks", "7-0")
    cosoco.install_hooks(driver, extension)
    assert driver.next_stage(settings, "xcsp-fastcop") == "original"
    assert driver.command_for(settings, tmp_path / "xcsp-fastcop/full/plan.json", tmp_path / "tasks", "7-0") == old
    new = driver.command_for(settings, tmp_path / "fastcop-cosoco/full/plan.json", tmp_path / "tasks", "7-0")
    assert new[:-1] == old[:-1]
    assert "/extra/driver.py --worker" in new[-1]
    assert [new[i + 1] for i, value in enumerate(new[:-1]) if value == "-t"] == ["idempotent", "besteffort"]
    assert settings["jobs"] == 2


def results(plan, directory, *, extra=False):
    for index, task in enumerate(plan["tasks"]):
        arms = []
        for solver in task["solvers"]:
            error = solver == "choco" and task["seed"] == 4
            arms.append({"solver": solver, "plan_sha256": plan["sha256"], "task_key": task["key"],
                         "status": "ERROR" if error else "SAT", "has_incumbent": not error,
                         "validation": {"valid": None if error else True}, "wall_seconds": 2,
                         "solved": False, "objective": None if error else 9 if extra else 10,
                         "par2": 360, "host": {"hostname": "node1"}})
        campaign.write(directory / "tasks" / f"{index:07d}" / "result.json",
                       {"complete": True, "plan_sha256": plan["sha256"], "task_key": task["key"], "arms": arms})


def test_comparison_retains_original_errors_and_does_not_multiply_extra_runs(tmp_path):
    parent, extension = fixture(tmp_path)
    added = cosoco.derive_plan(parent, parent, extension, "full", campaign)
    origin = Path(extension["parent_plans"]["full"]).parent
    target = tmp_path / "extra/full"
    results(parent, origin)
    results(added, target, extra=True)
    audit = report.audit(added, target)
    comparison = cosoco.combined_report(parent, added, audit, target, campaign, report)
    assert comparison["parent_solver_runs"] == 20 and comparison["supplement_solver_runs"] == 2
    assert len(comparison["parent_errors"]) == 2
    assert len(comparison["comparisons"]) == 4
    assert not comparison["timing_comparable"] and not comparison["cross_block_contradictions"]
    all_arms = [a for group in comparison["pairs"].values() for a in group["arms"]]
    assert sum(a["solver"] == "cosoco" for a in all_arms) == 2
    assert len(all_arms) == 22
    assert all(a["source_plan_sha256"] == parent["sha256"] for a in all_arms if a["solver"] != "cosoco")


def test_contradiction_against_a_nonreference_seed_is_reported(tmp_path):
    parent, extension = fixture(tmp_path)
    added = cosoco.derive_plan(parent, parent, extension, "full", campaign)
    origin = Path(extension["parent_plans"]["full"]).parent
    target = tmp_path / "extra/full"
    results(parent, origin)
    results(added, target, extra=True)
    path = origin / "tasks/0000003/result.json"
    value = campaign.read(path)
    value["arms"][0].update(status="OPTIMUM", objective=10)
    campaign.write(path, value)
    comparison = cosoco.combined_report(parent, added, report.audit(added, target), target, campaign, report)
    assert any(c["other_seed"] == 3 and c["reason"] == "better-witness-than-claimed-optimum"
               for c in comparison["cross_block_contradictions"])
