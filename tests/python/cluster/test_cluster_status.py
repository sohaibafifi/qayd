"""Progress accounting and bounded, per-family duration projections."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))
from cluster import status


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path):
    plan = {"tasks": [{"key": "task-0", "seconds": 60, "solvers": ["a", "b"],
                       "item": {"family": "f", "id": "i"}}]}
    plan["sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path = tmp_path / "full/plan.json"
    dump(path, plan)
    def arm(solver):
        return {"solver": solver, "plan_sha256": plan["sha256"], "task_key": "task-0",
                "status": "SAT", "has_incumbent": True, "validation": {"valid": True}, "wall_seconds": 3}
    return path, plan, arm


def test_partial_arms_reduce_work_without_counting_a_complete_paired_task(tmp_path):
    path, plan, arm = fixture(tmp_path)
    directory = path.parent / "tasks/0000000"
    dump(directory / "a.json", arm("a"))
    monitor = status.Monitor(tmp_path)
    sample = monitor.collect(path)
    assert sample["done"] == 0 and len(sample["rows"]) == 1
    assert sample["remaining_hours"] == 60 / 3600
    dump(directory / "result.json", {"complete": True, "plan_sha256": plan["sha256"],
                                     "task_key": "task-0", "arms": [arm("a"), arm("b")]})
    sample = monitor.collect(path)
    assert sample["done"] == 1 and len(sample["rows"]) == 2
    assert sample["remaining_hours"] == 0 and not sample["warnings"]


def test_incompatible_or_duplicate_records_never_count_as_progress(tmp_path):
    path, plan, arm = fixture(tmp_path)
    directory = path.parent / "tasks/0000000"
    wrong = {**arm("a"), "plan_sha256": "another-plan"}
    dump(directory / "a.json", wrong)
    sample = status.Monitor(tmp_path).collect(path)
    assert not sample["rows"] and sample["warnings"] and sample["remaining_hours"] == 120 / 3600
    dump(directory / "result.json", {"complete": True, "plan_sha256": plan["sha256"],
                                     "task_key": "task-0", "arms": [arm("a"), arm("a")]})
    assert status.Monitor(tmp_path).collect(path)["done"] == 0


def test_projection_requires_samples_for_each_family_and_rejects_errors():
    tasks = [{"solvers": ["a"], "seconds": 600, "item": {"family": f}} for f in ("fast", "slow") for _ in range(40)]
    rows = [{"solver": "a", "family": f, "budget": 600, "wall": wall,
             "instance": str(i), "error": False} for f, wall in (("fast", 2), ("slow", 120)) for i in range(20)]
    sample = {"plan": {"tasks": tasks}, "rows": rows, "warnings": []}
    assert status.projection(sample, 2) == (20 * 2 + 20 * 120) / 2 / 3600
    assert status.projection({**sample, "rows": rows[:20]}, 2) is None
    assert status.projection({**sample, "rows": [*rows, {**rows[0], "error": True}]}, 2) is None


def test_worker_capacity_comes_from_running_jobs_not_allocated_cores(tmp_path):
    dump(tmp_path / "unused.json", {})
    log = tmp_path / "waves/00001/1-0.12.stdout"
    log.parent.mkdir(parents=True)
    log.write_text("OAR 12: 7 workers, 16384 MiB per solver\n")
    monitor = status.Monitor(tmp_path)
    state = {"active_jobs": [{"Job_Id": 12, "state": "Running"}, {"Job_Id": 13, "state": "Waiting"}]}
    assert monitor.slots(state) == 7
    state["active_jobs"][1]["state"] = "Running"
    assert monitor.slots(state) is None


@pytest.mark.parametrize("stage", ["smoke", "gate", "full"])
def test_details_shows_a_queued_solver_before_its_first_result_without_mutation(tmp_path, capsys, stage):
    profile = "fastcop-cosoco"
    plan = {"tasks": [{"key": "extra", "seconds": 30, "solvers": ["cosoco"],
                       "item": {"family": "f", "id": "i"}}],
            "supplement": {"parent_plans": {"full": str(tmp_path / "xcsp-fastcop/full/plan.json")}}}
    plan["sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    dump(tmp_path / profile / stage / "plan.json", plan)
    dump(tmp_path / "settings.json", {"jobs": 2, "profiles": {profile: {}}})
    dump(tmp_path / "state.json", {"status": "running", "profile": "rcpsp", "stage": "full",
                                   "active_jobs": [{"Job_Id": 1, "state": "Running"}]})
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    status.display(status.Monitor(tmp_path), details=True, reference_slots=32)
    output = capsys.readouterr().out
    solver_line = next(line for line in output.splitlines() if line.startswith("  cosoco "))
    assert "0/1" in solver_line and "waiting to run" in solver_line
    assert f"{profile:15} {stage:12}" in output
    assert "Supplement to xcsp-fastcop" in output
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_details_distinguishes_an_active_empty_solver_from_a_queued_one(tmp_path, capsys):
    path, _plan, _arm = fixture(tmp_path)
    dump(tmp_path / "settings.json", {"jobs": 2, "profiles": {"active": {}}})
    dump(tmp_path / "state.json", {"status": "running", "profile": "active", "stage": "full",
                                   "active_jobs": [{"Job_Id": 1, "state": "Running"}]})
    (tmp_path / "active").mkdir()
    path.parent.rename(tmp_path / "active/full")
    status.display(status.Monitor(tmp_path), details=True, reference_slots=32)
    output = capsys.readouterr().out
    assert "running, no saved result yet" in output
    assert "waiting to run" not in output
