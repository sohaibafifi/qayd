"""Progress accounting and bounded, per-family duration projections."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "bench"))
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


def test_rejected_claims_are_errors_even_when_raw_results_are_cached(tmp_path):
    path, plan, arm = fixture(tmp_path)
    directory = path.parent / "tasks/0000000"
    dump(directory / "result.json", {"complete": True, "plan_sha256": plan["sha256"],
                                     "task_key": "task-0", "arms": [arm("a"), arm("b")]})
    original = (directory / "result.json").read_bytes()
    monitor = status.Monitor(tmp_path)
    assert monitor.collect(path)["rows"][0]["status"] == "SAT"
    dump(path.parent / "report.json", {"plan_sha256": plan["sha256"],
                                      "rejected_claims": [{"task": 0, "solver": "a"}]})
    rows = monitor.collect(path)["rows"]
    assert rows[0]["status"] == "ERROR" and rows[0]["error"]
    assert rows[1]["status"] == "SAT"
    dump(path.parent / "report.json", {"plan_sha256": "another-plan",
                                      "rejected_claims": [{"task": 0, "solver": "a"}]})
    assert monitor.collect(path)["rows"][0]["status"] == "SAT"
    assert (directory / "result.json").read_bytes() == original


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


def test_details_keeps_statuses_in_the_same_order_for_all_solvers(tmp_path, capsys):
    statuses = ["UNSAT", "ERROR", "SAT", "OPTIMUM", "UNKNOWN", "INVALID", "UNSUPPORTED",
                "Z_EXTRA", "A_EXTRA"]
    plan = {"tasks": [{"key": f"task-{i}", "seconds": 60, "solvers": ["a", "b"],
                       "item": {"family": "f", "id": str(i)}} for i in range(len(statuses))]}
    plan["sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    dump(tmp_path / "pb/full/plan.json", plan)
    dump(tmp_path / "settings.json", {"jobs": 2, "profiles": {"pb": {}}})
    dump(tmp_path / "state.json", {"status": "running", "profile": "pb", "stage": "full"})
    for i, task in enumerate(plan["tasks"]):
        arms = [{"solver": solver, "plan_sha256": plan["sha256"], "task_key": task["key"],
                 "status": value, "wall_seconds": 3}
                for solver, value in (("a", statuses[i]), ("b", statuses[-i - 1]))]
        dump(tmp_path / f"pb/full/tasks/{i:07d}/result.json",
             {"complete": True, "plan_sha256": plan["sha256"], "task_key": task["key"], "arms": arms})
    expected = ("statuses={'OPTIMUM': 1, 'SAT': 1, 'UNSAT': 1, 'UNKNOWN': 1, "
                "'UNSUPPORTED': 1, 'INVALID': 1, 'ERROR': 1, 'A_EXTRA': 1, 'Z_EXTRA': 1}")
    monitor = status.Monitor(tmp_path)
    for _ in range(2):
        status.display(monitor, details=True, reference_slots=32)
        lines = [line for line in capsys.readouterr().out.splitlines() if "statuses=" in line]
        assert len(lines) == 2
        assert all(line.endswith(expected) for line in lines)


@pytest.mark.parametrize("stage", ["smoke", "gate"])
def test_missing_full_plans_do_not_report_a_zero_duration(tmp_path, capsys, stage):
    path, _plan, _arm = fixture(tmp_path / "main")
    path.parent.rename(path.parent.with_name(stage))
    dump(tmp_path / "settings.json", {"jobs": 4, "profiles": {"main": {}, "hexaly": {}}})
    dump(tmp_path / "state.json", {"status": "running", "profile": "main", "stage": stage,
                                   "profile_errors": {"hexaly": "RG300 manifest missing"}})
    status.display(status.Monitor(tmp_path), reference_slots=64)
    output = capsys.readouterr().out
    assert "Full-campaign budget unavailable" in output
    assert "Full plans not yet prepared: hexaly, main." in output
    assert f"{'hexaly':15} BLOCKED" in output
    assert "At 64 continuously available workers" not in output
    assert "0.00 solver hours" not in output


def test_budget_labels_a_prepared_subtotal_when_another_full_plan_is_missing(tmp_path, capsys):
    fixture(tmp_path / "main")
    dump(tmp_path / "settings.json", {"jobs": 4, "profiles": {"main": {}, "hexaly": {}}})
    dump(tmp_path / "state.json", {"status": "running", "profile": "main", "stage": "full"})
    status.display(status.Monitor(tmp_path), reference_slots=32)
    output = capsys.readouterr().out
    assert "prepared, unblocked full profiles: 0.03 solver hours" in output
    assert "Full plans not yet prepared: hexaly. Their budgets are not included." in output


def test_shared_pool_marks_each_active_section_and_selection_limits_the_budget(tmp_path, capsys):
    for name in ("pb", "minizinc"):
        fixture(tmp_path / name)
    dump(tmp_path / "settings.json", {"jobs": 4, "profiles": {"pb": {}, "minizinc": {}},
                                      "selected_profiles": ["pb"]})
    dump(tmp_path / "state.json", {"status": "running", "profile": "multiple", "stage": "full",
                                   "active_profiles": ["pb", "minizinc"], "active_jobs": [{"state": "Running"}]})
    status.display(status.Monitor(tmp_path), details=True, reference_slots=32)
    output = capsys.readouterr().out
    assert "Active profiles: pb, minizinc" in output
    assert "Selected profiles: pb." in output
    assert "full/paused" in output
    assert "prepared, unblocked full profiles: 0.03 solver hours" in output
    assert output.count("running, no saved result yet") == 4
