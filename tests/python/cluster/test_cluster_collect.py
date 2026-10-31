"""Collection preserves provenance, partial work and failed outcomes without running jobs."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "bench"))
from cluster import collect


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def prepared(tmp_path, stage="full"):
    tasks = [{"key": f"task-{i}", "seed": i, "seconds": 180, "solvers": ["a", "b"],
              "item": {"id": str(i), "family": "f", "track": "xcsp-fastcop", "sense": "min"}}
             for i in range(2)]
    plan = {"root": "/another/checkout", "source": {"sha256": "frozen-sources"},
            "environment": "frozen-environment", "stage": stage, "tasks": tasks}
    plan["sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    directory = tmp_path / "campaign/fastcop" / stage
    write(directory / "plan.json", plan)
    def arm(index, solver, status="SAT"):
        valid = status == "SAT"
        return {"solver": solver, "task_key": tasks[index]["key"], "plan_sha256": plan["sha256"],
                "status": status, "objective": 10 if valid else None, "has_incumbent": valid,
                "validation": {"valid": True if valid else None}, "solved": status == "UNSAT",
                "proof_checked": False, "wall_seconds": 2, "cpu_seconds": 1.8, "peak_rss_kb": 20,
                "par2": 360, "host": {"hostname": "node", "oar_job_id": "42"}}
    arms = [arm(0, "a"), arm(0, "b", "ERROR")]
    write(directory / "tasks/0000000/result.json", {"complete": True, "task_key": tasks[0]["key"],
                                                   "plan_sha256": plan["sha256"], "arms": arms})
    write(directory / "tasks/0000000/a.json", arms[0])
    write(directory / "tasks/0000001/a.json", arm(1, "a"))
    return directory, plan, arm


def test_collect_copied_results_retains_errors_and_partial_arms_without_rewriting_inputs(tmp_path):
    directory, plan, _ = prepared(tmp_path)
    before = {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    destination = tmp_path / "exports"
    summary = collect.collect(tmp_path / "campaign", destination)
    assert summary["saved_runs"] == 3 and not summary["ready"]
    profile = summary["profiles"][0]
    assert profile["completed_tasks"] == 1 and profile["expected_tasks"] == 2
    assert profile["saved_runs"] == 3 and profile["partial_runs"] == 1 and len(profile["errors"]) == 1
    assert profile["missing_tasks"] == [1]
    rows = [json.loads(line) for line in (destination / "results.jsonl").read_text().splitlines()]
    assert sum(not row["task_complete"] for row in rows) == 1
    assert any(row["status"] == "ERROR" for row in rows)
    assert all(row["source_sha256"] == "frozen-sources" and row["plan_sha256"] == plan["sha256"] for row in rows)
    assert all(row["hostname"] == "node" and row["oar_job_id"] == "42" for row in rows)
    assert len(list(csv.DictReader((destination / "results.csv").read_text().splitlines()))) == 3
    for name, expected in summary["exports_sha256"].items():
        assert hashlib.sha256((destination / name).read_bytes()).hexdigest() == expected
    assert {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()} == before
    assert not (directory / "tasks/0000001/result.json").exists()


def test_wrong_partial_provenance_is_reported_and_never_counted(tmp_path):
    directory, _plan, arm = prepared(tmp_path)
    value = arm(1, "b")
    value["plan_sha256"] = "different"
    write(directory / "tasks/0000001/b.json", value)
    summary = collect.collect(tmp_path / "campaign", tmp_path / "exports")
    assert summary["saved_runs"] == 3
    assert any("incompatible partial arm" in error for error in summary["profiles"][0]["errors"])


def test_plan_hash_mismatch_stops_before_writing_any_export(tmp_path):
    directory, plan, _ = prepared(tmp_path)
    plan["tasks"][0]["seconds"] = 600
    write(directory / "plan.json", plan)
    with pytest.raises(ValueError, match="plan hash mismatch"):
        collect.collect(tmp_path / "campaign", tmp_path / "exports")
    assert not (tmp_path / "exports").exists()


def test_collection_keeps_correctness_contradictions_visible(tmp_path):
    directory, _plan, arm = prepared(tmp_path)
    path = directory / "tasks/0000000/result.json"
    value = json.loads(path.read_text())
    value["arms"][1] = arm(0, "b", "UNSAT")
    write(path, value)
    summary = collect.collect(tmp_path / "campaign", tmp_path / "exports")
    assert not summary["ready"]
    assert summary["profiles"][0]["contradictions"][0]["reason"] == "UNSAT-versus-verified-witness"


def test_stage_selection_keeps_qualification_and_full_runs_separate(tmp_path):
    prepared(tmp_path, "smoke")
    directory, _plan, _arm = prepared(tmp_path, "full")
    latest = collect.plans_at(tmp_path / "campaign")
    assert latest == [directory / "plan.json"]
    summary = collect.collect(tmp_path / "campaign", tmp_path / "exports", stage="all")
    assert {p["stage"] for p in summary["profiles"]} == {"smoke", "full"}
    assert len(summary["profiles"]) == 2


def test_export_cannot_overwrite_a_stage_and_missing_profiles_fail_closed(tmp_path):
    directory, _plan, _arm = prepared(tmp_path)
    with pytest.raises(ValueError, match="outside the input"):
        collect.collect(tmp_path / "campaign", directory)
    with pytest.raises(ValueError, match="no prepared plan"):
        collect.collect(tmp_path / "campaign", tmp_path / "exports", profiles=["absent"])
    with pytest.raises(ValueError, match="no latest plans"):
        collect.collect(tmp_path / "empty", tmp_path / "exports")
