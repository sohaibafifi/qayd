"""Public scheduling local-search profile coverage."""

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

cp = pytest.importorskip("qayd")


ROOT = Path(__file__).resolve().parents[2]
SCHEDULE_COUNTERS = (
    "schedule_moves_considered",
    "schedule_moves_accepted",
    "schedule_moves_rejected",
    "schedule_work_steps",
    "schedule_local_improvements",
    "schedule_global_improvements",
    "schedule_progress_publications",
    "schedule_incumbent_publication_attempts",
    "schedule_incumbent_publications",
    "schedule_incumbent_injection_attempts",
    "schedule_incumbent_injections",
    "schedule_incumbent_rejections",
    "schedule_incumbent_verifications",
    "schedule_incumbent_verification_rejections",
    "schedule_incumbent_verification_interruptions",
    "schedule_incumbent_incomplete_rejections",
    "schedule_restart_boundaries",
    "schedule_peak_buffered_candidates",
    "schedule_stalled_workers",
    "schedule_stalled_unused_work_steps",
    "schedule_worker_work_min",
    "schedule_worker_work_max",
    "schedule_work_budget_overruns",
    "schedule_elite_pool_size",
    "schedule_unused_work_steps",
    "schedule_incumbent_source_worker",
    "schedule_incumbent_source_round",
    "schedule_cycle_rejections",
    "schedule_reconstructions",
    "critical_path_updates",
    "schedule_window_rejections",
    "schedule_objective_rejections",
    "resource_profile_checks",
    "resource_candidate_scheduling_attempts",
    "resource_event_visits",
    "resource_peak_profile_events",
    "schedule_precedence_rejections",
    "schedule_infeasible_rejections",
    "schedule_justification_attempts",
    "schedule_delta_evaluations",
    "schedule_full_evaluations",
    "schedule_full_fallbacks",
    "schedule_topological_rebuilds",
    "schedule_oracle_validations",
    "schedule_oracle_mismatches",
    "schedule_dirty_cone_operations",
    "schedule_max_dirty_cone",
    "schedule_workspace_growths",
    "schedule_workspace_rollbacks",
    "schedule_alns_generation_attempts",
    "schedule_alns_moves_generated",
)


def _solve_job_shop(*, profile):
    model = cp.Model()
    tasks = model.tasks(range(6))
    durations = (3, 2, 2, 2, 1, 4)
    machines = (0, 1, 2, 1, 2, 0)
    for task in tasks:
        task.duration = durations[task.id]
        task.machine = machines[task.id]

    schedule = model.schedule(tasks, horizon=sum(durations))
    model.add(schedule[0].end <= schedule[1].start)
    model.add(schedule[1].end <= schedule[2].start)
    model.add(schedule[3].end <= schedule[4].start)
    model.add(schedule[4].end <= schedule[5].start)
    model.add(schedule.no_overlap(lambda task: task.machine))
    model.minimize(schedule.makespan())
    return model.solve(
        engine="ls",
        seed=11,
        threads=2,
        time_limit=1,
        max_iterations=16,
        profile=profile,
    )


def test_schedule_profile_is_present_only_when_requested():
    plain = _solve_job_shop(profile=False)
    profiled = _solve_job_shop(profile=True)

    assert plain.status == profiled.status == "SATISFIABLE"
    assert plain.candidates_evaluated is None
    assert plain.candidates_per_second is None
    assert plain.full_recompute_percentage is None
    assert plain.schedule_restart_work is None
    assert all(getattr(plain, field) is None for field in SCHEDULE_COUNTERS)

    assert profiled.candidates_evaluated == profiled.schedule_moves_considered
    assert profiled.schedule_moves_considered >= 0
    assert profiled.schedule_moves_accepted >= 0
    assert profiled.schedule_moves_rejected >= 0
    assert profiled.schedule_moves_accepted + profiled.schedule_moves_rejected == profiled.schedule_moves_considered
    assert profiled.schedule_cycle_rejections >= 0
    assert profiled.schedule_reconstructions >= 0
    assert profiled.critical_path_updates >= 0
    assert all(getattr(profiled, field) is not None and getattr(profiled, field) >= 0 for field in SCHEDULE_COUNTERS)
    assert profiled.schedule_global_improvements == profiled.schedule_progress_publications
    assert profiled.schedule_local_improvements >= profiled.schedule_global_improvements
    assert profiled.schedule_incumbent_publication_attempts == (
        profiled.schedule_incumbent_publications + profiled.schedule_incumbent_rejections
    )
    assert profiled.schedule_incumbent_injections <= profiled.schedule_incumbent_injection_attempts
    assert profiled.schedule_incumbent_verifications <= profiled.schedule_incumbent_publication_attempts
    assert profiled.schedule_incumbent_verification_rejections <= profiled.schedule_incumbent_rejections
    assert profiled.schedule_incumbent_verification_interruptions <= profiled.schedule_incumbent_rejections
    assert profiled.schedule_incumbent_incomplete_rejections <= profiled.schedule_incumbent_rejections
    assert profiled.schedule_restart_boundaries > 0
    assert profiled.schedule_restart_work == 2_048
    assert 1 <= profiled.schedule_peak_buffered_candidates <= 2
    assert profiled.schedule_worker_work_min <= profiled.schedule_worker_work_max
    assert profiled.schedule_work_budget_overruns == 0
    assert profiled.schedule_elite_pool_size == 1
    assert profiled.schedule_unused_work_steps + profiled.schedule_work_steps == 16
    assert profiled.schedule_incumbent_source_worker < 2
    assert math.isfinite(profiled.candidates_per_second)
    assert profiled.candidates_per_second >= 0.0
    assert math.isfinite(profiled.full_recompute_percentage)
    assert 0.0 <= profiled.full_recompute_percentage <= 100.0


@pytest.mark.parametrize(
    ("launcher", "shape_args"),
    (
        ("examples/python/scheduling/api/jssp.py", ("--jobs", "2", "--machines", "2")),
        ("examples/python/scheduling/native/jssp.py", ("--jobs", "2", "--machines", "2")),
        ("examples/python/scheduling/api/rcpsp.py", ("--tasks", "4", "--resources", "1")),
        ("examples/python/scheduling/native/rcpsp.py", ("--tasks", "4", "--resources", "1")),
    ),
)
def test_scheduling_launchers_emit_search_profile(launcher, shape_args):
    command = (
        sys.executable,
        str(ROOT / launcher),
        *shape_args,
        "--engine",
        "ls",
        "--seed",
        "7",
        "--time-limit",
        "1",
        "--threads",
        "2",
        "--max-iterations",
        "8",
        "--memory-limit-mb",
        "0",
        "--profile",
        "--json",
    )
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True, timeout=20)
    record = json.loads(completed.stdout)

    assert record["verified"] is True
    assert record["candidates_evaluated"] == record["schedule_moves_considered"]
    assert record["schedule_moves_accepted"] + record["schedule_moves_rejected"] == record["schedule_moves_considered"]
    assert record["schedule_global_improvements"] == record["schedule_progress_publications"]
    assert record["schedule_incumbent_publication_attempts"] == (
        record["schedule_incumbent_publications"] + record["schedule_incumbent_rejections"]
    )
    assert record["schedule_incumbent_injections"] <= record["schedule_incumbent_injection_attempts"]
    if launcher.endswith("/jssp.py"):
        assert record["schedule_restart_work"] == 2_048
    assert record["schedule_worker_work_min"] <= record["schedule_worker_work_max"]
    assert record["schedule_work_budget_overruns"] == 0
    assert record["schedule_elite_pool_size"] == 1
    assert record["schedule_unused_work_steps"] + record["schedule_work_steps"] == 8
    assert record["schedule_incumbent_source_worker"] < 2
    assert math.isfinite(record["candidates_per_second"])
    assert record["candidates_per_second"] >= 0.0
    assert math.isfinite(record["full_recompute_percentage"])
    assert 0.0 <= record["full_recompute_percentage"] <= 100.0
    assert all(record[field] is not None for field in SCHEDULE_COUNTERS)


def test_scheduling_verbose_output_includes_search_profile():
    command = (
        sys.executable,
        str(ROOT / "examples/python/scheduling/api/jssp.py"),
        "--jobs",
        "2",
        "--machines",
        "2",
        "--engine",
        "ls",
        "--time-limit",
        "1",
        "--max-iterations",
        "4",
        "--profile",
        "--verbose",
        "--json",
    )
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True, timeout=20)

    assert "schedule moves considered:" in completed.stdout
    assert "schedule moves accepted:" in completed.stdout
    assert "critical path updates:" in completed.stdout
    assert "schedule window rejections:" in completed.stdout
    assert "resource profile checks:" in completed.stdout
    assert "schedule justification attempts:" in completed.stdout
    assert "schedule delta evaluations:" in completed.stdout
    assert "schedule oracle mismatches:" in completed.stdout
    assert "schedule max dirty cone:" in completed.stdout
    assert "schedule restart work: 2048" in completed.stdout
