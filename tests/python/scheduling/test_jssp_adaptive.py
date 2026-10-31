"""The adaptive JSSP control reaches both public modeling paths and launchers."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

cp = pytest.importorskip("qayd")
ROOT = Path(__file__).resolve().parents[3]
JOBS = (
    ((0, 3), (1, 2), (2, 2)),
    ((1, 2), (2, 1), (0, 4)),
    ((2, 3), (0, 2), (1, 1)),
)


def job_shop(*, native=False):
    model = cp.Model()
    operations = [operation for job in JOBS for operation in job]
    horizon = sum(duration for _, duration in operations)
    if native:
        intervals = model.alternatives([[(machine, duration)] for machine, duration in operations], horizon)
        for job in range(len(JOBS)):
            for offset in range(2):
                model.precedence(intervals[3 * job + offset], intervals[3 * job + offset + 1])
        model.no_overlap_by_machine()
        model.minimize_makespan(intervals)
    else:
        tasks = model.tasks(range(len(operations)))
        for task, (machine, duration) in zip(tasks, operations):
            task.duration = duration
            task.machine = machine
        schedule = model.schedule(tasks, horizon=horizon)
        for job in range(len(JOBS)):
            for offset in range(2):
                model.add(schedule[3 * job + offset].end <= schedule[3 * job + offset + 1].start)
        model.add(schedule.no_overlap(lambda task: task.machine))
        model.minimize(schedule.makespan())
    return model


def replay(starts, objective):
    operations = [operation for job in JOBS for operation in job]
    assert len(starts) == len(operations)
    assert all(start >= 0 for start in starts)
    ends = [start + duration for start, (_, duration) in zip(starts, operations)]
    for job in range(len(JOBS)):
        for offset in range(2):
            assert ends[3 * job + offset] <= starts[3 * job + offset + 1]
    for left, (machine, _) in enumerate(operations):
        for right in range(left + 1, len(operations)):
            if operations[right][0] == machine:
                assert ends[left] <= starts[right] or ends[right] <= starts[left]
    assert objective == max(ends)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("iterations", [1, 17, 129])
def test_adaptive_jssp_publishes_complete_schedule_at_finite_work_limit(native, iterations):
    solution = job_shop(native=native).solve(
        engine="ls", threads=1, seed=11, time_limit=2, max_iterations=iterations,
        profile=True, schedule_jssp_search="tsab-adaptive",
    )
    assert solution.status == "SATISFIABLE"
    replay(solution.starts, solution.objective)
    assert solution.schedule_tsab_owner_worker_mask == 1
    assert solution.schedule_tsab_activations == 1
    assert solution.schedule_tsab_legacy_warmup_work_steps == 0
    assert solution.schedule_oracle_mismatches == 0
    assert solution.schedule_incumbent_verification_rejections == 0
    assert solution.schedule_work_budget_overruns == 0


def test_adaptive_jssp_does_not_require_profiling():
    solution = job_shop().solve(
        engine="ls", threads=1, seed=3, max_iterations=17,
        profile=False, schedule_jssp_search="tsab-adaptive",
    )
    assert solution.status == "SATISFIABLE"
    replay(solution.starts, solution.objective)


@pytest.mark.parametrize("threads", [2, 7])
def test_adaptive_jssp_rejects_other_worker_layouts(threads):
    with pytest.raises(ValueError, match="requires threads=1"):
        job_shop().solve(engine="ls", threads=threads, schedule_jssp_search="tsab-adaptive")


@pytest.mark.parametrize("control", ["schedule_path_relink", "schedule_lns_shadow"])
def test_adaptive_jssp_rejects_unqualified_hybrids(control):
    with pytest.raises(ValueError, match="cannot yet be combined"):
        job_shop().solve(engine="ls", threads=1, schedule_jssp_search="tsab-adaptive", **{control: True})


def test_adaptive_jssp_requires_selected_local_search_backend():
    with pytest.raises(ValueError, match="requires a selected scheduling local-search backend"):
        job_shop().solve(engine="exact", threads=1, schedule_jssp_search="tsab-adaptive")


def test_adaptive_jssp_rejects_flexible_machine_assignment():
    model = cp.Model()
    intervals = model.alternatives([[(0, 3), (1, 2)], [(0, 2), (1, 4)]], 10)
    model.no_overlap_by_machine()
    model.minimize_makespan(intervals)
    with pytest.raises(ValueError, match="supports only mandatory fixed-assignment JSSP"):
        model.solve(engine="ls", threads=1, schedule_jssp_search="tsab-adaptive")


@pytest.mark.parametrize("launcher", ["api", "native"])
def test_adaptive_jssp_launchers_report_strategy_and_verified_schedule(launcher):
    completed = subprocess.run(
        [
            sys.executable, str(ROOT / "examples/python/scheduling" / launcher / "jssp.py"),
            "--jobs", "8", "--machines", "6", "--seed", "1", "--engine", "ls",
            "--threads", "1", "--time-limit", "1", "--max-iterations", "129", "--profile",
            "--schedule-jssp-search", "tsab-adaptive", "--json", "--compact-json",
        ],
        cwd=ROOT, text=True, capture_output=True, check=True, timeout=20,
    )
    record = json.loads(completed.stdout)
    assert record["verified"] is True
    assert record["schedule_jssp_search"] == "tsab-adaptive"
    assert record["schedule_tsab_owner_worker_mask"] == 1
    assert record["schedule_tsab_activations"] == 1
    assert record["schedule_tsab_legacy_warmup_work_steps"] == 0
