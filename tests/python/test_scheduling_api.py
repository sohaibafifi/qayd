"""Scheduling convenience API surface."""

import pytest

cp = pytest.importorskip("qayd")


def test_schedule_resource_lambda_and_makespan():
    model = cp.Model()
    tasks = model.tasks([0, 1])
    for task in tasks:
        task.duration = 2
        task.demand = [1]

    schedule = model.schedule(tasks, horizon=6)
    model.add(schedule.resource(lambda task: task.demand[0]) <= 1)
    model.minimize(schedule.makespan())

    solution = model.solve(time_limit=2, seed=1, profile=True)

    assert solution.status == "OPTIMAL"
    assert solution.objective == 4
    assert sorted(solution.starts) == [0, 2]
    assert solution.backend_build_seconds is not None
    assert solution.time_to_first_feasible is not None


def test_schedule_precedence_from_end_to_start():
    model = cp.Model()
    tasks = model.tasks([0, 1])
    tasks[0].duration = 3
    tasks[1].duration = 2

    schedule = model.schedule(tasks, horizon=8)
    model.add(schedule[0].end <= schedule[1].start)
    model.minimize(schedule.makespan())

    solution = model.solve(time_limit=2, seed=1)

    assert solution.status == "OPTIMAL"
    assert solution.starts[0] + 3 <= solution.starts[1]
    assert solution.objective == 5


@pytest.mark.parametrize("schedule_cdcl", [False, True])
def test_optional_mode_cdcl_is_an_explicit_solve_argument(schedule_cdcl):
    model = cp.Model()
    tasks = model.tasks([0, 1])
    tasks[0].modes = [(0, 3), (1, 2)]
    tasks[1].modes = [(0, 2), (1, 4)]
    schedule = model.schedule(tasks, horizon=10)
    model.add(schedule.no_overlap())
    model.minimize(schedule.makespan())

    solution = model.solve(time_limit=2, schedule_cdcl=schedule_cdcl)

    assert solution.status == "OPTIMAL"
    assert solution.objective == 2


def test_fixed_schedule_uses_compact_ir_and_ls_profile():
    model = cp.Model()
    tasks = model.tasks(range(300))
    for task in tasks:
        task.duration = 1
        task.machine = task.id % 10
    schedule = model.schedule(tasks, horizon=300)
    model.add(schedule.no_overlap(lambda task: task.machine))
    model.minimize(schedule.makespan())

    solution = model.solve(engine="ls", time_limit=1, profile=True, memory_limit_mb=64)

    assert solution.status == "SATISFIABLE"
    assert len(solution.starts) == 300
    assert solution.constructor == "serial-sgs"
    assert solution.time_to_first_feasible is not None
    assert solution.time_to_first_feasible < 1
    assert solution.estimated_backend_bytes is not None


def test_exact_schedule_memory_limit_fails_before_lowering():
    model = cp.Model()
    tasks = model.tasks(range(60))
    for task in tasks:
        task.duration = 1
    schedule = model.schedule(tasks, horizon=60)
    model.add(schedule.no_overlap())
    model.minimize(schedule.makespan())

    with pytest.raises(ValueError, match="estimated exact backend"):
        model.solve(engine="exact", time_limit=1, memory_limit_mb=1)


def test_zero_time_limit_returns_before_exact_schedule_lowering():
    model = cp.Model()
    tasks = model.tasks(range(60))
    for task in tasks:
        task.duration = 1
    schedule = model.schedule(tasks, horizon=60)
    model.add(schedule.no_overlap())
    model.minimize(schedule.makespan())

    solution = model.solve(engine="auto", time_limit=0, profile=True, memory_limit_mb=1)

    assert solution.status == "UNKNOWN"
    assert solution.backend_build_seconds is None
    assert solution.estimated_backend_bytes is None
