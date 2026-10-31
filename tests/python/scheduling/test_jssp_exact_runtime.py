"""Requested scheduling controls must remain distinct from observed execution."""

import pytest

cp = pytest.importorskip("qayd")


def job_shop(moded):
    model = cp.Model()
    if moded:
        intervals = model.alternatives([[(0, 2)], [(1, 3)], [(1, 2)], [(0, 1)]], 8)
        model.precedence(intervals[0], intervals[1])
        model.precedence(intervals[2], intervals[3])
        model.no_overlap_by_machine()
        model.minimize_makespan(intervals)
    else:
        tasks = model.tasks(range(4))
        for task, machine, duration in zip(tasks, [0, 1, 1, 0], [2, 3, 2, 1]):
            task.machine = machine
            task.duration = duration
        schedule = model.schedule(tasks, horizon=8)
        model.add(schedule[0].end <= schedule[1].start)
        model.add(schedule[2].end <= schedule[3].start)
        model.add(schedule.no_overlap(lambda task: task.machine))
        model.minimize(schedule.makespan())
    return model


@pytest.mark.parametrize("moded,cdcl,kernel", [
    (False, False, "cdcl"), (False, True, "cdcl"),
    (True, False, "chronological-dfs"), (True, True, "cdcl"),
])
def test_exact_runtime_is_available_without_search_profiling(moded, cdcl, kernel):
    solution = job_shop(moded).solve(engine="exact", threads=1, schedule_cdcl=cdcl, time_limit=2, profile=False)
    assert solution.status == "OPTIMAL"
    assert solution.objectives == [5]
    assert solution.message is None
    assert solution.search_engine == "schedule-exact"
    assert solution.schedule_encoding == ("moded" if moded else "fixed")
    assert solution.schedule_search_kernel == kernel
    if kernel == "cdcl":
        assert solution.stats.lcg_atoms > 0


@pytest.mark.parametrize("moded", [False, True])
def test_zero_budget_does_not_claim_a_search_invocation(moded):
    solution = job_shop(moded).solve(engine="exact", threads=1, schedule_cdcl=True, time_limit=0, profile=True)
    assert solution.status == "UNKNOWN"
    assert solution.objectives == []
    assert solution.schedule_search_kernel is None
    assert solution.stats.lcg_atoms == 0
    assert solution.message is not None and "Deadline" in solution.message


@pytest.mark.parametrize("moded", [False, True])
def test_resource_bound_can_prove_infeasibility_before_search(moded):
    model = cp.Model()
    if moded:
        intervals = model.alternatives([[(0, 3)], [(0, 3)]], 5)
        model.no_overlap_by_machine()
        model.minimize_makespan(intervals)
    else:
        tasks = model.tasks(range(2))
        for task in tasks:
            task.machine = 0
            task.duration = 3
        schedule = model.schedule(tasks, horizon=5)
        model.add(schedule.no_overlap(lambda task: task.machine))
        model.minimize(schedule.makespan())
    solution = model.solve(engine="exact", threads=1, schedule_cdcl=True, time_limit=2, profile=True)
    assert solution.status == "UNSATISFIABLE"
    assert solution.objectives == []
    assert solution.search_engine == "schedule-exact"
    assert solution.schedule_encoding == ("moded" if moded else "fixed")
    assert solution.schedule_search_kernel is None
    assert solution.stats.lcg_atoms == 0


def test_local_search_does_not_inherit_the_exact_kernel_identity():
    solution = job_shop(True).solve(engine="ls", threads=1, max_iterations=1, profile=True)
    assert solution.search_engine == "schedule-ls"
    assert solution.schedule_encoding is None
    assert solution.schedule_search_kernel is None
    assert solution.search_incumbent_events is None


@pytest.mark.parametrize("moded", [False, True])
def test_exact_internal_events_are_separate_from_the_published_solution(moded):
    solution = job_shop(moded).solve(engine="exact", threads=1, schedule_cdcl=True, time_limit=2, profile=True)
    assert solution.status == "OPTIMAL"
    events = solution.search_incumbent_events
    assert events
    assert [event[0] for event in events] == sorted(event[0] for event in events)
    objectives = [event[1][0] for event in events]
    assert all(left > right for left, right in zip(objectives, objectives[1:]))
    assert objectives[-1] == solution.objective == 5
