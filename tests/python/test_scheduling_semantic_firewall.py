"""Semantic firewall between compact scheduling and native exact intervals."""

import pytest


cp = pytest.importorskip("qayd")


@pytest.mark.parametrize("calendar", [False, True])
def test_absent_native_interval_consumes_no_renewable_resource(calendar):
    model = cp.Model()
    absent = model.interval(4, 4, optional=True, name="absent")
    mandatory = model.interval(4, 4, name="mandatory")
    model.add(absent.presence == 0)
    demands = [(absent, 3), (mandatory, 1)]
    if calendar:
        model.resource_calendar(demands, 1, [(0, 4, 1)])
    else:
        model.resource(demands, 1)
    model.minimize_makespan([absent, mandatory])

    solution = model.solve(engine="auto")

    assert solution.status == "OPTIMAL"
    assert solution.objective == 4
    assert solution.bound_method == "exact proof"
    assert solution.presences == [False, True]
    assert solution.starts == [None, 0]


def test_compact_modes_preserve_machine_duration_and_resource_semantics():
    model = cp.Model()
    intervals = model.alternatives(
        [
            [(10, 4), (20, 2)],
            [(10, 3), (20, 5)],
        ],
        horizon=10,
    )
    model.no_overlap_by_machine()
    model.resource([(intervals[0], 1), (intervals[1], 1)], 1)
    model.minimize_makespan(intervals)

    solution = model.solve(engine="ls", seed=1, max_iterations=1_000)

    assert solution.status in {"SATISFIABLE", "OPTIMAL"}
    assert solution.objective == 5
    assert solution.machines == [20, 10]
    durations = [2, 3]
    starts = [int(start) for start in solution.starts]
    ends = [start + duration for start, duration in zip(starts, durations)]
    assert ends[0] <= starts[1] or ends[1] <= starts[0]


def _optional_model():
    model = cp.Model()
    interval = model.interval(2, 5, optional=True)
    model.add(interval.presence == 1)
    model.minimize_makespan([interval])
    return model


def _alternative_model():
    model = cp.Model()
    members = [model.interval(1, 5, optional=True), model.interval(2, 5, optional=True)]
    master = model.alternative(members)
    model.minimize_makespan([master])
    return model


def _sequence_model():
    model = cp.Model()
    intervals = model.intervals([1, 1], 5)
    model.sequence(intervals, [[0, 1], [1, 0]])
    model.minimize_makespan(intervals)
    return model


def _calendar_model():
    model = cp.Model()
    intervals = model.intervals([1, 1], 5)
    model.resource_calendar([(intervals[0], 1), (intervals[1], 1)], 0, [(2, 5, 1)])
    model.minimize_makespan(intervals)
    return model


def _state_model():
    model = cp.Model()
    intervals = model.intervals([1, 1], 5)
    model.state_function([(intervals[0], 0), (intervals[1], 1)], [[0, 1], [1, 0]])
    model.minimize_makespan(intervals)
    return model


@pytest.mark.parametrize(
    "builder",
    [_optional_model, _alternative_model, _sequence_model, _calendar_model, _state_model],
)
def test_native_interval_features_delegate_local_search_capability_to_the_orchestrator(builder):
    solution = builder().solve(engine="ls", time_limit=1, max_iterations=1_000)

    assert solution.status == "SATISFIABLE"
    assert solution.objective is not None


@pytest.mark.parametrize(
    ("builder", "expected_objective"),
    [(_sequence_model, 3), (_calendar_model, 4), (_state_model, 3)],
)
def test_advanced_native_features_stay_on_the_exact_auto_path(builder, expected_objective):
    solution = builder().solve(engine="auto")

    assert solution.status == "OPTIMAL"
    assert solution.objective == expected_objective
    assert solution.bound_method == "exact proof"


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (lambda model, intervals: model.sequence(intervals, [[0, 1], [1, 0]]), "sequence setups require native"),
        (lambda model, intervals: model.resource_calendar([(intervals[0], 1)], 1, []), "resource calendars require native"),
        (lambda model, intervals: model.state_function([(intervals[0], 0)], [[0]]), "state functions require native"),
    ],
)
def test_advanced_native_apis_reject_compact_intervals_explicitly(operation, message):
    model = cp.Model()
    intervals = model.schedule_intervals([1, 1], 5)

    with pytest.raises(ValueError, match=message):
        operation(model, intervals)
