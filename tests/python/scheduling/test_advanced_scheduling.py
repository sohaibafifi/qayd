import pytest

cp = pytest.importorskip("qayd")


def test_alternative_master_composes_with_schedule_constraints():
    model = cp.Model()
    cut = model.interval(3, 15)
    fast = model.interval(2, 15, optional=True)
    slow = model.interval(4, 15, optional=True)
    operation = model.alternative([fast, slow])
    pack = model.interval(2, 15)
    model.precedence(cut, operation)
    model.precedence(operation, pack)
    model.no_overlap([cut, operation, pack])
    model.minimize_makespan([cut, operation, pack])

    solution = model.solve(engine="exact")
    assert solution.status == "OPTIMAL"
    assert solution.objective == 7
    assert solution.value(fast.presence) == 1
    assert solution.value(slow.presence) == 0


def test_optional_alternative_can_be_absent():
    model = cp.Model()
    members = [model.interval(2, 10, optional=True), model.interval(3, 10, optional=True)]
    master = model.alternative(members, optional=True)
    model.add(master.presence == 0)

    solution = model.solve(engine="exact")
    assert solution.status == "SATISFIABLE"
    assert all(solution.value(member.presence) == 0 for member in members)


def test_sequence_enforces_asymmetric_setups_and_decodes_order():
    durations = [2, 1, 2]
    setups = [[0, 4, 1], [2, 0, 2], [1, 3, 0]]
    model = cp.Model()
    tasks = model.intervals(durations, 15)
    sequence = model.sequence(tasks, setups)
    model.minimize_makespan(tasks)

    solution = model.solve(engine="exact")
    order = sequence.order(solution)
    indices = [task.index for task in order]
    assert solution.status == "OPTIMAL"
    for before, after in zip(indices, indices[1:]):
        assert solution.starts[before] + durations[before] + setups[before][after] <= solution.starts[after]


def test_capacity_calendar_is_respected():
    durations = [3, 3, 2]
    demands = [2, 1, 2]
    calendar = [(3, 5, 1)]
    model = cp.Model()
    tasks = model.intervals(durations, 15)
    model.resource_calendar(list(zip(tasks, demands)), 3, calendar)
    model.minimize_makespan(tasks)

    solution = model.solve(engine="exact")
    assert solution.status == "OPTIMAL"
    ends = [solution.starts[index] + durations[index] for index in range(len(tasks))]
    for time in range(max(ends)):
        capacity = 1 if 3 <= time < 5 else 3
        usage = sum(demands[index] for index in range(len(tasks)) if solution.starts[index] <= time < ends[index])
        assert usage <= capacity


def test_state_function_allows_same_state_overlap_and_separates_changes():
    durations = [3, 2, 2, 1]
    states = [0, 1, 0, 2]
    transitions = [[0, 2, 3], [2, 0, 1], [3, 1, 0]]
    model = cp.Model()
    tasks = model.intervals(durations, 20)
    model.state_function(list(zip(tasks, states)), transitions)
    model.minimize_makespan(tasks)

    solution = model.solve(engine="exact")
    assert solution.status == "OPTIMAL"
    starts = solution.starts
    ends = [starts[index] + durations[index] for index in range(len(tasks))]
    for left in range(len(tasks)):
        for right in range(left + 1, len(tasks)):
            if states[left] == states[right]:
                continue
            assert (
                ends[left] + transitions[states[left]][states[right]] <= starts[right]
                or ends[right] + transitions[states[right]][states[left]] <= starts[left]
            )


@pytest.mark.parametrize(
    "builder, message",
    [
        (lambda model, tasks: model.sequence(tasks, [[0]]), "square"),
        (lambda model, tasks: model.resource_calendar([(tasks[0], 1)], 1, [(4, 3, 1)]), "start < end"),
        (lambda model, tasks: model.state_function([(tasks[0], 1)], [[0]]), "unknown state"),
    ],
)
def test_advanced_scheduling_rejects_invalid_shapes(builder, message):
    model = cp.Model()
    tasks = model.intervals([1, 1], 5)
    with pytest.raises(ValueError, match=message):
        builder(model, tasks)
