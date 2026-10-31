import pytest

import qayd as cp


def policy(variables, variable="input-order", value="min"):
    return cp.SearchPolicy([cp.SearchPhase(variables, variable, value)])


def test_search_policy_is_public_and_introspectable():
    model = cp.Model()
    x = model.bool_var("x")
    phase = cp.SearchPhase([x], "first-fail", "random-seeded")
    search_policy = cp.SearchPolicy([phase])

    assert [variable.index for variable in phase.variables] == [x.index]
    assert phase.variable_selector == "first-fail"
    assert phase.value_selector == "random-seeded"
    assert [variable.index for variable in search_policy.phases[0].variables] == [x.index]
    assert "SearchPhase" in repr(phase)
    assert "SearchPolicy" in repr(search_policy)


@pytest.mark.parametrize("selector", ["smallest", "dom_over_wdeg", ""])
def test_invalid_variable_selector_is_rejected(selector):
    model = cp.Model()
    x = model.bool_var()
    with pytest.raises(ValueError, match="unknown variable selector"):
        cp.SearchPhase([x], selector, "min")


@pytest.mark.parametrize("selector", ["middle", "random", ""])
def test_invalid_value_selector_is_rejected(selector):
    model = cp.Model()
    x = model.bool_var()
    with pytest.raises(ValueError, match="unknown value selector"):
        cp.SearchPhase([x], "auto", selector)


def test_search_phase_and_policy_require_typed_members():
    with pytest.raises(TypeError, match="IntVar"):
        cp.SearchPhase([1])
    with pytest.raises(TypeError, match="SearchPhase"):
        cp.SearchPolicy([object()])


def test_empty_phase_is_rejected_by_semantic_request_validation():
    model = cp.Model()
    model.bool_var()
    search_policy = cp.SearchPolicy([cp.SearchPhase([])])

    with pytest.raises(ValueError, match="empty variable scope"):
        model.solve(search_policy=search_policy)


@pytest.mark.parametrize(
    "phases",
    [
        lambda x: [cp.SearchPhase([x, x])],
        lambda x: [cp.SearchPhase([x]), cp.SearchPhase([x])],
    ],
)
def test_duplicate_phase_variables_are_rejected_by_semantic_request_validation(phases):
    model = cp.Model()
    x = model.bool_var()

    with pytest.raises(ValueError, match="appears in more than one phase|more than once"):
        model.solve(search_policy=cp.SearchPolicy(phases(x)))


def test_foreign_search_policy_variable_is_rejected_at_python_boundary():
    model = cp.Model()
    model.bool_var()
    foreign_model = cp.Model()
    foreign = foreign_model.bool_var()

    with pytest.raises(ValueError, match="different model"):
        model.solve(search_policy=policy([foreign]))


@pytest.mark.parametrize(
    "legacy",
    [
        {"search": []},
        {"hints": []},
        {"branch_order": []},
    ],
)
def test_explicit_search_policy_rejects_even_empty_legacy_guidance(legacy):
    model = cp.Model()
    x = model.bool_var()

    with pytest.raises(ValueError, match="cannot be combined"):
        model.solve(search_policy=cp.SearchPolicy([]), **legacy)


def test_none_search_policy_preserves_legacy_guidance():
    model = cp.Model()
    x = model.bool_var()
    solution = model.solve(search_policy=None, search=[x], hints=[(x, 1)])

    assert solution.status == "SATISFIABLE"
    assert solution.value(x) == 1


@pytest.mark.parametrize("variable_selector", ["auto", "input-order", "first-fail", "max-regret", "dom-wdeg", "activity"])
@pytest.mark.parametrize("value_selector", ["auto", "min", "max", "median", "random-seeded", "hint"])
def test_all_selectors_are_accepted(variable_selector, value_selector):
    model = cp.Model()
    x = model.int_var(0, 2)
    solution = model.solve(search_policy=policy([x], variable_selector, value_selector), seed=7)

    assert solution.status == "SATISFIABLE"
    assert solution.value(x) in (0, 1, 2)


def test_max_regret_uses_the_second_supported_value():
    model = cp.Model()
    first = model.int_var(values=[0, 2, 100])
    second = model.int_var(values=[0, 10, 11])
    model.add(first != second)

    solution = model.solve(search_policy=policy([first, second], "max-regret", "min"))

    assert solution.status == "SATISFIABLE"
    assert (solution.value(first), solution.value(second)) == (2, 0)


def test_ordered_phases_and_auto_fallback_assign_every_variable():
    model = cp.Model()
    x = model.bool_var("guided")
    y = model.bool_var("fallback")
    z = model.bool_var("fallback_constraint")
    model.add(y + z == 1)
    solution = model.solve(search_policy=policy([x], value="max"))

    assert solution.status == "SATISFIABLE"
    assert solution.value(x) == 1
    assert solution.value(y) + solution.value(z) == 1


def test_random_seeded_policy_is_reproducible():
    model = cp.Model()
    variables = model.int_vars(12, 0, 20)
    search_policy = policy(variables, value="random-seeded")

    first = model.solve(search_policy=search_policy, seed=42, threads=1)
    second = model.solve(search_policy=search_policy, seed=42, threads=1)

    assert first.status == second.status == "SATISFIABLE"
    assert first.assignment() == second.assignment()


def test_search_policy_preserves_unsat_proof():
    model = cp.Model()
    x = model.bool_var()
    model.add(x == 0)
    model.add(x == 1)

    solution = model.solve(search_policy=policy([x], value="max"))

    assert solution.status == "UNSATISFIABLE"


def test_search_policy_preserves_optimality_proof():
    model = cp.Model()
    x = model.int_var(0, 5)
    y = model.int_var(0, 5)
    model.add(x + y >= 4)
    model.minimize(x + y)

    solution = model.solve(search_policy=policy([y], value="max"))

    assert solution.status == "OPTIMAL"
    assert solution.objective == 4
    assert solution.value(x) + solution.value(y) == 4


def test_session_can_change_policy_between_epochs():
    model = cp.Model()
    x = model.int_var(0, 3)
    session = model.session()

    low = session.solve(search_policy=policy([x], value="min"), seed=9)
    high = session.solve(search_policy=policy([x], value="max"), seed=9)

    assert low.status == high.status == "SATISFIABLE"
    assert low.value(x) == 0
    assert high.value(x) == 3


def test_session_rejects_explicit_policy_with_legacy_guidance():
    model = cp.Model()
    x = model.bool_var()
    session = model.session()

    with pytest.raises(ValueError, match="cannot be combined"):
        session.solve(search_policy=policy([x]), search=[])


def test_solution_value_rejects_same_index_from_another_model():
    model = cp.Model()
    x = model.bool_var()
    other = cp.Model()
    foreign = other.bool_var()

    solution = model.solve()

    assert solution.value(x) in (0, 1)
    with pytest.raises(ValueError, match="different model"):
        solution.value(foreign)


def test_session_solution_value_rejects_same_index_from_another_model():
    model = cp.Model()
    x = model.bool_var()
    other = cp.Model()
    foreign = other.bool_var()

    solution = model.session().solve()

    assert solution.value(x) in (0, 1)
    with pytest.raises(ValueError, match="different model"):
        solution.value(foreign)
