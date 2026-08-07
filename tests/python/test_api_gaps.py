"""Field-reported API gaps: interval `.end`/`.duration` and `sum()` over
arithmetic expressions."""
import pytest

cp = pytest.importorskip("qayd")


def test_interval_end_is_start_plus_duration():
    model = cp.Model()
    iv = model.interval(3, 10, name="t")
    assert iv.duration == 3
    # `.end` is an expression usable directly in constraints.
    model.add(iv.end <= 5)
    solution = model.solve()
    assert solution.status in ("SATISFIABLE", "OPTIMAL")
    assert solution.value(iv.start) + 3 <= 5


def test_moded_interval_end_raises_clearly():
    model = cp.Model()
    (iv,) = model.alternatives([[(0, 2), (1, 4)]], horizon=10)
    with pytest.raises(ValueError, match="mode-dependent"):
        _ = iv.end


def test_sum_accepts_arithmetic_expressions():
    model = cp.Model()
    x = model.int_var(0, 5, name="x")
    y = model.int_var(0, 5, name="y")
    total = cp.sum([x + 1, y, 2])
    model.add(total == 10)
    model.add(x == 3)
    solution = model.solve()
    assert solution.status in ("SATISFIABLE", "OPTIMAL")
    assert solution.value(x) == 3
    assert (solution.value(x) + 1) + solution.value(y) + 2 == 10


def test_sum_rejects_junk_with_a_pointed_message():
    with pytest.raises(TypeError, match="terms or arithmetic operands"):
        cp.sum(["not-an-operand"])
    with pytest.raises(ValueError, match="no terms"):
        cp.sum([])


def test_list_terms_accept_scalar_multiplication_in_objectives():
    model = cp.Model()
    left, right = model.list_vars([1, 2], count=2)

    model.minimize(3 * cp.used(left) + cp.used(right) * 2)

    solution = model.solve()
    assert solution.status == "OPTIMAL"
    assert solution.objective == 2
    assert solution.lists[0] == []
    assert sorted(solution.lists[1]) == [1, 2]


def test_list_term_scalar_multiplication_applies_to_constraints():
    model = cp.Model()
    (route,) = model.list_vars([1, 2], count=1)

    model.add(2 * cp.sum(route, lambda i: i) <= 5)

    solution = model.solve()
    assert solution.status == "UNSATISFIABLE"


def test_max_of_accepts_list_terms_from_edges_and_items():
    model = cp.Model()
    (route,) = model.list_vars([1, 2, 3], count=1)
    distances = cp.matrix([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]])

    edge_term = cp.sum_edges(route, lambda i, j: distances[i][j], start=0, end=0)
    item_term = cp.sum(route, lambda i: 1)

    assert type(cp.max_of([edge_term, item_term])).__name__ == "Term"


def test_max_of_list_terms_solves_as_objective():
    model = cp.Model()
    left, right = model.list_vars([1, 2], count=2)

    model.minimize(cp.max_of([3 * cp.used(left), 2 * cp.used(right)]))

    solution = model.solve()
    assert solution.status == "OPTIMAL"
    assert solution.objective == 2
    assert solution.lists[0] == []
    assert sorted(solution.lists[1]) == [1, 2]


def test_max_of_list_terms_is_composable_in_weighted_sum():
    model = cp.Model()
    left, right = model.list_vars([1, 2], count=2)

    objective = 2 * cp.max_of([cp.used(left), cp.used(right)]) + 3 * cp.used(left)
    model.minimize(objective)

    solution = model.solve()
    assert solution.status == "OPTIMAL"
    assert solution.objective == 2
    assert solution.lists[0] == []
    assert sorted(solution.lists[1]) == [1, 2]


def test_multiple_max_of_list_terms_can_be_added():
    model = cp.Model()
    left, right = model.list_vars([1, 2], count=2)

    objective = cp.max_of([cp.used(left), cp.used(right)]) + cp.max_of([2 * cp.used(left), 3 * cp.used(right)])
    model.minimize(objective)

    solution = model.solve()
    assert solution.status == "OPTIMAL"
    assert solution.objective == 3
    assert sorted(solution.lists[0]) == [1, 2]
    assert solution.lists[1] == []


def test_ls_supports_max_of_list_terms_as_makespan():
    model = cp.Model()
    left, right = model.list_vars([1, 2, 3], count=2)

    left_load = cp.sum(left, lambda i: i)
    right_load = cp.sum(right, lambda i: i)
    model.minimize(cp.max_of([left_load, right_load]))

    solution = model.solve(engine="ls", time_limit=1, seed=1)
    assert solution.status in ("SATISFIABLE", "OPTIMAL")
    assert solution.objective == 3
    assert max(sum(route) for route in solution.lists) == 3
