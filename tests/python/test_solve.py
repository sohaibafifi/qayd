"""Tests for solve() honoring time_limit and seed on the integer backends.

Requires the compiled extension (`maturin develop --features python`); the tests
skip cleanly when it is not built.
"""

import os
import sys
import time
from random import Random

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
cp = pytest.importorskip("qayd")


def knapsack_model(coef_seed=7, n=55, hi=1000):
    """A 0/1 optimization with a knapsack cap and random conflicts. At n=55 with
    large coefficients it cannot prove optimality within a second; a small n
    finishes to OPTIMAL quickly."""
    rng = Random(coef_seed)
    model = cp.Model()
    xs = model.int_vars(n, 0, 1)
    value = [rng.randint(1, hi) for _ in range(n)]
    weight = [rng.randint(1, hi) for _ in range(n)]
    model.add(sum(weight[i] * xs[i] for i in range(n)) <= sum(weight) // 2)
    for _ in range(n):
        a, b = rng.randrange(n), rng.randrange(n)
        if a != b:
            model.add(xs[a] + xs[b] <= 1)
    model.maximize(sum(value[i] * xs[i] for i in range(n)))
    return model


def test_time_limit_stops_a_hard_solve():
    start = time.time()
    solution = knapsack_model().solve(time_limit=1, seed=1)
    elapsed = time.time() - start
    # Without an armed deadline this runs for far longer than the limit.
    assert elapsed < 5.0, f"solve ignored time_limit (took {elapsed:.1f}s)"
    assert solution.status in ("SATISFIABLE", "OPTIMAL")
    assert solution.objective is not None


def test_same_seed_is_deterministic():
    # Run a model that finishes to OPTIMAL (no time limit): with the search
    # complete, the same seed reproduces the solution *and* the stats exactly.
    # (Under a time limit, node counts vary with the wall-clock cutoff.)
    first = knapsack_model(n=20, hi=100).solve(seed=42)
    second = knapsack_model(n=20, hi=100).solve(seed=42)
    assert first.status == "OPTIMAL"
    assert first.objective == second.objective
    assert first.assignment() == second.assignment()
    assert first.stats.nodes == second.stats.nodes
    assert first.stats.failures == second.stats.failures
    assert first.stats.solutions == second.stats.solutions


def test_different_seeds_still_return():
    # Different seeds may diverge; the contract is only that each still returns a
    # feasible incumbent within the limit.
    for seed in (1, 2, 3):
        solution = knapsack_model().solve(time_limit=1, seed=seed)
        assert solution.objective is not None


def test_time_limit_on_local_search_backend():
    start = time.time()
    solution = knapsack_model().solve(engine="ls", time_limit=1, seed=1)
    assert time.time() - start < 5.0
    assert solution.status == "SATISFIABLE"


def test_session_solves_under_assumptions():
    model = cp.Model()
    x = model.bool_var("x")
    y = model.bool_var("y")
    model.add(x + y <= 1)

    session = model.session()
    solution = session.solve(assumptions=[x == 1])

    assert solution.status == "SATISFIABLE"
    assert solution.value(x) == 1
    assert solution.value(y) == 0


def test_session_reports_unsat_assumption_set():
    model = cp.Model()
    x = model.bool_var("x")
    y = model.bool_var("y")
    model.add(x + y <= 1)

    solution = model.session().solve(assumptions=[x == 1, y == 1])

    assert solution.status == "UNSATISFIABLE"


def test_integer_lexicographic_objective_uses_full_engine():
    model = cp.Model()
    x = model.bool_var("x")
    y = model.bool_var("y")
    model.minimize(x)
    model.then_maximize(y)

    solution = model.solve()

    assert solution.status == "OPTIMAL"
    assert solution.objectives == [0, 1]
    assert solution.value(x) == 0
    assert solution.value(y) == 1


def test_session_incumbent_callback_runs():
    model = cp.Model()
    x = model.int_var(0, 3, name="x")
    model.maximize(x)
    seen = []

    solution = model.session().solve(on_incumbent=lambda value, assignment: seen.append((value, assignment[x.index])))

    assert solution.status == "OPTIMAL"
    assert solution.objective == 3
    assert seen
    assert seen[-1] == (3, 3)


def test_session_exposes_nogood_snapshots():
    model = cp.Model()
    xs = model.int_vars(4, 0, 1)
    model.add(sum(xs) == 2)
    session = model.session()

    solution = session.solve(assumptions=[xs[0] == 1])

    assert solution.status == "SATISFIABLE"
    assert isinstance(session.raw_nogoods(limit=5), list)
    assert isinstance(session.nogoods(limit=5), list)


def test_branch_order_prioritizes_user_variables():
    model = cp.Model()
    x = model.bool_var("x")
    y = model.bool_var("y")
    model.add(x + y == 1)

    solution = model.solve(branch_order=[y], hints=[(y, 0)])

    assert solution.status == "SATISFIABLE"
    assert solution.value(y) == 0
    assert solution.value(x) == 1


def test_optional_interval_can_be_absent_under_makespan():
    model = cp.Model()
    required = model.interval(3, 6, name="required")
    optional = model.interval(3, 6, optional=True, name="optional")
    model.no_overlap([required, optional])
    model.minimize_makespan()

    solution = model.solve()

    assert solution.status == "OPTIMAL"
    assert solution.objective == 3
    assert solution.starts[required.index] == 0
    assert solution.starts[optional.index] is None
    assert solution.presences == [True, False]
    assert solution.value(optional.presence) == 0


def test_optional_interval_presence_literal_forces_scheduling():
    model = cp.Model()
    first = model.interval(3, 6)
    second = model.interval(3, 6, optional=True)
    model.no_overlap([first, second])
    model.add(second.presence == 1)
    model.minimize_makespan()

    solution = model.solve()

    assert solution.status == "OPTIMAL"
    assert solution.objective == 6
    assert solution.presences == [True, True]
    assert all(start is not None for start in solution.starts)


def test_native_interval_local_search_capability_is_decided_by_the_orchestrator():
    model = cp.Model()
    interval = model.interval(2, 6)

    solution = model.solve(engine="ls", time_limit=1, max_iterations=4)

    assert solution.status == "SATISFIABLE"
    assert solution.starts[interval.index] is not None


def test_independent_integer_and_list_families_are_decomposed_canonically():
    model = cp.Model()
    model.int_var(1, 1, name="fixed")
    model.list_vars([1, 2], count=1)

    solution = model.solve(engine="exact", time_limit=2)

    assert solution.status == "SATISFIABLE"
    assert solution.values == [1]
    assert solution.lists == [[1, 2]]


def test_collection_unsupported_status_is_rendered_without_frontend_reclassification():
    model = cp.Model()
    (route,) = model.list_vars([1, 2], count=1)
    distances = cp.matrix([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    model.add(cp.sum_edges(route, lambda i, j: distances[i][j], start=0, end=0) <= 10)

    solution = model.solve(engine="exact", time_limit=2)

    assert solution.status == "UNSUPPORTED"
