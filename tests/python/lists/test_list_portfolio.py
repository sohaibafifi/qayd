"""Multithread portfolio coverage for the Python list frontend."""

import pytest

cp = pytest.importorskip("qayd")


def _list_model():
    model = cp.Model()
    left, right = model.list_vars(range(12), count=2)
    model.minimize(cp.count(left) + cp.count(right))
    return model


def test_list_portfolio_is_exposed_by_explicit_arguments():
    solution = _list_model().solve(
        engine="ls", threads=2, time_limit=2, seed=17, max_iterations=12, profile=True
    )
    assert solution.status == "SATISFIABLE"
    assert sorted(item for sequence in solution.lists for item in sequence) == list(range(12))
    assert solution.alns_iterations is not None
    assert solution.candidates_evaluated is not None
    assert solution.candidates_per_second is not None
    assert solution.full_recompute_percentage is not None
    assert solution.anytime_checkpoints is not None
    assert solution.neighborhood_profile is not None
    counters = dict(solution.routing_counters)
    assert set(counters) == {
        "slices",
        "descent_slices",
        "alns_slices",
        "relink_slices",
        "global_scan_slices",
        "route_elimination_attempts",
        "ejection_chain_attempts",
        "chain_relocate_attempts",
        "guided_segment_exchange_attempts",
        "macro_candidates_built",
        "macro_budget_exhaustions",
        "elite_insertions",
        "elite_rejections",
        "path_relink_attempts",
        "path_relink_steps",
        "path_relink_budget_exhaustions",
    }
    assert all(value >= 0 for value in counters.values())


def test_threads_must_be_positive():
    with pytest.raises(ValueError, match="positive integer"):
        _list_model().solve(engine="ls", threads=0, time_limit=1)


def test_list_portfolio_is_selected_by_the_canonical_auto_plan():
    solution = _list_model().solve(engine="auto", threads=2, time_limit=1, max_iterations=4)
    assert solution.status == "SATISFIABLE"
    assert sorted(item for sequence in solution.lists for item in sequence) == list(range(12))
    assert solution.anytime_checkpoints is None
    assert solution.neighborhood_profile is None
    assert solution.routing_counters is None


def test_integer_frontend_forwards_portfolio_threads():
    model = cp.Model()
    value = model.int_var(0, 1)
    model.minimize(value)
    solution = model.solve(threads=2, time_limit=1)
    assert solution.status == "OPTIMAL"
    assert solution.value(value) == 0


def test_integer_frontend_requires_ls_for_iteration_limit():
    model = cp.Model()
    model.int_var(0, 1)
    with pytest.raises(ValueError, match="requires engine='ls'"):
        model.solve(max_iterations=1)


def test_integer_satisfaction_ls_accepts_threads_and_iteration_limit():
    model = cp.Model()
    value = model.int_var(0, 1)
    solution = model.solve(engine="ls", threads=2, max_iterations=1, time_limit=1)
    assert solution.status == "SATISFIABLE"
    assert solution.value(value) in (0, 1)
