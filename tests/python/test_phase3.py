import time

import pytest

cp = pytest.importorskip("qayd")


def _assignment_model(items):
    model = cp.Model()
    lists = model.list_vars(range(items), count=3)
    model.minimize(cp.sum(cp.used(sequence) for sequence in lists))
    return model


def test_auto_uses_exact_for_small_models_and_ls_for_large_ones():
    small = _assignment_model(8).solve(engine="auto", time_limit=2, seed=3)
    large = _assignment_model(30).solve(engine="auto", time_limit=2, seed=3, max_iterations=0)
    assert small.status == "OPTIMAL"
    assert large.status == "SATISFIABLE"
    assert sorted(item for sequence in large.lists for item in sequence) == list(range(30))


def test_zero_budget_covers_collection_preprocessing():
    model = _assignment_model(30)
    started = time.monotonic()
    solution = model.solve(engine="auto", time_limit=0)
    assert time.monotonic() - started < 0.5
    assert solution.status == "UNKNOWN"
    assert solution.lists is None


def test_zero_budget_covers_schedule_preprocessing():
    model = cp.Model()
    intervals = model.intervals([1] * 60, horizon=100)
    model.no_overlap(intervals)
    model.minimize_makespan(intervals)
    started = time.monotonic()
    solution = model.solve(engine="auto", time_limit=0)
    assert time.monotonic() - started < 0.5
    assert solution.status == "UNKNOWN"
    assert not solution.starts
