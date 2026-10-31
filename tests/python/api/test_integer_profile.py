"""Profile rendering and truth-value contracts for integer solves."""

import pytest

cp = pytest.importorskip("qayd")


INTEGER_LS_FIELDS = (
    "ls_moves",
    "ls_constraints",
    "ls_functionals",
    "ls_unsupported",
    "ls_rejected_incumbents",
    "ls_checkpoint_replays",
)


def _optimization_model():
    model = cp.Model()
    left, right = model.int_vars(2, 0, 4)
    model.add(left + right >= 3)
    model.minimize(left + right)
    return model


def test_integer_exact_profile_is_exposed_only_when_requested():
    plain = _optimization_model().solve(engine="exact")
    profiled = _optimization_model().solve(engine="exact", profile=True)

    assert plain.status == profiled.status == "OPTIMAL"
    assert plain.backend_build_seconds is None
    assert plain.estimated_backend_bytes is None
    assert profiled.backend_build_seconds is not None
    assert profiled.backend_build_seconds >= 0.0
    assert profiled.estimated_backend_bytes is not None
    assert profiled.estimated_backend_bytes > 0
    assert all(getattr(profiled, field) is None for field in INTEGER_LS_FIELDS)


def test_integer_ls_profile_preserves_engine_report_counters():
    plain = _optimization_model().solve(engine="ls", seed=7, max_iterations=16)
    profiled = _optimization_model().solve(engine="ls", seed=7, max_iterations=16, profile=True)

    assert plain.status == profiled.status == "SATISFIABLE"
    assert plain.backend_build_seconds is None
    assert plain.estimated_backend_bytes is None
    assert all(getattr(plain, field) is None for field in INTEGER_LS_FIELDS)
    assert profiled.backend_build_seconds is not None
    assert profiled.estimated_backend_bytes is not None
    assert profiled.ls_moves is not None
    assert profiled.ls_moves > 0
    assert profiled.ls_constraints == 1
    assert profiled.ls_functionals == 0
    assert profiled.ls_unsupported == 0
    assert profiled.ls_rejected_incumbents is not None
    assert profiled.ls_checkpoint_replays is not None


def test_only_feasible_solution_statuses_are_truthy():
    optimal = _optimization_model().solve(engine="exact")

    unknown_model = cp.Model()
    unknown_model.int_var(0, 1)
    unknown = unknown_model.solve(time_limit=0)

    unsupported_model = cp.Model()
    (route,) = unsupported_model.list_vars([1, 2], count=1)
    distances = cp.matrix([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    unsupported_model.add(cp.sum_edges(route, lambda i, j: distances[i][j], start=0, end=0) <= 10)
    unsupported = unsupported_model.solve(engine="exact")

    assert optimal.status == "OPTIMAL"
    assert optimal.is_sat()
    assert bool(optimal)
    assert unknown.status == "UNKNOWN"
    assert not unknown.is_sat()
    assert not unknown
    assert unsupported.status == "UNSUPPORTED"
    assert not unsupported.is_sat()
    assert not unsupported
