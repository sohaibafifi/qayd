"""List-domain resource scan API."""

import pytest

cp = pytest.importorskip("qayd")


def test_scan_resource_point_view_can_force_order():
    model = cp.Model()
    route, = model.list_vars([1, 2], count=1)
    demand = cp.array([0, 1, 3])

    load = cp.scan_resource(route, initial=0, delta=lambda prev, cur: demand[cur], boundary=0)
    model.add(load.after(1) <= 2)
    model.minimize(cp.sum_edges(route, lambda i, j: 1, start=0, end=0))

    solution = model.solve(time_limit=2, seed=1)

    assert solution.status in ("SATISFIABLE", "OPTIMAL")
    assert solution.lists[0].index(1) < solution.lists[0].index(2)


def test_scan_resource_custom_before_view():
    model = cp.Model()
    route, = model.list_vars([1, 2], count=1)
    demand = cp.array([0, 1, 3])

    load = cp.scan_resource(route, initial=0, delta=lambda prev, cur: demand[cur], boundary=0)
    load.view("before", lambda cur, state, prev: state - demand[cur])
    model.add(load.before(1) <= 0)
    model.minimize(cp.sum_edges(route, lambda i, j: 1, start=0, end=0))

    solution = model.solve(time_limit=2, seed=1)

    assert solution.status in ("SATISFIABLE", "OPTIMAL")
    assert solution.lists[0].index(1) < solution.lists[0].index(2)
