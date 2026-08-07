"""Routing named resource API."""

import pytest

cp = pytest.importorskip("qayd")


def test_route_resource_before_view_can_force_order():
    model = cp.Model()
    customers = model.customers([1, 2])
    for customer in customers:
        customer.recharge = 0

    travel = [
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ]
    energy = cp.matrix(
        [
            [0, 1, 1],
            [1, 0, 1],
            [1, 4, 0],
        ]
    )

    routes = model.routes(customers, vehicles=1, depot=0, travel=travel)
    fuel = routes.resource(
        "fuel",
        initial=3,
        bounds=(0, 3),
        consume=lambda prev, cur: energy[prev.id][cur.id],
        refill=lambda cur: cur.recharge,
    )

    model.add(fuel.within_bounds())
    for customer in customers:
        model.add(routes[customer].fuel.before >= 0)
    model.minimize(routes.total_distance())

    solution = model.solve(time_limit=2, seed=1)

    assert solution.status in ("SATISFIABLE", "OPTIMAL")
    assert solution.lists[0].index(1) < solution.lists[0].index(2)
