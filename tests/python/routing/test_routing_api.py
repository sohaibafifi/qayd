"""Routing convenience API surface."""

import pytest

cp = pytest.importorskip("qayd")


def test_routes_sum_distance_preserves_basic_tsp_shape():
    model = cp.Model()
    customers = model.customers([1, 2, 3])
    dist = [
        [0, 1, 2, 3],
        [1, 0, 1, 2],
        [2, 1, 0, 1],
        [3, 2, 1, 0],
    ]

    routes = model.routes(customers, vehicles=1, depot=0, travel=dist)
    model.minimize(routes.sum(lambda route: route.distance()))

    solution = model.solve(time_limit=2, seed=1)

    assert solution.status == "OPTIMAL"
    assert sorted(solution.lists[0]) == [1, 2, 3]
    assert solution.objective == 6


def test_customer_start_constraints_are_posted_per_visit():
    model = cp.Model()
    customers = model.customers([1, 2])
    for customer in customers:
        customer.service = 0
        customer.earliest = 0
        customer.latest = 20
        customer.demand = 1
    customers[1].latest = 2

    # Serving 2 before 1 reaches customer 1 at time 10, violating latest=2.
    dist = [
        [0, 1, 5],
        [1, 0, 1],
        [1, 5, 0],
    ]
    routes = model.routes(customers, vehicles=1, depot=0, travel=dist)

    for customer in customers:
        model.add(routes[customer].start >= customer.earliest)
        model.add(routes[customer].start <= customer.latest)
    for route in routes:
        model.add(route.sum(lambda customer: customer.demand) <= 2)
    model.add(routes[1].position < routes[2].position)
    model.minimize(routes.total_distance())

    solution = model.solve(time_limit=2, seed=1)

    assert solution.status in ("SATISFIABLE", "OPTIMAL")
    route = solution.lists[0]
    assert route.index(1) < route.index(2)
