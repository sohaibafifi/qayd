"""G8 acceptance matrix: the exact engine must solve+prove routing models that
combine optional lists, edge sums, scan constraints, and a scan value reified
into a linear objective. Every line asserts the exact engine reaches OPTIMAL and
matches an independent brute-force oracle.
"""

from itertools import permutations, product

import pytest

cp = pytest.importorskip("qayd")

CUST = [1, 2, 3]
DEPOT = 0
DIST = [
    [0, 2, 9, 5],
    [2, 0, 3, 7],
    [9, 3, 0, 4],
    [5, 7, 4, 0],
]
DEMAND = [0, 1, 2, 2]  # indexed by node id


# --- brute-force oracle -----------------------------------------------------


def _edge_cost(route):
    nodes = [DEPOT] + list(route) + [DEPOT]
    return sum(DIST[a][b] for a, b in zip(nodes, nodes[1:]))


def _scan_prefix_load(route):
    """Sum over stops of the cumulative load after each stop (order dependent)."""
    acc, total = 0, 0
    for cur in route:
        acc += DEMAND[cur]
        total += acc
    return total


def _assignments(count, optional):
    slots = list(range(count)) + ([-1] if optional else [])  # -1 == the pool
    for assign in product(slots, repeat=len(CUST)):
        routes = [[] for _ in range(count)]
        for cust, slot in zip(CUST, assign):
            if slot >= 0:
                routes[slot].append(cust)
        yield routes


def _orderings(routes):
    for perm in product(*[list(permutations(r)) for r in routes]):
        yield [list(p) for p in perm]


def brute_force(count, optional, objective, feasible=lambda routes: True, minimize=True):
    best = None
    for routes in _assignments(count, optional):
        for ordered in _orderings(routes):
            if not feasible(ordered):
                continue
            value = objective(ordered)
            if best is None or (value < best if minimize else value > best):
                best = value
    return best


# --- lambdas (mirror the oracle) --------------------------------------------


def _dist_matrix():
    return cp.matrix(DIST)


def _load_scan(route):
    demand = cp.array(DEMAND)
    return cp.scan_sum(
        route,
        lambda cur, acc, _prev: acc + demand[cur],
        lambda _cur, acc, _prev: acc,
        init=0,
        boundary=0,
    )


# --- the seven matrix lines -------------------------------------------------


def test_line1_sum_edges_max_of_non_optional():
    model = cp.Model()
    r0, r1 = model.list_vars(CUST, count=2)
    dist = _dist_matrix()
    model.minimize(
        cp.max_of(
            [
                cp.sum_edges(r0, lambda i, j: dist[i][j], start=DEPOT, end=DEPOT),
                cp.sum_edges(r1, lambda i, j: dist[i][j], start=DEPOT, end=DEPOT),
            ]
        )
    )
    sol = model.solve(engine="exact", time_limit=10)
    expected = brute_force(2, False, lambda routes: max(_edge_cost(r) for r in routes))
    assert sol.status == "OPTIMAL"
    assert sol.objective == expected


def test_line2_sum_edges_max_of_optional():
    model = cp.Model()
    r0, r1 = model.list_vars(CUST, count=2, optional=True)
    dist = _dist_matrix()
    model.minimize(
        cp.max_of(
            [
                cp.sum_edges(r0, lambda i, j: dist[i][j], start=DEPOT, end=DEPOT),
                cp.sum_edges(r1, lambda i, j: dist[i][j], start=DEPOT, end=DEPOT),
            ]
        )
    )
    sol = model.solve(engine="exact", time_limit=10)
    expected = brute_force(2, True, lambda routes: max(_edge_cost(r) for r in routes))
    assert sol.status == "OPTIMAL"
    assert sol.objective == expected


def test_line3_sum_edges_linear_non_optional():
    model = cp.Model()
    (route,) = model.list_vars(CUST, count=1)
    dist = _dist_matrix()
    model.minimize(cp.sum_edges(route, lambda i, j: dist[i][j], start=DEPOT, end=DEPOT))
    sol = model.solve(engine="exact", time_limit=10)
    expected = brute_force(1, False, lambda routes: _edge_cost(routes[0]))
    assert sol.status == "OPTIMAL"
    assert sol.objective == expected


def test_line4_sum_edges_linear_optional():
    # Optional single route + a per-dropped-item penalty, so leaving a customer
    # in the pool trades against the edge cost.
    penalty = 4
    model = cp.Model()
    (route,) = model.list_vars(CUST, count=1, optional=True)
    dist = _dist_matrix()
    served = cp.sum(route, lambda _i: 1)
    model.minimize(cp.sum_edges(route, lambda i, j: dist[i][j], start=DEPOT, end=DEPOT) + (-penalty) * served)

    def objective(routes):
        return _edge_cost(routes[0]) - penalty * len(routes[0])

    sol = model.solve(engine="exact", time_limit=10)
    expected = brute_force(1, True, objective)
    assert sol.status == "OPTIMAL"
    assert sol.objective == expected


def test_line5_sum_edges_scan_constraint_non_optional():
    cap = 10
    model = cp.Model()
    (route,) = model.list_vars(CUST, count=1)
    dist = _dist_matrix()
    model.add(_load_scan(route) <= cap)
    model.minimize(cp.sum_edges(route, lambda i, j: dist[i][j], start=DEPOT, end=DEPOT))

    def feasible(routes):
        return _scan_prefix_load(routes[0]) <= cap

    sol = model.solve(engine="exact", time_limit=10)
    expected = brute_force(1, False, lambda routes: _edge_cost(routes[0]), feasible)
    assert sol.status == "OPTIMAL"
    assert sol.objective == expected


def test_line6_sum_edges_scan_constraint_max_of_optional():
    cap = 6
    model = cp.Model()
    r0, r1 = model.list_vars(CUST, count=2, optional=True)
    dist = _dist_matrix()
    model.add(_load_scan(r0) <= cap)
    model.add(_load_scan(r1) <= cap)
    model.minimize(
        cp.max_of(
            [
                cp.sum_edges(r0, lambda i, j: dist[i][j], start=DEPOT, end=DEPOT),
                cp.sum_edges(r1, lambda i, j: dist[i][j], start=DEPOT, end=DEPOT),
            ]
        )
    )

    def feasible(routes):
        return all(_scan_prefix_load(r) <= cap for r in routes)

    sol = model.solve(engine="exact", time_limit=10)
    expected = brute_force(2, True, lambda routes: max(_edge_cost(r) for r in routes), feasible)
    assert sol.status == "OPTIMAL"
    assert sol.objective == expected


def test_line7_scan_in_linear_objective_non_optional():
    model = cp.Model()
    (route,) = model.list_vars(CUST, count=1)
    model.minimize(_load_scan(route))
    sol = model.solve(engine="exact", time_limit=10)
    expected = brute_force(1, False, lambda routes: _scan_prefix_load(routes[0]))
    assert sol.status == "OPTIMAL"
    assert sol.objective == expected


def test_line7_scan_reified_then_linear_objective_non_optional():
    # Same optimum, via the mandated reification primitive.
    model = cp.Model()
    (route,) = model.list_vars(CUST, count=1)
    value = model.list_value()
    model.add(_load_scan(route) == value)
    model.minimize(value)
    sol = model.solve(engine="exact", time_limit=10)
    expected = brute_force(1, False, lambda routes: _scan_prefix_load(routes[0]))
    assert sol.status == "OPTIMAL"
    assert sol.objective == expected


def test_combined_optional_edges_scan_constraint_reified_scan_objective():
    # The capstone: optional list + sum_edges + a scan CONSTRAINT + a scan value
    # reified into a linear objective term.
    cap = 6
    model = cp.Model()
    (route,) = model.list_vars(CUST, count=1, optional=True)
    dist = _dist_matrix()

    model.add(_load_scan(route) <= cap)
    lateness = model.list_value()
    model.add(_load_scan(route) == lateness)
    model.minimize(cp.sum_edges(route, lambda i, j: dist[i][j], start=DEPOT, end=DEPOT) + lateness)

    def feasible(routes):
        return _scan_prefix_load(routes[0]) <= cap

    def objective(routes):
        return _edge_cost(routes[0]) + _scan_prefix_load(routes[0])

    sol = model.solve(engine="exact", time_limit=10)
    expected = brute_force(1, True, objective, feasible)
    assert sol.status == "OPTIMAL"
    assert sol.objective == expected


# --- reification API surface ------------------------------------------------


def test_reification_accepts_eq_and_rejects_inequality():
    model = cp.Model()
    (route,) = model.list_vars(CUST, count=1)
    v = model.list_value()
    # `== value` reifies (a definition); it must not raise the integer-bound error.
    model.add(_load_scan(route) == v)
    # An inequality reification would need `v` to be a bounded optimisation
    # variable the enumeration does not range over, so it is rejected outright
    # rather than silently treated as `==`.
    with pytest.raises(ValueError, match="inequality reification is not supported"):
        _ = _load_scan(route) <= v
    with pytest.raises(ValueError, match="inequality reification is not supported"):
        _ = _load_scan(route) >= v


def test_reified_value_composes_in_weighted_sum():
    model = cp.Model()
    (route,) = model.list_vars(CUST, count=1)
    dist = _dist_matrix()
    v = model.list_value()
    model.add(_load_scan(route) == v)
    term = cp.sum_edges(route, lambda i, j: dist[i][j], start=DEPOT, end=DEPOT) + 2 * v
    model.minimize(term)
    sol = model.solve(engine="exact", time_limit=10)
    expected = brute_force(1, False, lambda routes: _edge_cost(routes[0]) + 2 * _scan_prefix_load(routes[0]))
    assert sol.status == "OPTIMAL"
    assert sol.objective == expected


def test_list_value_identity_uses_is():
    model = cp.Model()
    model.list_vars(CUST, count=1)
    v = model.list_value()
    w = model.list_value()
    assert v is v
    assert v is not w
