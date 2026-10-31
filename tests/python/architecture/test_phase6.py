import pytest

cp = pytest.importorskip("qayd")


def _cvrp_model():
    costs = [
        [0, 10, 10, 10, 10],
        [10, 0, 1, 1, 1],
        [10, 1, 0, 1, 1],
        [10, 1, 1, 0, 1],
        [10, 1, 1, 1, 0],
    ]
    distance = cp.matrix(costs)
    demand = cp.array([0, 1, 1, 1, 1])
    model = cp.Model()
    routes = model.list_vars([1, 2, 3, 4], count=2)
    model.minimize(cp.sum(cp.sum_edges(route, lambda i, j: distance[i][j], start=0, end=0) for route in routes))
    for route in routes:
        model.add(cp.sum(route, lambda item: demand[item]) <= 2)
    return model


def test_ls_exposes_certified_vrp_dual_and_gap():
    solution = _cvrp_model().solve(engine="ls", time_limit=2, seed=1)
    assert solution.status == "SATISFIABLE"
    assert solution.dual_bound == 41
    assert solution.dual_bound <= solution.objective
    assert solution.absolute_gap == solution.objective - solution.dual_bound
    assert solution.relative_gap == pytest.approx(solution.absolute_gap / solution.objective)
    assert solution.bound_method == "stabilized VRP column generation"


def test_exact_collection_reports_closed_gap():
    solution = _cvrp_model().solve(engine="exact", time_limit=10, seed=1)
    assert solution.status == "OPTIMAL"
    assert solution.dual_bound == solution.objective
    assert solution.absolute_gap == 0
    assert solution.relative_gap == 0.0
    assert "exact" in solution.bound_method


def test_schedule_ls_reports_critical_path_bound():
    model = cp.Model()
    intervals = model.alternatives([[(0, 3)], [(1, 4)]], horizon=20)
    model.precedence(intervals[0], intervals[1])
    model.minimize_makespan(intervals)
    solution = model.solve(engine="ls", time_limit=2, seed=1)
    assert solution.status == "SATISFIABLE"
    assert solution.dual_bound == 7
    assert solution.dual_bound <= solution.objective
    assert solution.bound_method == "critical-path/resource relaxation"


def test_generic_exact_api_solution_closes_the_gap():
    model = cp.Model()
    value = model.int_var(0, 5, name="value")
    model.minimize(value)
    solution = model.solve(time_limit=2)
    assert solution.status == "OPTIMAL"
    assert solution.objective == 0
    assert solution.dual_bound == 0
    assert solution.absolute_gap == 0
    assert solution.relative_gap == 0.0
    assert solution.bound_method == "exact proof"
