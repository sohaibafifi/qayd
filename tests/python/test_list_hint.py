"""Warm-start the LS list engine from a caller-supplied initial sequence per list."""

import pytest

cp = pytest.importorskip("qayd")


def _edge_cost(seq, rows):
    if not seq:
        return 0
    tour = [0] + list(seq) + [0]
    return sum(rows[tour[k]][tour[k + 1]] for k in range(len(tour) - 1))


def _model(n, a, optional=True):
    rows = [[abs(i - j) + 1 for j in range(n)] for i in range(n)]
    m = cp.Model()
    d = cp.matrix(rows)
    routes = m.list_vars(list(range(n)), count=a, optional=optional)
    m.minimize(cp.max_of([cp.sum_edges(r, lambda i, j: d[i][j], start=0, end=0) for r in routes]))
    return m, rows


def test_hint_seeds_incumbent_never_worse_than_hint():
    # The hint is the starting incumbent; LS only improves, so the result's
    # objective is never worse than the hint's own objective.
    n, a = 60, 6
    m, rows = _model(n, a)
    per = (n + a - 1) // a
    hint = [list(range(1 + i * per, min(1 + (i + 1) * per, n))) for i in range(a)]
    hint_obj = max(_edge_cost(s, rows) for s in hint)
    sol = m.solve(engine="ls", time_limit=2, seed=1, list_hint=hint)
    assert sol.status in ("SATISFIABLE", "OPTIMAL")
    assert sol.objective <= hint_obj


def test_hint_is_the_starting_incumbent():
    # With zero search iterations the solver returns exactly the seeded partition,
    # proving the hint becomes the initial incumbent rather than a random greedy
    # start. A full round-robin partition (every node named) leaves no leftovers.
    n, a = 30, 3
    m, _ = _model(n, a, optional=False)
    hint = [[v for v in range(n) if v % a == i] for i in range(a)]
    sol = m.solve(engine="ls", time_limit=5, seed=1, list_hint=hint, max_iterations=0)
    assert [sorted(r) for r in sol.lists] == [sorted(h) for h in hint]


def test_hint_completes_partition_with_unnamed_nodes():
    # A hint that names only some nodes is still a valid, complete start: the
    # unnamed nodes are pooled (optional model), so every node stays accounted for.
    n, a = 20, 3
    m, _ = _model(n, a, optional=True)
    sol = m.solve(engine="ls", time_limit=1, seed=1, list_hint=[[1, 2, 3], [4, 5]])
    assert sol.status in ("SATISFIABLE", "OPTIMAL")
    served = {c for route in sol.lists for c in route}
    assert served == set(range(n))  # every node lands in exactly one (real or pool) list


def test_hint_validation_rejects_bad_input():
    m, _ = _model(10, 2)
    with pytest.raises(ValueError, match="more than one list"):
        m.solve(engine="ls", time_limit=1, list_hint=[[1, 2], [1, 3]])
    with pytest.raises(ValueError, match="not in the list_vars universe"):
        m.solve(engine="ls", time_limit=1, list_hint=[[1, 2], [99]])
    with pytest.raises(ValueError, match="sequences but the model has"):
        m.solve(engine="ls", time_limit=1, list_hint=[[1], [2], [3], [4]])


def test_list_hint_rejected_for_integer_model():
    m = cp.Model()
    x = m.int_var(0, 5, name="x")
    m.minimize(x)
    with pytest.raises(ValueError, match="only supported for list_vars models"):
        m.solve(engine="ls", time_limit=1, list_hint=[[1]])
