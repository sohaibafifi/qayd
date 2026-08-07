"""Closing `end` boundary on a route scan: bound a resource over the CLOSED tour.

The battery must reach the depot (`end=0`) with charge >= 0, resetting to full at
chargers. The direct return is infeasible on a tight battery; a charger-mediated
return is feasible. `scan_sum(..., end=0)` folds the return edge, so exact search
matches a brute-force enumerator -- and rejects the over-optimistic route that an
end-less scan (which never sees the return arc) would accept.
"""

import itertools

import pytest

cp = pytest.importorskip("qayd")

# 0=depot, 1=customer (required), 2=charger (a costly detour beyond the customer).
POS = [0, 3, 5]
N = len(POS)
D = [[abs(POS[i] - POS[j]) for j in range(N)] for i in range(N)]
IS_CHARGER = [0, 0, 1]
REQ = [0, 1, 0]
CAP = 5
MOVE = 1
REWARD = 100


def _battery_ok(order, *, close):
    # acc starts full, drains MOVE*dist per move, resets to CAP at a charger. With
    # close=True the return edge to the depot (0) is folded in, matching end=0.
    acc, prev = CAP, 0
    for cur in (list(order) + [0]) if close else list(order):
        acc = CAP if IS_CHARGER[cur] else acc - MOVE * D[prev][cur]
        if acc < 0:
            return False
        prev = cur
    return True


def _brute(*, close):
    best = None
    for r in range(N):
        for subset in itertools.combinations([1, 2], r):
            for order in itertools.permutations(subset):
                if not _battery_ok(order, close=close):
                    continue
                tour = [0] + list(order) + [0]
                edge = sum(D[tour[k]][tour[k + 1]] for k in range(len(tour) - 1))
                obj = edge - REWARD * sum(REQ[c] for c in order)
                if best is None or obj < best:
                    best = obj
    assert best is not None  # the empty route is always feasible, so a best exists
    return best


def _solve(*, end):
    Dm, REQm, CHm = cp.matrix(D), cp.array(REQ), cp.array(IS_CHARGER)
    m = cp.Model()
    route = m.list_vars([1, 2], count=1, optional=True)[0]
    m.add(
        cp.scan_sum(
            route,
            step=lambda cur, acc, prev: cp.if_(CHm[cur], CAP, acc - MOVE * Dm[prev][cur]),
            emit=lambda cur, lvl, prev: cp.max(0, -lvl),  # shortfall below 0
            init=CAP,
            boundary=0,
            end=end,
        )
        <= 0
    )
    m.minimize(cp.sum_edges(route, lambda i, j: Dm[i][j], start=0, end=0) + cp.sum(route, lambda i: REQm[i]) * (-REWARD))
    return m.solve(engine="exact", time_limit=10)


def test_battery_close_matches_brute_and_visits_charger():
    brute = _brute(close=True)
    sol = _solve(end=0)
    assert sol.status == "OPTIMAL"
    assert sol.objective == brute  # exact-with-end == brute optimum over the closed tour
    served = sol.lists[0]
    assert 1 in served  # the required customer is served
    assert 2 in served  # only a charger-mediated return is feasible on the tight battery
    # The direct return the customer alone would take is genuinely infeasible.
    assert not _battery_ok((1,), close=True)


def test_end_bites_the_optimum():
    # An end-less scan never sees the return arc, so it would accept the cheaper but
    # unreturnable direct route -- a strictly better (over-optimistic) optimum. The
    # closed-tour optimum must be strictly worse, i.e. `end` actually constrains.
    assert _brute(close=True) > _brute(close=False)
    assert _solve(end=0).objective == _brute(close=True)


def test_ls_engine_respects_closing_battery():
    # The LS fold also sees the closing edge, so local search returns a route that
    # is battery-feasible over the CLOSED tour (violation includes the return arc).
    Dm, REQm, CHm = cp.matrix(D), cp.array(REQ), cp.array(IS_CHARGER)
    m = cp.Model()
    route = m.list_vars([1, 2], count=1, optional=True)[0]
    m.add(
        cp.scan_sum(
            route,
            step=lambda cur, acc, prev: cp.if_(CHm[cur], CAP, acc - MOVE * Dm[prev][cur]),
            emit=lambda cur, lvl, prev: cp.max(0, -lvl),
            init=CAP,
            boundary=0,
            end=0,
        )
        <= 0
    )
    m.minimize(cp.sum_edges(route, lambda i, j: Dm[i][j], start=0, end=0) + cp.sum(route, lambda i: REQm[i]) * (-REWARD))
    sol = m.solve(engine="ls", time_limit=2, seed=1)
    assert _battery_ok(sol.lists[0], close=True)  # LS never returns an unreturnable route as feasible


def test_scan_end_omitted_is_unchanged():
    # Without `end`, the scan stops at the last item: the direct-return route is
    # accepted (battery only checked at the customer), matching the end-less brute.
    sol = _solve(end=None)
    assert sol.status == "OPTIMAL"
    assert sol.objective == _brute(close=False)
