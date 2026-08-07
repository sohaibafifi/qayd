"""Cross-epoch SolveSession guarantees.

The session keeps learned nogoods across epochs and re-injects them by raw
atom id, so the eager atom layout must be frozen at session creation: a
per-epoch `search` may only guide branching, never change which fact an atom
id denotes (silent nogood reinterpretation) nor the eager atom count
(`set_base` layout panic).
"""
import pytest

cp = pytest.importorskip("qayd")


def build_perm_model(n=7, minimize=True):
    """n all-different vars over 0..n-1 with a weighted-sum objective: proving
    optimality walks through real conflicts, so the session keeps nogoods."""
    model = cp.Model()
    xs = [model.int_var(0, n - 1, name=f"x{i}") for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            model.add(xs[i] != xs[j])
    if minimize:
        model.minimize(sum((i + 1) * xs[i] for i in range(n)))
    return model, xs


def check_perm(solution, xs):
    vals = [solution.value(x) for x in xs]
    assert sorted(vals) == list(range(len(xs))), vals
    return vals


def test_search_subsets_keep_atom_layout_stable():
    # Regression: solving with the default search then a strict subset used to
    # rebuild a DIFFERENT eager atom layout while the session pool pinned the
    # first one — panicking in set_base, or silently re-reading stored nogoods
    # as facts about other variables when the counts happened to match.
    model = cp.Model()
    xs = [model.int_var(0, 2, name=f"v{i}") for i in range(3)]
    model.add(xs[1] != 1)  # only the middle var carries a constraint

    session = model.session()
    first = session.solve()
    assert first.status in ("SATISFIABLE", "OPTIMAL")
    assert first.value(xs[1]) != 1

    # A solution only carries the searched variables; assert on those.
    for subset in ([xs[0], xs[1]], [xs[1], xs[2]], [xs[2]]):
        solution = session.solve(search=subset)
        assert solution.status in ("SATISFIABLE", "OPTIMAL"), solution.status
        # `in` would call the overloaded `==` (which builds a constraint):
        # compare by identity instead.
        if any(v is xs[1] for v in subset):
            assert solution.value(xs[1]) != 1


def test_multi_epoch_assumptions_reuse_clauses_soundly():
    model, xs = build_perm_model()
    session = model.session()

    # Epoch 1: prove the free optimum — the B&B proof learns real nogoods
    # (perm-sum optimality takes hundreds of conflicts at n=7).
    base = session.solve()
    assert base.status == "OPTIMAL"
    base_vals = check_perm(base, xs)
    base_obj = base.objective
    assert session.learned_nogoods > 0, "the optimality proof must keep nogoods"

    # Epoch 2: assumptions + a different search subset — kept nogoods must not
    # poison it: still solvable, genuinely valid, objective no better than the
    # free optimum.
    pinned = session.solve(assumptions=[xs[0] == base_vals[0]])
    assert pinned.status == "OPTIMAL", pinned.status
    check_perm(pinned, xs)
    assert pinned.objective >= base_obj

    # Epoch 3: an assumption set that is infeasible (duplicate values under
    # allDifferent): must report UNSAT-under-assumptions, not crash.
    clash = session.solve(assumptions=[xs[0] == 0, xs[1] == 0])
    assert clash.status == "UNSATISFIABLE"

    # Epoch 4: back to free — epoch-3's assumptions must not linger, and the
    # optimum must be reproduced exactly.
    free = session.solve()
    assert free.status == "OPTIMAL"
    check_perm(free, xs)
    assert free.objective == base_obj

    # Epoch 5: subset search under fresh assumptions still sound.
    sub = session.solve(search=[xs[0], xs[3], xs[6]], assumptions=[xs[6] == 6])
    assert sub.status in ("SATISFIABLE", "OPTIMAL"), sub.status
    assert sub.value(xs[6]) == 6


def test_session_solve_releases_gil_and_reattaches_for_callback():
    # The solve region must run detached (background Python threads keep
    # ticking), while the incumbent callback reattaches per invocation.
    import threading
    import time

    model, xs = build_perm_model(n=9)
    session = model.session()

    ticks = []
    done = threading.Event()

    def heartbeat():
        while not done.is_set():
            ticks.append(None)
            time.sleep(0.005)

    incumbents = []
    thread = threading.Thread(target=heartbeat)
    thread.start()
    try:
        solution = session.solve(
            time_limit=1,
            on_incumbent=lambda value, assignment: incumbents.append(value),
        )
    finally:
        done.set()
        thread.join()

    assert solution.status in ("SATISFIABLE", "OPTIMAL"), solution.status
    assert incumbents, "the incumbent callback must fire"
    assert len(ticks) > 10, f"background thread starved: {len(ticks)} ticks"
