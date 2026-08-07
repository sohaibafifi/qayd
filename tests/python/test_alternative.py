"""The `alternative` constraint over caller-created optional intervals.

`m.alternative(members)` returns a master synchronised with the single member
that executes: `.start` is the shared start, `.end`/`.realized_duration`
follow the chosen member. Members stay first-class, so machine no-overlap and
presence-based costs are composed directly on them.
"""
import itertools

import pytest

cp = pytest.importorskip("qayd")


def make_task(model, durations, horizon, name):
    """One task, one optional member per duration; returns (master, members)."""
    members = [model.interval(d, horizon, optional=True, name=f"{name}.m{k}") for k, d in enumerate(durations)]
    return model.alternative(members, name=name), members


def test_basic_two_members_coherent():
    model = cp.Model()
    task, members = make_task(model, [2, 5], horizon=10, name="t")
    sol = model.solve()
    assert sol.status in ("SATISFIABLE", "OPTIMAL")
    presences = [sol.value(m.presence) for m in members]
    assert sorted(presences) == [0, 1]
    chosen = presences.index(1)
    assert sol.value(members[chosen].start) == sol.value(task.start)


def test_short_member_allows_late_start():
    # Regression: with an unconditional start equality, the long member's
    # horizon bound would cap the master even when the short one is chosen.
    model = cp.Model()
    task, members = make_task(model, [2, 8], horizon=10, name="t")
    model.add(members[0].presence == 1)  # choose the short member
    model.add(task.start >= 7)  # legal only because the short member fits
    sol = model.solve()
    assert sol.status in ("SATISFIABLE", "OPTIMAL")
    assert sol.value(task.start) >= 7
    assert sol.value(members[0].presence) == 1
    assert sol.value(members[1].presence) == 0


def _brute_force_optimum(dur, weight):
    best = None
    for assign in itertools.product((0, 1), repeat=len(dur)):
        by_machine = {0: [], 1: []}
        for i, mode in enumerate(assign):
            by_machine[mode].append(dur[i][mode])
        total_completion = 0
        for lengths in by_machine.values():
            clock = 0
            for length in sorted(lengths):  # SPT minimises ΣC_j on one machine
                clock += length
                total_completion += clock
        humans = sum(1 for mode in assign if mode == 0)
        obj = total_completion + weight * humans
        best = obj if best is None else min(best, obj)
    return best


def test_owner_use_case_matches_brute_force():
    # HCR-AMR-OSP shape: 3 tasks, member 0 = human, member 1 = robot, with
    # different durations; machine no-overlap composed directly on members;
    # custom objective = Σ end + w·Σ human-presence.
    dur = [[3, 5], [2, 4], [4, 2]]  # [human, robot] per task
    weight = 10
    model = cp.Model()
    tasks = [make_task(model, ds, horizon=30, name=f"t{i}") for i, ds in enumerate(dur)]
    model.no_overlap([members[0] for _, members in tasks])  # human machine
    model.no_overlap([members[1] for _, members in tasks])  # robot machine
    objective = cp.sum(
        [master.end for master, _ in tasks] + [weight * members[0].presence for _, members in tasks]
    )
    model.minimize(objective)

    sol = model.solve()
    assert sol.status == "OPTIMAL"
    assert sol.objective == _brute_force_optimum(dur, weight)


def test_end_usable_as_deadline():
    model = cp.Model()
    task, members = make_task(model, [2, 5], horizon=10, name="t")
    model.add(task.end <= 3)
    sol = model.solve()
    assert sol.status in ("SATISFIABLE", "OPTIMAL")
    assert sol.value(members[0].presence) == 1
    assert sol.value(members[1].presence) == 0


def test_alternative_getters_and_errors():
    model = cp.Model()
    plain = model.interval(2, 10)
    assert plain.members == []
    with pytest.raises(ValueError, match="alternative master"):
        _ = plain.realized_duration

    task, members = make_task(model, [2, 5], horizon=10, name="t")
    assert task.index is None
    assert len(task.members) == 2
    assert "t.end" in repr(task.end)
    with pytest.raises(ValueError, match="schedule its members"):
        model.no_overlap([task])
    with pytest.raises(ValueError, match="must be optional"):
        model.alternative([model.interval(2, 10)])
    with pytest.raises(ValueError, match="distinct"):
        model.alternative([members[0], members[0]])
    with pytest.raises(ValueError, match="at least one"):
        model.alternative([])


def test_solves_optimal_at_large_horizon():
    # The bounds channel must keep optimality proofs closing at realistic
    # horizons (the generic-implication encoding did not).
    dur = [(3, 5), (4, 6), (5, 7), (6, 8)]
    weight = 3
    model = cp.Model()
    tasks = [make_task(model, ds, horizon=2000, name=f"t{i}") for i, ds in enumerate(dur)]
    model.no_overlap([members[0] for _, members in tasks])
    model.no_overlap([members[1] for _, members in tasks])
    objective = cp.sum(
        [master.end for master, _ in tasks] + [weight * members[0].presence for _, members in tasks]
    )
    model.minimize(objective)
    sol = model.solve(time_limit=10)
    assert sol.status == "OPTIMAL", sol.status
