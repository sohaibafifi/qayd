import qayd as cp


def test_unordered_sets_rich_relations_and_unbounded_lexicographic_tiers():
    model = cp.Model()
    sets = model.set_vars([1, 2, 3, 4], 4)
    model.all_same_list([1, 2])
    model.different_list(2, 3)
    model.all_different_lists([1, 3, 4])
    model.list_distance(1, 3, min=1, max=2)
    model.precedence(1, 4)

    model.minimize(cp.sum(sets[0], lambda item: item))
    for tier in range(1, 10):
        model.then_minimize(cp.sum(sets[tier % 4], lambda item: item * tier))

    solution = model.solve(engine="exact", time_limit=5)
    assert solution.status == "OPTIMAL"
    assert len(solution.objectives) == 10
    assert all(values == sorted(values) for values in solution.lists)

    owner = {item: index for index, values in enumerate(solution.lists) for item in values}
    assert owner[1] == owner[2]
    assert owner[2] != owner[3]
    assert len({owner[1], owner[3], owner[4]}) == 3
    assert 1 <= abs(owner[1] - owner[3]) <= 2


def test_fixed_nonlinear_piecewise_and_external_python_surface():
    name = "phase7_python_double_v1"
    cp.register_external(name, lambda value: value * 2)

    model = cp.Model()
    (items,) = model.set_vars([2], 1)
    fixed_product = lambda: cp.mul_scaled(cp.fixed(1.5, scale=1_000), cp.fixed(2.0, scale=1_000), 1_000)
    objective = cp.sum(
        items,
        lambda item: item**3
        + fixed_product()
        + cp.piecewise(item, [(0, 0), (10, 100)])
        + cp.external(name, item)
        + 11 % item,
    )
    model.minimize(objective)

    solution = model.solve(engine="exact", time_limit=5)
    assert solution.status == "OPTIMAL"
    assert solution.objective == 3_033


def test_optional_interval_presence_is_returned_by_exact_scheduling():
    model = cp.Model()
    mandatory = model.interval(3, 10, name="mandatory")
    optional = model.interval(5, 10, optional=True, name="optional")
    model.minimize_makespan([mandatory, optional])

    solution = model.solve(engine="exact", time_limit=5)
    assert solution.status == "OPTIMAL"
    assert solution.objective == 3
    assert solution.presences == [True, False]
    assert solution.starts[0] is not None
    assert solution.starts[1] is None
