import itertools

import pytest

cp = pytest.importorskip("qayd")


def solve_objective(items, build):
    model = cp.Model()
    (route,) = model.list_vars(items, count=1)
    model.minimize(build(route))
    solution = model.solve(engine="exact", time_limit=5)
    assert solution.status == "OPTIMAL"
    return solution


def test_item_slot_and_minimum_maximum_reductions():
    solution = solve_objective(
        [2, 5, 9],
        lambda route: cp.minimum(route, lambda item: item * 2) + cp.maximum(route, lambda item: item * 3),
    )
    assert solution.objective == 31


def test_edge_and_item_pair_slots():
    edge_solution = solve_objective(
        [2],
        lambda route: cp.sum_edges(route, lambda source, target: source * 100 + target, start=7, end=8),
    )
    assert edge_solution.objective == 910

    pair_solution = solve_objective(
        [1, 2],
        lambda route: cp.item_pairs(route, lambda left, right: left * 10 + right),
    )
    assert pair_solution.objective == 66


def test_position_pair_slots_cover_items_and_positions():
    items = [2, 5]
    expected = min(
        sum((left * 10 + right) * (left_pos * 10 + right_pos) for left_pos, left in enumerate(order) for right_pos, right in enumerate(order))
        for order in itertools.permutations(items)
    )
    solution = solve_objective(
        items,
        lambda route: cp.pos_pairs(
            route,
            lambda left, right, left_pos, right_pos: (left * 10 + right) * (left_pos * 10 + right_pos),
        ),
    )
    assert solution.objective == expected == 544


def test_scan_and_select_share_slot_layout_and_arena_order():
    def scan(route):
        return cp.scan_sum(
            route,
            step=lambda current, accumulator, previous: current * 100 + accumulator * 10 + previous,
            emit=lambda current, accumulator, previous: current * 10_000 + accumulator * 10 + previous,
            init=5,
            boundary=7,
        )

    def selected(route):
        return cp.select_kth(
            route,
            0,
            step=lambda current, accumulator, previous: current * 100 + accumulator * 10 + previous,
            emit=lambda current, accumulator, previous: current * 10_000 + accumulator * 10 + previous,
            init=5,
            boundary=7,
        )

    solution = solve_objective([2], lambda route: scan(route) + selected(route))
    assert solution.objective == 45_154


def test_window_inner_and_emit_slots_share_one_arena():
    items = [1, 2, 3]
    expected = min(
        sum(sum(order[start : start + 2]) ** 2 for start in range(len(order) - 1))
        for order in itertools.permutations(items)
    )
    solution = solve_objective(
        items,
        lambda route: cp.windows(route, 2, inner=lambda item: item, emit=lambda total: total * total),
    )
    assert solution.objective == expected == 25


def test_arithmetic_comparison_and_helper_nodes_lower_correctly():
    def body(item):
        return (
            item // 2
            + 20 // item
            + cp.div_scaled(item * 10, 2, 10)
            + cp.min(item, 3)
            + cp.abs(item - 3)
            + cp.ne(item, 3)
            + (item < 3)
            + (item <= 3)
            + (item > 3)
            + (item >= 3)
        )

    solution = solve_objective([2, 3, 4], lambda route: cp.sum(route, body))
    assert solution.objective == 493


def test_lambda_body_coercion_errors_are_explicit():
    model = cp.Model()
    (route,) = model.list_vars([1], count=1)

    with pytest.raises(TypeError, match="lambda body may only combine lambda expressions and integers"):
        cp.sum(route, lambda _item: object())

    with pytest.raises(TypeError, match="lambda body may only combine lambda expressions and integers"):
        cp.sum(route, lambda item: item + "invalid")


def test_modular_power_is_rejected_during_lambda_compilation():
    model = cp.Model()
    (route,) = model.list_vars([1], count=1)

    with pytest.raises(ValueError, match="modular power is not supported"):
        cp.sum(route, lambda item: pow(item, 2, 3))


@pytest.mark.parametrize("scale", [0, -1])
def test_scaled_operations_reject_nonpositive_scale(scale):
    with pytest.raises(ValueError, match="scale must be positive"):
        cp.mul_scaled(1, 2, scale)
    with pytest.raises(ValueError, match="scale must be positive"):
        cp.div_scaled(1, 2, scale)


@pytest.mark.parametrize("points", [[], [(1, 1), (1, 2)], [(2, 1), (1, 2)]])
def test_piecewise_rejects_empty_or_unsorted_points(points):
    with pytest.raises(ValueError, match="strictly increasing"):
        cp.piecewise(1, points)


def test_external_requires_prior_registration():
    with pytest.raises(ValueError, match="is not registered"):
        cp.external("qayd_test_lambda_dsl_missing_external", 1)
