"""CSPLib prob040: multi-level distribution with Wagner-Whitin costs.

Specification: https://www.csplib.org/Problems/prob040/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args

DEFAULT_INSTANCE = {
    "parents": [-1, 0, 0],
    "holding_costs": [1, 2, 2],
    "order_costs": [3, 2, 2],
    "demands": [[0, 0, 0], [2, 0, 1], [0, 2, 1]],
}


@dataclass(frozen=True)
class DistributionInstance:
    parents: tuple[int, ...]
    holding_costs: tuple[int, ...]
    order_costs: tuple[int, ...]
    demands: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class DistributionModel:
    model: cp.Model
    instance: DistributionInstance
    orders: list[list[cp.IntVar]]
    inventory: list[list[cp.IntVar]]


def parse_instance(data: str | bytes) -> DistributionInstance:
    raw = json.loads(data)
    try:
        return DistributionInstance(
            tuple(int(value) for value in raw["parents"]),
            tuple(int(value) for value in raw["holding_costs"]),
            tuple(int(value) for value in raw["order_costs"]),
            tuple(tuple(int(value) for value in row) for row in raw["demands"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid distribution JSON instance") from error


def load_instance(path: str | Path) -> DistributionInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def build_model(instance: DistributionInstance) -> DistributionModel:
    node_count = len(instance.parents)
    if (
        node_count < 1
        or len(instance.holding_costs) != node_count
        or len(instance.order_costs) != node_count
    ):
        raise ValueError("stocking-point arrays have inconsistent dimensions")
    if len(instance.demands) != node_count or not instance.demands[0]:
        raise ValueError("demand rows have inconsistent dimensions")
    periods = len(instance.demands[0])
    if any(len(row) != periods for row in instance.demands):
        raise ValueError("demand rows have inconsistent horizons")
    roots = [node for node, parent in enumerate(instance.parents) if parent == -1]
    if len(roots) != 1 or any(
        parent >= node_count or parent == node
        for node, parent in enumerate(instance.parents)
    ):
        raise ValueError("parents must define one rooted supply tree")
    if any(value < 0 for value in (*instance.holding_costs, *instance.order_costs)):
        raise ValueError("costs must be non-negative")
    if any(demand < 0 for row in instance.demands for demand in row):
        raise ValueError("demands must be non-negative")
    total_demand = max(1, sum(demand for row in instance.demands for demand in row))
    children = [
        [node for node, parent in enumerate(instance.parents) if parent == current]
        for current in range(node_count)
    ]
    model = cp.Model()
    orders = [
        model.int_vars(periods, 0, total_demand, name=f"orders_{node}")
        for node in range(node_count)
    ]
    inventory = [
        model.int_vars(periods, 0, total_demand, name=f"inventory_{node}")
        for node in range(node_count)
    ]
    fixed_cost_terms = []
    holding_terms = []
    for node in range(node_count):
        for period in range(periods):
            previous = 0 if period == 0 else inventory[node][period - 1]
            outgoing = sum(orders[child][period] for child in children[node])
            model.add(
                inventory[node][period]
                == previous
                + orders[node][period]
                - instance.demands[node][period]
                - outgoing
            )
            used = model.bool_var(name=f"order_used_{node}_{period}")
            model.table(
                [orders[node][period], used],
                [(quantity, int(quantity > 0)) for quantity in range(total_demand + 1)],
            )
            fixed_cost_terms.append(instance.order_costs[node] * used)
            holding_terms.append(instance.holding_costs[node] * inventory[node][period])
    model.minimize(sum(fixed_cost_terms) + sum(holding_terms))
    return DistributionModel(model, instance, orders, inventory)


def decode(
    built: DistributionModel, solution: cp.Solution
) -> tuple[list[list[int]], list[list[int]]]:
    if not solution.is_sat():
        raise RuntimeError(f"no solution available, solver status is {solution.status}")
    orders = [[solution.value(variable) for variable in row] for row in built.orders]
    inventory = [
        [solution.value(variable) for variable in row] for row in built.inventory
    ]
    return orders, inventory


def validate(
    built: DistributionModel,
    orders: list[list[int]],
    inventory: list[list[int]],
    objective: int | None,
) -> None:
    node_count = len(built.instance.parents)
    periods = len(built.instance.demands[0])
    children = [
        [
            node
            for node, parent in enumerate(built.instance.parents)
            if parent == current
        ]
        for current in range(node_count)
    ]
    for node in range(node_count):
        for period in range(periods):
            previous = 0 if period == 0 else inventory[node][period - 1]
            expected = (
                previous + orders[node][period] - built.instance.demands[node][period]
            )
            expected -= sum(orders[child][period] for child in children[node])
            if inventory[node][period] != expected or expected < 0:
                raise AssertionError("an inventory balance is violated")
    cost = sum(
        built.instance.order_costs[node] * int(orders[node][period] > 0)
        + built.instance.holding_costs[node] * inventory[node][period]
        for node in range(node_count)
        for period in range(periods)
    )
    if objective is not None and cost != objective:
        raise AssertionError("the objective does not match distribution costs")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON distribution instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob040 nodes={len(instance.parents)} status={solution.status}")
    if not solution.is_sat():
        return 1
    orders, inventory = decode(built, solution)
    validate(built, orders, inventory, solution.objective)
    print(f"cost={solution.objective} orders={orders} inventory={inventory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
