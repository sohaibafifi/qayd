"""CSPLib prob034: warehouse location.

Specification: https://www.csplib.org/Problems/prob034/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args

DEFAULT_INSTANCE = {
    "fixed_cost": 30,
    "capacities": [1, 4, 2, 1, 3],
    "supply_costs": [
        [20, 24, 11, 25, 30],
        [28, 27, 82, 83, 74],
        [74, 97, 71, 96, 70],
        [2, 55, 73, 69, 61],
        [46, 96, 59, 83, 4],
        [42, 22, 29, 67, 59],
        [1, 5, 73, 59, 56],
        [10, 73, 13, 43, 96],
        [93, 35, 63, 85, 46],
        [47, 65, 55, 71, 95],
    ],
}


@dataclass(frozen=True)
class WarehouseInstance:
    fixed_cost: int
    capacities: tuple[int, ...]
    supply_costs: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class WarehouseModel:
    model: cp.Model
    instance: WarehouseInstance
    opened: list[cp.IntVar]
    assigned: list[list[cp.IntVar]]


def parse_instance(data: str | bytes) -> WarehouseInstance:
    raw = json.loads(data)
    try:
        return WarehouseInstance(
            int(raw["fixed_cost"]),
            tuple(int(value) for value in raw["capacities"]),
            tuple(tuple(int(value) for value in row) for row in raw["supply_costs"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid warehouse JSON instance") from error


def load_instance(path: str | Path) -> WarehouseInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def build_model(instance: WarehouseInstance) -> WarehouseModel:
    warehouse_count = len(instance.capacities)
    store_count = len(instance.supply_costs)
    if warehouse_count < 1 or store_count < 1:
        raise ValueError("the instance must contain warehouses and stores")
    if any(capacity < 0 for capacity in instance.capacities):
        raise ValueError("warehouse capacities must be non-negative")
    if any(len(row) != warehouse_count for row in instance.supply_costs):
        raise ValueError("every supply-cost row must contain one cost per warehouse")

    model = cp.Model()
    opened = [
        model.bool_var(name=f"open_{warehouse}") for warehouse in range(warehouse_count)
    ]
    assigned = [
        [
            model.bool_var(name=f"store_{store}_warehouse_{warehouse}")
            for warehouse in range(warehouse_count)
        ]
        for store in range(store_count)
    ]
    for store in range(store_count):
        model.add(sum(assigned[store]) == 1)
    for warehouse, capacity in enumerate(instance.capacities):
        model.add(
            sum(assigned[store][warehouse] for store in range(store_count))
            <= capacity * opened[warehouse]
        )

    fixed = instance.fixed_cost * sum(opened)
    supply = sum(
        instance.supply_costs[store][warehouse] * assigned[store][warehouse]
        for store in range(store_count)
        for warehouse in range(warehouse_count)
    )
    model.minimize(fixed + supply)
    return WarehouseModel(model, instance, opened, assigned)


def decode(built: WarehouseModel, solution: cp.Solution) -> tuple[list[int], list[int]]:
    if not solution.is_sat():
        raise RuntimeError(f"no solution available, solver status is {solution.status}")
    opened = [
        warehouse
        for warehouse, variable in enumerate(built.opened)
        if solution.value(variable)
    ]
    assignments = [
        next(
            warehouse
            for warehouse, variable in enumerate(row)
            if solution.value(variable)
        )
        for row in built.assigned
    ]
    return opened, assignments


def validate(
    built: WarehouseModel,
    opened: list[int],
    assignments: list[int],
    objective: int | None,
) -> None:
    if len(assignments) != len(built.instance.supply_costs):
        raise AssertionError("every store must have one assignment")
    for warehouse, capacity in enumerate(built.instance.capacities):
        load = assignments.count(warehouse)
        if load > capacity or (load and warehouse not in opened):
            raise AssertionError(
                "a warehouse capacity or opening constraint is violated"
            )
    cost = built.instance.fixed_cost * len(opened) + sum(
        built.instance.supply_costs[store][warehouse]
        for store, warehouse in enumerate(assignments)
    )
    if cost != objective:
        raise AssertionError("the objective does not match the decoded solution")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON warehouse instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)

    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(
        f"prob034 warehouses={len(instance.capacities)} stores={len(instance.supply_costs)} "
        f"status={solution.status}"
    )
    if not solution.is_sat():
        return 1
    opened, assignments = decode(built, solution)
    validate(built, opened, assignments, solution.objective)
    print(f"cost={solution.objective} opened={opened} assignments={assignments}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
