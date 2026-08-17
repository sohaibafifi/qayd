"""CSPLib prob038: steel mill slab design.

Specification: https://www.csplib.org/Problems/prob038/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values

DEFAULT_INSTANCE = {
    "capacities": [5, 8, 10],
    "orders": [
        {"weight": 4, "color": 0},
        {"weight": 3, "color": 1},
        {"weight": 2, "color": 0},
        {"weight": 2, "color": 2},
    ],
    "color_limit": 2,
}


@dataclass(frozen=True)
class SteelOrder:
    weight: int
    color: int


@dataclass(frozen=True)
class SteelMillInstance:
    capacities: tuple[int, ...]
    orders: tuple[SteelOrder, ...]
    color_limit: int


@dataclass(frozen=True)
class SteelMillModel:
    model: cp.Model
    instance: SteelMillInstance
    assignment: list[cp.IntVar]
    loads: list[cp.IntVar]


def parse_instance(data: str | bytes) -> SteelMillInstance:
    raw = json.loads(data)
    try:
        orders = tuple(
            SteelOrder(int(item["weight"]), int(item["color"]))
            for item in raw["orders"]
        )
        return SteelMillInstance(
            tuple(int(value) for value in raw["capacities"]),
            orders,
            int(raw.get("color_limit", 2)),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid steel-mill JSON instance") from error


def load_instance(path: str | Path) -> SteelMillInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def build_model(instance: SteelMillInstance) -> SteelMillModel:
    if not instance.capacities or not instance.orders or instance.color_limit < 1:
        raise ValueError("capacities, orders, and color_limit must be positive")
    capacities = tuple(sorted(set(instance.capacities)))
    if any(capacity < 1 for capacity in capacities) or any(
        order.weight < 1 for order in instance.orders
    ):
        raise ValueError("capacities and order weights must be positive")
    if any(
        order.weight > capacities[-1] or order.color < 0 for order in instance.orders
    ):
        raise ValueError("an order cannot be assigned to any slab")
    slab_count = len(instance.orders)
    model = cp.Model()
    assignment = model.int_vars(len(instance.orders), 0, slab_count - 1, name="slab")
    membership = [
        [
            model.bool_var(name=f"order_{order}_slab_{slab}")
            for slab in range(slab_count)
        ]
        for order in range(len(instance.orders))
    ]
    for order in range(len(instance.orders)):
        for slab in range(slab_count):
            model.table(
                [assignment[order], membership[order][slab]],
                [
                    (candidate, int(candidate == slab))
                    for candidate in range(slab_count)
                ],
            )
    maximum_capacity = capacities[-1]
    losses = [
        min(
            (capacity for capacity in (0, *capacities) if capacity >= load),
            default=maximum_capacity,
        )
        - load
        for load in range(maximum_capacity + 1)
    ]
    loads = []
    slab_losses = []
    colors = sorted({order.color for order in instance.orders})
    for slab in range(slab_count):
        load = model.int_var(0, maximum_capacity, name=f"load_{slab}")
        loss = model.int_var(0, maximum_capacity, name=f"loss_{slab}")
        model.add(
            load
            == sum(
                order.weight * membership[index][slab]
                for index, order in enumerate(instance.orders)
            )
        )
        model.table(
            [load, loss],
            [(value, losses[value]) for value in range(maximum_capacity + 1)],
        )
        color_used = []
        for color in colors:
            used = model.bool_var(name=f"slab_{slab}_color_{color}")
            members = [
                membership[index][slab]
                for index, order in enumerate(instance.orders)
                if order.color == color
            ]
            model.add(sum(members) >= used)
            model.add(sum(members) <= len(members) * used)
            color_used.append(used)
        model.add(sum(color_used) <= instance.color_limit)
        loads.append(load)
        slab_losses.append(loss)
    for slab in range(slab_count - 1):
        model.add(loads[slab] >= loads[slab + 1])
    model.minimize(sum(slab_losses))
    return SteelMillModel(model, instance, assignment, loads)


def decode(built: SteelMillModel, solution: cp.Solution) -> list[list[int]]:
    slabs = [[] for _ in built.loads]
    for order, slab in enumerate(values(solution, built.assignment)):
        slabs[slab].append(order)
    return [orders for orders in slabs if orders]


def validate(
    built: SteelMillModel, slabs: list[list[int]], objective: int | None
) -> None:
    assigned = [order for slab in slabs for order in slab]
    if sorted(assigned) != list(range(len(built.instance.orders))):
        raise AssertionError("orders are not assigned exactly once")
    capacities = sorted(set(built.instance.capacities))
    total_loss = 0
    for slab in slabs:
        load = sum(built.instance.orders[index].weight for index in slab)
        if load > capacities[-1]:
            raise AssertionError("a slab exceeds maximum capacity")
        if (
            len({built.instance.orders[index].color for index in slab})
            > built.instance.color_limit
        ):
            raise AssertionError("a slab contains too many colours")
        total_loss += (
            min(capacity for capacity in capacities if capacity >= load) - load
        )
    if objective is not None and total_loss != objective:
        raise AssertionError("the objective does not match total steel loss")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON steel-mill instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob038 orders={len(instance.orders)} status={solution.status}")
    if not solution.is_sat():
        return 1
    slabs = decode(built, solution)
    validate(built, slabs, solution.objective)
    print(f"loss={solution.objective} slabs={slabs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
