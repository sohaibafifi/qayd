"""CSPLib prob083: transshipment.

Specification: https://www.csplib.org/Problems/prob083/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args

DEFAULT_INSTANCE = {
    "warehouse_stock": [5, 5],
    "customer_demand": [4, 6],
    "warehouse_to_point_cost": [[1, 4], [3, 1]],
    "point_to_customer_cost": [[1, 5], [4, 1]],
}


@dataclass(frozen=True)
class TransshipmentInstance:
    warehouse_stock: tuple[int, ...]
    customer_demand: tuple[int, ...]
    warehouse_to_point_cost: tuple[tuple[int, ...], ...]
    point_to_customer_cost: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class TransshipmentModel:
    model: cp.Model
    instance: TransshipmentInstance
    inbound: list[list[cp.IntVar]]
    outbound: list[list[cp.IntVar]]


def parse_instance(data: str | bytes) -> TransshipmentInstance:
    raw = json.loads(data)
    try:
        return TransshipmentInstance(
            tuple(int(value) for value in raw["warehouse_stock"]),
            tuple(int(value) for value in raw["customer_demand"]),
            tuple(
                tuple(int(value) for value in row)
                for row in raw["warehouse_to_point_cost"]
            ),
            tuple(
                tuple(int(value) for value in row)
                for row in raw["point_to_customer_cost"]
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid transshipment JSON instance") from error


def load_instance(path: str | Path) -> TransshipmentInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def build_model(instance: TransshipmentInstance) -> TransshipmentModel:
    warehouses = len(instance.warehouse_stock)
    customers = len(instance.customer_demand)
    if (
        warehouses < 1
        or customers < 1
        or len(instance.warehouse_to_point_cost) != warehouses
    ):
        raise ValueError("the instance must contain warehouses, points, and customers")
    points = len(instance.warehouse_to_point_cost[0])
    if points < 1 or any(
        len(row) != points for row in instance.warehouse_to_point_cost
    ):
        raise ValueError("warehouse-to-point costs have inconsistent dimensions")
    if len(instance.point_to_customer_cost) != points or any(
        len(row) != customers for row in instance.point_to_customer_cost
    ):
        raise ValueError("point-to-customer costs have inconsistent dimensions")
    if sum(instance.warehouse_stock) < sum(instance.customer_demand):
        raise ValueError("total warehouse stock is smaller than total demand")

    total_demand = sum(instance.customer_demand)
    model = cp.Model()
    inbound = [
        [
            model.int_var(
                0,
                min(instance.warehouse_stock[warehouse], total_demand),
                name=f"in_{warehouse}_{point}",
            )
            for point in range(points)
        ]
        for warehouse in range(warehouses)
    ]
    outbound = [
        [
            model.int_var(
                0,
                instance.customer_demand[customer],
                name=f"out_{point}_{customer}",
            )
            for customer in range(customers)
        ]
        for point in range(points)
    ]
    for warehouse in range(warehouses):
        model.add(sum(inbound[warehouse]) <= instance.warehouse_stock[warehouse])
    for point in range(points):
        model.add(
            sum(inbound[warehouse][point] for warehouse in range(warehouses))
            == sum(outbound[point])
        )
    for customer in range(customers):
        model.add(
            sum(outbound[point][customer] for point in range(points))
            == instance.customer_demand[customer]
        )

    cost = sum(
        instance.warehouse_to_point_cost[warehouse][point] * inbound[warehouse][point]
        for warehouse in range(warehouses)
        for point in range(points)
    ) + sum(
        instance.point_to_customer_cost[point][customer] * outbound[point][customer]
        for point in range(points)
        for customer in range(customers)
    )
    model.minimize(cost)
    return TransshipmentModel(model, instance, inbound, outbound)


def decode(
    built: TransshipmentModel, solution: cp.Solution
) -> tuple[list[list[int]], list[list[int]]]:
    if not solution.is_sat():
        raise RuntimeError(f"no solution available, solver status is {solution.status}")
    inbound = [[solution.value(variable) for variable in row] for row in built.inbound]
    outbound = [
        [solution.value(variable) for variable in row] for row in built.outbound
    ]
    return inbound, outbound


def validate(
    built: TransshipmentModel,
    inbound: list[list[int]],
    outbound: list[list[int]],
    objective: int | None,
) -> None:
    for warehouse, stock in enumerate(built.instance.warehouse_stock):
        if sum(inbound[warehouse]) > stock:
            raise AssertionError("warehouse stock is exceeded")
    for point in range(len(outbound)):
        if sum(row[point] for row in inbound) != sum(outbound[point]):
            raise AssertionError("flow is not conserved at a transshipment point")
    for customer, demand in enumerate(built.instance.customer_demand):
        if sum(outbound[point][customer] for point in range(len(outbound))) != demand:
            raise AssertionError("customer demand is not met")
    cost = sum(
        built.instance.warehouse_to_point_cost[warehouse][point]
        * inbound[warehouse][point]
        for warehouse in range(len(inbound))
        for point in range(len(outbound))
    ) + sum(
        built.instance.point_to_customer_cost[point][customer]
        * outbound[point][customer]
        for point in range(len(outbound))
        for customer in range(len(built.instance.customer_demand))
    )
    if cost != objective:
        raise AssertionError("the objective does not match the decoded flow cost")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON transshipment instance")
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
        f"prob083 warehouses={len(instance.warehouse_stock)} status={solution.status}"
    )
    if not solution.is_sat():
        return 1
    inbound, outbound = decode(built, solution)
    validate(built, inbound, outbound, solution.objective)
    print(f"cost={solution.objective} inbound={inbound} outbound={outbound}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
