"""CSPLib prob086: capacitated vehicle routing.

Specification: https://www.csplib.org/Problems/prob086/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args

DEFAULT_INSTANCE = {
    "vehicles": 2,
    "capacity": 5,
    "demands": [0, 2, 3, 2, 2],
    "distances": [
        [0, 2, 3, 4, 3],
        [2, 0, 2, 3, 4],
        [3, 2, 0, 2, 3],
        [4, 3, 2, 0, 2],
        [3, 4, 3, 2, 0],
    ],
}


@dataclass(frozen=True)
class CvrpInstance:
    vehicles: int
    capacity: int
    demands: tuple[int, ...]
    distances: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class CvrpModel:
    model: cp.Model
    instance: CvrpInstance
    routes: list[list[cp.IntVar]]


def parse_instance(data: str | bytes) -> CvrpInstance:
    raw = json.loads(data)
    try:
        return CvrpInstance(
            int(raw["vehicles"]),
            int(raw["capacity"]),
            tuple(int(value) for value in raw["demands"]),
            tuple(tuple(int(value) for value in row) for row in raw["distances"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid CVRP JSON instance") from error


def load_instance(path: str | Path) -> CvrpInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def build_model(instance: CvrpInstance) -> CvrpModel:
    node_count = len(instance.demands)
    if instance.vehicles < 1 or instance.capacity < 1 or node_count < 2:
        raise ValueError("vehicles, capacity, and customers must be positive")
    if instance.demands[0] != 0 or any(
        demand < 0 or demand > instance.capacity for demand in instance.demands
    ):
        raise ValueError("demands are invalid")
    if len(instance.distances) != node_count or any(
        len(row) != node_count for row in instance.distances
    ):
        raise ValueError("the distance matrix dimensions are invalid")
    if any(distance < 0 for row in instance.distances for distance in row):
        raise ValueError("distances must be non-negative")
    customer_count = node_count - 1
    steps = customer_count
    model = cp.Model()
    routes = [
        model.int_vars(steps, 0, customer_count, name=f"vehicle_{vehicle}")
        for vehicle in range(instance.vehicles)
    ]
    flattened = [customer for route in routes for customer in route]
    zero_count = instance.vehicles * steps - customer_count
    model.cardinality(
        flattened,
        list(range(node_count)),
        [zero_count, *([1] * customer_count)],
        [zero_count, *([1] * customer_count)],
        closed=True,
    )
    travel_costs = []
    maximum_distance = max(distance for row in instance.distances for distance in row)
    demand_table = [(node, instance.demands[node]) for node in range(node_count)]
    distance_table = [
        (left, right, instance.distances[left][right])
        for left in range(node_count)
        for right in range(node_count)
    ]
    for vehicle, route in enumerate(routes):
        demands = []
        for step, customer in enumerate(route):
            if step < steps - 1:
                model.add((customer == 0).implies(route[step + 1] == 0))
            demand = model.int_var(
                0, instance.capacity, name=f"demand_{vehicle}_{step}"
            )
            model.table([customer, demand], demand_table)
            demands.append(demand)
        model.add(sum(demands) <= instance.capacity)
        first_cost = model.int_var(0, maximum_distance, name=f"travel_{vehicle}_0")
        model.table(
            [route[0], first_cost],
            [(right, instance.distances[0][right]) for right in range(node_count)],
        )
        travel_costs.append(first_cost)
        for step in range(steps - 1):
            cost = model.int_var(
                0, maximum_distance, name=f"travel_{vehicle}_{step + 1}"
            )
            model.table([route[step], route[step + 1], cost], distance_table)
            travel_costs.append(cost)
        last_cost = model.int_var(0, maximum_distance, name=f"travel_{vehicle}_{steps}")
        model.table(
            [route[-1], last_cost],
            [(left, instance.distances[left][0]) for left in range(node_count)],
        )
        travel_costs.append(last_cost)
    for vehicle in range(instance.vehicles - 1):
        model.add(routes[vehicle][0] >= routes[vehicle + 1][0])
    model.minimize(sum(travel_costs))
    return CvrpModel(model, instance, routes)


def decode(built: CvrpModel, solution: cp.Solution) -> list[list[int]]:
    if not solution.is_sat():
        raise RuntimeError(f"no solution available, solver status is {solution.status}")
    return [
        [
            solution.value(variable)
            for variable in route
            if solution.value(variable) != 0
        ]
        for route in built.routes
    ]


def route_cost(instance: CvrpInstance, routes: list[list[int]]) -> int:
    return sum(
        instance.distances[left][right]
        for route in routes
        for left, right in zip([0, *route], [*route, 0])
    )


def validate(built: CvrpModel, routes: list[list[int]], objective: int | None) -> None:
    customers = [customer for route in routes for customer in route]
    if sorted(customers) != list(range(1, len(built.instance.demands))):
        raise AssertionError("customers are not visited exactly once")
    if any(
        sum(built.instance.demands[customer] for customer in route)
        > built.instance.capacity
        for route in routes
    ):
        raise AssertionError("a vehicle capacity is exceeded")
    cost = route_cost(built.instance, routes)
    if objective is not None and cost != objective:
        raise AssertionError("the objective does not match route costs")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON CVRP instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob086 customers={len(instance.demands) - 1} status={solution.status}")
    if not solution.is_sat():
        return 1
    routes = decode(built, solution)
    validate(built, routes, solution.objective)
    print(f"distance={solution.objective} routes={routes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
