"""CSPLib prob075: product matrix travelling salesman problem.

Specification: https://www.csplib.org/Problems/prob075/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values

DEFAULT_INSTANCE = {"cost_vector": [2, 5, 3, 4], "price_vector": [3, 1, 4, 2]}


@dataclass(frozen=True)
class ProductTspInstance:
    cost_vector: tuple[int, ...]
    price_vector: tuple[int, ...]


@dataclass(frozen=True)
class ProductTspModel:
    model: cp.Model
    instance: ProductTspInstance
    tour: list[cp.IntVar]


def parse_instance(data: str | bytes) -> ProductTspInstance:
    raw = json.loads(data)
    try:
        return ProductTspInstance(
            tuple(int(value) for value in raw["cost_vector"]),
            tuple(int(value) for value in raw["price_vector"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid product-matrix TSP JSON instance") from error


def load_instance(path: str | Path) -> ProductTspInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def build_model(instance: ProductTspInstance) -> ProductTspModel:
    vertex_count = len(instance.cost_vector)
    if vertex_count < 2 or len(instance.price_vector) != vertex_count:
        raise ValueError("cost and price vectors must have equal non-trivial length")
    if any(value < 0 for value in (*instance.cost_vector, *instance.price_vector)):
        raise ValueError("vector entries must be non-negative")
    maximum_cost = max(instance.cost_vector) * max(instance.price_vector)
    model = cp.Model()
    tour = model.int_vars(vertex_count, 0, vertex_count - 1, name="tour")
    model.all_different(tour)
    model.add(tour[0] == 0)
    edge_costs = []
    table = [
        (left, right, instance.cost_vector[left] * instance.price_vector[right])
        for left in range(vertex_count)
        for right in range(vertex_count)
        if left != right
    ]
    for index in range(vertex_count):
        edge_cost = model.int_var(0, maximum_cost, name=f"edge_cost_{index}")
        model.table([tour[index], tour[(index + 1) % vertex_count], edge_cost], table)
        edge_costs.append(edge_cost)
    model.minimize(sum(edge_costs))
    return ProductTspModel(model, instance, tour)


def decode(built: ProductTspModel, solution: cp.Solution) -> list[int]:
    return values(solution, built.tour)


def tour_cost(instance: ProductTspInstance, tour: list[int]) -> int:
    return sum(
        instance.cost_vector[tour[index]]
        * instance.price_vector[tour[(index + 1) % len(tour)]]
        for index in range(len(tour))
    )


def validate(built: ProductTspModel, tour: list[int], objective: int | None) -> None:
    if sorted(tour) != list(range(len(built.instance.cost_vector))):
        raise AssertionError("the tour is not Hamiltonian")
    if objective is not None and tour_cost(built.instance, tour) != objective:
        raise AssertionError("the objective does not match the tour cost")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON product-matrix TSP instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob075 vertices={len(instance.cost_vector)} status={solution.status}")
    if not solution.is_sat():
        return 1
    tour = decode(built, solution)
    validate(built, tour, solution.objective)
    print(f"cost={solution.objective} tour={tour}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
