"""CSPLib prob071: diameter and degree bounded network design.

Specification: https://www.csplib.org/Problems/prob071/
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args

DEFAULT_INSTANCE = {
    "nodes": 4,
    "maximum_diameter": 2,
    "minimum_degree": 1,
    "edges": [[0, 1, 1], [0, 2, 2], [0, 3, 2], [1, 2, 1], [1, 3, 2], [2, 3, 1]],
}

Edge = tuple[int, int]


@dataclass(frozen=True)
class NetworkInstance:
    nodes: int
    maximum_diameter: int
    minimum_degree: int
    edges: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class NetworkModel:
    model: cp.Model
    instance: NetworkInstance
    selected: dict[Edge, cp.IntVar]
    costs: dict[Edge, int]


def parse_instance(data: str | bytes) -> NetworkInstance:
    raw = json.loads(data)
    try:
        return NetworkInstance(
            int(raw["nodes"]),
            int(raw["maximum_diameter"]),
            int(raw["minimum_degree"]),
            tuple((int(edge[0]), int(edge[1]), int(edge[2])) for edge in raw["edges"]),
        )
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError("invalid network-design JSON instance") from error


def load_instance(path: str | Path) -> NetworkInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def _edge(left: int, right: int) -> Edge:
    return (left, right) if left < right else (right, left)


def _paths(
    adjacency: list[list[int]], source: int, target: int, maximum_length: int
) -> list[tuple[Edge, ...]]:
    paths = []

    def visit(node: int, visited: set[int], edges: list[Edge]) -> None:
        if node == target:
            paths.append(tuple(edges))
            return
        if len(edges) == maximum_length:
            return
        for neighbour in adjacency[node]:
            if neighbour not in visited:
                visit(
                    neighbour, {*visited, neighbour}, [*edges, _edge(node, neighbour)]
                )

    visit(source, {source}, [])
    return paths


def build_model(instance: NetworkInstance) -> NetworkModel:
    if (
        instance.nodes < 2
        or instance.maximum_diameter < 1
        or instance.minimum_degree < 0
    ):
        raise ValueError("node, diameter, or degree bounds are invalid")
    costs = {}
    for left, right, cost in instance.edges:
        if (
            left < 0
            or right < 0
            or left >= instance.nodes
            or right >= instance.nodes
            or left == right
            or cost < 0
        ):
            raise ValueError("a candidate link is invalid")
        normalized = _edge(left, right)
        if normalized in costs:
            raise ValueError("candidate links must be unique")
        costs[normalized] = cost
    if not costs:
        raise ValueError("at least one candidate link is required")
    adjacency = [[] for _ in range(instance.nodes)]
    for left, right in costs:
        adjacency[left].append(right)
        adjacency[right].append(left)
    model = cp.Model()
    selected = {
        edge: model.bool_var(name=f"edge_{edge[0]}_{edge[1]}") for edge in costs
    }
    for node in range(instance.nodes):
        incident = [variable for edge, variable in selected.items() if node in edge]
        if len(incident) < instance.minimum_degree:
            witness = next(iter(selected.values()))
            model.add(witness == witness + 1)
        else:
            model.add(sum(incident) >= instance.minimum_degree)
    for source in range(instance.nodes):
        for target in range(source + 1, instance.nodes):
            path_variables = []
            for path_index, path in enumerate(
                _paths(adjacency, source, target, instance.maximum_diameter)
            ):
                active = model.bool_var(name=f"path_{source}_{target}_{path_index}")
                for edge in path:
                    model.add(active <= selected[edge])
                model.add(
                    active >= sum(selected[edge] for edge in path) - len(path) + 1
                )
                path_variables.append(active)
            if path_variables:
                model.add(sum(path_variables) >= 1)
            else:
                witness = next(iter(selected.values()))
                model.add(witness == witness + 1)
    model.minimize(sum(costs[edge] * variable for edge, variable in selected.items()))
    return NetworkModel(model, instance, selected, costs)


def decode(built: NetworkModel, solution: cp.Solution) -> frozenset[Edge]:
    if not solution.is_sat():
        raise RuntimeError(f"no solution available, solver status is {solution.status}")
    return frozenset(
        edge for edge, variable in built.selected.items() if solution.value(variable)
    )


def _distances(
    node_count: int, edges: frozenset[Edge], source: int
) -> list[int | None]:
    distances: list[int | None] = [None] * node_count
    distances[source] = 0
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for edge in edges:
            if node not in edge:
                continue
            neighbour = edge[1] if edge[0] == node else edge[0]
            if distances[neighbour] is None:
                distances[neighbour] = int(distances[node]) + 1
                queue.append(neighbour)
    return distances


def validate(
    built: NetworkModel, edges: frozenset[Edge], objective: int | None
) -> None:
    if not edges.issubset(built.costs):
        raise AssertionError("the network contains a non-candidate edge")
    for node in range(built.instance.nodes):
        if sum(node in edge for edge in edges) < built.instance.minimum_degree:
            raise AssertionError("a node violates the minimum degree")
        distances = _distances(built.instance.nodes, edges, node)
        if any(
            distance is None or distance > built.instance.maximum_diameter
            for distance in distances
        ):
            raise AssertionError("the network violates the diameter bound")
    cost = sum(built.costs[edge] for edge in edges)
    if objective is not None and cost != objective:
        raise AssertionError("the objective does not match selected link costs")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON network-design instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob071 nodes={instance.nodes} status={solution.status}")
    if not solution.is_sat():
        return 1
    edges = decode(built, solution)
    validate(built, edges, solution.objective)
    print(f"cost={solution.objective} edges={sorted(edges)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
