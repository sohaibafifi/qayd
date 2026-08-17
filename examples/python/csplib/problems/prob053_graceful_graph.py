"""CSPLib prob053: graceful graph labelling.

Specification: https://www.csplib.org/Problems/prob053/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values

Edge = tuple[int, int]


@dataclass(frozen=True)
class GracefulGraphModel:
    model: cp.Model
    vertex_count: int
    edges: tuple[Edge, ...]
    labels: list[cp.IntVar]
    differences: list[cp.IntVar]


def _normalize_edges(
    vertex_count: int, edges: list[Edge] | tuple[Edge, ...]
) -> tuple[Edge, ...]:
    normalized: set[Edge] = set()
    for left, right in edges:
        if left == right:
            raise ValueError("graceful graphs cannot contain loops")
        if left < 0 or right < 0 or left >= vertex_count or right >= vertex_count:
            raise ValueError("an edge endpoint is outside the graph")
        normalized.add((min(left, right), max(left, right)))
    return tuple(sorted(normalized))


def build_model(
    vertex_count: int, edges: list[Edge] | tuple[Edge, ...]
) -> GracefulGraphModel:
    if vertex_count < 1:
        raise ValueError("vertex_count must be positive")
    normalized = _normalize_edges(vertex_count, edges)
    edge_count = len(normalized)

    model = cp.Model()
    labels = model.int_vars(vertex_count, 0, edge_count, name="label")
    model.all_different(labels)
    differences: list[cp.IntVar] = []
    for index, (left, right) in enumerate(normalized):
        difference = model.int_var(1, edge_count, name=f"difference_{index}")
        model.add(difference == abs(labels[left] - labels[right]))
        differences.append(difference)
    if differences:
        model.all_different(differences)
    if vertex_count > 1:
        model.add(labels[0] < labels[-1])
    return GracefulGraphModel(model, vertex_count, normalized, labels, differences)


def decode(built: GracefulGraphModel, solution: cp.Solution) -> list[int]:
    return values(solution, built.labels)


def validate(labels: list[int], edges: tuple[Edge, ...]) -> None:
    edge_count = len(edges)
    if len(set(labels)) != len(labels) or any(
        label < 0 or label > edge_count for label in labels
    ):
        raise AssertionError("vertex labels must be distinct values from 0..q")
    differences = [abs(labels[left] - labels[right]) for left, right in edges]
    if sorted(differences) != list(range(1, edge_count + 1)):
        raise AssertionError("edge differences must be exactly 1..q")


def graph_edges(kind: str, vertex_count: int) -> list[Edge]:
    if kind == "path":
        return [(vertex, vertex + 1) for vertex in range(vertex_count - 1)]
    if kind == "cycle":
        return graph_edges("path", vertex_count) + (
            [(0, vertex_count - 1)] if vertex_count > 2 else []
        )
    if kind == "star":
        return [(0, vertex) for vertex in range(1, vertex_count)]
    if kind == "complete":
        return list(combinations(range(vertex_count), 2))
    raise ValueError(f"unknown graph kind: {kind}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vertices", type=int, default=8)
    parser.add_argument(
        "--graph", choices=("path", "cycle", "star", "complete"), default="path"
    )
    add_solver_arguments(parser)
    args = parser.parse_args(argv)

    built = build_model(args.vertices, graph_edges(args.graph, args.vertices))
    solution = solve_from_args(built.model, args)
    print(
        f"prob053 graph={args.graph} vertices={args.vertices} status={solution.status}"
    )
    if not solution.is_sat():
        return 1
    labels = decode(built, solution)
    validate(labels, built.edges)
    print(f"labels={labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
