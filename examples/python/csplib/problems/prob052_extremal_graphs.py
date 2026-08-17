"""CSPLib prob052: extremal graphs with small girth.

Specification: https://www.csplib.org/Problems/prob052/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations, permutations

import qayd as cp

from ..common import add_solver_arguments, solve_from_args

Edge = tuple[int, int]


@dataclass(frozen=True)
class ExtremalGraphModel:
    model: cp.Model
    vertex_count: int
    girth_limit: int
    edges: dict[Edge, cp.IntVar]
    forbidden_cycles: tuple[frozenset[Edge], ...]


def _edge(left: int, right: int) -> Edge:
    return (left, right) if left < right else (right, left)


def _cycles(vertex_count: int, girth_limit: int) -> tuple[frozenset[Edge], ...]:
    cycles = []
    for length in range(3, min(vertex_count, girth_limit) + 1):
        for vertices in combinations(range(vertex_count), length):
            start = vertices[0]
            for tail in permutations(vertices[1:]):
                cycle = (start, *tail)
                if cycle[1] > cycle[-1]:
                    continue
                cycle_edges = frozenset(
                    _edge(cycle[index], cycle[(index + 1) % length])
                    for index in range(length)
                )
                cycles.append(cycle_edges)
    return tuple(cycles)


def build_model(vertex_count: int, girth_limit: int) -> ExtremalGraphModel:
    if vertex_count < 1:
        raise ValueError("vertex_count must be positive")
    if girth_limit < 2:
        raise ValueError("girth_limit must be at least two")

    model = cp.Model()
    edges = {
        (left, right): model.bool_var(name=f"edge_{left}_{right}")
        for left, right in combinations(range(vertex_count), 2)
    }
    forbidden_cycles = _cycles(vertex_count, girth_limit)
    for cycle in forbidden_cycles:
        model.add(sum(edges[edge] for edge in cycle) <= len(cycle) - 1)
    if edges:
        model.maximize(sum(edges.values()))
    return ExtremalGraphModel(model, vertex_count, girth_limit, edges, forbidden_cycles)


def decode(built: ExtremalGraphModel, solution: cp.Solution) -> frozenset[Edge]:
    if not solution.is_sat():
        raise RuntimeError(f"no solution available, solver status is {solution.status}")
    return frozenset(
        edge for edge, variable in built.edges.items() if solution.value(variable)
    )


def validate(
    built: ExtremalGraphModel,
    selected_edges: frozenset[Edge],
    objective: int | None,
) -> None:
    valid_edges = set(combinations(range(built.vertex_count), 2))
    if not selected_edges.issubset(valid_edges):
        raise AssertionError("the graph contains an invalid edge")
    if any(cycle.issubset(selected_edges) for cycle in built.forbidden_cycles):
        raise AssertionError("the graph contains a forbidden short cycle")
    if objective is not None and len(selected_edges) != objective:
        raise AssertionError("the objective does not match the graph size")


def _canonical_graph(
    vertex_count: int, selected_edges: frozenset[Edge]
) -> tuple[Edge, ...]:
    return min(
        tuple(
            sorted(
                _edge(mapping[left], mapping[right]) for left, right in selected_edges
            )
        )
        for mapping in permutations(range(vertex_count))
    )


def count_non_isomorphic_extremal(
    vertex_count: int,
    girth_limit: int,
    edge_count: int,
) -> int:
    """Count extremal graphs exactly by exhaustive generation and relabeling."""

    all_edges = tuple(combinations(range(vertex_count), 2))
    forbidden_cycles = _cycles(vertex_count, girth_limit)
    canonical_graphs = set()
    for candidate in combinations(all_edges, edge_count):
        selected = frozenset(candidate)
        if any(cycle.issubset(selected) for cycle in forbidden_cycles):
            continue
        canonical_graphs.add(_canonical_graph(vertex_count, selected))
    return len(canonical_graphs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vertices", type=int, default=5)
    parser.add_argument("--girth-limit", type=int, default=3)
    parser.add_argument(
        "--count-nonisomorphic",
        action="store_true",
        help="exhaustively compute F_k(m), intended for small graphs",
    )
    add_solver_arguments(parser)
    args = parser.parse_args(argv)

    built = build_model(args.vertices, args.girth_limit)
    solution = solve_from_args(built.model, args)
    print(
        f"prob052 vertices={args.vertices} girth_limit={args.girth_limit} "
        f"status={solution.status}"
    )
    if not solution.is_sat():
        return 1
    selected_edges = decode(built, solution)
    validate(built, selected_edges, solution.objective)
    edge_count = len(selected_edges)
    print(
        f"f_{args.girth_limit}({args.vertices})={edge_count} edges={sorted(selected_edges)}"
    )
    if args.count_nonisomorphic:
        count = count_non_isomorphic_extremal(
            args.vertices,
            args.girth_limit,
            edge_count,
        )
        print(f"F_{args.girth_limit}({args.vertices})={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
