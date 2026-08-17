"""CSPLib prob074: maximum clique with DIMACS graph input.

Specification: https://www.csplib.org/Problems/prob074/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values

Edge = tuple[int, int]

SAMPLE_DIMACS = """\
c CSPLib prob074 sample
p edge 5 6
e 1 2
e 2 3
e 3 4
e 4 1
e 3 5
e 4 5
"""


@dataclass(frozen=True)
class Graph:
    vertex_count: int
    edges: frozenset[Edge]


@dataclass(frozen=True)
class MaximumCliqueModel:
    model: cp.Model
    graph: Graph
    selected: list[cp.IntVar]


def parse_dimacs(lines: Iterable[str]) -> Graph:
    vertex_count: int | None = None
    edges: set[Edge] = set()
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("c"):
            continue
        fields = line.split()
        if fields[0] == "p":
            if len(fields) < 4 or fields[1] not in {"edge", "col"}:
                raise ValueError(f"invalid DIMACS problem line at {line_number}")
            vertex_count = int(fields[2])
            if vertex_count < 0:
                raise ValueError("DIMACS vertex count cannot be negative")
            continue
        if fields[0] == "e":
            if vertex_count is None:
                raise ValueError("DIMACS edge appeared before the problem line")
            if len(fields) != 3:
                raise ValueError(f"invalid DIMACS edge line at {line_number}")
            left, right = int(fields[1]) - 1, int(fields[2]) - 1
            if left < 0 or right < 0 or left >= vertex_count or right >= vertex_count:
                raise ValueError(f"DIMACS edge endpoint out of range at {line_number}")
            if left != right:
                edges.add((min(left, right), max(left, right)))
            continue
        raise ValueError(f"unsupported DIMACS line at {line_number}: {line!r}")
    if vertex_count is None:
        raise ValueError("missing DIMACS problem line")
    return Graph(vertex_count, frozenset(edges))


def load_dimacs(path: str | Path) -> Graph:
    with Path(path).open(encoding="utf-8") as source:
        return parse_dimacs(source)


def build_model(graph: Graph) -> MaximumCliqueModel:
    if graph.vertex_count < 0:
        raise ValueError("vertex_count cannot be negative")
    for left, right in graph.edges:
        if left < 0 or right <= left or right >= graph.vertex_count:
            raise ValueError("graph edges must be normalized valid pairs")

    model = cp.Model()
    selected = [
        model.bool_var(name=f"selected_{vertex}")
        for vertex in range(graph.vertex_count)
    ]
    for left, right in combinations(range(graph.vertex_count), 2):
        if (left, right) not in graph.edges:
            model.add(selected[left] + selected[right] <= 1)
    if selected:
        model.maximize(sum(selected))
    return MaximumCliqueModel(model, graph, selected)


def decode(built: MaximumCliqueModel, solution: cp.Solution) -> list[int]:
    membership = values(solution, built.selected)
    return [vertex for vertex, selected in enumerate(membership) if selected]


def validate(clique: list[int], graph: Graph) -> None:
    if len(set(clique)) != len(clique) or any(
        vertex < 0 or vertex >= graph.vertex_count for vertex in clique
    ):
        raise AssertionError("clique vertices must be unique and valid")
    if any(
        (min(left, right), max(left, right)) not in graph.edges
        for left, right in combinations(clique, 2)
    ):
        raise AssertionError("two selected vertices are not adjacent")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="DIMACS .clq or .col file")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)

    graph = (
        load_dimacs(args.path)
        if args.path
        else parse_dimacs(SAMPLE_DIMACS.splitlines())
    )
    built = build_model(graph)
    solution = solve_from_args(built.model, args)
    print(
        f"prob074 vertices={graph.vertex_count} edges={len(graph.edges)} status={solution.status}"
    )
    if not solution.is_sat():
        return 1
    clique = decode(built, solution)
    validate(clique, graph)
    print(f"size={len(clique)} clique={[vertex + 1 for vertex in clique]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
