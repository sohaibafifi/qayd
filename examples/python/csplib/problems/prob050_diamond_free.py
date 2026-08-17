"""CSPLib prob050: diamond-free degree sequences.

Specification: https://www.csplib.org/Problems/prob050/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values

Edge = tuple[int, int]


@dataclass(frozen=True)
class DiamondFreeModel:
    model: cp.Model
    vertex_count: int
    edges: dict[Edge, cp.IntVar]
    degrees: list[cp.IntVar]


def build_model(vertex_count: int) -> DiamondFreeModel:
    if vertex_count < 2:
        raise ValueError("vertex_count must be at least two")
    allowed_degrees = [(degree,) for degree in range(3, vertex_count, 3)]

    model = cp.Model()
    edges = {
        (left, right): model.bool_var(name=f"edge_{left}_{right}")
        for left, right in combinations(range(vertex_count), 2)
    }
    degrees = model.int_vars(vertex_count, 1, vertex_count - 1, name="degree")
    if not allowed_degrees:
        model.add(degrees[0] != degrees[0])
    for vertex, degree in enumerate(degrees):
        incident = [variable for edge, variable in edges.items() if vertex in edge]
        model.add(degree == sum(incident))
        if allowed_degrees:
            model.table([degree], allowed_degrees)
    for index in range(vertex_count - 1):
        model.add(degrees[index] >= degrees[index + 1])

    model.add(sum(degrees) % 12 == 0)

    for vertices in combinations(range(vertex_count), 4):
        induced_edges = [
            edges[(left, right)] for left, right in combinations(vertices, 2)
        ]
        model.add(sum(induced_edges) <= 4)
    return DiamondFreeModel(model, vertex_count, edges, degrees)


def decode(
    built: DiamondFreeModel,
    solution: cp.Solution,
) -> tuple[list[int], frozenset[Edge]]:
    if not solution.is_sat():
        raise RuntimeError(f"no solution available, solver status is {solution.status}")
    degrees = values(solution, built.degrees)
    selected_edges = frozenset(
        edge for edge, variable in built.edges.items() if solution.value(variable)
    )
    return degrees, selected_edges


def validate(
    built: DiamondFreeModel,
    degrees: list[int],
    selected_edges: frozenset[Edge],
) -> None:
    if degrees != sorted(degrees, reverse=True):
        raise AssertionError("the degree sequence is not non-increasing")
    actual_degrees = [
        sum(vertex in edge for edge in selected_edges)
        for vertex in range(built.vertex_count)
    ]
    if degrees != actual_degrees:
        raise AssertionError("the degree sequence does not match the graph")
    if any(degree <= 0 or degree % 3 != 0 for degree in degrees):
        raise AssertionError("a degree is not a positive multiple of three")
    if sum(degrees) % 12 != 0:
        raise AssertionError("the degree sum is not a multiple of twelve")
    for vertices in combinations(range(built.vertex_count), 4):
        if sum(edge in selected_edges for edge in combinations(vertices, 2)) > 4:
            raise AssertionError("the graph contains a diamond")


def enumerate_degree_sequences(
    vertex_count: int,
    *,
    time_limit: int = 30,
) -> list[tuple[int, ...]]:
    """Enumerate every realizable degree sequence for the requested order."""

    built = build_model(vertex_count)
    sequences = []
    while True:
        solution = built.model.solve(
            engine="exact",
            time_limit=time_limit,
            seed=0,
            threads=1,
        )
        if solution.status == "UNSATISFIABLE":
            break
        if not solution.is_sat():
            raise RuntimeError(
                f"degree-sequence enumeration stopped with status {solution.status}"
            )
        degrees, selected_edges = decode(built, solution)
        validate(built, degrees, selected_edges)
        sequence = tuple(degrees)
        sequences.append(sequence)
        built.model.add(
            cp.any(
                [variable != value for variable, value in zip(built.degrees, sequence)]
            )
        )
    return sorted(sequences, reverse=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vertices", type=int, default=10)
    parser.add_argument(
        "--enumerate",
        action="store_true",
        help="enumerate all degree sequences instead of returning one witness",
    )
    add_solver_arguments(parser, time_limit=30)
    args = parser.parse_args(argv)

    if args.enumerate:
        sequences = enumerate_degree_sequences(
            args.vertices, time_limit=args.time_limit
        )
        print(f"prob050 vertices={args.vertices} sequences={len(sequences)}")
        for sequence in sequences:
            print(" ".join(map(str, sequence)))
        return 0

    built = build_model(args.vertices)
    solution = solve_from_args(built.model, args)
    print(f"prob050 vertices={args.vertices} status={solution.status}")
    if not solution.is_sat():
        return 1
    degrees, selected_edges = decode(built, solution)
    validate(built, degrees, selected_edges)
    print(f"degrees={degrees}")
    print(f"edges={sorted(selected_edges)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
