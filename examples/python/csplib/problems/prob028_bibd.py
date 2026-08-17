"""CSPLib prob028: balanced incomplete block designs.

Specification: https://www.csplib.org/Problems/prob028/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations

import qayd as cp

from ..common import add_solver_arguments, solve_from_args


@dataclass(frozen=True)
class BibdModel:
    model: cp.Model
    parameters: tuple[int, int, int, int, int]
    incidence: list[list[cp.IntVar]]


def build_model(v: int, b: int, r: int, k: int, lambda_: int) -> BibdModel:
    if min(v, b, r, k) < 1 or lambda_ < 0:
        raise ValueError("v, b, r, and k must be positive; lambda must be non-negative")
    model = cp.Model()
    incidence = [model.int_vars(b, 0, 1, name=f"object_{item}") for item in range(v)]

    if v * r != b * k or lambda_ * (v - 1) != r * (k - 1):
        model.add(incidence[0][0] != incidence[0][0])
        return BibdModel(model, (v, b, r, k, lambda_), incidence)

    for row in incidence:
        model.add(sum(row) == r)
    for block in range(b):
        model.add(sum(incidence[item][block] for item in range(v)) == k)

    and_table = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
    for first, second in combinations(range(v), 2):
        together: list[cp.IntVar] = []
        for block in range(b):
            both = model.bool_var(name=f"both_{first}_{second}_{block}")
            model.table(
                [incidence[first][block], incidence[second][block], both], and_table
            )
            together.append(both)
        model.add(sum(together) == lambda_)

    for item in range(v):
        model.add(incidence[item][0] == int(item < k))
    return BibdModel(model, (v, b, r, k, lambda_), incidence)


def decode(built: BibdModel, solution: cp.Solution) -> list[list[int]]:
    if not solution.is_sat():
        raise RuntimeError(f"no solution available, solver status is {solution.status}")
    return [[solution.value(cell) for cell in row] for row in built.incidence]


def validate(
    matrix: list[list[int]], parameters: tuple[int, int, int, int, int]
) -> None:
    v, b, r, k, lambda_ = parameters
    if len(matrix) != v or any(len(row) != b for row in matrix):
        raise AssertionError("incidence matrix has the wrong dimensions")
    if any(sum(row) != r for row in matrix):
        raise AssertionError("an object occurs in the wrong number of blocks")
    if any(sum(matrix[item][block] for item in range(v)) != k for block in range(b)):
        raise AssertionError("a block has the wrong size")
    for first, second in combinations(range(v), 2):
        together = sum(
            matrix[first][block] * matrix[second][block] for block in range(b)
        )
        if together != lambda_:
            raise AssertionError(
                "an object pair occurs together the wrong number of times"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v", type=int, default=7)
    parser.add_argument("--b", type=int, default=7)
    parser.add_argument("--r", type=int, default=3)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--lambda", dest="lambda_", type=int, default=1)
    add_solver_arguments(parser, time_limit=30)
    args = parser.parse_args(argv)

    built = build_model(args.v, args.b, args.r, args.k, args.lambda_)
    solution = solve_from_args(built.model, args)
    print(f"prob028 parameters={built.parameters} status={solution.status}")
    if not solution.is_sat():
        return 1
    matrix = decode(built, solution)
    validate(matrix, built.parameters)
    for row in matrix:
        print(" ".join(map(str, row)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
