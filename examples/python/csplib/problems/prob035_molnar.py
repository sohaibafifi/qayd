"""CSPLib prob035: Molnar's determinant problem.

Specification: https://www.csplib.org/Problems/prob035/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product

import qayd as cp

from ..common import add_solver_arguments, solve_from_args


@dataclass(frozen=True)
class MolnarModel:
    model: cp.Model
    size: int
    bound: int
    nonzero: bool
    positive_squared_determinant: bool
    matrix: list[list[cp.IntVar]]


def determinant(matrix: list[list[int]] | tuple[tuple[int, ...], ...]) -> int:
    if len(matrix) == 1:
        return matrix[0][0]
    return sum(
        (-1) ** column
        * matrix[0][column]
        * determinant(
            [
                [value for index, value in enumerate(row) if index != column]
                for row in matrix[1:]
            ]
        )
        for column in range(len(matrix))
    )


def build_model(
    size: int = 2,
    bound: int = 1,
    *,
    nonzero: bool = False,
    positive_squared_determinant: bool = False,
) -> MolnarModel:
    if size < 2 or bound < 1:
        raise ValueError("size must be at least two and bound must be positive")
    domain = tuple(
        value for value in range(-bound, bound + 1) if not nonzero or value != 0
    )
    candidate_count = len(domain) ** (size * size)
    if candidate_count > 5_000_000:
        raise ValueError("the requested extensional determinant domain is too large")
    allowed = []
    for flattened in product(domain, repeat=size * size):
        matrix = [list(flattened[row * size : (row + 1) * size]) for row in range(size)]
        if any(abs(matrix[index][index]) == 1 for index in range(size)):
            continue
        if determinant(matrix) != 1:
            continue
        squared = [[value * value for value in row] for row in matrix]
        squared_determinant = determinant(squared)
        if squared_determinant == 1 or (
            squared_determinant == -1 and not positive_squared_determinant
        ):
            allowed.append(flattened)

    model = cp.Model()
    matrix = [
        model.int_vars(size, -bound, bound, name=f"row_{row}") for row in range(size)
    ]
    unary = [(value,) for value in domain]
    for variable in (entry for row in matrix for entry in row):
        model.table([variable], unary)
    if allowed:
        model.table([entry for row in matrix for entry in row], allowed)
    else:
        model.add(matrix[0][0] != matrix[0][0])
    return MolnarModel(
        model,
        size,
        bound,
        nonzero,
        positive_squared_determinant,
        matrix,
    )


def decode(built: MolnarModel, solution: cp.Solution) -> list[list[int]]:
    if not solution.is_sat():
        raise RuntimeError(f"no solution available, solver status is {solution.status}")
    return [[solution.value(variable) for variable in row] for row in built.matrix]


def validate(built: MolnarModel, matrix: list[list[int]]) -> None:
    if len(matrix) != built.size or any(len(row) != built.size for row in matrix):
        raise AssertionError("the matrix dimensions are invalid")
    if any(abs(matrix[index][index]) == 1 for index in range(built.size)):
        raise AssertionError("a diagonal entry equals plus or minus one")
    if built.nonzero and any(value == 0 for row in matrix for value in row):
        raise AssertionError("the nonzero variant contains a zero")
    if determinant(matrix) != 1:
        raise AssertionError("the matrix determinant is not one")
    squared_determinant = determinant(
        [[value * value for value in row] for row in matrix]
    )
    expected = {1} if built.positive_squared_determinant else {-1, 1}
    if squared_determinant not in expected:
        raise AssertionError("the squared-entry determinant is invalid")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=2)
    parser.add_argument("--bound", type=int, default=1)
    parser.add_argument("--nonzero", action="store_true")
    parser.add_argument("--positive-squared-determinant", action="store_true")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    built = build_model(
        args.size,
        args.bound,
        nonzero=args.nonzero,
        positive_squared_determinant=args.positive_squared_determinant,
    )
    solution = solve_from_args(built.model, args)
    print(f"prob035 size={args.size} bound={args.bound} status={solution.status}")
    if not solution.is_sat():
        return 1
    matrix = decode(built, solution)
    validate(built, matrix)
    for row in matrix:
        print(" ".join(map(str, row)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
