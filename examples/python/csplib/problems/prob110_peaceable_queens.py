"""CSPLib prob110: peaceably co-existing armies of queens.

Specification: https://www.csplib.org/Problems/prob110/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations

import qayd as cp

from ..common import add_solver_arguments, solve_from_args

Square = tuple[int, int]


@dataclass(frozen=True)
class PeaceableQueensModel:
    model: cp.Model
    size: int
    black: list[list[cp.IntVar]]
    white: list[list[cp.IntVar]]


def _attack(left: Square, right: Square) -> bool:
    return (
        left[0] == right[0]
        or left[1] == right[1]
        or abs(left[0] - right[0]) == abs(left[1] - right[1])
    )


def build_model(size: int) -> PeaceableQueensModel:
    if size < 1:
        raise ValueError("size must be positive")
    model = cp.Model()
    black = [model.int_vars(size, 0, 1, name=f"black_{row}") for row in range(size)]
    white = [model.int_vars(size, 0, 1, name=f"white_{row}") for row in range(size)]
    squares = [(row, column) for row in range(size) for column in range(size)]
    for row, column in squares:
        model.add(black[row][column] + white[row][column] <= 1)
    for left, right in combinations(squares, 2):
        if _attack(left, right):
            model.add(black[left[0]][left[1]] + white[right[0]][right[1]] <= 1)
            model.add(white[left[0]][left[1]] + black[right[0]][right[1]] <= 1)
    black_count = sum(variable for row in black for variable in row)
    white_count = sum(variable for row in white for variable in row)
    model.add(black_count == white_count)
    model.maximize(black_count)
    return PeaceableQueensModel(model, size, black, white)


def decode(
    built: PeaceableQueensModel, solution: cp.Solution
) -> tuple[set[Square], set[Square]]:
    if not solution.is_sat():
        raise RuntimeError(f"no solution available, solver status is {solution.status}")
    black = {
        (row, column)
        for row in range(built.size)
        for column in range(built.size)
        if solution.value(built.black[row][column])
    }
    white = {
        (row, column)
        for row in range(built.size)
        for column in range(built.size)
        if solution.value(built.white[row][column])
    }
    return black, white


def validate(
    built: PeaceableQueensModel,
    black: set[Square],
    white: set[Square],
    objective: int | None,
) -> None:
    board = {(row, column) for row in range(built.size) for column in range(built.size)}
    if (
        not black.issubset(board)
        or not white.issubset(board)
        or black.intersection(white)
    ):
        raise AssertionError("an army placement is invalid")
    if len(black) != len(white):
        raise AssertionError("the armies have different sizes")
    if any(_attack(left, right) for left in black for right in white):
        raise AssertionError("opposing queens attack each other")
    if objective is not None and len(black) != objective:
        raise AssertionError("the objective does not match the army size")


def render(size: int, black: set[Square], white: set[Square]) -> str:
    return "\n".join(
        "".join(
            "B" if (row, column) in black else "W" if (row, column) in white else "."
            for column in range(size)
        )
        for row in range(size)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=5)
    add_solver_arguments(parser, time_limit=30)
    args = parser.parse_args(argv)
    built = build_model(args.size)
    solution = solve_from_args(built.model, args)
    print(f"prob110 size={args.size} status={solution.status}")
    if not solution.is_sat():
        return 1
    black, white = decode(built, solution)
    validate(built, black, white, solution.objective)
    print(f"army_size={solution.objective}")
    print(render(args.size, black, white))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
