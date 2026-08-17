"""CSPLib prob032: maximum-density still life.

Specification: https://www.csplib.org/Problems/prob032/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product

import qayd as cp

from ..common import add_solver_arguments, solve_from_args


@dataclass(frozen=True)
class StillLifeModel:
    model: cp.Model
    size: int
    cells: list[list[cp.IntVar]]


def _neighbours(cells: list[list[cp.IntVar]], row: int, column: int) -> list[cp.IntVar]:
    size = len(cells)
    return [
        cells[other_row][other_column]
        for other_row in range(max(0, row - 1), min(size, row + 2))
        for other_column in range(max(0, column - 1), min(size, column + 2))
        if (other_row, other_column) != (row, column)
    ]


def build_model(size: int) -> StillLifeModel:
    if size < 1:
        raise ValueError("size must be positive")
    model = cp.Model()
    cells = [model.int_vars(size, 0, 1, name=f"row_{row}") for row in range(size)]

    for row in range(size):
        for column in range(size):
            variables = [cells[row][column], *_neighbours(cells, row, column)]
            allowed = []
            for state in product((0, 1), repeat=len(variables)):
                alive = state[0]
                count = sum(state[1:])
                next_alive = int(count == 3 or (alive == 1 and count == 2))
                if next_alive == alive:
                    allowed.append(state)
            model.table(variables, allowed)

    for row in range(-1, size + 1):
        for column in range(-1, size + 1):
            if 0 <= row < size and 0 <= column < size:
                continue
            neighbours = _neighbours(cells, row, column)
            if neighbours:
                model.add(sum(neighbours) != 3)
    model.maximize(sum(cell for row in cells for cell in row))
    return StillLifeModel(model, size, cells)


def decode(built: StillLifeModel, solution: cp.Solution) -> list[list[int]]:
    if not solution.is_sat():
        raise RuntimeError(f"no solution available, solver status is {solution.status}")
    return [[solution.value(cell) for cell in row] for row in built.cells]


def validate(grid: list[list[int]]) -> None:
    size = len(grid)
    if size == 0 or any(len(row) != size for row in grid):
        raise AssertionError("the still-life grid must be non-empty and square")
    for row in range(-1, size + 1):
        for column in range(-1, size + 1):
            alive = int(
                0 <= row < size and 0 <= column < size and grid[row][column] == 1
            )
            count = sum(
                grid[other_row][other_column]
                for other_row in range(max(0, row - 1), min(size, row + 2))
                for other_column in range(max(0, column - 1), min(size, column + 2))
                if (other_row, other_column) != (row, column)
            )
            next_alive = int(count == 3 or (alive == 1 and count == 2))
            if next_alive != alive:
                raise AssertionError("the pattern changes under the Game of Life rules")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=3)
    add_solver_arguments(parser, time_limit=30)
    args = parser.parse_args(argv)

    built = build_model(args.size)
    solution = solve_from_args(built.model, args)
    print(f"prob032 size={args.size} status={solution.status}")
    if not solution.is_sat():
        return 1
    grid = decode(built, solution)
    validate(grid)
    print(f"live={solution.objective}")
    for row in grid:
        print("".join("#" if cell else "." for cell in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
