"""CSPLib prob057: Killer Sudoku and its square-order generalization.

Specification: https://www.csplib.org/Problems/prob057/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from math import isqrt
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args

Cell = tuple[int, int]
Cage = tuple[int, tuple[Cell, ...]]

DEFAULT_INSTANCE = {
    "size": 4,
    "cages": [
        {"total": 3, "cells": [[0, 0], [0, 1]]},
        {"total": 7, "cells": [[0, 2], [0, 3]]},
        {"total": 7, "cells": [[1, 0], [1, 1]]},
        {"total": 3, "cells": [[1, 2], [1, 3]]},
        {"total": 3, "cells": [[2, 0], [2, 1]]},
        {"total": 7, "cells": [[2, 2], [2, 3]]},
        {"total": 7, "cells": [[3, 0], [3, 1]]},
        {"total": 3, "cells": [[3, 2], [3, 3]]},
    ],
}


@dataclass(frozen=True)
class KillerSudokuModel:
    model: cp.Model
    size: int
    cages: tuple[Cage, ...]
    cells: list[list[cp.IntVar]]


def _connected(cells: tuple[Cell, ...]) -> bool:
    unseen = set(cells)
    frontier = {unseen.pop()}
    while frontier:
        row, column = frontier.pop()
        neighbours = {
            (row - 1, column),
            (row + 1, column),
            (row, column - 1),
            (row, column + 1),
        }
        reached = unseen.intersection(neighbours)
        unseen.difference_update(reached)
        frontier.update(reached)
    return not unseen


def parse_instance(data: str | bytes) -> tuple[int, tuple[Cage, ...]]:
    raw = json.loads(data)
    try:
        size = int(raw["size"])
        cages = tuple(
            (
                int(cage["total"]),
                tuple((int(cell[0]), int(cell[1])) for cell in cage["cells"]),
            )
            for cage in raw["cages"]
        )
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise ValueError("invalid Killer Sudoku JSON instance") from error
    return size, cages


def load_instance(path: str | Path) -> tuple[int, tuple[Cage, ...]]:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def build_model(size: int, cages: tuple[Cage, ...]) -> KillerSudokuModel:
    box_size = isqrt(size)
    if size < 1 or box_size * box_size != size:
        raise ValueError("size must be a positive perfect square")
    all_cells = [(row, column) for row in range(size) for column in range(size)]
    covered = [cell for _, cage_cells in cages for cell in cage_cells]
    if sorted(covered) != all_cells:
        raise ValueError("cages must cover every cell exactly once")
    for _, cage_cells in cages:
        if not cage_cells or not _connected(cage_cells):
            raise ValueError("every cage must be non-empty and orthogonally connected")

    model = cp.Model()
    cells = [model.int_vars(size, 1, size, name=f"row_{row}") for row in range(size)]
    for row in cells:
        model.all_different(row)
    for column in range(size):
        model.all_different([cells[row][column] for row in range(size)])
    for box_row in range(box_size):
        for box_column in range(box_size):
            box = [
                cells[row][column]
                for row in range(box_row * box_size, (box_row + 1) * box_size)
                for column in range(box_column * box_size, (box_column + 1) * box_size)
            ]
            model.all_different(box)
    for total, cage_cells in cages:
        variables = [cells[row][column] for row, column in cage_cells]
        model.add(sum(variables) == total)
        if len(variables) > 1:
            model.all_different(variables)
    return KillerSudokuModel(model, size, cages, cells)


def decode(built: KillerSudokuModel, solution: cp.Solution) -> list[list[int]]:
    if not solution.is_sat():
        raise RuntimeError(f"no solution available, solver status is {solution.status}")
    return [[solution.value(cell) for cell in row] for row in built.cells]


def validate(built: KillerSudokuModel, grid: list[list[int]]) -> None:
    expected = list(range(1, built.size + 1))
    if any(sorted(row) != expected for row in grid):
        raise AssertionError("a row is not a permutation of 1..n")
    if any(
        sorted(grid[row][column] for row in range(built.size)) != expected
        for column in range(built.size)
    ):
        raise AssertionError("a column is not a permutation of 1..n")
    box_size = isqrt(built.size)
    for box_row in range(box_size):
        for box_column in range(box_size):
            values = [
                grid[row][column]
                for row in range(box_row * box_size, (box_row + 1) * box_size)
                for column in range(box_column * box_size, (box_column + 1) * box_size)
            ]
            if sorted(values) != expected:
                raise AssertionError("a box is not a permutation of 1..n")
    for total, cage_cells in built.cages:
        values = [grid[row][column] for row, column in cage_cells]
        if sum(values) != total or len(values) != len(set(values)):
            raise AssertionError("a cage violates its sum or distinctness rule")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON Killer Sudoku instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)

    if args.path:
        size, cages = load_instance(args.path)
    else:
        size, cages = parse_instance(json.dumps(DEFAULT_INSTANCE))
    built = build_model(size, cages)
    solution = solve_from_args(built.model, args)
    print(f"prob057 size={size} cages={len(cages)} status={solution.status}")
    if not solution.is_sat():
        return 1
    grid = decode(built, solution)
    validate(built, grid)
    for row in grid:
        print(" ".join(map(str, row)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
