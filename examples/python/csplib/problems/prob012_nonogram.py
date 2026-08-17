"""CSPLib prob012: Nonograms.

Specification: https://www.csplib.org/Problems/prob012/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import qayd as cp

from ..common import add_solver_arguments, solve_from_args

Clues = list[list[int]]


@dataclass(frozen=True)
class NonogramModel:
    model: cp.Model
    row_clues: Clues
    column_clues: Clues
    cells: list[list[cp.IntVar]]


def line_patterns(length: int, clues: list[int]) -> list[tuple[int, ...]]:
    if length < 0 or any(clue <= 0 for clue in clues):
        raise ValueError("line length must be non-negative and clues must be positive")
    if not clues:
        return [(0,) * length]

    patterns: list[tuple[int, ...]] = []

    def place(clue_index: int, next_position: int, filled: tuple[int, ...]) -> None:
        clue = clues[clue_index]
        remaining = clues[clue_index + 1 :]
        required_after = sum(remaining) + len(remaining)
        latest_start = length - clue - required_after
        for start in range(next_position, latest_start + 1):
            block = tuple(range(start, start + clue))
            new_filled = filled + block
            if clue_index + 1 == len(clues):
                occupied = set(new_filled)
                patterns.append(
                    tuple(int(position in occupied) for position in range(length))
                )
            else:
                place(clue_index + 1, start + clue + 1, new_filled)

    place(0, 0, ())
    return patterns


def build_model(row_clues: Clues, column_clues: Clues) -> NonogramModel:
    height = len(row_clues)
    width = len(column_clues)
    if height < 1 or width < 1:
        raise ValueError("a Nonogram grid must be non-empty")

    model = cp.Model()
    cells = [model.int_vars(width, 0, 1, name=f"row_{row}") for row in range(height)]
    for row, clues in enumerate(row_clues):
        patterns = line_patterns(width, clues)
        if patterns:
            model.table(cells[row], patterns)
        else:
            model.add(cells[row][0] != cells[row][0])
    for column, clues in enumerate(column_clues):
        variables = [cells[row][column] for row in range(height)]
        patterns = line_patterns(height, clues)
        if patterns:
            model.table(variables, patterns)
        else:
            model.add(variables[0] != variables[0])
    return NonogramModel(
        model,
        [list(row) for row in row_clues],
        [list(column) for column in column_clues],
        cells,
    )


def parse_clues(text: str) -> Clues:
    """Parse lines separated by semicolons, with comma-separated block sizes."""

    result: Clues = []
    for line in text.split(";"):
        stripped = line.strip()
        if stripped in {"", ".", "-"}:
            result.append([])
        else:
            result.append([int(value) for value in stripped.split(",")])
    return result


def decode(built: NonogramModel, solution: cp.Solution) -> list[list[int]]:
    if not solution.is_sat():
        raise RuntimeError(f"no solution available, solver status is {solution.status}")
    return [[solution.value(cell) for cell in row] for row in built.cells]


def _runs(line: list[int]) -> list[int]:
    result: list[int] = []
    current = 0
    for value in line + [0]:
        if value:
            current += 1
        elif current:
            result.append(current)
            current = 0
    return result


def validate(grid: list[list[int]], row_clues: Clues, column_clues: Clues) -> None:
    if len(grid) != len(row_clues) or any(
        len(row) != len(column_clues) for row in grid
    ):
        raise AssertionError("grid dimensions do not match the clues")
    if any(_runs(row) != clues for row, clues in zip(grid, row_clues)):
        raise AssertionError("a row does not match its clues")
    for column, clues in enumerate(column_clues):
        if _runs([grid[row][column] for row in range(len(grid))]) != clues:
            raise AssertionError("a column does not match its clues")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", default="1;3;5;3;1")
    parser.add_argument("--columns", default="1;3;5;3;1")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)

    row_clues = parse_clues(args.rows)
    column_clues = parse_clues(args.columns)
    built = build_model(row_clues, column_clues)
    solution = solve_from_args(built.model, args)
    print(f"prob012 size={len(row_clues)}x{len(column_clues)} status={solution.status}")
    if not solution.is_sat():
        return 1
    grid = decode(built, solution)
    validate(grid, row_clues, column_clues)
    for row in grid:
        print("".join("#" if cell else "." for cell in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
