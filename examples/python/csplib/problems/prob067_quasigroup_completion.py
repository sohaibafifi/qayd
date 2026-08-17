"""CSPLib prob067: quasigroup completion.

Specification: https://www.csplib.org/Problems/prob067/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values

Grid = list[list[int | None]]


@dataclass(frozen=True)
class QuasigroupCompletionModel:
    model: cp.Model
    clues: Grid
    table: list[list[cp.IntVar]]


def _validate_grid(grid: Grid) -> None:
    order = len(grid)
    if order < 1 or any(len(row) != order for row in grid):
        raise ValueError("the clue grid must be a non-empty square")
    for row in grid:
        for clue in row:
            if clue is not None and (clue < 0 or clue >= order):
                raise ValueError("clues must be in 0..order-1")


def build_model(grid: Grid) -> QuasigroupCompletionModel:
    _validate_grid(grid)
    order = len(grid)
    clues = [list(row) for row in grid]
    model = cp.Model()
    flat = model.int_vars(order * order, 0, order - 1, name="cell")
    table = [flat[row * order : (row + 1) * order] for row in range(order)]
    for row in table:
        model.all_different(row)
    for column in range(order):
        model.all_different([table[row][column] for row in range(order)])
    for row in range(order):
        for column in range(order):
            clue = clues[row][column]
            if clue is not None:
                model.add(table[row][column] == clue)
    return QuasigroupCompletionModel(model, clues, table)


def parse_grid(text: str) -> Grid:
    """Parse a one-based grid such as ``1,.,.,4;.,.,2,.;3,.,1,.;.,3,.,.``."""

    rows: Grid = []
    for raw_row in text.split(";"):
        tokens = raw_row.replace(",", " ").split()
        row: list[int | None] = []
        for token in tokens:
            if token in {".", "_", "-"}:
                row.append(None)
            else:
                value = int(token)
                if value < 1:
                    raise ValueError("textual grid clues use one-based values")
                row.append(value - 1)
        rows.append(row)
    _validate_grid(rows)
    return rows


def decode(built: QuasigroupCompletionModel, solution: cp.Solution) -> list[list[int]]:
    return [values(solution, row) for row in built.table]


def validate(table: list[list[int]], clues: Grid) -> None:
    order = len(table)
    expected = list(range(order))
    if order == 0 or any(sorted(row) != expected for row in table):
        raise AssertionError("every row must be a permutation")
    if any(
        sorted(table[row][column] for row in range(order)) != expected
        for column in range(order)
    ):
        raise AssertionError("every column must be a permutation")
    for row in range(order):
        for column in range(order):
            clue = clues[row][column]
            if clue is not None and table[row][column] != clue:
                raise AssertionError("the completion changed a clue")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", default="1,.,.,4;.,.,2,.;3,.,1,.;.,3,.,.")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)

    built = build_model(parse_grid(args.grid))
    solution = solve_from_args(built.model, args)
    print(f"prob067 order={len(built.table)} status={solution.status}")
    if not solution.is_sat():
        return 1
    table = decode(built, solution)
    validate(table, built.clues)
    for row in table:
        print(" ".join(str(value + 1) for value in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
