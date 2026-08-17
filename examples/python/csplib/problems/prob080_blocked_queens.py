"""CSPLib prob080: blocked n-queens.

Specification: https://www.csplib.org/Problems/prob080/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values
from .prob079_queens_completion import Square, post_queen_constraints


@dataclass(frozen=True)
class BlockedQueensModel:
    model: cp.Model
    size: int
    blocked: frozenset[Square]
    rows: list[cp.IntVar]


def build_model(
    size: int, *, blocked: set[Square] | frozenset[Square] = frozenset()
) -> BlockedQueensModel:
    if size < 1:
        raise ValueError("size must be positive")
    normalized = frozenset(blocked)
    if any(
        row < 0 or column < 0 or row >= size or column >= size
        for row, column in normalized
    ):
        raise ValueError("a blocked coordinate is outside the board")

    model = cp.Model()
    rows = model.int_vars(size, 0, size - 1, name="row")
    post_queen_constraints(model, size, tuple(range(size)), rows)
    for row, column in normalized:
        model.add(rows[column] != row)
    return BlockedQueensModel(model, size, normalized, rows)


def decode(built: BlockedQueensModel, solution: cp.Solution) -> list[int]:
    return values(solution, built.rows)


def validate(rows: list[int], *, blocked: frozenset[Square]) -> None:
    size = len(rows)
    if sorted(rows) != list(range(size)):
        raise AssertionError("there must be exactly one queen per row and column")
    if len({row + column for column, row in enumerate(rows)}) != size:
        raise AssertionError("two queens share a sum diagonal")
    if len({row - column for column, row in enumerate(rows)}) != size:
        raise AssertionError("two queens share a difference diagonal")
    if any((row, column) in blocked for column, row in enumerate(rows)):
        raise AssertionError("a queen lies on a blocked square")


def _parse_square(value: str) -> Square:
    try:
        row, column = value.split(",", maxsplit=1)
        return int(row), int(column)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected ROW,COLUMN") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument(
        "--blocked",
        type=_parse_square,
        action="append",
        help="blocked ROW,COLUMN, zero based",
    )
    add_solver_arguments(parser)
    args = parser.parse_args(argv)

    blocked = frozenset(args.blocked if args.blocked is not None else [(0, 0)])
    built = build_model(args.size, blocked=blocked)
    solution = solve_from_args(built.model, args)
    print(f"prob080 size={args.size} blocked={len(blocked)} status={solution.status}")
    if not solution.is_sat():
        return 1
    rows = decode(built, solution)
    validate(rows, blocked=blocked)
    print(f"queens={[(row, column) for column, row in enumerate(rows)]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
