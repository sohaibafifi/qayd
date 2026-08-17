"""CSPLib prob054: n-queens.

Specification: https://www.csplib.org/Problems/prob054/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values


@dataclass(frozen=True)
class NQueensModel:
    model: cp.Model
    rows: list[cp.IntVar]


def build_model(size: int) -> NQueensModel:
    if size < 1:
        raise ValueError("size must be positive")

    model = cp.Model()
    rows = model.int_vars(size, 0, size - 1, name="row")
    model.all_different(rows)
    for left in range(size):
        for right in range(left + 1, size):
            distance = right - left
            model.add(rows[left] != rows[right] + distance)
            model.add(rows[left] != rows[right] - distance)
    if size > 1:
        model.add(rows[0] < rows[-1])
    return NQueensModel(model, rows)


def decode(built: NQueensModel, solution: cp.Solution) -> list[int]:
    return values(solution, built.rows)


def validate(rows: list[int]) -> None:
    size = len(rows)
    if sorted(rows) != list(range(size)):
        raise AssertionError("there must be exactly one queen in each row and column")
    for left in range(size):
        for right in range(left + 1, size):
            if abs(rows[left] - rows[right]) == right - left:
                raise AssertionError(
                    f"queens in columns {left} and {right} share a diagonal"
                )


def render(rows: list[int]) -> str:
    return "\n".join(
        " ".join("Q" if rows[column] == row else "." for column in range(len(rows)))
        for row in range(len(rows))
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=8)
    add_solver_arguments(parser)
    args = parser.parse_args(argv)

    built = build_model(args.size)
    solution = solve_from_args(built.model, args)
    print(f"prob054 size={args.size} status={solution.status}")
    if not solution.is_sat():
        return 1
    rows = decode(built, solution)
    validate(rows)
    print(f"rows={rows}")
    print(render(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
