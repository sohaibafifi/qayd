"""CSPLib prob076: Costas arrays.

Specification: https://www.csplib.org/Problems/prob076/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values


@dataclass(frozen=True)
class CostasArrayModel:
    model: cp.Model
    permutation: list[cp.IntVar]
    difference_rows: list[list[cp.IntVar]]


def build_model(order: int) -> CostasArrayModel:
    if order < 1:
        raise ValueError("order must be positive")

    model = cp.Model()
    permutation = model.int_vars(order, 0, order - 1, name="mark_row")
    model.all_different(permutation)
    difference_rows: list[list[cp.IntVar]] = []
    for offset in range(1, order):
        row: list[cp.IntVar] = []
        for column in range(order - offset):
            difference = model.int_var(
                -(order - 1), order - 1, name=f"d_{offset}_{column}"
            )
            model.add(difference == permutation[column] - permutation[column + offset])
            row.append(difference)
        if len(row) > 1:
            model.all_different(row)
        difference_rows.append(row)
    if order > 1:
        model.add(permutation[0] < permutation[-1])
    return CostasArrayModel(model, permutation, difference_rows)


def decode(built: CostasArrayModel, solution: cp.Solution) -> list[int]:
    return values(solution, built.permutation)


def validate(permutation: list[int]) -> None:
    order = len(permutation)
    if sorted(permutation) != list(range(order)):
        raise AssertionError("a Costas array must have one mark per row and column")
    vectors: set[tuple[int, int]] = set()
    for left in range(order):
        for right in range(left + 1, order):
            vector = (right - left, permutation[right] - permutation[left])
            if vector in vectors:
                raise AssertionError(
                    "two pairs of marks have the same displacement vector"
                )
            vectors.add(vector)


def render(permutation: list[int]) -> str:
    return "\n".join(
        " ".join(
            "X" if permutation[column] == row else "."
            for column in range(len(permutation))
        )
        for row in range(len(permutation))
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=int, default=6)
    add_solver_arguments(parser)
    args = parser.parse_args(argv)

    built = build_model(args.order)
    solution = solve_from_args(built.model, args)
    print(f"prob076 order={args.order} status={solution.status}")
    if not solution.is_sat():
        return 1
    permutation = decode(built, solution)
    validate(permutation)
    print(f"permutation={[value + 1 for value in permutation]}")
    print(render(permutation))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
