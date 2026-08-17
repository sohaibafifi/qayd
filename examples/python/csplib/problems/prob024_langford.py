"""CSPLib prob024: Langford's number problem.

Specification: https://www.csplib.org/Problems/prob024/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values


@dataclass(frozen=True)
class LangfordModel:
    model: cp.Model
    pair_count: int
    first: list[cp.IntVar]
    second: list[cp.IntVar]


def build_model(pair_count: int) -> LangfordModel:
    if pair_count < 2:
        raise ValueError("pair_count must be at least two")

    model = cp.Model()
    length = 2 * pair_count
    first: list[cp.IntVar] = []
    second: list[cp.IntVar] = []
    for number in range(1, pair_count + 1):
        left = model.int_var(0, length - number - 2, name=f"first_{number}")
        right = model.int_var(number + 1, length - 1, name=f"second_{number}")
        model.add(right == left + number + 1)
        first.append(left)
        second.append(right)
    model.all_different(first + second)

    if pair_count > 1:
        model.add(first[-1] <= (pair_count - 2) // 2)
    return LangfordModel(model, pair_count, first, second)


def decode(built: LangfordModel, solution: cp.Solution) -> list[int]:
    left_positions = values(solution, built.first)
    right_positions = values(solution, built.second)
    sequence = [0] * (2 * built.pair_count)
    for number, (left, right) in enumerate(
        zip(left_positions, right_positions), start=1
    ):
        sequence[left] = number
        sequence[right] = number
    return sequence


def validate(sequence: list[int], *, pair_count: int) -> None:
    if len(sequence) != 2 * pair_count:
        raise AssertionError(
            "a Langford sequence must contain two copies of every number"
        )
    for number in range(1, pair_count + 1):
        positions = [index for index, value in enumerate(sequence) if value == number]
        if len(positions) != 2:
            raise AssertionError(f"number {number} must occur exactly twice")
        if positions[1] - positions[0] != number + 1:
            raise AssertionError(
                f"the two {number} values must have {number} values between them"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=4)
    add_solver_arguments(parser)
    args = parser.parse_args(argv)

    built = build_model(args.pairs)
    solution = solve_from_args(built.model, args)
    print(f"prob024 pairs={args.pairs} status={solution.status}")
    if not solution.is_sat():
        return 1
    sequence = decode(built, solution)
    validate(sequence, pair_count=args.pairs)
    print(f"sequence={sequence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
