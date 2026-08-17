"""CSPLib prob049: equal-sums number partitioning.

Specification: https://www.csplib.org/Problems/prob049/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values


@dataclass(frozen=True)
class NumberPartitionModel:
    model: cp.Model
    size: int
    highest_power: int
    in_second_set: list[cp.IntVar]


def build_model(size: int, *, highest_power: int = 2) -> NumberPartitionModel:
    if size < 1:
        raise ValueError("size must be positive")
    if highest_power < 0:
        raise ValueError("highest_power must be non-negative")

    model = cp.Model()
    in_second_set = [
        model.bool_var(name=f"in_b_{number}") for number in range(1, size + 1)
    ]

    totals = [
        sum(number**power for number in range(1, size + 1))
        for power in range(highest_power + 1)
    ]
    if any(total % 2 for total in totals):
        model.add(in_second_set[0] != in_second_set[0])
    else:
        for power, total in enumerate(totals):
            model.add(
                sum(
                    (number**power) * in_second_set[number - 1]
                    for number in range(1, size + 1)
                )
                == total // 2
            )
        model.add(in_second_set[0] == 0)
    return NumberPartitionModel(model, size, highest_power, in_second_set)


def decode(
    built: NumberPartitionModel, solution: cp.Solution
) -> tuple[list[int], list[int]]:
    membership = values(solution, built.in_second_set)
    first = [
        number for number, in_second in enumerate(membership, start=1) if not in_second
    ]
    second = [
        number for number, in_second in enumerate(membership, start=1) if in_second
    ]
    return first, second


def validate(
    first: list[int], second: list[int], *, size: int, highest_power: int
) -> None:
    if sorted(first + second) != list(range(1, size + 1)) or set(first).intersection(
        second
    ):
        raise AssertionError("the sets must partition 1..N")
    for power in range(highest_power + 1):
        left = sum(number**power for number in first)
        right = sum(number**power for number in second)
        if left != right:
            raise AssertionError(f"power-{power} sums differ: {left} != {right}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--power", type=int, default=2)
    add_solver_arguments(parser)
    args = parser.parse_args(argv)

    built = build_model(args.size, highest_power=args.power)
    solution = solve_from_args(built.model, args)
    print(f"prob049 size={args.size} power={args.power} status={solution.status}")
    if not solution.is_sat():
        return 1
    first, second = decode(built, solution)
    validate(first, second, size=args.size, highest_power=args.power)
    print(f"A={first}")
    print(f"B={second}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
