"""CSPLib prob041: the n-fractions puzzle.

Specification: https://www.csplib.org/Problems/prob041/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
from math import ceil

import qayd as cp

from ..common import add_solver_arguments, solve_from_args


@dataclass(frozen=True)
class FractionsModel:
    model: cp.Model
    term_count: int
    target: int
    digits: list[list[cp.IntVar]]


def build_model(term_count: int = 3, target: int = 1) -> FractionsModel:
    if term_count < 1 or target < 1:
        raise ValueError("term_count and target must be positive")
    model = cp.Model()
    digits = [
        model.int_vars(3, 1, 9, name=f"fraction_{index}") for index in range(term_count)
    ]
    denominators = [10 * term[1] + term[2] for term in digits]

    def multiply(expressions):
        result = 1
        for expression in expressions:
            result = result * expression
        return result

    common_denominator = multiply(denominators)
    scaled_numerators = [
        digits[index][0]
        * multiply(
            denominator
            for other, denominator in enumerate(denominators)
            if other != index
        )
        for index in range(term_count)
    ]
    model.add(sum(scaled_numerators) == target * common_denominator)

    flattened = [digit for term in digits for digit in term]
    model.cardinality(
        flattened,
        list(range(1, 10)),
        [1] * 9,
        [ceil(term_count / 3)] * 9,
        closed=True,
    )
    for index in range(term_count - 1):
        left = 100 * digits[index][0] + 10 * digits[index][1] + digits[index][2]
        right = (
            100 * digits[index + 1][0]
            + 10 * digits[index + 1][1]
            + digits[index + 1][2]
        )
        model.add(left <= right)
    return FractionsModel(model, term_count, target, digits)


def decode(built: FractionsModel, solution: cp.Solution) -> list[tuple[int, int, int]]:
    if not solution.is_sat():
        raise RuntimeError(f"no solution available, solver status is {solution.status}")
    return [tuple(solution.value(variable) for variable in row) for row in built.digits]


def validate(built: FractionsModel, fractions: list[tuple[int, int, int]]) -> None:
    if len(fractions) != built.term_count:
        raise AssertionError("the number of fractions is invalid")
    digits = [digit for fraction in fractions for digit in fraction]
    upper = ceil(built.term_count / 3)
    if any(not 1 <= digit <= 9 for digit in digits):
        raise AssertionError("a digit is outside 1 through 9")
    if any(not 1 <= digits.count(value) <= upper for value in range(1, 10)):
        raise AssertionError("digit occurrence bounds are violated")
    total = sum((Fraction(x, 10 * y + z) for x, y, z in fractions), start=Fraction(0))
    if total != built.target:
        raise AssertionError("the fractions do not sum to the target")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terms", type=int, default=3)
    parser.add_argument("--target", type=int, default=1)
    add_solver_arguments(parser, time_limit=30)
    args = parser.parse_args(argv)
    built = build_model(args.terms, args.target)
    solution = solve_from_args(built.model, args)
    print(f"prob041 terms={args.terms} target={args.target} status={solution.status}")
    if not solution.is_sat():
        return 1
    fractions = decode(built, solution)
    validate(built, fractions)
    print(" + ".join(f"{x}/{y}{z}" for x, y, z in fractions), f"= {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
