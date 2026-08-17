"""CSPLib prob084: two-circulant-core Hadamard Legendre pairs.

Specification: https://www.csplib.org/Problems/prob084/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values


@dataclass(frozen=True)
class HadamardModel:
    model: cp.Model
    order: int
    first: list[cp.IntVar]
    second: list[cp.IntVar]


def build_model(order: int) -> HadamardModel:
    if order < 3 or order % 2 == 0:
        raise ValueError("order must be odd and at least three")
    model = cp.Model()
    first = model.int_vars(order, -1, 1, name="first")
    second = model.int_vars(order, -1, 1, name="second")
    unary = [(-1,), (1,)]
    for variable in (*first, *second):
        model.table([variable], unary)
    model.add(sum(first) == 1)
    model.add(sum(second) == 1)
    product_table = [
        (left, right, left * right) for left in (-1, 1) for right in (-1, 1)
    ]
    for shift in range(1, (order - 1) // 2 + 1):
        products = []
        for sequence_index, sequence in enumerate((first, second)):
            for index in range(order):
                product_variable = model.int_var(
                    -1,
                    1,
                    name=f"product_{sequence_index}_{shift}_{index}",
                )
                model.table(
                    [
                        sequence[index],
                        sequence[(index + shift) % order],
                        product_variable,
                    ],
                    product_table,
                )
                products.append(product_variable)
        model.add(sum(products) == -2)
    model.add(first[0] == 1)
    return HadamardModel(model, order, first, second)


def decode(built: HadamardModel, solution: cp.Solution) -> tuple[list[int], list[int]]:
    return values(solution, built.first), values(solution, built.second)


def validate(built: HadamardModel, first: list[int], second: list[int]) -> None:
    if len(first) != built.order or len(second) != built.order:
        raise AssertionError("the sequence lengths are invalid")
    if any(value not in (-1, 1) for value in (*first, *second)):
        raise AssertionError("a sequence value is not -1 or 1")
    if sum(first) != 1 or sum(second) != 1:
        raise AssertionError("a sequence sum is not one")
    for shift in range(1, (built.order - 1) // 2 + 1):
        correlation = sum(
            first[index] * first[(index + shift) % built.order]
            for index in range(built.order)
        )
        correlation += sum(
            second[index] * second[(index + shift) % built.order]
            for index in range(built.order)
        )
        if correlation != -2:
            raise AssertionError("the Legendre autocorrelation condition is violated")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=int, default=5)
    add_solver_arguments(parser, time_limit=30)
    args = parser.parse_args(argv)
    built = build_model(args.order)
    solution = solve_from_args(built.model, args)
    print(f"prob084 order={args.order} status={solution.status}")
    if not solution.is_sat():
        return 1
    first, second = decode(built, solution)
    validate(built, first, second)
    print(f"first={first} second={second}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
