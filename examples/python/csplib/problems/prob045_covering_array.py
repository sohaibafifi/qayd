"""CSPLib prob045: covering arrays.

Specification: https://www.csplib.org/Problems/prob045/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations, product

import qayd as cp

from ..common import add_solver_arguments, solve_from_args


@dataclass(frozen=True)
class CoveringArrayModel:
    model: cp.Model
    strength: int
    row_count: int
    alphabet_size: int
    column_count: int
    array: list[list[cp.IntVar]]


def build_model(
    strength: int,
    row_count: int,
    alphabet_size: int,
    column_count: int,
) -> CoveringArrayModel:
    if not 1 <= strength <= row_count:
        raise ValueError("strength must be between one and row_count")
    if alphabet_size < 2 or column_count < 1:
        raise ValueError("alphabet_size and column_count must be positive")
    model = cp.Model()
    array = [
        model.int_vars(column_count, 0, alphabet_size - 1, name=f"row_{row}")
        for row in range(row_count)
    ]
    for chosen_rows in combinations(range(row_count), strength):
        for requested in product(range(alphabet_size), repeat=strength):
            matches = []
            allowed = [
                (*values, int(values == requested))
                for values in product(range(alphabet_size), repeat=strength)
            ]
            for column in range(column_count):
                match = model.bool_var(
                    name=f"cover_{'_'.join(map(str, chosen_rows))}_{'_'.join(map(str, requested))}_{column}"
                )
                model.table(
                    [*(array[row][column] for row in chosen_rows), match], allowed
                )
                matches.append(match)
            model.add(sum(matches) >= 1)
    for row in range(row_count):
        model.add(array[row][0] == 0)
    return CoveringArrayModel(
        model,
        strength,
        row_count,
        alphabet_size,
        column_count,
        array,
    )


def decode(built: CoveringArrayModel, solution: cp.Solution) -> list[list[int]]:
    if not solution.is_sat():
        raise RuntimeError(f"no solution available, solver status is {solution.status}")
    return [[solution.value(variable) for variable in row] for row in built.array]


def validate(built: CoveringArrayModel, array: list[list[int]]) -> None:
    if len(array) != built.row_count or any(
        len(row) != built.column_count for row in array
    ):
        raise AssertionError("the covering array dimensions are invalid")
    for chosen_rows in combinations(range(built.row_count), built.strength):
        observed = {
            tuple(array[row][column] for row in chosen_rows)
            for column in range(built.column_count)
        }
        expected = set(product(range(built.alphabet_size), repeat=built.strength))
        if observed != expected:
            raise AssertionError("a row subset does not cover every value tuple")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strength", type=int, default=2)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--alphabet", type=int, default=2)
    parser.add_argument("--columns", type=int, default=4)
    add_solver_arguments(parser, time_limit=30)
    args = parser.parse_args(argv)
    built = build_model(args.strength, args.rows, args.alphabet, args.columns)
    solution = solve_from_args(built.model, args)
    print(
        f"prob045 parameters={(args.strength, args.rows, args.alphabet, args.columns)} status={solution.status}"
    )
    if not solution.is_sat():
        return 1
    array = decode(built, solution)
    validate(built, array)
    for row in array:
        print(" ".join(map(str, row)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
