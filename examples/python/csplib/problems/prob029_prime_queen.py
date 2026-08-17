"""CSPLib prob029: prime queen attacking problem.

Specification: https://www.csplib.org/Problems/prob029/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values


@dataclass(frozen=True)
class PrimeQueenModel:
    model: cp.Model
    size: int
    locations: list[cp.IntVar]
    queen: cp.IntVar
    primes: tuple[int, ...]
    free: list[cp.IntVar]


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    divisor = 2
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 1
    return True


def _knight_moves(size: int) -> list[tuple[int, int]]:
    return [
        (left, right)
        for left in range(size * size)
        for right in range(size * size)
        if sorted(
            (
                abs(left // size - right // size),
                abs(left % size - right % size),
            )
        )
        == [1, 2]
    ]


def _queen_attacks(size: int, queen: int, target: int) -> bool:
    if queen == target:
        return False
    queen_row, queen_column = divmod(queen, size)
    target_row, target_column = divmod(target, size)
    return (
        queen_row == target_row
        or queen_column == target_column
        or abs(queen_row - target_row) == abs(queen_column - target_column)
    )


def build_model(size: int) -> PrimeQueenModel:
    if size < 1:
        raise ValueError("size must be positive")
    cell_count = size * size
    model = cp.Model()
    locations = model.int_vars(cell_count, 0, cell_count - 1, name="number_location")
    queen = model.int_var(0, cell_count - 1, name="queen")
    model.all_different(locations)

    moves = _knight_moves(size)
    for number in range(cell_count - 1):
        model.table([locations[number], locations[number + 1]], moves)

    primes = tuple(number for number in range(2, cell_count + 1) if _is_prime(number))
    free = [model.bool_var(name=f"prime_{prime}_free") for prime in primes]
    attack_table = [
        (
            queen_cell,
            target_cell,
            int(not _queen_attacks(size, queen_cell, target_cell)),
        )
        for queen_cell in range(cell_count)
        for target_cell in range(cell_count)
    ]
    for prime, variable in zip(primes, free):
        model.table([queen, locations[prime - 1], variable], attack_table)
    if free:
        model.minimize(sum(free))
    return PrimeQueenModel(model, size, locations, queen, primes, free)


def decode(built: PrimeQueenModel, solution: cp.Solution) -> tuple[list[int], int]:
    if not solution.is_sat():
        raise RuntimeError(f"no solution available, solver status is {solution.status}")
    return values(solution, built.locations), solution.value(built.queen)


def validate(
    built: PrimeQueenModel,
    locations: list[int],
    queen: int,
    objective: int | None,
) -> None:
    cell_count = built.size * built.size
    if sorted(locations) != list(range(cell_count)):
        raise AssertionError("the numbered cells are not a permutation of the board")
    moves = set(_knight_moves(built.size))
    if any(pair not in moves for pair in zip(locations, locations[1:])):
        raise AssertionError("consecutive numbers are not connected by knight moves")
    if queen < 0 or queen >= cell_count:
        raise AssertionError("the queen cell is invalid")
    free_count = sum(
        not _queen_attacks(built.size, queen, locations[prime - 1])
        for prime in built.primes
    )
    if objective is not None and free_count != objective:
        raise AssertionError("the objective does not match the number of free primes")


def render(built: PrimeQueenModel, locations: list[int], queen: int) -> str:
    numbers = [0] * (built.size * built.size)
    for number, cell in enumerate(locations, start=1):
        numbers[cell] = number
    width = len(str(len(numbers)))
    return "\n".join(
        " ".join(
            f"{numbers[row * built.size + column]:>{width}}"
            + ("Q" if row * built.size + column == queen else " ")
            for column in range(built.size)
        )
        for row in range(built.size)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=5)
    add_solver_arguments(parser, time_limit=30)
    args = parser.parse_args(argv)

    built = build_model(args.size)
    solution = solve_from_args(built.model, args)
    print(f"prob029 size={args.size} status={solution.status}")
    if not solution.is_sat():
        return 1
    locations, queen = decode(built, solution)
    validate(built, locations, queen, solution.objective)
    free_count = sum(
        not _queen_attacks(built.size, queen, locations[prime - 1])
        for prime in built.primes
    )
    print(f"free_primes={free_count} queen={divmod(queen, built.size)}")
    print(render(built, locations, queen))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
