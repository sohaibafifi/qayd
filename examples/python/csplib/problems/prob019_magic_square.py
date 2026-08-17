"""CSPLib prob019: magic squares and magic sequences.

Specification: https://www.csplib.org/Problems/prob019/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values


@dataclass(frozen=True)
class MagicSquareModel:
    model: cp.Model
    order: int
    cells: list[list[cp.IntVar]]


@dataclass(frozen=True)
class MagicSequenceModel:
    model: cp.Model
    sequence: list[cp.IntVar]


def build_magic_square(order: int) -> MagicSquareModel:
    if order < 1:
        raise ValueError("order must be positive")

    model = cp.Model()
    flat = model.int_vars(order * order, 1, order * order, name="cell")
    cells = [flat[row * order : (row + 1) * order] for row in range(order)]
    model.all_different(flat)
    magic_sum = order * (order * order + 1) // 2
    for row in cells:
        model.add(sum(row) == magic_sum)
    for column in range(order):
        model.add(sum(cells[row][column] for row in range(order)) == magic_sum)
    model.add(sum(cells[index][index] for index in range(order)) == magic_sum)
    model.add(
        sum(cells[index][order - index - 1] for index in range(order)) == magic_sum
    )

    if order > 1:
        model.add(cells[0][0] < cells[0][-1])
    return MagicSquareModel(model, order, cells)


def decode_magic_square(
    built: MagicSquareModel, solution: cp.Solution
) -> list[list[int]]:
    return [values(solution, row) for row in built.cells]


def validate_magic_square(square: list[list[int]]) -> None:
    order = len(square)
    if order == 0 or any(len(row) != order for row in square):
        raise AssertionError("a magic square must be a non-empty square matrix")
    flat = [value for row in square for value in row]
    if sorted(flat) != list(range(1, order * order + 1)):
        raise AssertionError("cells must contain 1..n^2 exactly once")
    expected = order * (order * order + 1) // 2
    lines = list(square)
    lines.extend(
        [[square[row][column] for row in range(order)] for column in range(order)]
    )
    lines.append([square[index][index] for index in range(order)])
    lines.append([square[index][order - index - 1] for index in range(order)])
    if any(sum(line) != expected for line in lines):
        raise AssertionError(
            "every row, column, and main diagonal must have the magic sum"
        )


def build_magic_sequence(length: int) -> MagicSequenceModel:
    if length < 1:
        raise ValueError("length must be positive")

    model = cp.Model()
    sequence = model.int_vars(length, 0, length - 1, name="value")
    indicators = [
        [model.bool_var(name=f"at_{position}_is_{value}") for value in range(length)]
        for position in range(length)
    ]
    for position in range(length):
        for value in range(length):
            allowed = [
                (candidate, int(candidate == value)) for candidate in range(length)
            ]
            model.table([sequence[position], indicators[position][value]], allowed)
    for value in range(length):
        model.add(
            sum(indicators[position][value] for position in range(length))
            == sequence[value]
        )
    model.add(sum(sequence) == length)
    return MagicSequenceModel(model, sequence)


def decode_magic_sequence(
    built: MagicSequenceModel, solution: cp.Solution
) -> list[int]:
    return values(solution, built.sequence)


def validate_magic_sequence(sequence: list[int]) -> None:
    if any(value < 0 or value >= len(sequence) for value in sequence):
        raise AssertionError("magic-sequence values must be in 0..n-1")
    counts = [sequence.count(value) for value in range(len(sequence))]
    if counts != sequence:
        raise AssertionError("each entry must equal its value's occurrence count")


def build_model(
    size: int = 3, *, variant: str = "square"
) -> MagicSquareModel | MagicSequenceModel:
    """Build the magic-square or magic-sequence variant."""

    if variant == "square":
        return build_magic_square(size)
    if variant == "sequence":
        return build_magic_sequence(size)
    raise ValueError("variant must be 'square' or 'sequence'")


def decode(
    built: MagicSquareModel | MagicSequenceModel,
    solution: cp.Solution,
) -> list[list[int]] | list[int]:
    if isinstance(built, MagicSquareModel):
        return decode_magic_square(built, solution)
    return decode_magic_sequence(built, solution)


def validate(
    built: MagicSquareModel | MagicSequenceModel,
    result: list[list[int]] | list[int],
) -> None:
    if isinstance(built, MagicSquareModel):
        validate_magic_square(result)
    else:
        validate_magic_sequence(result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("square", "sequence"), default="square")
    parser.add_argument("--size", type=int)
    add_solver_arguments(parser)
    args = parser.parse_args(argv)

    if args.variant == "square":
        size = 3 if args.size is None else args.size
        built = build_magic_square(size)
        solution = solve_from_args(built.model, args)
        print(f"prob019 variant=square order={size} status={solution.status}")
        if not solution.is_sat():
            return 1
        square = decode_magic_square(built, solution)
        validate_magic_square(square)
        for row in square:
            print(" ".join(map(str, row)))
        return 0

    size = 10 if args.size is None else args.size
    built = build_magic_sequence(size)
    solution = solve_from_args(built.model, args)
    print(f"prob019 variant=sequence length={size} status={solution.status}")
    if not solution.is_sat():
        return 1
    sequence = decode_magic_sequence(built, solution)
    validate_magic_sequence(sequence)
    print(f"sequence={sequence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
