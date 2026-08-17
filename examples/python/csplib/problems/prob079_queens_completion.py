"""CSPLib prob079: n-queens completion and excluded diagonals.

Specification: https://www.csplib.org/Problems/prob079/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import qayd as cp

from ..common import add_solver_arguments, solve_from_args

Square = tuple[int, int]


@dataclass(frozen=True)
class QueensVariantModel:
    model: cp.Model
    size: int
    columns: tuple[int, ...]
    allowed_rows: tuple[int, ...]
    rows: list[cp.IntVar]
    preplaced: tuple[Square, ...] = ()
    excluded_sums: tuple[int, ...] = ()
    excluded_differences: tuple[int, ...] = ()


def post_queen_constraints(
    model: cp.Model,
    size: int,
    columns: tuple[int, ...],
    rows: list[cp.IntVar],
) -> None:
    model.all_different(rows)
    sum_diagonals: list[cp.IntVar] = []
    difference_diagonals: list[cp.IntVar] = []
    for column, row in zip(columns, rows):
        sum_diagonal = model.int_var(0, 2 * size - 2, name=f"sum_diag_{column}")
        difference_diagonal = model.int_var(
            -(size - 1), size - 1, name=f"diff_diag_{column}"
        )
        model.add(sum_diagonal == row + column)
        model.add(difference_diagonal == row - column)
        sum_diagonals.append(sum_diagonal)
        difference_diagonals.append(difference_diagonal)
    if len(rows) > 1:
        model.all_different(sum_diagonals)
        model.all_different(difference_diagonals)


def _check_squares(size: int, squares: tuple[Square, ...]) -> None:
    if any(
        row < 0 or column < 0 or row >= size or column >= size
        for row, column in squares
    ):
        raise ValueError("a queen coordinate is outside the board")


def build_completion_model(
    size: int, *, preplaced: tuple[Square, ...] = ()
) -> QueensVariantModel:
    if size < 1:
        raise ValueError("size must be positive")
    _check_squares(size, preplaced)

    model = cp.Model()
    columns = tuple(range(size))
    allowed_rows = tuple(range(size))
    rows = model.int_vars(size, 0, size - 1, name="row")
    post_queen_constraints(model, size, columns, rows)
    for row, column in preplaced:
        model.add(rows[column] == row)
    return QueensVariantModel(
        model, size, columns, allowed_rows, rows, preplaced=preplaced
    )


def _normalized_subset(
    size: int, values: tuple[int, ...] | None, name: str
) -> tuple[int, ...]:
    normalized = tuple(range(size)) if values is None else tuple(sorted(set(values)))
    if any(value < 0 or value >= size for value in normalized):
        raise ValueError(f"{name} contains an index outside the board")
    return normalized


def build_excluded_diagonals_model(
    size: int,
    *,
    allowed_rows: tuple[int, ...] | None = None,
    allowed_columns: tuple[int, ...] | None = None,
    excluded_sums: tuple[int, ...] = (),
    excluded_differences: tuple[int, ...] = (),
) -> QueensVariantModel:
    if size < 1:
        raise ValueError("size must be positive")
    row_values = _normalized_subset(size, allowed_rows, "allowed_rows")
    columns = _normalized_subset(size, allowed_columns, "allowed_columns")
    if len(row_values) != len(columns):
        raise ValueError("allowed row and column sets must have equal cardinality")
    if not row_values:
        raise ValueError("at least one row and column must be allowed")

    model = cp.Model()
    rows = [
        model.int_var(values=row_values, name=f"row_{column}") for column in columns
    ]
    post_queen_constraints(model, size, columns, rows)
    for column, row in zip(columns, rows):
        for diagonal in excluded_sums:
            model.add(row + column != diagonal)
        for diagonal in excluded_differences:
            model.add(row - column != diagonal)
    return QueensVariantModel(
        model,
        size,
        columns,
        row_values,
        rows,
        excluded_sums=tuple(sorted(set(excluded_sums))),
        excluded_differences=tuple(sorted(set(excluded_differences))),
    )


def build_model(
    size: int = 8,
    *,
    variant: str = "completion",
    preplaced: tuple[Square, ...] = (),
    allowed_rows: tuple[int, ...] | None = None,
    allowed_columns: tuple[int, ...] | None = None,
    excluded_sums: tuple[int, ...] = (),
    excluded_differences: tuple[int, ...] = (),
) -> QueensVariantModel:
    """Build the completion or excluded-diagonals variant."""

    if variant == "completion":
        return build_completion_model(size, preplaced=preplaced)
    if variant == "excluded":
        return build_excluded_diagonals_model(
            size,
            allowed_rows=allowed_rows,
            allowed_columns=allowed_columns,
            excluded_sums=excluded_sums,
            excluded_differences=excluded_differences,
        )
    raise ValueError("variant must be 'completion' or 'excluded'")


def decode(built: QueensVariantModel, solution: cp.Solution) -> dict[int, int]:
    if not solution.is_sat():
        raise RuntimeError(f"no solution available, solver status is {solution.status}")
    return {
        column: solution.value(row) for column, row in zip(built.columns, built.rows)
    }


def validate(built: QueensVariantModel, row_by_column: dict[int, int]) -> None:
    if set(row_by_column) != set(built.columns):
        raise AssertionError("the assignment must contain every allowed column")
    rows = list(row_by_column.values())
    if set(rows) != set(built.allowed_rows):
        raise AssertionError("the assignment must contain every allowed row")
    sums = [row + column for column, row in row_by_column.items()]
    differences = [row - column for column, row in row_by_column.items()]
    if len(set(sums)) != len(sums) or len(set(differences)) != len(differences):
        raise AssertionError("two queens share a diagonal")
    if set(built.preplaced).difference(
        (row, column) for column, row in row_by_column.items()
    ):
        raise AssertionError("a preplaced queen is missing")
    if set(sums).intersection(built.excluded_sums):
        raise AssertionError("a queen lies on an excluded sum diagonal")
    if set(differences).intersection(built.excluded_differences):
        raise AssertionError("a queen lies on an excluded difference diagonal")


def _parse_pair(value: str) -> tuple[int, int]:
    try:
        left, right = value.split(",", maxsplit=1)
        return int(left), int(right)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected two comma-separated integers"
        ) from error


def _parse_indices(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    return tuple(int(item) for item in value.split(",") if item.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=("completion", "excluded"), default="completion"
    )
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument(
        "--queen",
        type=_parse_pair,
        action="append",
        help="preplaced ROW,COLUMN, zero based",
    )
    parser.add_argument("--allowed-rows", help="comma-separated zero-based row indices")
    parser.add_argument(
        "--allowed-columns", help="comma-separated zero-based column indices"
    )
    parser.add_argument("--exclude-sum", type=int, action="append", default=[])
    parser.add_argument("--exclude-difference", type=int, action="append", default=[])
    add_solver_arguments(parser)
    args = parser.parse_args(argv)

    if args.variant == "completion":
        preplaced = tuple(args.queen if args.queen is not None else [(0, 0)])
        built = build_completion_model(args.size, preplaced=preplaced)
    else:
        built = build_excluded_diagonals_model(
            args.size,
            allowed_rows=_parse_indices(args.allowed_rows),
            allowed_columns=_parse_indices(args.allowed_columns),
            excluded_sums=tuple(args.exclude_sum),
            excluded_differences=tuple(args.exclude_difference),
        )
    solution = solve_from_args(built.model, args)
    print(f"prob079 variant={args.variant} size={args.size} status={solution.status}")
    if not solution.is_sat():
        return 1
    row_by_column = decode(built, solution)
    validate(built, row_by_column)
    print(f"queens={sorted((row, column) for column, row in row_by_column.items())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
