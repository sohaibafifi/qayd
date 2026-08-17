"""CSPLib prob132: layout problem.

Specification: https://www.csplib.org/Problems/prob132/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args

DEFAULT_INSTANCE = {
    "grid": [[1, 1, 1, 1], [1, 1, 0, 1], [1, 1, 1, 1]],
    "shapes": [
        [[1, 1]],
        [[1], [1]],
        [[1, 1], [1, 0]],
    ],
}


@dataclass(frozen=True)
class LayoutInstance:
    grid: tuple[tuple[int, ...], ...]
    shapes: tuple[tuple[tuple[int, ...], ...], ...]


@dataclass(frozen=True)
class ShapePlacement:
    row: int
    column: int
    cells: frozenset[tuple[int, int]]


@dataclass(frozen=True)
class LayoutModel:
    model: cp.Model
    instance: LayoutInstance
    placements: list[list[ShapePlacement]]
    selected: list[list[cp.IntVar]]


def parse_instance(data: str | bytes) -> LayoutInstance:
    raw = json.loads(data)
    try:
        grid = tuple(tuple(int(value) for value in row) for row in raw["grid"])
        shapes = tuple(
            tuple(tuple(int(value) for value in row) for row in shape)
            for shape in raw["shapes"]
        )
        return LayoutInstance(grid, shapes)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid layout JSON instance") from error


def load_instance(path: str | Path) -> LayoutInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def _placements(
    instance: LayoutInstance, shape: tuple[tuple[int, ...], ...]
) -> list[ShapePlacement]:
    rows = len(instance.grid)
    columns = len(instance.grid[0])
    height = len(shape)
    width = len(shape[0])
    result = []
    for row in range(rows - height + 1):
        for column in range(columns - width + 1):
            cells = frozenset(
                (row + local_row, column + local_column)
                for local_row in range(height)
                for local_column in range(width)
                if shape[local_row][local_column]
            )
            if cells and all(
                instance.grid[cell_row][cell_column] for cell_row, cell_column in cells
            ):
                result.append(ShapePlacement(row, column, cells))
    return result


def build_model(instance: LayoutInstance) -> LayoutModel:
    if not instance.grid or not instance.grid[0] or not instance.shapes:
        raise ValueError("the grid and shape list must be non-empty")
    columns = len(instance.grid[0])
    if any(
        len(row) != columns or any(value not in (0, 1) for value in row)
        for row in instance.grid
    ):
        raise ValueError("the grid is not a rectangular binary matrix")
    if any(
        not shape
        or not shape[0]
        or any(
            len(row) != len(shape[0]) or any(value not in (0, 1) for value in row)
            for row in shape
        )
        for shape in instance.shapes
    ):
        raise ValueError("a shape is not a rectangular binary matrix")
    placements = [_placements(instance, shape) for shape in instance.shapes]
    if any(not options for options in placements):
        raise ValueError("a shape has no valid placement")
    model = cp.Model()
    selected = [
        [
            model.bool_var(name=f"shape_{shape}_placement_{index}")
            for index in range(len(options))
        ]
        for shape, options in enumerate(placements)
    ]
    for variables in selected:
        model.add(sum(variables) == 1)
    for row in range(len(instance.grid)):
        for column in range(columns):
            covering = [
                selected[shape][index]
                for shape, options in enumerate(placements)
                for index, placement in enumerate(options)
                if (row, column) in placement.cells
            ]
            if covering:
                model.add(sum(covering) <= 1)
    return LayoutModel(model, instance, placements, selected)


def decode(built: LayoutModel, solution: cp.Solution) -> list[ShapePlacement]:
    if not solution.is_sat():
        raise RuntimeError(f"no solution available, solver status is {solution.status}")
    return [
        next(
            placement
            for placement, variable in zip(options, variables)
            if solution.value(variable)
        )
        for options, variables in zip(built.placements, built.selected)
    ]


def validate(built: LayoutModel, placements: list[ShapePlacement]) -> None:
    if len(placements) != len(built.instance.shapes):
        raise AssertionError("the number of shape placements is invalid")
    occupied = set()
    for shape, placement in zip(built.instance.shapes, placements):
        expected = {
            (placement.row + row, placement.column + column)
            for row in range(len(shape))
            for column in range(len(shape[0]))
            if shape[row][column]
        }
        if expected != set(placement.cells):
            raise AssertionError("a shape placement does not match its bitmap")
        if any(not built.instance.grid[row][column] for row, column in expected):
            raise AssertionError("a shape covers an unavailable grid cell")
        if occupied.intersection(expected):
            raise AssertionError("two shapes overlap")
        occupied.update(expected)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON layout instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob132 shapes={len(instance.shapes)} status={solution.status}")
    if not solution.is_sat():
        return 1
    placements = decode(built, solution)
    validate(built, placements)
    print(f"origins={[(placement.row, placement.column) for placement in placements]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
