"""CSPLib prob023: magic hexagons.

Specification: https://www.csplib.org/Problems/prob023/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import qayd as cp

from ..common import add_solver_arguments, solve_from_args

Coordinate = tuple[int, int]


@dataclass(frozen=True)
class MagicHexagonModel:
    model: cp.Model
    order: int
    magic_sum: int | None
    cells: dict[Coordinate, cp.IntVar]


def coordinates(order: int) -> tuple[Coordinate, ...]:
    radius = order - 1
    return tuple(
        (q, r)
        for r in range(-radius, radius + 1)
        for q in range(-radius, radius + 1)
        if -radius <= -q - r <= radius
    )


def build_model(order: int) -> MagicHexagonModel:
    if order < 1:
        raise ValueError("order must be positive")
    coords = coordinates(order)
    cell_count = len(coords)
    diameter = 2 * order - 1
    total = cell_count * (cell_count + 1) // 2
    magic_sum = total // diameter if total % diameter == 0 else None

    model = cp.Model()
    cells = {
        coord: model.int_var(1, cell_count, name=f"cell_{coord[0]}_{coord[1]}")
        for coord in coords
    }
    model.all_different(cells.values())
    if magic_sum is None:
        witness = next(iter(cells.values()))
        model.add(witness == witness + 1)
        return MagicHexagonModel(model, order, None, cells)

    radius = order - 1
    for axis in range(3):
        for coordinate in range(-radius, radius + 1):
            line = [
                variable
                for (q, r), variable in cells.items()
                if (q, r, -q - r)[axis] == coordinate
            ]
            model.add(sum(line) == magic_sum)
    if order > 1:
        model.add(cells[(-radius, 0)] < cells[(radius, 0)])
    return MagicHexagonModel(model, order, magic_sum, cells)


def decode(built: MagicHexagonModel, solution: cp.Solution) -> dict[Coordinate, int]:
    if not solution.is_sat():
        raise RuntimeError(f"no solution available, solver status is {solution.status}")
    return {
        coordinate: solution.value(variable)
        for coordinate, variable in built.cells.items()
    }


def validate(values_by_coordinate: dict[Coordinate, int], *, order: int) -> None:
    coords = coordinates(order)
    cell_count = len(coords)
    if set(values_by_coordinate) != set(coords):
        raise AssertionError("the hexagon has the wrong cells")
    if sorted(values_by_coordinate.values()) != list(range(1, cell_count + 1)):
        raise AssertionError("the hexagon must contain 1..cell_count exactly once")
    radius = order - 1
    line_sums = []
    for axis in range(3):
        for coordinate in range(-radius, radius + 1):
            line_sums.append(
                sum(
                    value
                    for (q, r), value in values_by_coordinate.items()
                    if (q, r, -q - r)[axis] == coordinate
                )
            )
    if len(set(line_sums)) != 1:
        raise AssertionError("all three families of lines must have the same sum")


def render(values_by_coordinate: dict[Coordinate, int], *, order: int) -> str:
    radius = order - 1
    width = len(str(len(values_by_coordinate)))
    lines = []
    for r in range(-radius, radius + 1):
        row = [
            (q, r) for q in range(-radius, radius + 1) if (q, r) in values_by_coordinate
        ]
        indent = " " * (abs(r) * (width + 1) // 2)
        lines.append(
            indent + " ".join(f"{values_by_coordinate[cell]:>{width}}" for cell in row)
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=int, default=3)
    add_solver_arguments(parser, time_limit=30)
    args = parser.parse_args(argv)

    built = build_model(args.order)
    solution = solve_from_args(built.model, args)
    print(
        f"prob023 order={args.order} magic_sum={built.magic_sum} status={solution.status}"
    )
    if not solution.is_sat():
        return 1
    values_by_coordinate = decode(built, solution)
    validate(values_by_coordinate, order=args.order)
    print(render(values_by_coordinate, order=args.order))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
