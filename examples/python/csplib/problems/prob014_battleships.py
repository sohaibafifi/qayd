"""CSPLib prob014: Solitaire Battleships.

Specification: https://www.csplib.org/Problems/prob014/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args

DEFAULT_INSTANCE = {
    "row_counts": [3, 0, 1, 1, 1],
    "column_counts": [2, 1, 1, 0, 2],
    "fleet": [3, 2, 1],
    "clues": [
        {"row": 0, "column": 0, "kind": "left"},
        {"row": 1, "column": 1, "kind": "water"},
        {"row": 4, "column": 0, "kind": "submarine"},
    ],
}


@dataclass(frozen=True)
class Clue:
    row: int
    column: int
    kind: str


@dataclass(frozen=True)
class BattleshipsInstance:
    row_counts: tuple[int, ...]
    column_counts: tuple[int, ...]
    fleet: tuple[int, ...]
    clues: tuple[Clue, ...]


@dataclass(frozen=True)
class Placement:
    cells: tuple[tuple[int, int], ...]
    roles: tuple[str, ...]


@dataclass(frozen=True)
class BattleshipsModel:
    model: cp.Model
    instance: BattleshipsInstance
    placements: list[list[Placement]]
    selected: list[list[cp.IntVar]]


def parse_instance(data: str | bytes) -> BattleshipsInstance:
    raw = json.loads(data)
    try:
        clues = tuple(
            Clue(int(clue["row"]), int(clue["column"]), str(clue["kind"]).lower())
            for clue in raw.get("clues", [])
        )
        return BattleshipsInstance(
            tuple(int(value) for value in raw["row_counts"]),
            tuple(int(value) for value in raw["column_counts"]),
            tuple(int(value) for value in raw["fleet"]),
            clues,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid Battleships JSON instance") from error


def load_instance(path: str | Path) -> BattleshipsInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def _placements(rows: int, columns: int, length: int) -> list[Placement]:
    if length == 1:
        return [
            Placement(((row, column),), ("submarine",))
            for row in range(rows)
            for column in range(columns)
        ]

    placements = []
    for row in range(rows):
        for column in range(columns - length + 1):
            cells = tuple((row, column + offset) for offset in range(length))
            roles = ("left", *("middle" for _ in range(length - 2)), "right")
            placements.append(Placement(cells, roles))
    for row in range(rows - length + 1):
        for column in range(columns):
            cells = tuple((row + offset, column) for offset in range(length))
            roles = ("top", *("middle" for _ in range(length - 2)), "bottom")
            placements.append(Placement(cells, roles))
    return placements


def _touch(left: Placement, right: Placement) -> bool:
    return any(
        max(abs(left_row - right_row), abs(left_column - right_column)) <= 1
        for left_row, left_column in left.cells
        for right_row, right_column in right.cells
    )


def _role_matches(actual: str, requested: str) -> bool:
    if requested in {"occupied", "ship"}:
        return True
    if requested == "end":
        return actual in {"left", "right", "top", "bottom"}
    return actual == requested


def build_model(instance: BattleshipsInstance) -> BattleshipsModel:
    rows = len(instance.row_counts)
    columns = len(instance.column_counts)
    if rows < 1 or columns < 1 or not instance.fleet:
        raise ValueError("the board and fleet must be non-empty")
    if any(length < 1 or length > max(rows, columns) for length in instance.fleet):
        raise ValueError("a ship has an invalid length")
    if any(count < 0 or count > columns for count in instance.row_counts):
        raise ValueError("a row count is invalid")
    if any(count < 0 or count > rows for count in instance.column_counts):
        raise ValueError("a column count is invalid")
    occupied = sum(instance.fleet)
    if sum(instance.row_counts) != occupied or sum(instance.column_counts) != occupied:
        raise ValueError("row and column counts must equal the fleet area")

    allowed_clues = {
        "water",
        "occupied",
        "ship",
        "submarine",
        "middle",
        "end",
        "left",
        "right",
        "top",
        "bottom",
    }
    if any(
        clue.row < 0
        or clue.row >= rows
        or clue.column < 0
        or clue.column >= columns
        or clue.kind not in allowed_clues
        for clue in instance.clues
    ):
        raise ValueError("a clue is invalid")

    model = cp.Model()
    placements = [_placements(rows, columns, length) for length in instance.fleet]
    selected = [
        [
            model.bool_var(name=f"ship_{ship}_placement_{index}")
            for index in range(len(options))
        ]
        for ship, options in enumerate(placements)
    ]
    for variables in selected:
        model.add(sum(variables) == 1)

    for first in range(len(instance.fleet)):
        for second in range(first + 1, len(instance.fleet)):
            for left_index, left in enumerate(placements[first]):
                for right_index, right in enumerate(placements[second]):
                    if _touch(left, right):
                        model.add(
                            selected[first][left_index] + selected[second][right_index]
                            <= 1
                        )

    for row, count in enumerate(instance.row_counts):
        model.add(
            sum(
                variable * sum(cell_row == row for cell_row, _ in placement.cells)
                for options, variables in zip(placements, selected)
                for placement, variable in zip(options, variables)
            )
            == count
        )
    for column, count in enumerate(instance.column_counts):
        model.add(
            sum(
                variable
                * sum(cell_column == column for _, cell_column in placement.cells)
                for options, variables in zip(placements, selected)
                for placement, variable in zip(options, variables)
            )
            == count
        )

    for clue in instance.clues:
        covering = []
        compatible = []
        for options, variables in zip(placements, selected):
            for placement, variable in zip(options, variables):
                for cell, role in zip(placement.cells, placement.roles):
                    if cell != (clue.row, clue.column):
                        continue
                    covering.append(variable)
                    if _role_matches(role, clue.kind):
                        compatible.append(variable)
        if clue.kind == "water":
            model.add(sum(covering) == 0)
        elif compatible:
            model.add(sum(compatible) == 1)
        else:
            model.add(selected[0][0] != selected[0][0])
    return BattleshipsModel(model, instance, placements, selected)


def decode(built: BattleshipsModel, solution: cp.Solution) -> list[Placement]:
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


def validate(built: BattleshipsModel, ships: list[Placement]) -> None:
    if len(ships) != len(built.instance.fleet):
        raise AssertionError("the decoded fleet has the wrong size")
    if any(
        len(ship.cells) != length for ship, length in zip(ships, built.instance.fleet)
    ):
        raise AssertionError("a decoded ship has the wrong length")
    for first in range(len(ships)):
        for second in range(first + 1, len(ships)):
            if _touch(ships[first], ships[second]):
                raise AssertionError("two ships touch")

    rows = len(built.instance.row_counts)
    columns = len(built.instance.column_counts)
    occupied = {cell for ship in ships for cell in ship.cells}
    row_counts = [
        sum((row, column) in occupied for column in range(columns))
        for row in range(rows)
    ]
    column_counts = [
        sum((row, column) in occupied for row in range(rows))
        for column in range(columns)
    ]
    if row_counts != list(built.instance.row_counts):
        raise AssertionError("row counts do not match")
    if column_counts != list(built.instance.column_counts):
        raise AssertionError("column counts do not match")

    roles = {cell: role for ship in ships for cell, role in zip(ship.cells, ship.roles)}
    for clue in built.instance.clues:
        actual = roles.get((clue.row, clue.column))
        if clue.kind == "water" and actual is not None:
            raise AssertionError("a water clue is occupied")
        if clue.kind != "water" and (
            actual is None or not _role_matches(actual, clue.kind)
        ):
            raise AssertionError("a ship clue does not match")


def render(built: BattleshipsModel, ships: list[Placement]) -> str:
    rows = len(built.instance.row_counts)
    columns = len(built.instance.column_counts)
    occupied = {cell for ship in ships for cell in ship.cells}
    return "\n".join(
        "".join("#" if (row, column) in occupied else "." for column in range(columns))
        for row in range(rows)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON Battleships instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)

    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(
        f"prob014 rows={len(instance.row_counts)} columns={len(instance.column_counts)} "
        f"status={solution.status}"
    )
    if not solution.is_sat():
        return 1
    ships = decode(built, solution)
    validate(built, ships)
    print(render(built, ships))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
