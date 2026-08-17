"""CSPLib prob085: bookshelves.

Specification: https://www.csplib.org/Problems/prob085/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values

DEFAULT_INSTANCE = {
    "planks": [24, 24, 18],
    "thickness": 1,
    "cut_loss": 1,
    "clearance": 5,
    "min_width": 6,
    "max_width": 12,
}


@dataclass(frozen=True)
class BookshelfInstance:
    planks: tuple[int, ...]
    thickness: int
    cut_loss: int
    clearance: int
    min_width: int
    max_width: int


@dataclass(frozen=True)
class BookshelfModel:
    model: cp.Model
    instance: BookshelfInstance
    configurations: tuple[tuple[int, int, int], ...]
    configuration: cp.IntVar
    plank_for_piece: list[cp.IntVar]


def parse_instance(data: str | bytes) -> BookshelfInstance:
    raw = json.loads(data)
    try:
        return BookshelfInstance(
            tuple(int(value) for value in raw["planks"]),
            int(raw["thickness"]),
            int(raw["cut_loss"]),
            int(raw["clearance"]),
            int(raw["min_width"]),
            int(raw["max_width"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid bookshelves JSON instance") from error


def load_instance(path: str | Path) -> BookshelfInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def build_model(instance: BookshelfInstance) -> BookshelfModel:
    if not instance.planks or any(length < 1 for length in instance.planks):
        raise ValueError("plank lengths must be positive")
    if instance.thickness < 0 or instance.cut_loss < 0 or instance.clearance < 1:
        raise ValueError("thickness, cut loss, or clearance is invalid")
    if not 1 <= instance.min_width <= instance.max_width:
        raise ValueError("shelf-width bounds are invalid")
    total_wood = sum(instance.planks)
    max_shelves = max(
        1, total_wood // max(1, instance.min_width + 2 * instance.clearance)
    )
    configurations = tuple(
        (shelves, width, shelves * (instance.clearance + instance.thickness))
        for shelves in range(1, max_shelves + 1)
        for width in range(instance.min_width, instance.max_width + 1)
        if shelves * width + 2 * shelves * (instance.clearance + instance.thickness)
        <= total_wood
    )
    if not configurations:
        raise ValueError("no bookshelf configuration fits the total wood")
    model = cp.Model()
    configuration = model.int_var(0, len(configurations) - 1, name="configuration")
    piece_count = max_shelves + 2
    inactive_plank = len(instance.planks)
    plank_for_piece = model.int_vars(piece_count, 0, inactive_plank, name="plank")
    contributions = [[None for _ in instance.planks] for _ in range(piece_count)]
    for piece in range(piece_count):
        active = model.bool_var(name=f"active_{piece}")
        model.table(
            [configuration, active, plank_for_piece[piece]],
            [
                (index, int(piece < shelves + 2), plank)
                for index, (shelves, _, _) in enumerate(configurations)
                for plank in (
                    range(len(instance.planks))
                    if piece < shelves + 2
                    else [inactive_plank]
                )
            ],
        )
        for plank in range(len(instance.planks)):
            contribution = model.int_var(
                0, max(instance.planks), name=f"piece_{piece}_plank_{plank}"
            )
            model.table(
                [configuration, plank_for_piece[piece], contribution],
                [
                    (
                        index,
                        assigned,
                        (height if piece < 2 else width) + instance.cut_loss
                        if piece < shelves + 2 and assigned == plank
                        else 0,
                    )
                    for index, (shelves, width, height) in enumerate(configurations)
                    for assigned in (
                        range(len(instance.planks))
                        if piece < shelves + 2
                        else [inactive_plank]
                    )
                ],
            )
            contributions[piece][plank] = contribution
    for plank, length in enumerate(instance.planks):
        model.add(
            sum(contributions[piece][plank] for piece in range(piece_count)) <= length
        )
    capacity = model.int_var(
        0, max(shelves * width for shelves, width, _ in configurations), name="capacity"
    )
    model.table(
        [configuration, capacity],
        [
            (index, shelves * width)
            for index, (shelves, width, _) in enumerate(configurations)
        ],
    )
    model.maximize(capacity)
    return BookshelfModel(
        model, instance, configurations, configuration, plank_for_piece
    )


def decode(
    built: BookshelfModel, solution: cp.Solution
) -> tuple[int, int, int, list[list[int]]]:
    index = solution.value(built.configuration)
    shelves, width, height = built.configurations[index]
    cuts = [[] for _ in built.instance.planks]
    for piece, plank in enumerate(values(solution, built.plank_for_piece)):
        if plank < len(cuts) and piece < shelves + 2:
            cuts[plank].append(height if piece < 2 else width)
    return shelves, width, height, cuts


def validate(
    built: BookshelfModel,
    result: tuple[int, int, int, list[list[int]]],
    objective: int | None,
) -> None:
    shelves, width, height, cuts = result
    if (shelves, width, height) not in built.configurations:
        raise AssertionError("the bookshelf dimensions are invalid")
    pieces = sorted(piece for plank in cuts for piece in plank)
    if pieces != sorted([height, height, *([width] * shelves)]):
        raise AssertionError("the required side and shelf pieces are not cut")
    for length, plank in zip(built.instance.planks, cuts):
        if sum(plank) + built.instance.cut_loss * len(plank) > length:
            raise AssertionError("a cutting pattern exceeds its plank")
    if objective is not None and shelves * width != objective:
        raise AssertionError("the objective does not match shelf capacity")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON bookshelves instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob085 planks={len(instance.planks)} status={solution.status}")
    if not solution.is_sat():
        return 1
    result = decode(built, solution)
    validate(built, result, solution.objective)
    print(
        f"capacity={solution.objective} shelves={result[0]} width={result[1]} cuts={result[3]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
