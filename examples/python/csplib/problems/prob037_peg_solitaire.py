"""CSPLib prob037: peg solitaire.

Specification: https://www.csplib.org/Problems/prob037/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values

DEFAULT_INSTANCE = {"initial": ["xxo"], "target": ["oox"]}
Cell = tuple[int, int]
Move = tuple[Cell, Cell, Cell]


@dataclass(frozen=True)
class PegInstance:
    initial: tuple[str, ...]
    target: tuple[str, ...]


@dataclass(frozen=True)
class PegModel:
    model: cp.Model
    instance: PegInstance
    holes: tuple[Cell, ...]
    transitions: tuple[Move, ...]
    moves: list[cp.IntVar]


def parse_instance(data: str | bytes) -> PegInstance:
    raw = json.loads(data)
    try:
        return PegInstance(
            tuple(str(row) for row in raw["initial"]),
            tuple(str(row) for row in raw["target"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid peg-solitaire JSON instance") from error


def load_instance(path: str | Path) -> PegInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def _transitions(holes: tuple[Cell, ...]) -> tuple[Move, ...]:
    hole_set = set(holes)
    moves = []
    for row, column in holes:
        for row_delta, column_delta in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            middle = (row + row_delta, column + column_delta)
            target = (row + 2 * row_delta, column + 2 * column_delta)
            if middle in hole_set and target in hole_set:
                moves.append(((row, column), middle, target))
    return tuple(moves)


def build_model(instance: PegInstance) -> PegModel:
    if (
        not instance.initial
        or not instance.initial[0]
        or len(instance.initial) != len(instance.target)
    ):
        raise ValueError("initial and target boards must be non-empty")
    width = len(instance.initial[0])
    if any(len(row) != width for row in (*instance.initial, *instance.target)):
        raise ValueError("boards must be rectangular and have equal dimensions")
    allowed_characters = {"x", "o", "#"}
    if any(
        character not in allowed_characters
        for row in (*instance.initial, *instance.target)
        for character in row
    ):
        raise ValueError("boards may contain only x, o, and #")
    holes = tuple(
        (row, column)
        for row in range(len(instance.initial))
        for column in range(width)
        if instance.initial[row][column] != "#"
    )
    if any(
        (instance.initial[row][column] == "#") != (instance.target[row][column] == "#")
        for row in range(len(instance.initial))
        for column in range(width)
    ):
        raise ValueError("initial and target boards have different hole sets")
    initial_pegs = sum(instance.initial[row][column] == "x" for row, column in holes)
    target_pegs = sum(instance.target[row][column] == "x" for row, column in holes)
    move_count = initial_pegs - target_pegs
    transitions = _transitions(holes)
    if move_count < 0 or (move_count > 0 and not transitions):
        raise ValueError("the target has an invalid peg count")

    model = cp.Model()
    states = [
        model.int_vars(len(holes), 0, 1, name=f"state_{step}")
        for step in range(move_count + 1)
    ]
    moves = (
        model.int_vars(move_count, 0, len(transitions) - 1, name="move")
        if move_count
        else []
    )
    for index, (row, column) in enumerate(holes):
        model.add(states[0][index] == int(instance.initial[row][column] == "x"))
        model.add(states[-1][index] == int(instance.target[row][column] == "x"))
    for step in range(move_count):
        for hole_index, hole in enumerate(holes):
            allowed = []
            for move_index, (origin, middle, target) in enumerate(transitions):
                if hole in (origin, middle):
                    allowed.append((move_index, 1, 0))
                elif hole == target:
                    allowed.append((move_index, 0, 1))
                else:
                    allowed.extend(((move_index, 0, 0), (move_index, 1, 1)))
            model.table(
                [moves[step], states[step][hole_index], states[step + 1][hole_index]],
                allowed,
            )
    return PegModel(model, instance, holes, transitions, moves)


def decode(built: PegModel, solution: cp.Solution) -> list[Move]:
    return [built.transitions[index] for index in values(solution, built.moves)]


def validate(built: PegModel, moves: list[Move]) -> None:
    occupied = {
        (row, column)
        for row, column in built.holes
        if built.instance.initial[row][column] == "x"
    }
    for origin, middle, target in moves:
        if origin not in occupied or middle not in occupied or target in occupied:
            raise AssertionError("a decoded jump is illegal")
        occupied.remove(origin)
        occupied.remove(middle)
        occupied.add(target)
    expected = {
        (row, column)
        for row, column in built.holes
        if built.instance.target[row][column] == "x"
    }
    if occupied != expected:
        raise AssertionError("the move sequence does not reach the target board")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON peg-solitaire instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob037 holes={len(built.holes)} status={solution.status}")
    if not solution.is_sat():
        return 1
    moves = decode(built, solution)
    validate(built, moves)
    print(f"moves={moves}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
