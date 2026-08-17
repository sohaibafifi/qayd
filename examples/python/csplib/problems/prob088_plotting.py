"""CSPLib prob088: Plotting puzzle planning.

Specification: https://www.csplib.org/Problems/prob088/
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args

DEFAULT_INSTANCE = {
    "grid": [[1, 2], [1, 2]],
    "hand": 1,
    "colours": 2,
    "shots": 2,
    "target_remaining": 2,
    "paths": [[0, 1], [2, 3], [0, 2], [1, 3]],
}


@dataclass(frozen=True)
class PlottingInstance:
    grid: tuple[tuple[int, ...], ...]
    hand: int
    colours: int
    shots: int
    target_remaining: int
    paths: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class PlottingModel:
    model: cp.Model
    instance: PlottingInstance
    states: list[list[cp.IntVar]]
    hands: list[cp.IntVar]
    actions: list[cp.IntVar]


def parse_instance(data: str | bytes) -> PlottingInstance:
    raw = json.loads(data)
    try:
        return PlottingInstance(
            tuple(tuple(int(value) for value in row) for row in raw["grid"]),
            int(raw["hand"]),
            int(raw["colours"]),
            int(raw["shots"]),
            int(raw["target_remaining"]),
            tuple(tuple(int(cell) for cell in path) for path in raw["paths"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid Plotting JSON instance") from error


def load_instance(path: str | Path) -> PlottingInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def apply_gravity(state: list[int], rows: int, columns: int, cell: int) -> None:
    column = cell % columns
    column_values = [
        state[row * columns + column]
        for row in range(rows)
        if state[row * columns + column] != 0
    ]
    column_values = [0] * (rows - len(column_values)) + column_values
    for row, value in enumerate(column_values):
        state[row * columns + column] = value


def transition(
    instance: PlottingInstance, state: tuple[int, ...], hand: int, action: int
) -> tuple[tuple[int, ...], int]:
    if action == 0:
        return state, hand
    grid = list(state)
    consumed = False
    rows, columns = len(instance.grid), len(instance.grid[0])
    for cell in instance.paths[action - 1]:
        block = grid[cell]
        if block == 0:
            continue
        if block == hand:
            grid[cell] = 0
            apply_gravity(grid, rows, columns, cell)
            consumed = True
            continue
        if consumed:
            grid[cell], hand = hand, block
        return tuple(grid), hand
    return tuple(grid), hand


def build_model(instance: PlottingInstance) -> PlottingModel:
    rows = len(instance.grid)
    columns = len(instance.grid[0]) if rows else 0
    cells = rows * columns
    if rows < 1 or columns < 1 or any(len(row) != columns for row in instance.grid):
        raise ValueError("the Plotting grid must be rectangular and non-empty")
    if (
        instance.colours < 1
        or instance.shots < 1
        or not 1 <= instance.hand <= instance.colours
    ):
        raise ValueError("colours, shots, or hand block are invalid")
    if any(
        value < 0 or value > instance.colours for row in instance.grid for value in row
    ):
        raise ValueError("a grid block has an invalid colour")
    if any(
        not path or any(cell < 0 or cell >= cells for cell in path)
        for path in instance.paths
    ):
        raise ValueError("a shot path is invalid")
    state_count = (
        (instance.colours + 1) ** cells * instance.colours * (len(instance.paths) + 1)
    )
    if state_count > 300_000:
        raise ValueError("the extensional Plotting state space is too large")
    model = cp.Model()
    states = [
        model.int_vars(cells, 0, instance.colours, name=f"state_{step}")
        for step in range(instance.shots + 1)
    ]
    hands = [
        model.int_var(1, instance.colours, name=f"hand_{step}")
        for step in range(instance.shots + 1)
    ]
    actions = model.int_vars(instance.shots, 0, len(instance.paths), name="action")
    flat_initial = [value for row in instance.grid for value in row]
    for variable, value in zip(states[0], flat_initial):
        model.add(variable == value)
    model.add(hands[0] == instance.hand)
    table = []
    for state in itertools.product(range(instance.colours + 1), repeat=cells):
        for hand in range(1, instance.colours + 1):
            for action in range(len(instance.paths) + 1):
                next_state, next_hand = transition(instance, state, hand, action)
                table.append((*state, hand, action, *next_state, next_hand))
    for step in range(instance.shots):
        model.table(
            [
                *states[step],
                hands[step],
                actions[step],
                *states[step + 1],
                hands[step + 1],
            ],
            table,
        )
    occupied = []
    for cell, variable in enumerate(states[-1]):
        flag = model.bool_var(name=f"occupied_{cell}")
        model.table(
            [variable, flag],
            [(value, int(value != 0)) for value in range(instance.colours + 1)],
        )
        occupied.append(flag)
    model.add(sum(occupied) <= instance.target_remaining)
    used = []
    for step, action in enumerate(actions):
        flag = model.bool_var(name=f"used_{step}")
        model.table(
            [action, flag],
            [(value, int(value != 0)) for value in range(len(instance.paths) + 1)],
        )
        used.append(flag)
    model.minimize(sum(used))
    return PlottingModel(model, instance, states, hands, actions)


def decode(
    built: PlottingModel, solution: cp.Solution
) -> tuple[list[int], list[tuple[list[int], int]]]:
    actions = [solution.value(variable) for variable in built.actions]
    history = [
        ([solution.value(variable) for variable in state], solution.value(hand))
        for state, hand in zip(built.states, built.hands)
    ]
    return actions, history


def validate(
    built: PlottingModel,
    result: tuple[list[int], list[tuple[list[int], int]]],
    objective: int | None,
) -> None:
    actions, history = result
    for step, action in enumerate(actions):
        expected = transition(
            built.instance, tuple(history[step][0]), history[step][1], action
        )
        if expected != (tuple(history[step + 1][0]), history[step + 1][1]):
            raise AssertionError("a Plotting transition is invalid")
    if sum(value != 0 for value in history[-1][0]) > built.instance.target_remaining:
        raise AssertionError("the target number of remaining blocks is not reached")
    if objective is not None and sum(action != 0 for action in actions) != objective:
        raise AssertionError("the objective does not match the number of shots")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON Plotting instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob088 cells={sum(map(len, instance.grid))} status={solution.status}")
    if not solution.is_sat():
        return 1
    result = decode(built, solution)
    validate(built, result, solution.objective)
    print(f"shots={solution.objective} actions={result[0]} final={result[1][-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
