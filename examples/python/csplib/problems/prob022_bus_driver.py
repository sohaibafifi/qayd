"""CSPLib prob022: bus driver scheduling as set partitioning.

Specification: https://www.csplib.org/Problems/prob022/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values

SAMPLE_ORLIB = """4 4 2
1 2 1 2
1 2 3 4
1 2 1 3
1 2 2 4
"""


@dataclass(frozen=True)
class BusDriverInstance:
    task_count: int
    shifts: tuple[frozenset[int], ...]
    costs: tuple[int, ...]
    known_minimum: int | None


@dataclass(frozen=True)
class BusDriverModel:
    model: cp.Model
    instance: BusDriverInstance
    selected: list[cp.IntVar]


def parse_orlib(data: str | bytes) -> BusDriverInstance:
    text = data.decode() if isinstance(data, bytes) else data
    tokens = text.split()
    try:
        task_count, shift_count, known_minimum = map(int, tokens[:3])
        cursor = 3
        shifts = []
        costs = []
        for _ in range(shift_count):
            cost = int(tokens[cursor])
            covered_count = int(tokens[cursor + 1])
            cursor += 2
            covered = frozenset(
                int(value) - 1 for value in tokens[cursor : cursor + covered_count]
            )
            cursor += covered_count
            costs.append(cost)
            shifts.append(covered)
    except (IndexError, TypeError, ValueError) as error:
        raise ValueError("invalid ORLIB set-partitioning instance") from error
    if cursor != len(tokens):
        raise ValueError("unexpected trailing data in ORLIB instance")
    return BusDriverInstance(task_count, tuple(shifts), tuple(costs), known_minimum)


def load_instance(path: str | Path) -> BusDriverInstance:
    return parse_orlib(Path(path).read_text(encoding="utf-8"))


def build_model(instance: BusDriverInstance) -> BusDriverModel:
    if (
        instance.task_count < 1
        or not instance.shifts
        or len(instance.shifts) != len(instance.costs)
    ):
        raise ValueError("the task and shift sets must be non-empty")
    if any(
        not shift or min(shift) < 0 or max(shift) >= instance.task_count
        for shift in instance.shifts
    ):
        raise ValueError("a shift covers an invalid task")
    model = cp.Model()
    selected = [
        model.bool_var(name=f"shift_{index}") for index in range(len(instance.shifts))
    ]
    for task in range(instance.task_count):
        covering = [
            selected[index]
            for index, shift in enumerate(instance.shifts)
            if task in shift
        ]
        if not covering:
            model.add(selected[0] != selected[0])
        else:
            model.add(sum(covering) == 1)
    model.minimize(
        sum(instance.costs[index] * variable for index, variable in enumerate(selected))
    )
    return BusDriverModel(model, instance, selected)


def decode(built: BusDriverModel, solution: cp.Solution) -> list[int]:
    return [
        index
        for index, selected in enumerate(values(solution, built.selected))
        if selected
    ]


def validate(built: BusDriverModel, selected: list[int], objective: int | None) -> None:
    for task in range(built.instance.task_count):
        if sum(task in built.instance.shifts[index] for index in selected) != 1:
            raise AssertionError("a task is not covered exactly once")
    cost = sum(built.instance.costs[index] for index in selected)
    if objective is not None and cost != objective:
        raise AssertionError("the objective does not match selected shift costs")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="ORLIB set-partitioning instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = load_instance(args.path) if args.path else parse_orlib(SAMPLE_ORLIB)
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(
        f"prob022 tasks={instance.task_count} shifts={len(instance.shifts)} status={solution.status}"
    )
    if not solution.is_sat():
        return 1
    selected = decode(built, solution)
    validate(built, selected, solution.objective)
    print(f"selected={selected} cost={solution.objective}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
