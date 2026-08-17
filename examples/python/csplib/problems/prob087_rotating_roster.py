"""CSPLib prob087: rotating rostering.

Specification: https://www.csplib.org/Problems/prob087/
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values

DEFAULT_INSTANCE = {
    "requirements": [
        [1, 1, 1, 1, 1, 2, 2],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
    ],
    "minimum_run": 1,
    "maximum_run": 4,
}


@dataclass(frozen=True)
class RosterInstance:
    requirements: tuple[tuple[int, ...], ...]
    minimum_run: int
    maximum_run: int


@dataclass(frozen=True)
class RosterModel:
    model: cp.Model
    instance: RosterInstance
    weeks: int
    shifts: list[cp.IntVar]


def parse_instance(data: str | bytes) -> RosterInstance:
    raw = json.loads(data)
    try:
        return RosterInstance(
            tuple(tuple(int(value) for value in row) for row in raw["requirements"]),
            int(raw["minimum_run"]),
            int(raw["maximum_run"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid rotating-roster JSON instance") from error


def load_instance(path: str | Path) -> RosterInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def build_model(instance: RosterInstance) -> RosterModel:
    if len(instance.requirements) != 4 or any(
        len(row) != 7 for row in instance.requirements
    ):
        raise ValueError("requirements must be a 4 by 7 matrix")
    weeks = sum(instance.requirements[shift][0] for shift in range(4))
    if weeks < 1 or any(
        sum(instance.requirements[shift][day] for shift in range(4)) != weeks
        for day in range(7)
    ):
        raise ValueError("daily requirements must sum to the number of weeks")
    if not 1 <= instance.minimum_run <= instance.maximum_run:
        raise ValueError("run-length bounds are invalid")
    days = weeks * 7
    model = cp.Model()
    shifts = model.int_vars(days, 0, 3, name="shift")
    for weekday in range(7):
        variables = [shifts[week * 7 + weekday] for week in range(weeks)]
        counts = [instance.requirements[shift][weekday] for shift in range(4)]
        model.cardinality(variables, [0, 1, 2, 3], counts, counts, closed=True)
    for week in range(weeks):
        model.add(shifts[week * 7 + 5] == shifts[week * 7 + 6])
    order = [
        (left, right)
        for left in range(4)
        for right in range(4)
        if right == 0 or left <= right
    ]
    for day in range(days):
        model.table([shifts[day], shifts[(day + 1) % days]], order)
        max_window = [
            shifts[(day + offset) % days] for offset in range(instance.maximum_run + 1)
        ]
        model.table(
            max_window,
            [
                row
                for row in itertools.product(range(4), repeat=len(max_window))
                if len(set(row)) > 1
            ],
        )
        rest_flags = []
        for offset in range(min(14, days)):
            flag = model.bool_var(name=f"rest_{day}_{offset}")
            model.table(
                [shifts[(day + offset) % days], flag],
                [(shift, int(shift == 0)) for shift in range(4)],
            )
            rest_flags.append(flag)
        model.add(sum(rest_flags) >= 2)
        if instance.minimum_run > 1:
            run_window = [
                shifts[(day - offset) % days]
                for offset in range(instance.minimum_run - 1, -1, -1)
            ]
            run_window.append(shifts[(day + 1) % days])
            allowed_runs = [
                row
                for row in itertools.product(range(4), repeat=len(run_window))
                if row[-1] == row[-2] or len(set(row[:-1])) == 1
            ]
            model.table(run_window, allowed_runs)
    return RosterModel(model, instance, weeks, shifts)


def decode(built: RosterModel, solution: cp.Solution) -> list[list[int]]:
    flat = values(solution, built.shifts)
    return [flat[start : start + 7] for start in range(0, len(flat), 7)]


def validate(built: RosterModel, roster: list[list[int]]) -> None:
    flat = [shift for week in roster for shift in week]
    if len(roster) != built.weeks or any(len(week) != 7 for week in roster):
        raise AssertionError("the rotating roster has invalid dimensions")
    for weekday in range(7):
        column = [roster[week][weekday] for week in range(built.weeks)]
        if [column.count(shift) for shift in range(4)] != [
            row[weekday] for row in built.instance.requirements
        ]:
            raise AssertionError("a weekday does not meet staffing requirements")
    if any(week[5] != week[6] for week in roster):
        raise AssertionError("Saturday and Sunday shifts differ")
    days = len(flat)
    for day in range(days):
        left, right = flat[day], flat[(day + 1) % days]
        if right != 0 and left > right:
            raise AssertionError("the forward shift order is violated")
        if all(
            flat[(day + offset) % days] == left
            for offset in range(built.instance.maximum_run + 1)
        ):
            raise AssertionError("a shift run is too long")
        if sum(flat[(day + offset) % days] == 0 for offset in range(min(14, days))) < 2:
            raise AssertionError("a rolling rest requirement is violated")
        if right != left and any(
            flat[(day - offset) % days] != left
            for offset in range(built.instance.minimum_run)
        ):
            raise AssertionError("a shift run is too short")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON rotating-roster instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob087 weeks={built.weeks} status={solution.status}")
    if not solution.is_sat():
        return 1
    roster = decode(built, solution)
    validate(built, roster)
    print(f"roster={roster}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
