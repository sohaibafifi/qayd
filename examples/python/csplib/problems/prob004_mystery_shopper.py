"""CSPLib prob004: mystery shopper scheduling.

Specification: https://www.csplib.org/Problems/prob004/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args

DEFAULT_INSTANCE = {"visitor_groups": [2, 2], "visitee_groups": [2, 2]}


@dataclass(frozen=True)
class MysteryShopperInstance:
    visitor_groups: tuple[int, ...]
    visitee_groups: tuple[int, ...]


@dataclass(frozen=True)
class MysteryShopperModel:
    model: cp.Model
    instance: MysteryShopperInstance
    weeks: int
    visitor_for_visitee: list[list[cp.IntVar]]


def parse_instance(data: str | bytes) -> MysteryShopperInstance:
    raw = json.loads(data)
    try:
        return MysteryShopperInstance(
            tuple(int(value) for value in raw["visitor_groups"]),
            tuple(int(value) for value in raw["visitee_groups"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid mystery-shopper JSON instance") from error


def load_instance(path: str | Path) -> MysteryShopperInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def _group_of(sizes: tuple[int, ...], total: int) -> list[int]:
    groups = [group for group, size in enumerate(sizes) for _ in range(size)]
    if len(groups) < total:
        groups.extend([len(sizes)] * (total - len(groups)))
    return groups


def build_model(instance: MysteryShopperInstance) -> MysteryShopperModel:
    if not instance.visitor_groups or not instance.visitee_groups:
        raise ValueError("visitor and visitee groups must be non-empty")
    if any(size < 1 for size in (*instance.visitor_groups, *instance.visitee_groups)):
        raise ValueError("group sizes must be positive")
    visitor_count = sum(instance.visitor_groups)
    visitee_count = sum(instance.visitee_groups)
    if visitor_count < visitee_count:
        raise ValueError("there must be at least as many visitors as visitees")
    weeks = len(instance.visitor_groups)
    visitor_group = _group_of(instance.visitor_groups, visitor_count)
    visitee_group = _group_of(instance.visitee_groups, visitor_count)
    model = cp.Model()
    visitor_for_visitee = [
        model.int_vars(weeks, 0, visitor_count - 1, name=f"visitee_{visitee}")
        for visitee in range(visitor_count)
    ]
    visitee_for_visitor = [
        model.int_vars(weeks, 0, visitor_count - 1, name=f"visitor_{visitor}")
        for visitor in range(visitor_count)
    ]
    for week in range(weeks):
        visitors = [
            visitor_for_visitee[visitee][week] for visitee in range(visitor_count)
        ]
        visitees = [
            visitee_for_visitor[visitor][week] for visitor in range(visitor_count)
        ]
        model.all_different(visitors)
        model.all_different(visitees)
        model.channel(visitors, visitees)
    for visitee in range(visitor_count):
        group_values = []
        for week in range(weeks):
            group = model.int_var(
                0,
                len(instance.visitor_groups) - 1,
                name=f"visitor_group_{visitee}_{week}",
            )
            model.element_const(
                visitor_group, visitor_for_visitee[visitee][week], group
            )
            group_values.append(group)
        model.all_different(group_values)
    for visitor in range(visitor_count):
        groups = []
        for week in range(weeks):
            group = model.int_var(
                0, len(set(visitee_group)) - 1, name=f"visitee_group_{visitor}_{week}"
            )
            model.element_const(
                visitee_group, visitee_for_visitor[visitor][week], group
            )
            groups.append(group)
        model.all_different(groups)
    return MysteryShopperModel(model, instance, weeks, visitor_for_visitee)


def decode(built: MysteryShopperModel, solution: cp.Solution) -> list[list[int]]:
    if not solution.is_sat():
        raise RuntimeError(f"no solution available, solver status is {solution.status}")
    visitor_count = len(built.visitor_for_visitee)
    return [
        [
            solution.value(built.visitor_for_visitee[visitee][week])
            for visitee in range(visitor_count)
        ]
        for week in range(built.weeks)
    ]


def validate(built: MysteryShopperModel, schedule: list[list[int]]) -> None:
    visitor_count = sum(built.instance.visitor_groups)
    if len(schedule) != built.weeks or any(
        sorted(week) != list(range(visitor_count)) for week in schedule
    ):
        raise AssertionError("a weekly assignment is not a bijection")
    visitor_group = _group_of(built.instance.visitor_groups, visitor_count)
    visitee_group = _group_of(built.instance.visitee_groups, visitor_count)
    for visitee in range(visitor_count):
        groups = [visitor_group[schedule[week][visitee]] for week in range(built.weeks)]
        if len(set(groups)) != len(groups):
            raise AssertionError("a visitee sees the same visitor group twice")
    for visitor in range(visitor_count):
        groups = []
        for week in range(built.weeks):
            visitee = schedule[week].index(visitor)
            groups.append(visitee_group[visitee])
        if len(set(groups)) != len(groups):
            raise AssertionError("a visitor sees the same visitee group twice")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON mystery-shopper instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob004 weeks={built.weeks} status={solution.status}")
    if not solution.is_sat():
        return 1
    schedule = decode(built, solution)
    validate(built, schedule)
    print(f"schedule={schedule}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
