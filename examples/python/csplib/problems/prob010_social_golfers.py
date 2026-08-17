"""CSPLib prob010: social golfers.

Specification: https://www.csplib.org/Problems/prob010/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations

import qayd as cp

from ..common import add_solver_arguments, solve_from_args


@dataclass(frozen=True)
class SocialGolfersModel:
    model: cp.Model
    groups: int
    group_size: int
    weeks: int
    assignments: list[list[cp.IntVar]]


def build_model(groups: int, group_size: int, weeks: int) -> SocialGolfersModel:
    if groups < 1 or group_size < 1 or weeks < 1:
        raise ValueError("groups, group_size, and weeks must be positive")
    golfer_count = groups * group_size
    model = cp.Model()
    assignments = [
        model.int_vars(golfer_count, 0, groups - 1, name=f"week_{week}")
        for week in range(weeks)
    ]
    for week in range(weeks):
        model.cardinality(
            assignments[week],
            list(range(groups)),
            [group_size] * groups,
            [group_size] * groups,
            closed=True,
        )

    equality_table = [
        (left, right, int(left == right))
        for left in range(groups)
        for right in range(groups)
    ]
    for first, second in combinations(range(golfer_count), 2):
        together: list[cp.IntVar] = []
        for week in range(weeks):
            same_group = model.bool_var(name=f"together_{first}_{second}_{week}")
            model.table(
                [assignments[week][first], assignments[week][second], same_group],
                equality_table,
            )
            together.append(same_group)
        model.add(sum(together) <= 1)

    for golfer in range(golfer_count):
        model.add(assignments[0][golfer] == golfer // group_size)
    for week in range(1, weeks):
        model.add(assignments[week][0] == 0)
    return SocialGolfersModel(model, groups, group_size, weeks, assignments)


def decode(built: SocialGolfersModel, solution: cp.Solution) -> list[list[list[int]]]:
    if not solution.is_sat():
        raise RuntimeError(f"no solution available, solver status is {solution.status}")
    schedule: list[list[list[int]]] = []
    for week in built.assignments:
        groups = [[] for _ in range(built.groups)]
        for golfer, variable in enumerate(week):
            groups[solution.value(variable)].append(golfer)
        schedule.append(groups)
    return schedule


def validate(schedule: list[list[list[int]]], *, groups: int, group_size: int) -> None:
    golfer_count = groups * group_size
    met: dict[tuple[int, int], int] = {}
    for week in schedule:
        if len(week) != groups or any(len(group) != group_size for group in week):
            raise AssertionError("every week must contain equally sized groups")
        if sorted(golfer for group in week for golfer in group) != list(
            range(golfer_count)
        ):
            raise AssertionError("every golfer must play exactly once per week")
        for group in week:
            for pair in combinations(sorted(group), 2):
                met[pair] = met.get(pair, 0) + 1
                if met[pair] > 1:
                    raise AssertionError("two golfers met more than once")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=int, default=3)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--weeks", type=int, default=5)
    add_solver_arguments(parser)
    args = parser.parse_args(argv)

    built = build_model(args.groups, args.group_size, args.weeks)
    solution = solve_from_args(built.model, args)
    print(
        f"prob010 groups={args.groups} group_size={args.group_size} "
        f"weeks={args.weeks} status={solution.status}"
    )
    if not solution.is_sat():
        return 1
    schedule = decode(built, solution)
    validate(schedule, groups=args.groups, group_size=args.group_size)
    for week, groups in enumerate(schedule, start=1):
        print(f"week {week}: {groups}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
