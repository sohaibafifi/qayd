"""CSPLib prob026: sports tournament scheduling.

Specification: https://www.csplib.org/Problems/prob026/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations

import qayd as cp

from ..common import add_solver_arguments, solve_from_args

Match = tuple[int, int]


@dataclass(frozen=True)
class TournamentModel:
    model: cp.Model
    team_count: int
    home: list[list[cp.IntVar]]
    away: list[list[cp.IntVar]]


def build_model(team_count: int) -> TournamentModel:
    if team_count < 2 or team_count % 2:
        raise ValueError("team_count must be positive and even")
    weeks = team_count - 1
    periods = team_count // 2
    model = cp.Model()
    home = [
        model.int_vars(periods, 0, team_count - 1, name=f"home_{week}")
        for week in range(weeks)
    ]
    away = [
        model.int_vars(periods, 0, team_count - 1, name=f"away_{week}")
        for week in range(weeks)
    ]
    for week in range(weeks):
        model.all_different([*home[week], *away[week]])

    for first, second in combinations(range(team_count), 2):
        occurrences = []
        table = [
            (left, right, int({left, right} == {first, second}))
            for left in range(team_count)
            for right in range(team_count)
            if left != right
        ]
        for week in range(weeks):
            for period in range(periods):
                occurrence = model.bool_var(
                    name=f"pair_{first}_{second}_{week}_{period}"
                )
                model.table([home[week][period], away[week][period], occurrence], table)
                occurrences.append(occurrence)
        model.add(sum(occurrences) == 1)

    for team in range(team_count):
        table = [
            (left, right, int(team in (left, right)))
            for left in range(team_count)
            for right in range(team_count)
            if left != right
        ]
        for period in range(periods):
            occurrences = []
            for week in range(weeks):
                occurrence = model.bool_var(name=f"team_{team}_{week}_{period}")
                model.table([home[week][period], away[week][period], occurrence], table)
                occurrences.append(occurrence)
            model.add(sum(occurrences) <= 2)

    for period in range(periods):
        model.add(home[0][period] == 2 * period)
        model.add(away[0][period] == 2 * period + 1)
    return TournamentModel(model, team_count, home, away)


def decode(built: TournamentModel, solution: cp.Solution) -> list[list[Match]]:
    if not solution.is_sat():
        raise RuntimeError(f"no solution available, solver status is {solution.status}")
    return [
        [
            (
                solution.value(built.home[week][period]),
                solution.value(built.away[week][period]),
            )
            for period in range(built.team_count // 2)
        ]
        for week in range(built.team_count - 1)
    ]


def validate(schedule: list[list[Match]], team_count: int) -> None:
    weeks = team_count - 1
    periods = team_count // 2
    if len(schedule) != weeks or any(len(week) != periods for week in schedule):
        raise AssertionError("the schedule dimensions are invalid")
    played_pairs = []
    for week in schedule:
        teams = [team for match in week for team in match]
        if sorted(teams) != list(range(team_count)):
            raise AssertionError("a team does not play exactly once in a week")
        played_pairs.extend(frozenset(match) for match in week)
    expected_pairs = [frozenset(pair) for pair in combinations(range(team_count), 2)]
    if sorted(played_pairs, key=lambda pair: tuple(sorted(pair))) != expected_pairs:
        raise AssertionError("not every team pair plays exactly once")
    for period in range(periods):
        for team in range(team_count):
            if sum(team in schedule[week][period] for week in range(weeks)) > 2:
                raise AssertionError("a team appears too often in one period")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teams", type=int, default=6)
    add_solver_arguments(parser, time_limit=30)
    args = parser.parse_args(argv)
    built = build_model(args.teams)
    solution = solve_from_args(built.model, args)
    print(f"prob026 teams={args.teams} status={solution.status}")
    if not solution.is_sat():
        return 1
    schedule = decode(built, solution)
    validate(schedule, args.teams)
    for week, matches in enumerate(schedule, start=1):
        print(f"week {week}: {matches}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
