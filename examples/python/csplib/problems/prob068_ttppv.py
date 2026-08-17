"""CSPLib prob068: travelling tournament with predefined venues.

Specification: https://www.csplib.org/Problems/prob068/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args

DEFAULT_INSTANCE = {
    "venues": [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]],
    "distances": [[0, 1, 2, 1], [1, 0, 1, 2], [2, 1, 0, 1], [1, 2, 1, 0]],
    "maximum_consecutive": 3,
}


@dataclass(frozen=True)
class TtppvInstance:
    venues: tuple[tuple[int, ...], ...]
    distances: tuple[tuple[int, ...], ...]
    maximum_consecutive: int


@dataclass(frozen=True)
class TtppvModel:
    model: cp.Model
    instance: TtppvInstance
    opponents: list[list[cp.IntVar]]
    at_home: list[list[cp.IntVar]]


def parse_instance(data: str | bytes) -> TtppvInstance:
    raw = json.loads(data)
    try:
        return TtppvInstance(
            tuple(tuple(int(value) for value in row) for row in raw["venues"]),
            tuple(tuple(int(value) for value in row) for row in raw["distances"]),
            int(raw.get("maximum_consecutive", 3)),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid TTPPV JSON instance") from error


def load_instance(path: str | Path) -> TtppvInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def build_model(instance: TtppvInstance) -> TtppvModel:
    team_count = len(instance.venues)
    if team_count < 2 or team_count % 2:
        raise ValueError("an even number of teams is required")
    if len(instance.distances) != team_count or any(
        len(row) != team_count for row in (*instance.venues, *instance.distances)
    ):
        raise ValueError("venue and distance matrices must be square")
    if any(value < 0 for row in instance.distances for value in row):
        raise ValueError("distances must be non-negative")
    if any(
        instance.venues[left][right] not in (0, 1)
        or (
            left != right
            and instance.venues[left][right] + instance.venues[right][left] != 1
        )
        for left in range(team_count)
        for right in range(team_count)
    ):
        raise ValueError("predefined venues are invalid")
    rounds = team_count - 1
    model = cp.Model()
    opponents = [
        model.int_vars(rounds, 0, team_count - 1, name=f"opponent_{team}")
        for team in range(team_count)
    ]
    at_home = [
        model.int_vars(rounds, 0, 1, name=f"home_{team}") for team in range(team_count)
    ]
    locations = [
        model.int_vars(rounds, 0, team_count - 1, name=f"location_{team}")
        for team in range(team_count)
    ]
    for team in range(team_count):
        model.all_different(opponents[team])
        for round_index in range(rounds):
            model.table(
                [
                    opponents[team][round_index],
                    at_home[team][round_index],
                    locations[team][round_index],
                ],
                [
                    (
                        opponent,
                        instance.venues[team][opponent],
                        team if instance.venues[team][opponent] else opponent,
                    )
                    for opponent in range(team_count)
                    if opponent != team
                ],
            )
        window = instance.maximum_consecutive + 1
        for start in range(rounds - window + 1):
            home_count = sum(at_home[team][start : start + window])
            model.add(home_count >= 1)
            model.add(home_count <= window - 1)
    for round_index in range(rounds):
        column = [opponents[team][round_index] for team in range(team_count)]
        for team in range(team_count):
            reciprocal = model.int_var(
                0, team_count - 1, name=f"reciprocal_{team}_{round_index}"
            )
            model.element(column, opponents[team][round_index], reciprocal)
            model.add(reciprocal == team)

    maximum_distance = max(value for row in instance.distances for value in row)
    travel_costs = []
    distance_table = [
        (left, right, instance.distances[left][right])
        for left in range(team_count)
        for right in range(team_count)
    ]
    for team in range(team_count):
        first = model.int_var(0, maximum_distance, name=f"travel_{team}_0")
        model.table(
            [locations[team][0], first],
            [
                (location, instance.distances[team][location])
                for location in range(team_count)
            ],
        )
        travel_costs.append(first)
        for round_index in range(rounds - 1):
            cost = model.int_var(
                0, maximum_distance, name=f"travel_{team}_{round_index + 1}"
            )
            model.table(
                [locations[team][round_index], locations[team][round_index + 1], cost],
                distance_table,
            )
            travel_costs.append(cost)
        last = model.int_var(0, maximum_distance, name=f"travel_{team}_{rounds}")
        model.table(
            [locations[team][-1], last],
            [
                (location, instance.distances[location][team])
                for location in range(team_count)
            ],
        )
        travel_costs.append(last)
    model.minimize(sum(travel_costs))
    return TtppvModel(model, instance, opponents, at_home)


def decode(
    built: TtppvModel, solution: cp.Solution
) -> tuple[list[list[int]], list[list[int]]]:
    if not solution.is_sat():
        raise RuntimeError(f"no solution available, solver status is {solution.status}")
    opponents = [
        [solution.value(variable) for variable in row] for row in built.opponents
    ]
    at_home = [[solution.value(variable) for variable in row] for row in built.at_home]
    return opponents, at_home


def travel_cost(
    instance: TtppvInstance, opponents: list[list[int]], at_home: list[list[int]]
) -> int:
    total = 0
    for team in range(len(opponents)):
        locations = [
            team if home else opponent
            for opponent, home in zip(opponents[team], at_home[team])
        ]
        total += sum(
            instance.distances[left][right]
            for left, right in zip([team, *locations], [*locations, team])
        )
    return total


def validate(
    built: TtppvModel,
    opponents: list[list[int]],
    at_home: list[list[int]],
    objective: int | None,
) -> None:
    team_count = len(opponents)
    rounds = team_count - 1
    for team in range(team_count):
        if sorted(opponents[team]) != [
            other for other in range(team_count) if other != team
        ]:
            raise AssertionError("a team does not play every opponent exactly once")
        for round_index in range(rounds):
            opponent = opponents[team][round_index]
            if opponents[opponent][round_index] != team:
                raise AssertionError("a round opponent relation is not reciprocal")
            if at_home[team][round_index] != built.instance.venues[team][opponent]:
                raise AssertionError("a game violates its predefined venue")
        window = built.instance.maximum_consecutive + 1
        if any(
            sum(at_home[team][start : start + window]) in (0, window)
            for start in range(rounds - window + 1)
        ):
            raise AssertionError("a team has too many consecutive home or away games")
    if (
        objective is not None
        and travel_cost(built.instance, opponents, at_home) != objective
    ):
        raise AssertionError("the objective does not match total team travel")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON TTPPV instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob068 teams={len(instance.venues)} status={solution.status}")
    if not solution.is_sat():
        return 1
    opponents, at_home = decode(built, solution)
    validate(built, opponents, at_home, solution.objective)
    print(f"travel={solution.objective} opponents={opponents}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
