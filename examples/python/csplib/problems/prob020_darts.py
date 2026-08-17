"""CSPLib prob020: darts tournament order of play.

Specification: https://www.csplib.org/Problems/prob020/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values

DEFAULT_INSTANCE = {
    "throwers": 6,
    "prizes": 3,
    "dartboards": 2,
    "ranking": [0, 1, 2, 3, 4, 5],
}


@dataclass(frozen=True)
class DartsInstance:
    throwers: int
    prizes: int
    dartboards: int
    ranking: tuple[int, ...]


@dataclass(frozen=True)
class DartsModel:
    model: cp.Model
    instance: DartsInstance
    matches: tuple[tuple[int, int], ...]
    times: list[cp.IntVar]
    boards: list[cp.IntVar]


def parse_instance(data: str | bytes) -> DartsInstance:
    raw = json.loads(data)
    try:
        throwers = int(raw["throwers"])
        return DartsInstance(
            throwers,
            int(raw["prizes"]),
            int(raw["dartboards"]),
            tuple(int(value) for value in raw.get("ranking", range(throwers))),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid darts-tournament JSON instance") from error


def load_instance(path: str | Path) -> DartsInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def build_model(instance: DartsInstance) -> DartsModel:
    if (
        instance.throwers < 2
        or not 1 <= instance.prizes < instance.throwers
        or instance.dartboards < 1
    ):
        raise ValueError("thrower, prize, or dartboard count is invalid")
    if sorted(instance.ranking) != list(range(instance.throwers)):
        raise ValueError("ranking must be a permutation of throwers")
    matches = tuple(zip(instance.ranking, instance.ranking[1:]))
    match_count = len(matches)
    model = cp.Model()
    times = model.int_vars(match_count, 0, match_count - 1, name="time")
    boards = model.int_vars(match_count, 0, instance.dartboards - 1, name="board")
    slots = model.int_vars(
        match_count, 0, match_count * instance.dartboards - 1, name="slot"
    )
    for match in range(match_count):
        model.add(slots[match] == instance.dartboards * times[match] + boards[match])
    model.all_different(slots)
    for first in range(match_count):
        for second in range(first + 1, match_count):
            if set(matches[first]) & set(matches[second]):
                model.add(times[first] != times[second])
    board_counts = []
    for board in range(instance.dartboards):
        count = model.int_var(0, match_count, name=f"board_count_{board}")
        flags = []
        for match in range(match_count):
            selected = model.bool_var(name=f"match_{match}_board_{board}")
            model.table(
                [boards[match], selected],
                [
                    (candidate, int(candidate == board))
                    for candidate in range(instance.dartboards)
                ],
            )
            flags.append(selected)
        model.add(count == sum(flags))
        board_counts.append(count)
    maximum_load = model.int_var(0, match_count, name="maximum_board_load")
    for count in board_counts:
        model.add(maximum_load >= count)
    discovery = []
    for thrower in instance.ranking[: instance.prizes]:
        incident = [
            times[index] for index, match in enumerate(matches) if thrower in match
        ]
        discovered = model.int_var(0, match_count - 1, name=f"discovery_{thrower}")
        if len(incident) == 1:
            model.add(discovered == incident[0])
        else:
            model.table(
                [*incident, discovered],
                [
                    (left, right, max(left, right))
                    for left in range(match_count)
                    for right in range(match_count)
                ],
            )
        discovery.append(discovered)
    excitement_bound = instance.prizes * match_count
    model.minimize(maximum_load * (excitement_bound + 1) - sum(discovery))
    return DartsModel(model, instance, matches, times, boards)


def decode(built: DartsModel, solution: cp.Solution) -> list[tuple[int, int, int, int]]:
    times = values(solution, built.times)
    boards = values(solution, built.boards)
    return sorted(
        (times[index], boards[index], match[0], match[1])
        for index, match in enumerate(built.matches)
    )


def tournament_metrics(
    built: DartsModel, schedule: list[tuple[int, int, int, int]]
) -> tuple[int, int]:
    loads = [
        sum(board == candidate for _, board, _, _ in schedule)
        for candidate in range(built.instance.dartboards)
    ]
    discoveries = []
    for thrower in built.instance.ranking[: built.instance.prizes]:
        discoveries.append(
            max(time for time, _, left, right in schedule if thrower in (left, right))
        )
    return max(loads), sum(discoveries)


def validate(
    built: DartsModel, schedule: list[tuple[int, int, int, int]], objective: int | None
) -> None:
    if sorted((left, right) for _, _, left, right in schedule) != sorted(built.matches):
        raise AssertionError("the comparison certificate is incomplete")
    if len({(time, board) for time, board, _, _ in schedule}) != len(schedule):
        raise AssertionError("two matches share a dartboard and time")
    for first, (time, _, left, right) in enumerate(schedule):
        for other_time, _, other_left, other_right in schedule[first + 1 :]:
            if time == other_time and {left, right} & {other_left, other_right}:
                raise AssertionError("a thrower plays two simultaneous matches")
    load, excitement = tournament_metrics(built, schedule)
    expected = load * (built.instance.prizes * len(built.matches) + 1) - excitement
    if objective is not None and expected != objective:
        raise AssertionError(
            "the objective does not match board load and discovery times"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON darts-tournament instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob020 throwers={instance.throwers} status={solution.status}")
    if not solution.is_sat():
        return 1
    schedule = decode(built, solution)
    validate(built, schedule, solution.objective)
    print(f"objective={solution.objective} schedule={schedule}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
