"""CSPLib prob078: train traffic rescheduling.

Specification: https://www.csplib.org/Problems/prob078/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values

DEFAULT_INSTANCE = {
    "horizon": 12,
    "headway": 1,
    "trains": [
        {
            "scheduled": 0,
            "routes": [
                {"sections": [0, 1], "durations": [2, 2]},
                {"sections": [2, 1], "durations": [3, 2]},
            ],
        },
        {
            "scheduled": 1,
            "routes": [
                {"sections": [1, 0], "durations": [2, 2]},
                {"sections": [3, 0], "durations": [2, 2]},
            ],
        },
        {"scheduled": 2, "routes": [{"sections": [0, 3], "durations": [2, 1]}]},
    ],
}


@dataclass(frozen=True)
class TrainRoute:
    sections: tuple[int, ...]
    durations: tuple[int, ...]


@dataclass(frozen=True)
class Train:
    scheduled: int
    routes: tuple[TrainRoute, ...]


@dataclass(frozen=True)
class TrainInstance:
    horizon: int
    headway: int
    trains: tuple[Train, ...]


@dataclass(frozen=True)
class TrainModel:
    model: cp.Model
    instance: TrainInstance
    routes: list[cp.IntVar]
    starts: list[cp.IntVar]


def parse_instance(data: str | bytes) -> TrainInstance:
    raw = json.loads(data)
    try:
        trains = tuple(
            Train(
                int(item["scheduled"]),
                tuple(
                    TrainRoute(
                        tuple(int(value) for value in route["sections"]),
                        tuple(int(value) for value in route["durations"]),
                    )
                    for route in item["routes"]
                ),
            )
            for item in raw["trains"]
        )
        return TrainInstance(int(raw["horizon"]), int(raw.get("headway", 0)), trains)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid train-rescheduling JSON instance") from error


def load_instance(path: str | Path) -> TrainInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def occupancies(route: TrainRoute, start: int) -> dict[int, tuple[int, int]]:
    result = {}
    offset = 0
    for section, duration in zip(route.sections, route.durations):
        result[section] = (start + offset, start + offset + duration)
        offset += duration
    return result


def routes_compatible(
    left: TrainRoute, left_start: int, right: TrainRoute, right_start: int, headway: int
) -> bool:
    left_use = occupancies(left, left_start)
    right_use = occupancies(right, right_start)
    return all(
        left_use[section][1] + headway <= right_use[section][0]
        or right_use[section][1] + headway <= left_use[section][0]
        for section in left_use.keys() & right_use.keys()
    )


def build_model(instance: TrainInstance) -> TrainModel:
    if instance.horizon < 1 or instance.headway < 0 or not instance.trains:
        raise ValueError("horizon, headway, and trains are invalid")
    if any(
        train.scheduled < 0
        or not train.routes
        or any(
            not route.sections
            or len(route.sections) != len(route.durations)
            or any(value < 1 for value in route.durations)
            for route in train.routes
        )
        for train in instance.trains
    ):
        raise ValueError("a train or route is invalid")
    model = cp.Model()
    routes = [
        model.int_var(0, len(train.routes) - 1, name=f"route_{index}")
        for index, train in enumerate(instance.trains)
    ]
    starts = [
        model.int_var(
            train.scheduled,
            instance.horizon - min(sum(route.durations) for route in train.routes),
            name=f"start_{index}",
        )
        for index, train in enumerate(instance.trains)
    ]
    for train, item in enumerate(instance.trains):
        allowed = [
            (route, start)
            for route, path in enumerate(item.routes)
            for start in range(
                item.scheduled, instance.horizon - sum(path.durations) + 1
            )
        ]
        model.table([routes[train], starts[train]], allowed)
    for first in range(len(instance.trains)):
        for second in range(first + 1, len(instance.trains)):
            allowed = []
            for left_route, left in enumerate(instance.trains[first].routes):
                for right_route, right in enumerate(instance.trains[second].routes):
                    for left_start in range(
                        instance.trains[first].scheduled,
                        instance.horizon - sum(left.durations) + 1,
                    ):
                        for right_start in range(
                            instance.trains[second].scheduled,
                            instance.horizon - sum(right.durations) + 1,
                        ):
                            if routes_compatible(
                                left, left_start, right, right_start, instance.headway
                            ):
                                allowed.append(
                                    (left_route, left_start, right_route, right_start)
                                )
            model.table(
                [routes[first], starts[first], routes[second], starts[second]], allowed
            )
    model.minimize(
        sum(
            starts[index] - train.scheduled
            for index, train in enumerate(instance.trains)
        )
    )
    return TrainModel(model, instance, routes, starts)


def decode(built: TrainModel, solution: cp.Solution) -> tuple[list[int], list[int]]:
    return values(solution, built.routes), values(solution, built.starts)


def validate(
    built: TrainModel, result: tuple[list[int], list[int]], objective: int | None
) -> None:
    routes, starts = result
    for train, (route, start) in enumerate(zip(routes, starts)):
        item = built.instance.trains[train]
        if (
            start < item.scheduled
            or start + sum(item.routes[route].durations) > built.instance.horizon
        ):
            raise AssertionError("a train lies outside the planning horizon")
    for first in range(len(routes)):
        for second in range(first + 1, len(routes)):
            if not routes_compatible(
                built.instance.trains[first].routes[routes[first]],
                starts[first],
                built.instance.trains[second].routes[routes[second]],
                starts[second],
                built.instance.headway,
            ):
                raise AssertionError("two trains conflict on a track section")
    delay = sum(
        start - train.scheduled for start, train in zip(starts, built.instance.trains)
    )
    if objective is not None and delay != objective:
        raise AssertionError("the objective does not match total departure delay")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON train-rescheduling instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob078 trains={len(instance.trains)} status={solution.status}")
    if not solution.is_sat():
        return 1
    result = decode(built, solution)
    validate(built, result, solution.objective)
    print(f"delay={solution.objective} routes={result[0]} starts={result[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
