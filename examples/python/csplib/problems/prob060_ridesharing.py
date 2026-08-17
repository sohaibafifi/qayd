"""CSPLib prob060: ridesharing.

Specification: https://www.csplib.org/Problems/prob060/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values

DEFAULT_INSTANCE = {
    "travel_times": [
        [0, 10, 20, 30],
        [10, 0, 10, 20],
        [20, 10, 0, 10],
        [30, 20, 10, 0],
    ],
    "drivers": [
        {"user": 1, "route": [0, 1, 2, 3], "earliest": 0, "latest": 50, "seats": 2},
        {"user": 2, "route": [1, 2, 3], "earliest": 5, "latest": 45, "seats": 1},
    ],
    "riders": [
        {
            "user": 3,
            "origin": 1,
            "destination": 2,
            "earliest": 5,
            "latest": 35,
            "load": 1,
            "reward": 2,
        },
        {
            "user": 4,
            "origin": 2,
            "destination": 3,
            "earliest": 15,
            "latest": 50,
            "load": 1,
            "reward": 2,
        },
        {
            "user": 2,
            "origin": 1,
            "destination": 3,
            "earliest": 5,
            "latest": 45,
            "load": 1,
            "reward": 3,
        },
    ],
}


@dataclass(frozen=True)
class Driver:
    user: int
    route: tuple[int, ...]
    earliest: int
    latest: int
    seats: int


@dataclass(frozen=True)
class Rider:
    user: int
    origin: int
    destination: int
    earliest: int
    latest: int
    load: int
    reward: int


@dataclass(frozen=True)
class RidesharingInstance:
    travel_times: tuple[tuple[int, ...], ...]
    drivers: tuple[Driver, ...]
    riders: tuple[Rider, ...]


@dataclass(frozen=True)
class RidesharingModel:
    model: cp.Model
    instance: RidesharingInstance
    assignment: list[cp.IntVar]
    times: list[list[cp.IntVar]]


def parse_instance(data: str | bytes) -> RidesharingInstance:
    raw = json.loads(data)
    try:
        drivers = tuple(
            Driver(
                int(item["user"]),
                tuple(int(value) for value in item["route"]),
                int(item["earliest"]),
                int(item["latest"]),
                int(item["seats"]),
            )
            for item in raw["drivers"]
        )
        riders = tuple(
            Rider(
                int(item["user"]),
                int(item["origin"]),
                int(item["destination"]),
                int(item["earliest"]),
                int(item["latest"]),
                int(item.get("load", 1)),
                int(item.get("reward", 1)),
            )
            for item in raw["riders"]
        )
        return RidesharingInstance(
            tuple(tuple(int(value) for value in row) for row in raw["travel_times"]),
            drivers,
            riders,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid ridesharing JSON instance") from error


def load_instance(path: str | Path) -> RidesharingInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def route_positions(driver: Driver, rider: Rider) -> tuple[int, int] | None:
    try:
        pickup = driver.route.index(rider.origin)
        drop = driver.route.index(rider.destination, pickup + 1)
        return pickup, drop
    except ValueError:
        return None


def build_model(instance: RidesharingInstance) -> RidesharingModel:
    location_count = len(instance.travel_times)
    if location_count < 1 or any(
        len(row) != location_count for row in instance.travel_times
    ):
        raise ValueError("travel_times must be a square matrix")
    if not instance.drivers or not instance.riders:
        raise ValueError("drivers and riders must be non-empty")
    model = cp.Model()
    sentinel = len(instance.drivers)
    times = []
    for driver_index, driver in enumerate(instance.drivers):
        if (
            len(driver.route) < 2
            or driver.seats < 1
            or any(place < 0 or place >= location_count for place in driver.route)
        ):
            raise ValueError("a driver offer is invalid")
        row = model.int_vars(
            len(driver.route),
            driver.earliest,
            driver.latest,
            name=f"time_{driver_index}",
        )
        model.add(row[0] >= driver.earliest)
        model.add(row[-1] <= driver.latest)
        for position in range(len(driver.route) - 1):
            model.add(
                row[position + 1]
                >= row[position]
                + instance.travel_times[driver.route[position]][
                    driver.route[position + 1]
                ]
            )
        times.append(row)
    assignment = model.int_vars(len(instance.riders), 0, sentinel, name="driver")
    served = []
    compatible_by_rider = []
    for rider_index, rider in enumerate(instance.riders):
        compatible = [
            driver
            for driver, offer in enumerate(instance.drivers)
            if offer.user != rider.user
            and route_positions(offer, rider) is not None
            and rider.load <= offer.seats
        ]
        compatible_by_rider.append(compatible)
        model.table(
            [assignment[rider_index]], [(driver,) for driver in [*compatible, sentinel]]
        )
        selected = model.bool_var(name=f"served_{rider_index}")
        model.table(
            [assignment[rider_index], selected],
            [(driver, int(driver != sentinel)) for driver in [*compatible, sentinel]],
        )
        served.append(selected)
        for driver in compatible:
            pickup, drop = route_positions(instance.drivers[driver], rider)
            uses = model.bool_var(name=f"rider_{rider_index}_driver_{driver}")
            model.table(
                [assignment[rider_index], uses],
                [
                    (candidate, int(candidate == driver))
                    for candidate in [*compatible, sentinel]
                ],
            )
            model.add((uses == 1).implies(times[driver][pickup] >= rider.earliest))
            model.add((uses == 1).implies(times[driver][drop] <= rider.latest))
    for driver, offer in enumerate(instance.drivers):
        for segment in range(len(offer.route) - 1):
            passengers = []
            for rider_index, rider in enumerate(instance.riders):
                positions = route_positions(offer, rider)
                if (
                    driver not in compatible_by_rider[rider_index]
                    or positions is None
                    or not positions[0] <= segment < positions[1]
                ):
                    continue
                selected = model.bool_var(name=f"load_{driver}_{segment}_{rider_index}")
                model.table(
                    [assignment[rider_index], selected],
                    [
                        (candidate, int(candidate == driver))
                        for candidate in [*compatible_by_rider[rider_index], sentinel]
                    ],
                )
                passengers.append(rider.load * selected)
            if passengers:
                model.add(sum(passengers) <= offer.seats)
    driver_by_user = {
        driver.user: index for index, driver in enumerate(instance.drivers)
    }
    for shifter, rider in enumerate(instance.riders):
        if rider.user not in driver_by_user:
            continue
        own_driver = driver_by_user[rider.user]
        domain = [*compatible_by_rider[shifter], sentinel]
        for other in range(len(instance.riders)):
            other_domain = [*compatible_by_rider[other], sentinel]
            model.table(
                [assignment[shifter], assignment[other]],
                [
                    (left, right)
                    for left in domain
                    for right in other_domain
                    if left == sentinel or right != own_driver
                ],
            )
    model.maximize(
        sum(rider.reward * selected for rider, selected in zip(instance.riders, served))
    )
    return RidesharingModel(model, instance, assignment, times)


def decode(
    built: RidesharingModel, solution: cp.Solution
) -> tuple[list[int], list[list[int]]]:
    return values(solution, built.assignment), [
        values(solution, row) for row in built.times
    ]


def validate(
    built: RidesharingModel,
    result: tuple[list[int], list[list[int]]],
    objective: int | None,
) -> None:
    assignment, times = result
    sentinel = len(built.instance.drivers)
    for driver, (offer, row) in enumerate(zip(built.instance.drivers, times)):
        if row[0] < offer.earliest or row[-1] > offer.latest:
            raise AssertionError("a driver route violates its time window")
        for position in range(len(offer.route) - 1):
            if (
                row[position + 1]
                < row[position]
                + built.instance.travel_times[offer.route[position]][
                    offer.route[position + 1]
                ]
            ):
                raise AssertionError("a driver route violates travel time")
        for segment in range(len(offer.route) - 1):
            load = 0
            for rider_index, rider in enumerate(built.instance.riders):
                positions = route_positions(offer, rider)
                if (
                    assignment[rider_index] == driver
                    and positions is not None
                    and positions[0] <= segment < positions[1]
                ):
                    load += rider.load
            if load > offer.seats:
                raise AssertionError("a car capacity is exceeded")
    for rider_index, (rider, driver) in enumerate(
        zip(built.instance.riders, assignment)
    ):
        if driver == sentinel:
            continue
        positions = route_positions(built.instance.drivers[driver], rider)
        if (
            positions is None
            or times[driver][positions[0]] < rider.earliest
            or times[driver][positions[1]] > rider.latest
        ):
            raise AssertionError("a rider trip constraint is violated")
    driver_by_user = {
        driver.user: index for index, driver in enumerate(built.instance.drivers)
    }
    for shifter, rider in enumerate(built.instance.riders):
        if (
            rider.user in driver_by_user
            and assignment[shifter] != sentinel
            and driver_by_user[rider.user] in assignment
        ):
            raise AssertionError(
                "a shifter is simultaneously a rider and a serving driver"
            )
    reward = sum(
        rider.reward
        for rider, driver in zip(built.instance.riders, assignment)
        if driver != sentinel
    )
    if objective is not None and reward != objective:
        raise AssertionError("the objective does not match served-rider reward")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON ridesharing instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob060 riders={len(instance.riders)} status={solution.status}")
    if not solution.is_sat():
        return 1
    result = decode(built, solution)
    validate(built, result, solution.objective)
    print(f"reward={solution.objective} assignment={result[0]} times={result[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
