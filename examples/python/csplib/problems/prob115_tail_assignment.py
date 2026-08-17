"""CSPLib prob115: tail assignment.

Specification: https://www.csplib.org/Problems/prob115/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values

DEFAULT_INSTANCE = {
    "minimum_turn": 1,
    "tails": [
        {"name": "A", "airport": 0, "available": 0, "max_flight_time": 8},
        {"name": "B", "airport": 1, "available": 0, "max_flight_time": 8},
    ],
    "flights": [
        {
            "origin": 0,
            "destination": 1,
            "departure": 1,
            "arrival": 3,
            "forbidden_tails": [],
        },
        {
            "origin": 1,
            "destination": 0,
            "departure": 1,
            "arrival": 3,
            "forbidden_tails": [],
        },
        {
            "origin": 1,
            "destination": 2,
            "departure": 4,
            "arrival": 6,
            "forbidden_tails": [1],
        },
        {
            "origin": 0,
            "destination": 1,
            "departure": 4,
            "arrival": 6,
            "forbidden_tails": [0],
        },
    ],
}


@dataclass(frozen=True)
class Tail:
    name: str
    airport: int
    available: int
    max_flight_time: int


@dataclass(frozen=True)
class Flight:
    origin: int
    destination: int
    departure: int
    arrival: int
    forbidden_tails: frozenset[int]


@dataclass(frozen=True)
class TailInstance:
    minimum_turn: int
    tails: tuple[Tail, ...]
    flights: tuple[Flight, ...]


@dataclass(frozen=True)
class TailModel:
    model: cp.Model
    instance: TailInstance
    assignment: list[cp.IntVar]
    predecessor: list[cp.IntVar]


def parse_instance(data: str | bytes) -> TailInstance:
    raw = json.loads(data)
    try:
        tails = tuple(
            Tail(
                str(item["name"]),
                int(item["airport"]),
                int(item["available"]),
                int(item["max_flight_time"]),
            )
            for item in raw["tails"]
        )
        flights = tuple(
            Flight(
                int(item["origin"]),
                int(item["destination"]),
                int(item["departure"]),
                int(item["arrival"]),
                frozenset(int(value) for value in item.get("forbidden_tails", [])),
            )
            for item in raw["flights"]
        )
        return TailInstance(int(raw.get("minimum_turn", 0)), tails, flights)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid tail-assignment JSON instance") from error


def load_instance(path: str | Path) -> TailInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def legal_connection(instance: TailInstance, before: int, after: int) -> bool:
    left, right = instance.flights[before], instance.flights[after]
    return (
        left.destination == right.origin
        and left.arrival + instance.minimum_turn <= right.departure
    )


def build_model(instance: TailInstance) -> TailModel:
    tail_count = len(instance.tails)
    flight_count = len(instance.flights)
    if tail_count < 1 or flight_count < 1 or instance.minimum_turn < 0:
        raise ValueError("tails, flights, and minimum turn are invalid")
    if any(
        flight.departure < 0 or flight.arrival <= flight.departure
        for flight in instance.flights
    ):
        raise ValueError("a flight time is invalid")
    if any(tail.max_flight_time < 0 for tail in instance.tails):
        raise ValueError("a tail flight-time limit is invalid")
    model = cp.Model()
    assignment = model.int_vars(flight_count, 0, tail_count - 1, name="tail")
    predecessor = model.int_vars(
        flight_count, 0, flight_count + tail_count - 1, name="predecessor"
    )
    model.all_different(predecessor)
    for flight, item in enumerate(instance.flights):
        allowed_tails = [
            tail for tail in range(tail_count) if tail not in item.forbidden_tails
        ]
        if not allowed_tails:
            raise ValueError(f"flight {flight} is forbidden for every tail")
        model.table([assignment[flight]], [(tail,) for tail in allowed_tails])
        allowed_predecessors = []
        for tail in allowed_tails:
            source = instance.tails[tail]
            if source.airport == item.origin and source.available <= item.departure:
                allowed_predecessors.append((tail, flight_count + tail))
            allowed_predecessors.extend(
                (tail, before)
                for before in range(flight_count)
                if legal_connection(instance, before, flight)
            )
        model.table([assignment[flight], predecessor[flight]], allowed_predecessors)
        for before in range(flight_count):
            selected = model.bool_var(name=f"predecessor_{before}_of_{flight}")
            model.table(
                [predecessor[flight], selected],
                [
                    (candidate, int(candidate == before))
                    for candidate in range(flight_count + tail_count)
                ],
            )
            model.add((selected == 1).implies(assignment[flight] == assignment[before]))
    for tail, item in enumerate(instance.tails):
        memberships = []
        for flight in range(flight_count):
            selected = model.bool_var(name=f"flight_{flight}_tail_{tail}")
            model.table(
                [assignment[flight], selected],
                [
                    (candidate, int(candidate == tail))
                    for candidate in range(tail_count)
                ],
            )
            memberships.append(selected)
        model.add(
            sum(
                (flight.arrival - flight.departure) * selected
                for flight, selected in zip(instance.flights, memberships)
            )
            <= item.max_flight_time
        )
    return TailModel(model, instance, assignment, predecessor)


def decode(built: TailModel, solution: cp.Solution) -> tuple[list[int], list[int]]:
    return values(solution, built.assignment), values(solution, built.predecessor)


def validate(built: TailModel, result: tuple[list[int], list[int]]) -> None:
    assignment, predecessor = result
    if len(set(predecessor)) != len(predecessor):
        raise AssertionError("two flights use the same predecessor")
    flight_count = len(built.instance.flights)
    for flight, tail in enumerate(assignment):
        if tail in built.instance.flights[flight].forbidden_tails:
            raise AssertionError("a tail operates a forbidden flight")
        previous = predecessor[flight]
        if previous < flight_count:
            if assignment[previous] != tail or not legal_connection(
                built.instance, previous, flight
            ):
                raise AssertionError("a flight connection is invalid")
        else:
            source_tail = previous - flight_count
            source = built.instance.tails[source_tail]
            item = built.instance.flights[flight]
            if (
                source_tail != tail
                or source.airport != item.origin
                or source.available > item.departure
            ):
                raise AssertionError("a route does not start at its carry-in activity")
    for tail, item in enumerate(built.instance.tails):
        total = sum(
            flight.arrival - flight.departure
            for flight, assigned in zip(built.instance.flights, assignment)
            if assigned == tail
        )
        if total > item.max_flight_time:
            raise AssertionError("a tail exceeds its flight-time maintenance limit")


def routes(built: TailModel, result: tuple[list[int], list[int]]) -> list[list[int]]:
    assignment, predecessor = result
    successors = {previous: flight for flight, previous in enumerate(predecessor)}
    flight_count = len(assignment)
    answer = []
    for tail in range(len(built.instance.tails)):
        route = []
        cursor = flight_count + tail
        while cursor in successors:
            cursor = successors[cursor]
            route.append(cursor)
        answer.append(route)
    return answer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON tail-assignment instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob115 flights={len(instance.flights)} status={solution.status}")
    if not solution.is_sat():
        return 1
    result = decode(built, solution)
    validate(built, result)
    print(f"routes={routes(built, result)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
