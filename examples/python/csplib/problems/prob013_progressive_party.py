"""CSPLib prob013: progressive party problem.

Specification: https://www.csplib.org/Problems/prob013/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values

DEFAULT_INSTANCE = {
    "periods": 2,
    "boats": [
        {"capacity": 6, "crew": 2},
        {"capacity": 6, "crew": 2},
        {"capacity": 4, "crew": 2},
        {"capacity": 4, "crew": 2},
    ],
}


@dataclass(frozen=True)
class Boat:
    capacity: int
    crew: int


@dataclass(frozen=True)
class PartyInstance:
    periods: int
    boats: tuple[Boat, ...]


@dataclass(frozen=True)
class PartyModel:
    model: cp.Model
    instance: PartyInstance
    hosts: list[cp.IntVar]
    visits: list[list[cp.IntVar]]


def parse_instance(data: str | bytes) -> PartyInstance:
    raw = json.loads(data)
    try:
        boats = tuple(
            Boat(int(item["capacity"]), int(item["crew"])) for item in raw["boats"]
        )
        return PartyInstance(int(raw["periods"]), boats)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid progressive-party JSON instance") from error


def load_instance(path: str | Path) -> PartyInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def build_model(instance: PartyInstance) -> PartyModel:
    boat_count = len(instance.boats)
    if instance.periods < 1 or boat_count < 2:
        raise ValueError("periods and boats must be non-empty")
    if any(boat.crew < 1 or boat.capacity < boat.crew for boat in instance.boats):
        raise ValueError("a boat capacity or crew size is invalid")
    model = cp.Model()
    hosts = [model.bool_var(name=f"host_{boat}") for boat in range(boat_count)]
    visits = [
        model.int_vars(instance.periods, 0, boat_count - 1, name=f"crew_{boat}")
        for boat in range(boat_count)
    ]
    membership = [
        [
            [
                model.bool_var(name=f"crew_{crew}_period_{period}_host_{host}")
                for host in range(boat_count)
            ]
            for period in range(instance.periods)
        ]
        for crew in range(boat_count)
    ]
    for crew in range(boat_count):
        for period in range(instance.periods):
            self_visit = model.bool_var(name=f"self_{crew}_{period}")
            model.table(
                [visits[crew][period], self_visit],
                [(host, int(host == crew)) for host in range(boat_count)],
            )
            model.add(self_visit == hosts[crew])
            selected_host = model.bool_var(name=f"selected_host_{crew}_{period}")
            model.element(hosts, visits[crew][period], selected_host)
            model.add(selected_host == 1)
            for host in range(boat_count):
                model.table(
                    [visits[crew][period], membership[crew][period][host]],
                    [
                        (candidate, int(candidate == host))
                        for candidate in range(boat_count)
                    ],
                )
        for first in range(instance.periods):
            for second in range(first + 1, instance.periods):
                model.add(
                    (hosts[crew] == 0).implies(
                        visits[crew][first] != visits[crew][second]
                    )
                )
    for period in range(instance.periods):
        for host, boat in enumerate(instance.boats):
            model.add(
                sum(
                    instance.boats[crew].crew * membership[crew][period][host]
                    for crew in range(boat_count)
                )
                <= boat.capacity
            )
    same_table = [
        (left, right, int(left == right))
        for left in range(boat_count)
        for right in range(boat_count)
    ]
    for first in range(boat_count):
        for second in range(first + 1, boat_count):
            meetings = []
            for period in range(instance.periods):
                meet = model.bool_var(name=f"meet_{first}_{second}_{period}")
                model.table(
                    [visits[first][period], visits[second][period], meet], same_table
                )
                meetings.append(meet)
            model.add(sum(meetings) <= 1)
    model.minimize(sum(hosts))
    return PartyModel(model, instance, hosts, visits)


def decode(
    built: PartyModel, solution: cp.Solution
) -> tuple[list[int], list[list[int]]]:
    host_boats = [
        index for index, value in enumerate(values(solution, built.hosts)) if value
    ]
    visits = [[solution.value(variable) for variable in row] for row in built.visits]
    return host_boats, visits


def validate(
    built: PartyModel,
    hosts: list[int],
    visits: list[list[int]],
    objective: int | None,
) -> None:
    host_set = set(hosts)
    if any(len(row) != built.instance.periods for row in visits):
        raise AssertionError("a crew schedule has the wrong length")
    for crew, row in enumerate(visits):
        if crew in host_set and any(host != crew for host in row):
            raise AssertionError("a host crew leaves its boat")
        if crew not in host_set and len(set(row)) != len(row):
            raise AssertionError("a guest crew revisits a host")
        if any(host not in host_set for host in row):
            raise AssertionError("a crew visits a non-host boat")
    for period in range(built.instance.periods):
        for host in hosts:
            load = sum(
                built.instance.boats[crew].crew
                for crew in range(len(visits))
                if visits[crew][period] == host
            )
            if load > built.instance.boats[host].capacity:
                raise AssertionError("a host capacity is exceeded")
    for first in range(len(visits)):
        for second in range(first + 1, len(visits)):
            if (
                sum(
                    visits[first][period] == visits[second][period]
                    for period in range(built.instance.periods)
                )
                > 1
            ):
                raise AssertionError("two guest crews meet more than once")
    if objective is not None and len(hosts) != objective:
        raise AssertionError("the objective does not match the host count")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON progressive-party instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob013 boats={len(instance.boats)} status={solution.status}")
    if not solution.is_sat():
        return 1
    hosts, visits = decode(built, solution)
    validate(built, hosts, visits, solution.objective)
    print(f"hosts={hosts} visits={visits}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
