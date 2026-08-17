"""CSPLib prob091: medical appointment sequence scheduling.

Specification: https://www.csplib.org/Problems/prob091/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values

DEFAULT_INSTANCE = {
    "slots_per_day": 4,
    "days": 4,
    "resources": [
        {"type": "doctor", "available": list(range(16)), "workload": 1},
        {"type": "doctor", "available": list(range(16)), "workload": 0},
        {"type": "room", "available": list(range(16)), "workload": 0},
    ],
    "appointments": [
        {"duration": 1, "types": ["doctor", "room"]},
        {"duration": 2, "types": ["doctor", "room"]},
        {"duration": 1, "types": ["doctor"]},
    ],
    "order": [0, 1, 2],
    "gaps": [
        {"before": 0, "after": 1, "minimum": 1, "maximum": 5},
        {"before": 1, "after": 2, "minimum": 0, "maximum": 4},
    ],
    "continuity": [[0, 0, 1, 0]],
    "unavailable_days": [],
}


@dataclass(frozen=True)
class SequenceResource:
    kind: str
    available: frozenset[int]
    workload: int


@dataclass(frozen=True)
class SequenceAppointment:
    duration: int
    types: tuple[str, ...]


@dataclass(frozen=True)
class MasspInstance:
    slots_per_day: int
    days: int
    resources: tuple[SequenceResource, ...]
    appointments: tuple[SequenceAppointment, ...]
    order: tuple[int, ...]
    gaps: tuple[tuple[int, int, int, int], ...]
    continuity: tuple[tuple[int, int, int, int], ...]
    unavailable_days: frozenset[int]


@dataclass(frozen=True)
class MasspModel:
    model: cp.Model
    instance: MasspInstance
    starts: list[cp.IntVar]
    resources: list[list[cp.IntVar]]


def parse_instance(data: str | bytes) -> MasspInstance:
    raw = json.loads(data)
    try:
        resources = tuple(
            SequenceResource(
                str(item["type"]),
                frozenset(int(slot) for slot in item["available"]),
                int(item.get("workload", 0)),
            )
            for item in raw["resources"]
        )
        appointments = tuple(
            SequenceAppointment(
                int(item["duration"]), tuple(str(value) for value in item["types"])
            )
            for item in raw["appointments"]
        )
        gaps = tuple(
            (
                int(item["before"]),
                int(item["after"]),
                int(item["minimum"]),
                int(item["maximum"]),
            )
            for item in raw.get("gaps", [])
        )
        return MasspInstance(
            int(raw["slots_per_day"]),
            int(raw["days"]),
            resources,
            appointments,
            tuple(int(value) for value in raw.get("order", [])),
            gaps,
            tuple(
                tuple(int(value) for value in row) for row in raw.get("continuity", [])
            ),
            frozenset(int(value) for value in raw.get("unavailable_days", [])),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid MASSP JSON instance") from error


def load_instance(path: str | Path) -> MasspInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def valid_resource_starts(
    instance: MasspInstance, appointment: int, resource: int
) -> list[int]:
    duration = instance.appointments[appointment].duration
    horizon = instance.days * instance.slots_per_day
    return [
        start
        for start in range(horizon - duration + 1)
        if start // instance.slots_per_day not in instance.unavailable_days
        and start % instance.slots_per_day + duration <= instance.slots_per_day
        and all(
            start + offset in instance.resources[resource].available
            for offset in range(duration)
        )
    ]


def build_model(instance: MasspInstance) -> MasspModel:
    count = len(instance.appointments)
    horizon = instance.days * instance.slots_per_day
    if (
        instance.slots_per_day < 1
        or instance.days < 1
        or not instance.resources
        or count < 1
    ):
        raise ValueError("calendar, resources, and appointments must be non-empty")
    if sorted(instance.order) != list(range(count)):
        raise ValueError("order must be a permutation of appointments")
    model = cp.Model()
    starts = [
        model.int_var(0, horizon - item.duration, name=f"start_{index}")
        for index, item in enumerate(instance.appointments)
    ]
    assigned: list[list[cp.IntVar]] = []
    for appointment, item in enumerate(instance.appointments):
        if item.duration < 1 or not item.types:
            raise ValueError("an appointment is invalid")
        row = model.int_vars(
            len(item.types),
            0,
            len(instance.resources) - 1,
            name=f"resource_{appointment}",
        )
        model.all_different(row)
        for requirement, kind in enumerate(item.types):
            allowed = [
                (resource, start)
                for resource, candidate in enumerate(instance.resources)
                if candidate.kind == kind
                for start in valid_resource_starts(instance, appointment, resource)
            ]
            if not allowed:
                raise ValueError(f"no resource can satisfy appointment {appointment}")
            model.table([row[requirement], starts[appointment]], allowed)
        assigned.append(row)
    for left, right in zip(instance.order, instance.order[1:]):
        model.add(starts[left] + instance.appointments[left].duration <= starts[right])
    for before, after, minimum, maximum in instance.gaps:
        if not (0 <= before < count and 0 <= after < count and 0 <= minimum <= maximum):
            raise ValueError("an appointment gap is invalid")
        end = starts[before] + instance.appointments[before].duration
        model.add(starts[after] >= end + minimum)
        model.add(starts[after] <= end + maximum)
    for left, left_req, right, right_req in instance.continuity:
        try:
            model.add(assigned[left][left_req] == assigned[right][right_req])
        except IndexError as error:
            raise ValueError("a continuity reference is invalid") from error
    maximum_workload = model.int_var(
        0,
        sum(item.duration for item in instance.appointments)
        + max(item.workload for item in instance.resources),
        name="maximum_workload",
    )
    for resource, item in enumerate(instance.resources):
        contributions = []
        for appointment, row in enumerate(assigned):
            for requirement, variable in enumerate(row):
                selected = model.bool_var(
                    name=f"selected_{appointment}_{requirement}_{resource}"
                )
                model.table(
                    [variable, selected],
                    [
                        (candidate, int(candidate == resource))
                        for candidate in range(len(instance.resources))
                    ],
                )
                contributions.append(
                    instance.appointments[appointment].duration * selected
                )
        model.add(maximum_workload >= item.workload + sum(contributions))
    model.minimize(maximum_workload)
    return MasspModel(model, instance, starts, assigned)


def decode(
    built: MasspModel, solution: cp.Solution
) -> tuple[list[int], list[list[int]]]:
    return values(solution, built.starts), [
        values(solution, row) for row in built.resources
    ]


def validate(
    built: MasspModel, result: tuple[list[int], list[list[int]]], objective: int | None
) -> None:
    starts, assigned = result
    for appointment, (start, resources) in enumerate(zip(starts, assigned)):
        if len(set(resources)) != len(resources):
            raise AssertionError("an appointment reuses a resource")
        for requirement, resource in enumerate(resources):
            if (
                built.instance.resources[resource].kind
                != built.instance.appointments[appointment].types[requirement]
            ):
                raise AssertionError("a resource has the wrong type")
            if start not in valid_resource_starts(
                built.instance, appointment, resource
            ):
                raise AssertionError("a resource is unavailable")
    for left, right in zip(built.instance.order, built.instance.order[1:]):
        if starts[left] + built.instance.appointments[left].duration > starts[right]:
            raise AssertionError("the appointment order is violated")
    for before, after, minimum, maximum in built.instance.gaps:
        gap = (
            starts[after]
            - starts[before]
            - built.instance.appointments[before].duration
        )
        if not minimum <= gap <= maximum:
            raise AssertionError("an appointment gap is violated")
    for left, left_req, right, right_req in built.instance.continuity:
        if assigned[left][left_req] != assigned[right][right_req]:
            raise AssertionError("continuity of care is violated")
    workloads = [item.workload for item in built.instance.resources]
    for appointment, resources in enumerate(assigned):
        for resource in resources:
            workloads[resource] += built.instance.appointments[appointment].duration
    if objective is not None and max(workloads) != objective:
        raise AssertionError("the objective does not match maximum workload")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON MASSP instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob091 appointments={len(instance.appointments)} status={solution.status}")
    if not solution.is_sat():
        return 1
    result = decode(built, solution)
    validate(built, result, solution.objective)
    print(
        f"maximum_workload={solution.objective} starts={result[0]} resources={result[1]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
