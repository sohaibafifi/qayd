"""CSPLib prob089: medical appointment scheduling.

Specification: https://www.csplib.org/Problems/prob089/
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
    "days": 3,
    "duration": 2,
    "required_types": ["doctor", "room"],
    "resources": [
        {"type": "doctor", "available": [0, 1, 2, 4, 5, 6], "workload": 2},
        {"type": "doctor", "available": [4, 5, 6, 8, 9, 10], "workload": 0},
        {"type": "room", "available": [0, 1, 4, 5, 8, 9], "workload": 0},
    ],
    "undesired_days": [1],
    "preferred_days": [2],
    "preferred_resources": [1],
    "preferred_times": [{"weekday": 2, "start": 0, "end": 1}],
}


@dataclass(frozen=True)
class MedicalResource:
    kind: str
    available: frozenset[int]
    workload: int


@dataclass(frozen=True)
class MaspInstance:
    slots_per_day: int
    days: int
    duration: int
    required_types: tuple[str, ...]
    resources: tuple[MedicalResource, ...]
    undesired_days: frozenset[int]
    preferred_days: frozenset[int]
    preferred_resources: frozenset[int]
    preferred_times: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class MaspModel:
    model: cp.Model
    instance: MaspInstance
    start: cp.IntVar
    resources: list[cp.IntVar]


def parse_instance(data: str | bytes) -> MaspInstance:
    raw = json.loads(data)
    try:
        resources = tuple(
            MedicalResource(
                str(item["type"]),
                frozenset(int(slot) for slot in item["available"]),
                int(item.get("workload", 0)),
            )
            for item in raw["resources"]
        )
        preferred_times = tuple(
            (int(item["weekday"]), int(item["start"]), int(item["end"]))
            for item in raw.get("preferred_times", [])
        )
        return MaspInstance(
            int(raw["slots_per_day"]),
            int(raw["days"]),
            int(raw["duration"]),
            tuple(str(value) for value in raw["required_types"]),
            resources,
            frozenset(int(value) for value in raw.get("undesired_days", [])),
            frozenset(int(value) for value in raw.get("preferred_days", [])),
            frozenset(int(value) for value in raw.get("preferred_resources", [])),
            preferred_times,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid MASP JSON instance") from error


def load_instance(path: str | Path) -> MaspInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def feasible_starts(instance: MaspInstance, resource: int) -> list[int]:
    horizon = instance.days * instance.slots_per_day
    return [
        start
        for start in range(horizon - instance.duration + 1)
        if start // instance.slots_per_day not in instance.undesired_days
        and start % instance.slots_per_day + instance.duration <= instance.slots_per_day
        and all(
            start + offset in instance.resources[resource].available
            for offset in range(instance.duration)
        )
    ]


def preference_penalty(instance: MaspInstance, start: int, assigned: list[int]) -> int:
    day = start // instance.slots_per_day
    slot = start % instance.slots_per_day
    penalty = int(bool(instance.preferred_days) and day not in instance.preferred_days)
    penalty += (
        sum(resource not in instance.preferred_resources for resource in assigned)
        if instance.preferred_resources
        else 0
    )
    penalty += sum(
        weekday == day % 7 and not low <= slot <= high
        for weekday, low, high in instance.preferred_times
    )
    return penalty


def build_model(instance: MaspInstance) -> MaspModel:
    if (
        instance.slots_per_day < 1
        or instance.days < 1
        or instance.duration < 1
        or not instance.required_types
    ):
        raise ValueError("calendar and required resource types must be non-empty")
    if not instance.resources or any(
        resource.workload < 0 for resource in instance.resources
    ):
        raise ValueError("resources and workloads are invalid")
    horizon = instance.days * instance.slots_per_day
    model = cp.Model()
    start = model.int_var(0, horizon - instance.duration, name="start")
    assigned = model.int_vars(
        len(instance.required_types), 0, len(instance.resources) - 1, name="resource"
    )
    model.all_different(assigned)
    for requirement, kind in enumerate(instance.required_types):
        allowed = [
            (resource, slot)
            for resource, item in enumerate(instance.resources)
            if item.kind == kind
            for slot in feasible_starts(instance, resource)
        ]
        if not allowed:
            raise ValueError(f"no resource can satisfy required type {kind!r}")
        model.table([assigned[requirement], start], allowed)
    maximum_penalty = 1 + len(assigned) + len(instance.preferred_times)
    penalty = model.int_var(0, maximum_penalty, name="preference_penalty")
    rows = []
    for slot in range(horizon - instance.duration + 1):
        for selection in __import__("itertools").product(
            range(len(instance.resources)), repeat=len(assigned)
        ):
            if len(set(selection)) == len(selection):
                rows.append(
                    (
                        slot,
                        *selection,
                        preference_penalty(instance, slot, list(selection)),
                    )
                )
    model.table([start, *assigned, penalty], rows)
    model.minimize(penalty)
    return MaspModel(model, instance, start, assigned)


def decode(built: MaspModel, solution: cp.Solution) -> tuple[int, list[int]]:
    return solution.value(built.start), values(solution, built.resources)


def validate(
    built: MaspModel, result: tuple[int, list[int]], objective: int | None
) -> None:
    start, assigned = result
    if len(set(assigned)) != len(assigned):
        raise AssertionError("a resource is selected more than once")
    for requirement, resource in enumerate(assigned):
        if (
            built.instance.resources[resource].kind
            != built.instance.required_types[requirement]
        ):
            raise AssertionError("a selected resource has the wrong type")
        if start not in feasible_starts(built.instance, resource):
            raise AssertionError("a selected resource is unavailable")
    if (
        objective is not None
        and preference_penalty(built.instance, start, assigned) != objective
    ):
        raise AssertionError("the objective does not match preference violations")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON MASP instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(
        f"prob089 requirements={len(instance.required_types)} status={solution.status}"
    )
    if not solution.is_sat():
        return 1
    result = decode(built, solution)
    validate(built, result, solution.objective)
    print(f"violations={solution.objective} start={result[0]} resources={result[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
