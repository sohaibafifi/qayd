"""CSPLib prob072: target tracking in distributed sensor networks.

Specification: https://www.csplib.org/Problems/prob072/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args

DEFAULT_INSTANCE = {
    "sensor_count": 6,
    "visible": [[0, 1, 3, 4], [1, 2, 4, 5]],
    "compatible": [
        [0, 1, 1, 1, 1, 1],
        [1, 0, 1, 1, 1, 1],
        [1, 1, 0, 1, 1, 1],
        [1, 1, 1, 0, 1, 1],
        [1, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 0],
    ],
}


@dataclass(frozen=True)
class TrackingInstance:
    sensor_count: int
    visible: tuple[tuple[int, ...], ...]
    compatible: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class TrackingModel:
    model: cp.Model
    instance: TrackingInstance
    sensors: list[list[cp.IntVar]]


def parse_instance(data: str | bytes) -> TrackingInstance:
    raw = json.loads(data)
    try:
        return TrackingInstance(
            int(raw["sensor_count"]),
            tuple(tuple(int(value) for value in row) for row in raw["visible"]),
            tuple(tuple(int(value) for value in row) for row in raw["compatible"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid target-tracking JSON instance") from error


def load_instance(path: str | Path) -> TrackingInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def build_model(instance: TrackingInstance) -> TrackingModel:
    if instance.sensor_count < 3 or not instance.visible:
        raise ValueError("at least three sensors and one target are required")
    if len(instance.compatible) != instance.sensor_count or any(
        len(row) != instance.sensor_count for row in instance.compatible
    ):
        raise ValueError("the compatibility matrix dimensions are invalid")
    if any(
        len(row) < 3 or min(row) < 0 or max(row) >= instance.sensor_count
        for row in instance.visible
    ):
        raise ValueError("a target visibility set is invalid")
    model = cp.Model()
    sensors = [
        model.int_vars(3, 0, instance.sensor_count - 1, name=f"target_{target}")
        for target in range(len(instance.visible))
    ]
    model.all_different([sensor for row in sensors for sensor in row])
    compatible_pairs = [
        (left, right)
        for left in range(instance.sensor_count)
        for right in range(instance.sensor_count)
        if left != right and instance.compatible[left][right]
    ]
    for target, row in enumerate(sensors):
        allowed = [(sensor,) for sensor in sorted(set(instance.visible[target]))]
        for sensor in row:
            model.table([sensor], allowed)
        model.add(row[0] < row[1])
        model.add(row[1] < row[2])
        for first in range(3):
            for second in range(first + 1, 3):
                model.table([row[first], row[second]], compatible_pairs)
    return TrackingModel(model, instance, sensors)


def decode(built: TrackingModel, solution: cp.Solution) -> list[tuple[int, int, int]]:
    if not solution.is_sat():
        raise RuntimeError(f"no solution available, solver status is {solution.status}")
    return [
        tuple(solution.value(variable) for variable in row) for row in built.sensors
    ]


def validate(built: TrackingModel, sensors: list[tuple[int, int, int]]) -> None:
    flattened = [sensor for row in sensors for sensor in row]
    if len(flattened) != 3 * len(built.instance.visible) or len(set(flattened)) != len(
        flattened
    ):
        raise AssertionError("sensors are not assigned disjointly")
    for target, row in enumerate(sensors):
        if len(set(row)) != 3 or any(
            sensor not in built.instance.visible[target] for sensor in row
        ):
            raise AssertionError("a target has an invalid visible sensor triple")
        if any(
            not built.instance.compatible[row[first]][row[second]]
            for first in range(3)
            for second in range(first + 1, 3)
        ):
            raise AssertionError("a target sensor triple is not mutually compatible")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON target-tracking instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob072 targets={len(instance.visible)} status={solution.status}")
    if not solution.is_sat():
        return 1
    sensors = decode(built, solution)
    validate(built, sensors)
    print(f"sensors={sensors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
