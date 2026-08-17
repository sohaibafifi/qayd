"""CSPLib prob008: vessel loading.

Specification: https://www.csplib.org/Problems/prob008/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values

DEFAULT_INSTANCE = {
    "deck": [7, 5],
    "containers": [
        {"width": 3, "height": 2, "class": 0},
        {"width": 2, "height": 2, "class": 1},
        {"width": 2, "height": 3, "class": 0},
    ],
    "separation_x": [[0, 1], [1, 0]],
    "separation_y": [[0, 0], [0, 0]],
}


@dataclass(frozen=True)
class Container:
    width: int
    height: int
    container_class: int


@dataclass(frozen=True)
class VesselInstance:
    width: int
    height: int
    containers: tuple[Container, ...]
    separation_x: tuple[tuple[int, ...], ...]
    separation_y: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class VesselModel:
    model: cp.Model
    instance: VesselInstance
    x: list[cp.IntVar]
    y: list[cp.IntVar]


def parse_instance(data: str | bytes) -> VesselInstance:
    raw = json.loads(data)
    try:
        containers = tuple(
            Container(int(item["width"]), int(item["height"]), int(item["class"]))
            for item in raw["containers"]
        )
        return VesselInstance(
            int(raw["deck"][0]),
            int(raw["deck"][1]),
            containers,
            tuple(tuple(int(value) for value in row) for row in raw["separation_x"]),
            tuple(tuple(int(value) for value in row) for row in raw["separation_y"]),
        )
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError("invalid vessel-loading JSON instance") from error


def load_instance(path: str | Path) -> VesselInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def build_model(instance: VesselInstance) -> VesselModel:
    if instance.width < 1 or instance.height < 1 or not instance.containers:
        raise ValueError("the deck and container set must be non-empty")
    class_count = len(instance.separation_x)
    if class_count < 1 or len(instance.separation_y) != class_count:
        raise ValueError("separation matrices have inconsistent dimensions")
    if any(
        len(row) != class_count
        for row in (*instance.separation_x, *instance.separation_y)
    ):
        raise ValueError("separation matrices must be square")
    for item in instance.containers:
        if (
            item.width < 1
            or item.height < 1
            or item.width > instance.width
            or item.height > instance.height
            or item.container_class < 0
            or item.container_class >= class_count
        ):
            raise ValueError("a container is invalid")

    model = cp.Model()
    x = [
        model.int_var(0, instance.width - item.width, name=f"x_{index}")
        for index, item in enumerate(instance.containers)
    ]
    y = [
        model.int_var(0, instance.height - item.height, name=f"y_{index}")
        for index, item in enumerate(instance.containers)
    ]
    for first in range(len(instance.containers)):
        left = instance.containers[first]
        for second in range(first + 1, len(instance.containers)):
            right = instance.containers[second]
            separation_x = instance.separation_x[left.container_class][
                right.container_class
            ]
            separation_y = instance.separation_y[left.container_class][
                right.container_class
            ]
            model.add(
                cp.any(
                    [
                        x[first] + left.width + separation_x <= x[second],
                        x[second] + right.width + separation_x <= x[first],
                        y[first] + left.height + separation_y <= y[second],
                        y[second] + right.height + separation_y <= y[first],
                    ]
                )
            )
    return VesselModel(model, instance, x, y)


def decode(built: VesselModel, solution: cp.Solution) -> list[tuple[int, int]]:
    return list(zip(values(solution, built.x), values(solution, built.y)))


def validate(built: VesselModel, origins: list[tuple[int, int]]) -> None:
    if len(origins) != len(built.instance.containers):
        raise AssertionError("the number of decoded origins is invalid")
    for (x, y), item in zip(origins, built.instance.containers):
        if (
            x < 0
            or y < 0
            or x + item.width > built.instance.width
            or y + item.height > built.instance.height
        ):
            raise AssertionError("a container lies outside the deck")
    for first in range(len(origins)):
        left = built.instance.containers[first]
        for second in range(first + 1, len(origins)):
            right = built.instance.containers[second]
            sx = built.instance.separation_x[left.container_class][
                right.container_class
            ]
            sy = built.instance.separation_y[left.container_class][
                right.container_class
            ]
            separated = (
                origins[first][0] + left.width + sx <= origins[second][0]
                or origins[second][0] + right.width + sx <= origins[first][0]
                or origins[first][1] + left.height + sy <= origins[second][1]
                or origins[second][1] + right.height + sy <= origins[first][1]
            )
            if not separated:
                raise AssertionError("two containers overlap or violate separation")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON vessel-loading instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob008 containers={len(instance.containers)} status={solution.status}")
    if not solution.is_sat():
        return 1
    origins = decode(built, solution)
    validate(built, origins)
    print(f"origins={origins}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
