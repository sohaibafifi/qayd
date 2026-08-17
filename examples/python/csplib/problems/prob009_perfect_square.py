"""CSPLib prob009: perfect square placement.

Specification: https://www.csplib.org/Problems/prob009/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values

DEFAULT_INSTANCE = {"outer_size": 1, "square_sizes": [1]}


@dataclass(frozen=True)
class PerfectSquareInstance:
    outer_size: int
    square_sizes: tuple[int, ...]


@dataclass(frozen=True)
class PerfectSquareModel:
    model: cp.Model
    instance: PerfectSquareInstance
    x: list[cp.IntVar]
    y: list[cp.IntVar]


def parse_instance(data: str | bytes) -> PerfectSquareInstance:
    raw = json.loads(data)
    try:
        return PerfectSquareInstance(
            int(raw["outer_size"]),
            tuple(int(value) for value in raw["square_sizes"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid perfect-square JSON instance") from error


def load_instance(path: str | Path) -> PerfectSquareInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def build_model(instance: PerfectSquareInstance) -> PerfectSquareModel:
    if instance.outer_size < 1 or not instance.square_sizes:
        raise ValueError("outer_size and square_sizes must be non-empty")
    if len(set(instance.square_sizes)) != len(instance.square_sizes):
        raise ValueError("a perfect placement requires distinct square sizes")
    if any(size < 1 or size > instance.outer_size for size in instance.square_sizes):
        raise ValueError("a square size is invalid")
    if sum(size * size for size in instance.square_sizes) != instance.outer_size**2:
        raise ValueError("square areas must exactly equal the outer-square area")

    model = cp.Model()
    x = [
        model.int_var(0, instance.outer_size - size, name=f"x_{index}")
        for index, size in enumerate(instance.square_sizes)
    ]
    y = [
        model.int_var(0, instance.outer_size - size, name=f"y_{index}")
        for index, size in enumerate(instance.square_sizes)
    ]
    for first, left_size in enumerate(instance.square_sizes):
        for second in range(first + 1, len(instance.square_sizes)):
            right_size = instance.square_sizes[second]
            model.add(
                cp.any(
                    [
                        x[first] + left_size <= x[second],
                        x[second] + right_size <= x[first],
                        y[first] + left_size <= y[second],
                        y[second] + right_size <= y[first],
                    ]
                )
            )
    largest = max(
        range(len(instance.square_sizes)), key=instance.square_sizes.__getitem__
    )
    model.add(x[largest] <= (instance.outer_size - instance.square_sizes[largest]) // 2)
    model.add(y[largest] <= x[largest])
    return PerfectSquareModel(model, instance, x, y)


def decode(built: PerfectSquareModel, solution: cp.Solution) -> list[tuple[int, int]]:
    return list(zip(values(solution, built.x), values(solution, built.y)))


def validate(built: PerfectSquareModel, origins: list[tuple[int, int]]) -> None:
    if len(origins) != len(built.instance.square_sizes):
        raise AssertionError("the number of square origins is invalid")
    for (x, y), size in zip(origins, built.instance.square_sizes):
        if (
            x < 0
            or y < 0
            or x + size > built.instance.outer_size
            or y + size > built.instance.outer_size
        ):
            raise AssertionError("a square lies outside the outer square")
    for first, left_size in enumerate(built.instance.square_sizes):
        for second in range(first + 1, len(origins)):
            right_size = built.instance.square_sizes[second]
            separated = (
                origins[first][0] + left_size <= origins[second][0]
                or origins[second][0] + right_size <= origins[first][0]
                or origins[first][1] + left_size <= origins[second][1]
                or origins[second][1] + right_size <= origins[first][1]
            )
            if not separated:
                raise AssertionError("two squares overlap")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON perfect-square instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob009 squares={len(instance.square_sizes)} status={solution.status}")
    if not solution.is_sat():
        return 1
    origins = decode(built, solution)
    validate(built, origins)
    print(f"origins={origins}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
