"""CSPLib prob070: distributed channel assignment.

Specification: https://www.csplib.org/Problems/prob070/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values

DEFAULT_INSTANCE = {
    "access_points": 4,
    "channels": 4,
    "overlap_cost": [100, 50, 10, 0],
    "edges": [[0, 1, 2], [1, 2, 1], [2, 3, 2], [0, 3, 1]],
}


@dataclass(frozen=True)
class ChannelInstance:
    access_points: int
    channels: int
    overlap_cost: tuple[int, ...]
    edges: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class ChannelModel:
    model: cp.Model
    instance: ChannelInstance
    assignment: list[cp.IntVar]


def parse_instance(data: str | bytes) -> ChannelInstance:
    raw = json.loads(data)
    try:
        return ChannelInstance(
            int(raw["access_points"]),
            int(raw["channels"]),
            tuple(int(value) for value in raw["overlap_cost"]),
            tuple((int(edge[0]), int(edge[1]), int(edge[2])) for edge in raw["edges"]),
        )
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError("invalid channel-assignment JSON instance") from error


def load_instance(path: str | Path) -> ChannelInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def build_model(instance: ChannelInstance) -> ChannelModel:
    if instance.access_points < 1 or instance.channels < 1 or not instance.overlap_cost:
        raise ValueError("access points, channels, and overlap costs must be non-empty")
    if any(cost < 0 for cost in instance.overlap_cost):
        raise ValueError("overlap costs must be non-negative")
    if any(
        left < 0
        or right < 0
        or left >= instance.access_points
        or right >= instance.access_points
        or left == right
        or weight < 0
        for left, right, weight in instance.edges
    ):
        raise ValueError("an interference edge is invalid")
    model = cp.Model()
    assignment = model.int_vars(
        instance.access_points, 0, instance.channels - 1, name="channel"
    )
    maximum_edge_cost = max(instance.overlap_cost) * max(
        (edge[2] for edge in instance.edges), default=0
    )
    costs = []
    for edge_index, (left, right, weight) in enumerate(instance.edges):
        cost = model.int_var(0, maximum_edge_cost, name=f"interference_{edge_index}")
        model.table(
            [assignment[left], assignment[right], cost],
            [
                (
                    left_channel,
                    right_channel,
                    weight
                    * instance.overlap_cost[
                        min(
                            abs(left_channel - right_channel),
                            len(instance.overlap_cost) - 1,
                        )
                    ],
                )
                for left_channel in range(instance.channels)
                for right_channel in range(instance.channels)
            ],
        )
        costs.append(cost)
    model.add(assignment[0] == 0)
    model.minimize(sum(costs))
    return ChannelModel(model, instance, assignment)


def decode(built: ChannelModel, solution: cp.Solution) -> list[int]:
    return values(solution, built.assignment)


def interference_cost(instance: ChannelInstance, assignment: list[int]) -> int:
    return sum(
        weight
        * instance.overlap_cost[
            min(
                abs(assignment[left] - assignment[right]),
                len(instance.overlap_cost) - 1,
            )
        ]
        for left, right, weight in instance.edges
    )


def validate(built: ChannelModel, assignment: list[int], objective: int | None) -> None:
    if len(assignment) != built.instance.access_points or any(
        channel < 0 or channel >= built.instance.channels for channel in assignment
    ):
        raise AssertionError("a channel assignment is invalid")
    if (
        objective is not None
        and interference_cost(built.instance, assignment) != objective
    ):
        raise AssertionError("the objective does not match total interference")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON channel-assignment instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob070 access_points={instance.access_points} status={solution.status}")
    if not solution.is_sat():
        return 1
    assignment = decode(built, solution)
    validate(built, assignment, solution.objective)
    print(f"interference={solution.objective} channels={assignment}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
