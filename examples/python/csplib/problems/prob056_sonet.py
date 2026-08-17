"""CSPLib prob056: synchronous optical network ring design.

Specification: https://www.csplib.org/Problems/prob056/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args

DEFAULT_INSTANCE = {
    "demands": [
        [0, 2, 0, 1],
        [2, 0, 1, 0],
        [0, 1, 0, 2],
        [1, 0, 2, 0],
    ],
    "rings": 2,
    "node_limit": 3,
    "channel_limit": 3,
}


@dataclass(frozen=True)
class SonetInstance:
    demands: tuple[tuple[int, ...], ...]
    rings: int
    node_limit: int
    channel_limit: int | None


@dataclass(frozen=True)
class SonetModel:
    model: cp.Model
    instance: SonetInstance
    installed: list[list[cp.IntVar]]
    flows: dict[tuple[int, int, int], cp.IntVar]


def parse_instance(data: str | bytes) -> SonetInstance:
    raw = json.loads(data)
    try:
        channel_limit = raw.get("channel_limit")
        return SonetInstance(
            tuple(tuple(int(value) for value in row) for row in raw["demands"]),
            int(raw["rings"]),
            int(raw["node_limit"]),
            None if channel_limit is None else int(channel_limit),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid SONET JSON instance") from error


def load_instance(path: str | Path) -> SonetInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def build_model(instance: SonetInstance) -> SonetModel:
    node_count = len(instance.demands)
    if node_count < 2 or any(len(row) != node_count for row in instance.demands):
        raise ValueError("demands must be a square matrix with at least two nodes")
    if instance.rings < 1 or instance.node_limit < 2:
        raise ValueError("rings must be positive and node_limit at least two")
    for left in range(node_count):
        if instance.demands[left][left] != 0:
            raise ValueError("diagonal demands must be zero")
        for right in range(left + 1, node_count):
            if (
                instance.demands[left][right] < 0
                or instance.demands[left][right] != instance.demands[right][left]
            ):
                raise ValueError("demands must be non-negative and symmetric")

    model = cp.Model()
    installed = [
        [model.bool_var(name=f"ring_{ring}_node_{node}") for node in range(node_count)]
        for ring in range(instance.rings)
    ]
    for ring in range(instance.rings):
        model.add(sum(installed[ring]) <= instance.node_limit)

    flows: dict[tuple[int, int, int], cp.IntVar] = {}
    for left, right in combinations(range(node_count), 2):
        demand = instance.demands[left][right]
        if demand == 0:
            continue
        pair_flows = []
        for ring in range(instance.rings):
            flow = model.int_var(0, demand, name=f"flow_{ring}_{left}_{right}")
            model.add(flow <= demand * installed[ring][left])
            model.add(flow <= demand * installed[ring][right])
            flows[(ring, left, right)] = flow
            pair_flows.append(flow)
        model.add(sum(pair_flows) == demand)
    if instance.channel_limit is not None:
        if instance.channel_limit < 0:
            raise ValueError("channel_limit must be non-negative")
        for ring in range(instance.rings):
            ring_load = sum(
                flow for (flow_ring, _, _), flow in flows.items() if flow_ring == ring
            )
            model.add(ring_load <= instance.channel_limit)
    model.minimize(sum(variable for ring in installed for variable in ring))
    return SonetModel(model, instance, installed, flows)


def decode(
    built: SonetModel,
    solution: cp.Solution,
) -> tuple[list[list[int]], dict[tuple[int, int, int], int]]:
    if not solution.is_sat():
        raise RuntimeError(f"no solution available, solver status is {solution.status}")
    rings = [
        [node for node, variable in enumerate(row) if solution.value(variable)]
        for row in built.installed
    ]
    flows = {key: solution.value(variable) for key, variable in built.flows.items()}
    return rings, flows


def validate(
    built: SonetModel,
    rings: list[list[int]],
    flows: dict[tuple[int, int, int], int],
    objective: int | None,
) -> None:
    if any(len(nodes) > built.instance.node_limit for nodes in rings):
        raise AssertionError("a ring exceeds its node limit")
    node_sets = [set(nodes) for nodes in rings]
    for left, right in combinations(range(len(built.instance.demands)), 2):
        demand = built.instance.demands[left][right]
        delivered = sum(
            flows.get((ring, left, right), 0) for ring in range(built.instance.rings)
        )
        if delivered != demand:
            raise AssertionError("a demand is not fully carried")
        for ring in range(built.instance.rings):
            if flows.get((ring, left, right), 0) and not {left, right}.issubset(
                node_sets[ring]
            ):
                raise AssertionError("traffic uses a ring without both endpoint ADMs")
    if built.instance.channel_limit is not None:
        for ring in range(built.instance.rings):
            load = sum(
                value for (flow_ring, _, _), value in flows.items() if flow_ring == ring
            )
            if load > built.instance.channel_limit:
                raise AssertionError("a ring exceeds its channel limit")
    if sum(map(len, rings)) != objective:
        raise AssertionError("the objective does not match the ADM count")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON SONET instance")
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
        f"prob056 nodes={len(instance.demands)} rings={instance.rings} status={solution.status}"
    )
    if not solution.is_sat():
        return 1
    rings, flows = decode(built, solution)
    validate(built, rings, flows, solution.objective)
    print(f"adms={solution.objective} rings={rings} flows={flows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
