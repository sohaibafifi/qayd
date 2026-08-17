"""CSPLib prob048: minimum energy broadcast.

Specification: https://www.csplib.org/Problems/prob048/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values

DEFAULT_INSTANCE = {
    "source": 0,
    "energy": [
        [0, 2, 5, 0],
        [2, 0, 2, 4],
        [5, 2, 0, 1],
        [0, 4, 1, 0],
    ],
}


@dataclass(frozen=True)
class BroadcastInstance:
    source: int
    energy: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class BroadcastModel:
    model: cp.Model
    instance: BroadcastInstance
    parents: list[cp.IntVar]
    hops: list[cp.IntVar]
    powers: list[cp.IntVar]


def parse_instance(data: str | bytes) -> BroadcastInstance:
    raw = json.loads(data)
    try:
        return BroadcastInstance(
            int(raw["source"]),
            tuple(tuple(int(value) for value in row) for row in raw["energy"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid minimum-energy-broadcast JSON instance") from error


def load_instance(path: str | Path) -> BroadcastInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def build_model(instance: BroadcastInstance) -> BroadcastModel:
    node_count = len(instance.energy)
    if node_count < 1 or any(len(row) != node_count for row in instance.energy):
        raise ValueError("energy must be a non-empty square matrix")
    if instance.source < 0 or instance.source >= node_count:
        raise ValueError("source is invalid")
    if any(cost < 0 for row in instance.energy for cost in row):
        raise ValueError("energy costs must be non-negative")
    maximum_energy = max(cost for row in instance.energy for cost in row)
    model = cp.Model()
    parents = model.int_vars(node_count, 0, node_count - 1, name="parent")
    hops = model.int_vars(node_count, 0, node_count - 1, name="hop")
    powers = model.int_vars(node_count, 0, maximum_energy, name="power")
    model.add(parents[instance.source] == instance.source)
    model.add(hops[instance.source] == 0)
    for node in range(node_count):
        if node == instance.source:
            continue
        allowed_parents = [
            parent
            for parent in range(node_count)
            if parent != node and instance.energy[parent][node] > 0
        ]
        if not allowed_parents:
            model.add(parents[node] != parents[node])
            continue
        model.table([parents[node]], [(parent,) for parent in allowed_parents])
        parent_hop = model.int_var(0, node_count - 1, name=f"parent_hop_{node}")
        model.element(hops, parents[node], parent_hop)
        model.add(hops[node] == parent_hop + 1)
    for transmitter in range(node_count):
        for child in range(node_count):
            if child == instance.source or child == transmitter:
                continue
            is_child = model.bool_var(name=f"parent_{transmitter}_child_{child}")
            model.table(
                [parents[child], is_child],
                [(parent, int(parent == transmitter)) for parent in range(node_count)],
            )
            model.add(
                powers[transmitter] >= instance.energy[transmitter][child] * is_child
            )
    model.minimize(sum(powers))
    return BroadcastModel(model, instance, parents, hops, powers)


def decode(
    built: BroadcastModel, solution: cp.Solution
) -> tuple[list[int], list[int], list[int]]:
    return (
        values(solution, built.parents),
        values(solution, built.hops),
        values(solution, built.powers),
    )


def validate(
    built: BroadcastModel,
    parents: list[int],
    hops: list[int],
    powers: list[int],
    objective: int | None,
) -> None:
    source = built.instance.source
    if parents[source] != source or hops[source] != 0:
        raise AssertionError("the source root is invalid")
    for node in range(len(parents)):
        if node == source:
            continue
        parent = parents[node]
        if parent == node or built.instance.energy[parent][node] <= 0:
            raise AssertionError("a broadcast-tree edge is invalid")
        if hops[node] != hops[parent] + 1:
            raise AssertionError("hop counts do not define a rooted tree")
    expected_powers = [
        max(
            (
                built.instance.energy[node][child]
                for child in range(len(parents))
                if child != source and parents[child] == node
            ),
            default=0,
        )
        for node in range(len(parents))
    ]
    if powers != expected_powers:
        raise AssertionError("transmitter powers do not match their children")
    if objective is not None and sum(powers) != objective:
        raise AssertionError("the objective does not match total broadcast energy")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", nargs="?", help="JSON minimum-energy-broadcast instance"
    )
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob048 nodes={len(instance.energy)} status={solution.status}")
    if not solution.is_sat():
        return 1
    parents, hops, powers = decode(built, solution)
    validate(built, parents, hops, powers, solution.objective)
    print(f"energy={solution.objective} parents={parents} powers={powers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
