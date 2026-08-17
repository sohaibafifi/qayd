"""CSPLib prob051: tank allocation.

Specification: https://www.csplib.org/Problems/prob051/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args

DEFAULT_INSTANCE = {
    "tank_capacities": [4, 3, 3, 4],
    "cargo_volumes": [5, 4],
    "allowed": [[1, 1], [1, 0], [0, 1], [1, 1]],
    "adjacent_tanks": [[0, 1], [1, 2], [2, 3]],
    "incompatible_cargoes": [[0, 1]],
}


@dataclass(frozen=True)
class TankInstance:
    capacities: tuple[int, ...]
    cargo_volumes: tuple[int, ...]
    allowed: tuple[tuple[int, ...], ...]
    adjacent_tanks: tuple[tuple[int, int], ...]
    incompatible_cargoes: frozenset[frozenset[int]]


@dataclass(frozen=True)
class TankModel:
    model: cp.Model
    instance: TankInstance
    cargo: list[cp.IntVar]
    amount: list[cp.IntVar]


def parse_instance(data: str | bytes) -> TankInstance:
    raw = json.loads(data)
    try:
        return TankInstance(
            tuple(int(value) for value in raw["tank_capacities"]),
            tuple(int(value) for value in raw["cargo_volumes"]),
            tuple(tuple(int(value) for value in row) for row in raw["allowed"]),
            tuple(
                (int(pair[0]), int(pair[1])) for pair in raw.get("adjacent_tanks", [])
            ),
            frozenset(
                frozenset((int(pair[0]), int(pair[1])))
                for pair in raw.get("incompatible_cargoes", [])
            ),
        )
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError("invalid tank-allocation JSON instance") from error


def load_instance(path: str | Path) -> TankInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def build_model(instance: TankInstance) -> TankModel:
    tank_count = len(instance.capacities)
    cargo_count = len(instance.cargo_volumes)
    if tank_count < 1 or cargo_count < 1 or len(instance.allowed) != tank_count:
        raise ValueError("tanks and cargoes must be non-empty")
    if any(capacity < 1 for capacity in instance.capacities) or any(
        volume < 1 for volume in instance.cargo_volumes
    ):
        raise ValueError("tank capacities and cargo volumes must be positive")
    if any(
        len(row) != cargo_count or any(value not in (0, 1) for value in row)
        for row in instance.allowed
    ):
        raise ValueError("the tank compatibility matrix is invalid")
    if any(
        min(pair) < 0 or max(pair) >= tank_count for pair in instance.adjacent_tanks
    ):
        raise ValueError("an adjacent-tank pair is invalid")
    model = cp.Model()
    cargo = model.int_vars(tank_count, 0, cargo_count, name="cargo")
    amount = [
        model.int_var(0, capacity, name=f"amount_{tank}")
        for tank, capacity in enumerate(instance.capacities)
    ]
    contributions = [[None for _ in range(tank_count)] for _ in range(cargo_count)]
    empty = []
    for tank, capacity in enumerate(instance.capacities):
        allowed_values = [
            0,
            *(
                cargo_index + 1
                for cargo_index, allowed in enumerate(instance.allowed[tank])
                if allowed
            ),
        ]
        model.table([cargo[tank]], [(value,) for value in allowed_values])
        model.table(
            [cargo[tank], amount[tank]],
            [
                (0, 0),
                *(
                    (cargo_index, value)
                    for cargo_index in allowed_values[1:]
                    for value in range(1, capacity + 1)
                ),
            ],
        )
        is_empty = model.bool_var(name=f"empty_{tank}")
        model.table(
            [cargo[tank], is_empty],
            [(value, int(value == 0)) for value in allowed_values],
        )
        empty.append(is_empty)
        for cargo_index in range(cargo_count):
            contribution = model.int_var(
                0, capacity, name=f"tank_{tank}_cargo_{cargo_index}"
            )
            model.table(
                [cargo[tank], amount[tank], contribution],
                [
                    (assigned, value, value if assigned == cargo_index + 1 else 0)
                    for assigned in allowed_values
                    for value in (range(1, capacity + 1) if assigned else (0,))
                ],
            )
            contributions[cargo_index][tank] = contribution
    for cargo_index, volume in enumerate(instance.cargo_volumes):
        model.add(sum(contributions[cargo_index]) == volume)
    forbidden = {
        (left + 1, right + 1)
        for pair in instance.incompatible_cargoes
        for left in pair
        for right in pair
        if left != right
    }
    adjacency_table = [
        (left, right)
        for left in range(cargo_count + 1)
        for right in range(cargo_count + 1)
        if (left, right) not in forbidden
    ]
    for left, right in instance.adjacent_tanks:
        model.table([cargo[left], cargo[right]], adjacency_table)
    model.maximize(
        sum(instance.capacities[tank] * empty[tank] for tank in range(tank_count))
    )
    return TankModel(model, instance, cargo, amount)


def decode(built: TankModel, solution: cp.Solution) -> list[tuple[int | None, int]]:
    if not solution.is_sat():
        raise RuntimeError(f"no solution available, solver status is {solution.status}")
    return [
        (
            None if solution.value(cargo) == 0 else solution.value(cargo) - 1,
            solution.value(amount),
        )
        for cargo, amount in zip(built.cargo, built.amount)
    ]


def validate(
    built: TankModel, allocation: list[tuple[int | None, int]], objective: int | None
) -> None:
    totals = [0] * len(built.instance.cargo_volumes)
    for tank, (cargo, amount) in enumerate(allocation):
        if cargo is None:
            if amount != 0:
                raise AssertionError("an empty tank contains cargo")
            continue
        if (
            not built.instance.allowed[tank][cargo]
            or not 1 <= amount <= built.instance.capacities[tank]
        ):
            raise AssertionError("a tank allocation is incompatible")
        totals[cargo] += amount
    if totals != list(built.instance.cargo_volumes):
        raise AssertionError("cargo volumes are not fully allocated")
    for left, right in built.instance.adjacent_tanks:
        cargoes = frozenset((allocation[left][0], allocation[right][0]))
        if None not in cargoes and cargoes in built.instance.incompatible_cargoes:
            raise AssertionError("incompatible cargoes occupy adjacent tanks")
    unused = sum(
        built.instance.capacities[tank]
        for tank, (cargo, _) in enumerate(allocation)
        if cargo is None
    )
    if objective is not None and unused != objective:
        raise AssertionError("the objective does not match unused tank volume")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON tank-allocation instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob051 tanks={len(instance.capacities)} status={solution.status}")
    if not solution.is_sat():
        return 1
    allocation = decode(built, solution)
    validate(built, allocation, solution.objective)
    print(f"unused_capacity={solution.objective} allocation={allocation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
