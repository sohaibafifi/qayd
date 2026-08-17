"""CSPLib prob131: production line sequencing.

Specification: https://www.csplib.org/Problems/prob131/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values

DEFAULT_INSTANCE = {
    "capacities": [[2, 1], [1, 2]],
    "batches": [
        {"lines": [0], "line_on": 0, "line_off": 1, "attributes": {"model": 0}},
        {"lines": [0, 1], "line_on": 0, "line_off": 0, "attributes": {"model": 1}},
        {"lines": [1], "line_on": 0, "line_off": 1, "attributes": {"model": 0}},
        {"lines": [0, 1], "line_on": 1, "line_off": 1, "attributes": {"model": 1}},
        {"lines": [0], "line_on": 0, "line_off": 0, "attributes": {"model": 0}},
        {"lines": [1], "line_on": 1, "line_off": 1, "attributes": {"model": 1}},
    ],
    "distributions": [
        {"attribute": "model", "value": 0, "day": 0, "minimum": 2, "maximum": 2},
        {"attribute": "model", "value": 1, "day": 1, "minimum": 2, "maximum": 2},
    ],
    "batting_orders": [{"attribute": "model", "order": [0, 1]}],
}


@dataclass(frozen=True)
class Batch:
    lines: frozenset[int]
    line_on: int
    line_off: int
    attributes: tuple[tuple[str, int], ...]

    def attribute(self, name: str) -> int:
        return dict(self.attributes)[name]


@dataclass(frozen=True)
class Distribution:
    attribute: str
    value: int
    day: int
    minimum: int
    maximum: int


@dataclass(frozen=True)
class ProductionInstance:
    capacities: tuple[tuple[int, ...], ...]
    batches: tuple[Batch, ...]
    distributions: tuple[Distribution, ...]
    batting_orders: tuple[tuple[str, tuple[int, ...]], ...]


@dataclass(frozen=True)
class ProductionModel:
    model: cp.Model
    instance: ProductionInstance
    slots: tuple[tuple[int, int], ...]
    sequence: list[cp.IntVar]


def parse_instance(data: str | bytes) -> ProductionInstance:
    raw = json.loads(data)
    try:
        batches = tuple(
            Batch(
                frozenset(int(value) for value in item["lines"]),
                int(item["line_on"]),
                int(item["line_off"]),
                tuple(
                    sorted(
                        (str(key), int(value))
                        for key, value in item["attributes"].items()
                    )
                ),
            )
            for item in raw["batches"]
        )
        distributions = tuple(
            Distribution(
                str(item["attribute"]),
                int(item["value"]),
                int(item["day"]),
                int(item["minimum"]),
                int(item["maximum"]),
            )
            for item in raw.get("distributions", [])
        )
        batting = tuple(
            (str(item["attribute"]), tuple(int(value) for value in item["order"]))
            for item in raw.get("batting_orders", [])
        )
        return ProductionInstance(
            tuple(tuple(int(value) for value in row) for row in raw["capacities"]),
            batches,
            distributions,
            batting,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid production-line JSON instance") from error


def load_instance(path: str | Path) -> ProductionInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def build_model(instance: ProductionInstance) -> ProductionModel:
    line_count = len(instance.capacities)
    if (
        line_count < 1
        or not instance.batches
        or any(not row for row in instance.capacities)
    ):
        raise ValueError("lines, capacities, and batches must be non-empty")
    day_count = len(instance.capacities[0])
    if any(
        len(row) != day_count or any(value < 0 for value in row)
        for row in instance.capacities
    ):
        raise ValueError(
            "capacity rows must have equal lengths and non-negative values"
        )
    slots = tuple(
        (line, day)
        for day in range(day_count)
        for line in range(line_count)
        for _ in range(instance.capacities[line][day])
    )
    if len(slots) != len(instance.batches):
        raise ValueError("total production capacity must equal the number of batches")
    if any(
        not batch.lines or batch.line_on < 0 or batch.line_off < batch.line_on
        for batch in instance.batches
    ):
        raise ValueError("a batch is invalid")
    model = cp.Model()
    sequence = model.int_vars(len(slots), 0, len(instance.batches) - 1, name="batch")
    model.all_different(sequence)
    for slot, (line, day) in enumerate(slots):
        allowed = [
            batch
            for batch, item in enumerate(instance.batches)
            if line in item.lines and item.line_on <= day <= item.line_off
        ]
        model.table([sequence[slot]], [(batch,) for batch in allowed])
    for index, distribution in enumerate(instance.distributions):
        if (
            not 0 <= distribution.day < day_count
            or not 0 <= distribution.minimum <= distribution.maximum
        ):
            raise ValueError("a distribution constraint is invalid")
        matching = []
        for slot, (_, day) in enumerate(slots):
            if day != distribution.day:
                continue
            selected = model.bool_var(name=f"distribution_{index}_{slot}")
            model.table(
                [sequence[slot], selected],
                [
                    (
                        batch,
                        int(
                            instance.batches[batch].attribute(distribution.attribute)
                            == distribution.value
                        ),
                    )
                    for batch in range(len(instance.batches))
                ],
            )
            matching.append(selected)
        model.add(sum(matching) >= distribution.minimum)
        model.add(sum(matching) <= distribution.maximum)
    for attribute, order in instance.batting_orders:
        rank = {value: index for index, value in enumerate(order)}
        allowed_pairs = [
            (left, right)
            for left in range(len(instance.batches))
            for right in range(len(instance.batches))
            if rank[instance.batches[left].attribute(attribute)]
            <= rank[instance.batches[right].attribute(attribute)]
        ]
        for slot in range(len(slots) - 1):
            if slots[slot] == slots[slot + 1]:
                model.table([sequence[slot], sequence[slot + 1]], allowed_pairs)
    return ProductionModel(model, instance, slots, sequence)


def decode(built: ProductionModel, solution: cp.Solution) -> list[int]:
    return values(solution, built.sequence)


def validate(built: ProductionModel, sequence: list[int]) -> None:
    if sorted(sequence) != list(range(len(built.instance.batches))):
        raise AssertionError("batches are not assigned exactly once")
    for batch, (line, day) in zip(sequence, built.slots):
        item = built.instance.batches[batch]
        if line not in item.lines or not item.line_on <= day <= item.line_off:
            raise AssertionError("a batch violates its line or date restriction")
    for distribution in built.instance.distributions:
        count = sum(
            built.instance.batches[batch].attribute(distribution.attribute)
            == distribution.value
            for batch, (_, day) in zip(sequence, built.slots)
            if day == distribution.day
        )
        if not distribution.minimum <= count <= distribution.maximum:
            raise AssertionError("an even-distribution constraint is violated")
    for attribute, order in built.instance.batting_orders:
        rank = {value: index for index, value in enumerate(order)}
        for slot in range(len(sequence) - 1):
            if built.slots[slot] == built.slots[slot + 1]:
                left = built.instance.batches[sequence[slot]].attribute(attribute)
                right = built.instance.batches[sequence[slot + 1]].attribute(attribute)
                if rank[left] > rank[right]:
                    raise AssertionError("a batting-order constraint is violated")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON production-line instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob131 batches={len(instance.batches)} status={solution.status}")
    if not solution.is_sat():
        return 1
    sequence = decode(built, solution)
    validate(built, sequence)
    print(f"slots={built.slots} sequence={sequence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
