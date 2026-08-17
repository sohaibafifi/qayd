"""CSPLib prob116: Vellino's coloured-bin problem.

Specification: https://www.csplib.org/Problems/prob116/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args

MATERIALS = ("glass", "plastic", "steel", "wood", "copper")
COLORS = ("unused", "red", "blue", "green")
DEFAULT_INSTANCE = {"capacities": [3, 3, 3], "demands": [1, 1, 1, 1, 1]}


@dataclass(frozen=True)
class VellinoInstance:
    capacities: tuple[int, int, int]
    demands: tuple[int, int, int, int, int]


@dataclass(frozen=True)
class VellinoModel:
    model: cp.Model
    instance: VellinoInstance
    colors: list[cp.IntVar]
    contents: list[list[cp.IntVar]]


def parse_instance(data: str | bytes) -> VellinoInstance:
    raw = json.loads(data)
    try:
        capacities = tuple(int(value) for value in raw["capacities"])
        demands = tuple(int(value) for value in raw["demands"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid Vellino JSON instance") from error
    if len(capacities) != 3 or len(demands) != 5:
        raise ValueError("Vellino requires three bin colours and five materials")
    return VellinoInstance(capacities, demands)


def load_instance(path: str | Path) -> VellinoInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def build_model(instance: VellinoInstance) -> VellinoModel:
    if any(capacity < 1 for capacity in instance.capacities) or any(
        demand < 0 for demand in instance.demands
    ):
        raise ValueError("capacities must be positive and demands non-negative")
    bin_count = max(1, sum(instance.demands))
    maximum_capacity = max(instance.capacities)
    model = cp.Model()
    colors = model.int_vars(bin_count, 0, 3, name="color")
    contents = [
        [
            model.int_var(
                0, min(maximum_capacity, demand), name=f"bin_{bin_index}_{material}"
            )
            for material, demand in enumerate(instance.demands)
        ]
        for bin_index in range(bin_count)
    ]
    used = []
    for bin_index in range(bin_count):
        capacity = model.int_var(0, maximum_capacity, name=f"capacity_{bin_index}")
        is_used = model.bool_var(name=f"used_{bin_index}")
        model.element_const([0, *instance.capacities], colors[bin_index], capacity)
        model.table(
            [colors[bin_index], is_used],
            [(color, int(color != 0)) for color in range(4)],
        )
        model.add(sum(contents[bin_index]) <= capacity)
        model.add((colors[bin_index] == 0).iff(sum(contents[bin_index]) == 0))

        glass, plastic, steel, wood, copper = contents[bin_index]
        model.add((colors[bin_index] == 1).implies((plastic == 0) & (steel == 0)))
        model.add((colors[bin_index] == 2).implies((wood == 0) & (plastic == 0)))
        model.add((colors[bin_index] == 3).implies((steel == 0) & (glass == 0)))
        model.add((colors[bin_index] == 1).implies(wood <= 1))
        model.add((colors[bin_index] == 3).implies(wood <= 2))
        model.add((wood > 0).implies(plastic > 0))
        model.add((glass > 0).implies(copper == 0))
        model.add((copper > 0).implies(plastic == 0))
        used.append(is_used)
    for material, demand in enumerate(instance.demands):
        model.add(
            sum(contents[bin_index][material] for bin_index in range(bin_count))
            == demand
        )
    for bin_index in range(bin_count - 1):
        model.add(colors[bin_index] >= colors[bin_index + 1])
    model.minimize(sum(used))
    return VellinoModel(model, instance, colors, contents)


def decode(built: VellinoModel, solution: cp.Solution) -> list[tuple[str, list[int]]]:
    if not solution.is_sat():
        raise RuntimeError(f"no solution available, solver status is {solution.status}")
    bins = []
    for color, row in zip(built.colors, built.contents):
        color_value = solution.value(color)
        if color_value:
            bins.append(
                (COLORS[color_value], [solution.value(variable) for variable in row])
            )
    return bins


def validate(
    built: VellinoModel, bins: list[tuple[str, list[int]]], objective: int | None
) -> None:
    totals = [0] * 5
    for color_name, contents in bins:
        color = COLORS.index(color_name)
        if color == 0 or len(contents) != 5:
            raise AssertionError("a decoded bin is invalid")
        if sum(contents) > built.instance.capacities[color - 1]:
            raise AssertionError("a bin exceeds its capacity")
        glass, plastic, steel, wood, copper = contents
        if color == 1 and (plastic or steel or wood > 1):
            raise AssertionError("a red-bin rule is violated")
        if color == 2 and (wood or plastic):
            raise AssertionError("a blue-bin rule is violated")
        if color == 3 and (steel or glass or wood > 2):
            raise AssertionError("a green-bin rule is violated")
        if wood and not plastic:
            raise AssertionError("wood is stored without plastic")
        if glass and copper:
            raise AssertionError("glass is stored with copper")
        if copper and plastic:
            raise AssertionError("copper is stored with plastic")
        totals = [total + value for total, value in zip(totals, contents)]
    if totals != list(built.instance.demands):
        raise AssertionError("material demands are not met")
    if objective is not None and len(bins) != objective:
        raise AssertionError("the objective does not match the used-bin count")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON Vellino instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob116 status={solution.status}")
    if not solution.is_sat():
        return 1
    bins = decode(built, solution)
    validate(built, bins, solution.objective)
    print(f"used_bins={solution.objective} bins={bins}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
