"""CSPLib prob133: zero-one knapsack.

Specification: https://www.csplib.org/Problems/prob133/
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values

SAMPLE_ESSENCE = """\
language Essence 1.3
letting items be new type enum {a,b,c,d,e}
letting capacity be 100
letting gain be function
    ( a --> 10
    , b --> 20
    , c --> 40
    , d --> 40
    , e --> 50
    )
letting weight be function
    ( a --> 15
    , b --> 25
    , c --> 45
    , d --> 50
    , e --> 60
    )
"""


@dataclass(frozen=True)
class KnapsackInstance:
    names: tuple[str, ...]
    weights: tuple[int, ...]
    gains: tuple[int, ...]
    capacity: int


@dataclass(frozen=True)
class KnapsackModel:
    model: cp.Model
    instance: KnapsackInstance
    selected: list[cp.IntVar]


def parse_essence_parameter(text: str) -> KnapsackInstance:
    capacity_match = re.search(
        r"\bletting\s+capacity\s+be\s+(-?\d+)", text, flags=re.IGNORECASE
    )
    items_match = re.search(
        r"\bletting\s+items\b.*?\{([^}]+)\}", text, flags=re.IGNORECASE | re.DOTALL
    )
    if capacity_match is None or items_match is None:
        raise ValueError("missing items or capacity declaration")
    names = tuple(
        item.strip() for item in items_match.group(1).split(",") if item.strip()
    )

    functions: dict[str, dict[str, int]] = {}
    pattern = re.compile(
        r"\bletting\s+(gain|weight)\s+be\s+function\s*(.*?)"
        r"(?=\n\s*letting\b|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(text):
        function_name = match.group(1).lower()
        pairs = re.findall(r"([A-Za-z_]\w*)\s*-->\s*(-?\d+)", match.group(2))
        functions[function_name] = {name: int(value) for name, value in pairs}
    if set(functions) != {"gain", "weight"}:
        raise ValueError("missing gain or weight function")
    if any(set(functions[name]) != set(names) for name in ("gain", "weight")):
        raise ValueError("gain and weight functions must define every item")
    return KnapsackInstance(
        names,
        tuple(functions["weight"][name] for name in names),
        tuple(functions["gain"][name] for name in names),
        int(capacity_match.group(1)),
    )


def load_essence_parameter(path: str | Path) -> KnapsackInstance:
    return parse_essence_parameter(Path(path).read_text(encoding="utf-8"))


def build_model(instance: KnapsackInstance) -> KnapsackModel:
    item_count = len(instance.names)
    if len(set(instance.names)) != item_count:
        raise ValueError("item names must be unique")
    if len(instance.weights) != item_count or len(instance.gains) != item_count:
        raise ValueError("names, weights, and gains must have equal length")
    if instance.capacity < 0 or any(weight <= 0 for weight in instance.weights):
        raise ValueError("capacity must be non-negative and weights must be positive")

    model = cp.Model()
    selected = [model.bool_var(name=f"selected_{name}") for name in instance.names]
    model.add(
        sum(weight * variable for weight, variable in zip(instance.weights, selected))
        <= instance.capacity
    )
    if selected:
        model.maximize(
            sum(gain * variable for gain, variable in zip(instance.gains, selected))
        )
    return KnapsackModel(model, instance, selected)


def decode(built: KnapsackModel, solution: cp.Solution) -> list[int]:
    membership = values(solution, built.selected)
    return [index for index, selected in enumerate(membership) if selected]


def validate(built: KnapsackModel, selected: list[int], objective: int | None) -> None:
    invalid_index = any(
        index < 0 or index >= len(built.instance.names) for index in selected
    )
    if len(set(selected)) != len(selected) or invalid_index:
        raise AssertionError("selected item indices must be unique and valid")
    weight = sum(built.instance.weights[index] for index in selected)
    gain = sum(built.instance.gains[index] for index in selected)
    if weight > built.instance.capacity:
        raise AssertionError("the selected items exceed capacity")
    if objective != gain:
        raise AssertionError("the reported objective does not equal the selected gain")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="CSPLib Essence parameter file")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)

    instance = (
        load_essence_parameter(args.path)
        if args.path
        else parse_essence_parameter(SAMPLE_ESSENCE)
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(
        f"prob133 items={len(instance.names)} capacity={instance.capacity} status={solution.status}"
    )
    if not solution.is_sat():
        return 1
    selected = decode(built, solution)
    validate(built, selected, solution.objective)
    print(
        f"gain={solution.objective} items={[instance.names[index] for index in selected]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
