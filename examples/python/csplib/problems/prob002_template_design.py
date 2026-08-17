"""CSPLib prob002: template design.

Specification: https://www.csplib.org/Problems/prob002/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args

DEFAULT_INSTANCE = {"slots": 3, "templates": 2, "demands": [4, 5, 6]}


@dataclass(frozen=True)
class TemplateInstance:
    slots: int
    templates: int
    demands: tuple[int, ...]


@dataclass(frozen=True)
class TemplateModel:
    model: cp.Model
    instance: TemplateInstance
    layout: list[list[cp.IntVar]]
    pressings: list[cp.IntVar]
    production: list[list[cp.IntVar]]


def parse_instance(data: str | bytes) -> TemplateInstance:
    raw = json.loads(data)
    try:
        return TemplateInstance(
            int(raw["slots"]),
            int(raw["templates"]),
            tuple(int(value) for value in raw["demands"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid template-design JSON instance") from error


def load_instance(path: str | Path) -> TemplateInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def build_model(instance: TemplateInstance) -> TemplateModel:
    if instance.slots < 1 or instance.templates < 1 or not instance.demands:
        raise ValueError("slots, templates, and demands must be positive")
    if any(demand < 0 for demand in instance.demands):
        raise ValueError("demands must be non-negative")
    maximum_pressings = max(instance.demands, default=0)
    model = cp.Model()
    layout = [
        model.int_vars(
            len(instance.demands), 0, instance.slots, name=f"layout_{template}"
        )
        for template in range(instance.templates)
    ]
    pressings = model.int_vars(
        instance.templates,
        0,
        maximum_pressings,
        name="pressings",
    )
    production = [
        [
            model.int_var(
                0,
                instance.slots * maximum_pressings,
                name=f"production_{template}_{variation}",
            )
            for variation in range(len(instance.demands))
        ]
        for template in range(instance.templates)
    ]
    products = [
        (slots, count, slots * count)
        for slots in range(instance.slots + 1)
        for count in range(maximum_pressings + 1)
    ]
    for template in range(instance.templates):
        model.add(sum(layout[template]) == instance.slots)
        for variation in range(len(instance.demands)):
            model.table(
                [
                    layout[template][variation],
                    pressings[template],
                    production[template][variation],
                ],
                products,
            )
    for variation, demand in enumerate(instance.demands):
        model.add(
            sum(
                production[template][variation]
                for template in range(instance.templates)
            )
            >= demand
        )
    for template in range(instance.templates - 1):
        model.add(pressings[template] >= pressings[template + 1])
    model.minimize(sum(pressings))
    return TemplateModel(model, instance, layout, pressings, production)


def decode(
    built: TemplateModel, solution: cp.Solution
) -> tuple[list[list[int]], list[int]]:
    if not solution.is_sat():
        raise RuntimeError(f"no solution available, solver status is {solution.status}")
    layout = [[solution.value(variable) for variable in row] for row in built.layout]
    pressings = [solution.value(variable) for variable in built.pressings]
    return layout, pressings


def validate(
    built: TemplateModel,
    layout: list[list[int]],
    pressings: list[int],
    objective: int | None,
) -> None:
    if (
        len(layout) != built.instance.templates
        or len(pressings) != built.instance.templates
    ):
        raise AssertionError("the decoded number of templates is invalid")
    if any(sum(row) != built.instance.slots for row in layout):
        raise AssertionError("a template does not use every slot")
    for variation, demand in enumerate(built.instance.demands):
        produced = sum(
            layout[index][variation] * pressings[index] for index in range(len(layout))
        )
        if produced < demand:
            raise AssertionError("a variation demand is not met")
    if objective is not None and sum(pressings) != objective:
        raise AssertionError("the objective does not match total pressings")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON template-design instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob002 templates={instance.templates} status={solution.status}")
    if not solution.is_sat():
        return 1
    layout, pressings = decode(built, solution)
    validate(built, layout, pressings, solution.objective)
    print(f"pressings={pressings} total={solution.objective} layout={layout}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
