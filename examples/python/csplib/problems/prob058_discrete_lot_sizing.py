"""CSPLib prob058: discrete lot sizing and scheduling.

Specification: https://www.csplib.org/Problems/prob058/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values

DEFAULT_INSTANCE = {
    "holding_cost": 2,
    "demand": [[0, 1, 0, 0, 1], [1, 0, 0, 0, 1]],
    "changeover": [[0, 5], [3, 0]],
}


@dataclass(frozen=True)
class LotSizingInstance:
    holding_cost: int
    demand: tuple[tuple[int, ...], ...]
    changeover: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class LotSizingModel:
    model: cp.Model
    instance: LotSizingInstance
    plan: list[cp.IntVar]


def parse_instance(data: str | bytes) -> LotSizingInstance:
    raw = json.loads(data)
    try:
        return LotSizingInstance(
            int(raw["holding_cost"]),
            tuple(tuple(int(value) for value in row) for row in raw["demand"]),
            tuple(tuple(int(value) for value in row) for row in raw["changeover"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid discrete-lot-sizing JSON instance") from error


def load_instance(path: str | Path) -> LotSizingInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def build_model(instance: LotSizingInstance) -> LotSizingModel:
    item_count = len(instance.demand)
    if item_count < 1 or not instance.demand[0] or instance.holding_cost < 0:
        raise ValueError("items, periods, and holding cost are invalid")
    periods = len(instance.demand[0])
    if any(
        len(row) != periods or any(value < 0 for value in row)
        for row in instance.demand
    ):
        raise ValueError("demand rows are invalid")
    if len(instance.changeover) != item_count or any(
        len(row) != item_count for row in instance.changeover
    ):
        raise ValueError("the changeover matrix dimensions are invalid")
    if any(value < 0 for row in instance.changeover for value in row):
        raise ValueError("changeover costs must be non-negative")
    if sum(sum(row) for row in instance.demand) > periods:
        raise ValueError("total unit demand exceeds the production horizon")
    model = cp.Model()
    plan = model.int_vars(periods, 0, item_count, name="production")
    produced = [
        [
            model.bool_var(name=f"period_{period}_item_{item}")
            for period in range(periods)
        ]
        for item in range(item_count)
    ]
    for item in range(item_count):
        for period in range(periods):
            model.table(
                [plan[period], produced[item][period]],
                [(value, int(value == item + 1)) for value in range(item_count + 1)],
            )
        model.add(sum(produced[item]) == sum(instance.demand[item]))
        for period in range(periods):
            model.add(
                sum(produced[item][: period + 1])
                >= sum(instance.demand[item][: period + 1])
            )

    holding_terms = []
    for item in range(item_count):
        for period in range(periods):
            inventory = sum(produced[item][: period + 1]) - sum(
                instance.demand[item][: period + 1]
            )
            holding_terms.append(instance.holding_cost * inventory)
    maximum_changeover = max(value for row in instance.changeover for value in row)
    setup_state = model.int_vars(periods + 1, 0, item_count, name="setup")
    model.add(setup_state[0] == 0)
    changeover_terms = []
    transition_table = []
    for previous in range(item_count + 1):
        for production in range(item_count + 1):
            following = previous if production == 0 else production
            cost = (
                0
                if previous == 0 or production == 0
                else instance.changeover[previous - 1][production - 1]
            )
            transition_table.append((previous, production, following, cost))
    for period in range(periods):
        cost = model.int_var(0, maximum_changeover, name=f"changeover_{period}")
        model.table(
            [setup_state[period], plan[period], setup_state[period + 1], cost],
            transition_table,
        )
        changeover_terms.append(cost)
    model.minimize(sum(holding_terms) + sum(changeover_terms))
    return LotSizingModel(model, instance, plan)


def decode(built: LotSizingModel, solution: cp.Solution) -> list[int]:
    return values(solution, built.plan)


def schedule_cost(instance: LotSizingInstance, plan: list[int]) -> int:
    item_count = len(instance.demand)
    inventory = [0] * item_count
    holding = 0
    changeover = 0
    setup = 0
    for period, production in enumerate(plan):
        if production:
            inventory[production - 1] += 1
            if setup:
                changeover += instance.changeover[setup - 1][production - 1]
            setup = production
        for item in range(item_count):
            inventory[item] -= instance.demand[item][period]
            if inventory[item] < 0:
                raise AssertionError("an order is produced after its due date")
        holding += instance.holding_cost * sum(inventory)
    if any(inventory):
        raise AssertionError("production does not exactly match total demand")
    return holding + changeover


def validate(built: LotSizingModel, plan: list[int], objective: int | None) -> None:
    if len(plan) != len(built.instance.demand[0]):
        raise AssertionError("the production plan has an invalid horizon")
    cost = schedule_cost(built.instance, plan)
    if objective is not None and cost != objective:
        raise AssertionError("the objective does not match lot-sizing cost")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON discrete-lot-sizing instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob058 periods={len(instance.demand[0])} status={solution.status}")
    if not solution.is_sat():
        return 1
    plan = decode(built, solution)
    validate(built, plan, solution.objective)
    print(f"cost={solution.objective} plan={plan}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
