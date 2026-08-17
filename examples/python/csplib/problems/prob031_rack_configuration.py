"""CSPLib prob031: rack configuration.

Specification: https://www.csplib.org/Problems/prob031/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values

DEFAULT_INSTANCE = {
    "card_powers": [4, 3, 3, 2],
    "rack_models": [
        {"power": 8, "connectors": 3, "price": 7},
        {"power": 12, "connectors": 4, "price": 10},
    ],
    "max_racks": 4,
}


@dataclass(frozen=True)
class RackType:
    power: int
    connectors: int
    price: int


@dataclass(frozen=True)
class RackInstance:
    card_powers: tuple[int, ...]
    rack_models: tuple[RackType, ...]
    max_racks: int


@dataclass(frozen=True)
class RackModel:
    model: cp.Model
    instance: RackInstance
    card_rack: list[cp.IntVar]
    rack_type: list[cp.IntVar]


def parse_instance(data: str | bytes) -> RackInstance:
    raw = json.loads(data)
    try:
        rack_models = tuple(
            RackType(int(item["power"]), int(item["connectors"]), int(item["price"]))
            for item in raw["rack_models"]
        )
        return RackInstance(
            tuple(int(value) for value in raw["card_powers"]),
            rack_models,
            int(raw.get("max_racks", len(raw["card_powers"]))),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid rack-configuration JSON instance") from error


def load_instance(path: str | Path) -> RackInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def build_model(instance: RackInstance) -> RackModel:
    if not instance.card_powers or not instance.rack_models or instance.max_racks < 1:
        raise ValueError("cards, rack models, and max_racks must be non-empty")
    if any(power < 1 for power in instance.card_powers):
        raise ValueError("card powers must be positive")
    if any(
        item.power < 1 or item.connectors < 1 or item.price < 0
        for item in instance.rack_models
    ):
        raise ValueError("a rack model is invalid")

    model = cp.Model()
    card_rack = model.int_vars(
        len(instance.card_powers), 0, instance.max_racks - 1, name="card_rack"
    )
    rack_type = model.int_vars(
        instance.max_racks, 0, len(instance.rack_models), name="rack_type"
    )
    powers = [0, *(item.power for item in instance.rack_models)]
    connectors = [0, *(item.connectors for item in instance.rack_models)]
    prices = [0, *(item.price for item in instance.rack_models)]
    maximum_power = max(powers)
    maximum_connectors = max(connectors)
    maximum_price = max(prices)
    rack_prices = []
    membership = [
        [
            model.bool_var(name=f"card_{card}_rack_{rack}")
            for rack in range(instance.max_racks)
        ]
        for card in range(len(instance.card_powers))
    ]
    for card in range(len(instance.card_powers)):
        for rack in range(instance.max_racks):
            model.table(
                [card_rack[card], membership[card][rack]],
                [
                    (candidate, int(candidate == rack))
                    for candidate in range(instance.max_racks)
                ],
            )
    for rack in range(instance.max_racks):
        power = model.int_var(0, maximum_power, name=f"rack_{rack}_power")
        connector_count = model.int_var(
            0, maximum_connectors, name=f"rack_{rack}_connectors"
        )
        price = model.int_var(0, maximum_price, name=f"rack_{rack}_price")
        used = model.bool_var(name=f"rack_{rack}_used")
        model.element_const(powers, rack_type[rack], power)
        model.element_const(connectors, rack_type[rack], connector_count)
        model.element_const(prices, rack_type[rack], price)
        model.table(
            [rack_type[rack], used],
            [
                (candidate, int(candidate > 0))
                for candidate in range(len(instance.rack_models) + 1)
            ],
        )
        assigned_count = sum(
            membership[card][rack] for card in range(len(instance.card_powers))
        )
        assigned_power = sum(
            instance.card_powers[card] * membership[card][rack]
            for card in range(len(instance.card_powers))
        )
        model.add(assigned_count <= connector_count)
        model.add(assigned_power <= power)
        model.add(assigned_count >= used)
        rack_prices.append(price)
    for rack in range(instance.max_racks - 1):
        model.add(rack_type[rack] >= rack_type[rack + 1])
    model.minimize(sum(rack_prices))
    return RackModel(model, instance, card_rack, rack_type)


def decode(built: RackModel, solution: cp.Solution) -> tuple[list[int], list[int]]:
    return values(solution, built.card_rack), values(solution, built.rack_type)


def validate(
    built: RackModel,
    card_rack: list[int],
    rack_type: list[int],
    objective: int | None,
) -> None:
    if (
        len(card_rack) != len(built.instance.card_powers)
        or len(rack_type) != built.instance.max_racks
    ):
        raise AssertionError("decoded rack dimensions are invalid")
    total_price = 0
    for rack, encoded_type in enumerate(rack_type):
        cards = [card for card, assigned in enumerate(card_rack) if assigned == rack]
        if encoded_type == 0:
            if cards:
                raise AssertionError("a card is assigned to an unused rack")
            continue
        selected_type = built.instance.rack_models[encoded_type - 1]
        if len(cards) > selected_type.connectors:
            raise AssertionError("rack connector capacity is exceeded")
        if (
            sum(built.instance.card_powers[card] for card in cards)
            > selected_type.power
        ):
            raise AssertionError("rack power capacity is exceeded")
        total_price += selected_type.price
    if objective is not None and total_price != objective:
        raise AssertionError("the objective does not match rack prices")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON rack-configuration instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob031 cards={len(instance.card_powers)} status={solution.status}")
    if not solution.is_sat():
        return 1
    card_rack, rack_type = decode(built, solution)
    validate(built, card_rack, rack_type, solution.objective)
    print(f"cost={solution.objective} rack_type={rack_type} card_rack={card_rack}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
