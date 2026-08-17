"""CSPLib prob063: winner determination in a combinatorial auction.

Specification: https://www.csplib.org/Problems/prob063/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values

DEFAULT_INSTANCE = {
    "item_count": 4,
    "bids": [
        {"value": 10, "items": [0, 1]},
        {"value": 12, "items": [1, 2]},
        {"value": 7, "items": [3]},
        {"value": 14, "items": [0, 2]},
        {"value": 11, "items": [2, 3]},
    ],
}


@dataclass(frozen=True)
class Bid:
    value: int
    items: frozenset[int]


@dataclass(frozen=True)
class AuctionInstance:
    item_count: int
    bids: tuple[Bid, ...]


@dataclass(frozen=True)
class AuctionModel:
    model: cp.Model
    instance: AuctionInstance
    accepted: list[cp.IntVar]


def parse_instance(data: str | bytes) -> AuctionInstance:
    raw = json.loads(data)
    try:
        bids = tuple(
            Bid(int(bid["value"]), frozenset(int(item) for item in bid["items"]))
            for bid in raw["bids"]
        )
        return AuctionInstance(int(raw["item_count"]), bids)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid auction JSON instance") from error


def load_instance(path: str | Path) -> AuctionInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def build_model(instance: AuctionInstance) -> AuctionModel:
    if instance.item_count < 1:
        raise ValueError("item_count must be positive")
    if any(not bid.items for bid in instance.bids):
        raise ValueError("bids must contain at least one item")
    if any(
        item < 0 or item >= instance.item_count
        for bid in instance.bids
        for item in bid.items
    ):
        raise ValueError("a bid contains an invalid item")

    model = cp.Model()
    accepted = [
        model.bool_var(name=f"accepted_{index}") for index in range(len(instance.bids))
    ]
    for item in range(instance.item_count):
        containing = [
            accepted[index]
            for index, bid in enumerate(instance.bids)
            if item in bid.items
        ]
        if containing:
            model.add(sum(containing) <= 1)
    if accepted:
        model.maximize(
            sum(bid.value * accepted[index] for index, bid in enumerate(instance.bids))
        )
    return AuctionModel(model, instance, accepted)


def decode(built: AuctionModel, solution: cp.Solution) -> list[int]:
    membership = values(solution, built.accepted)
    return [index for index, accepted in enumerate(membership) if accepted]


def validate(built: AuctionModel, accepted: list[int], objective: int | None) -> None:
    used: set[int] = set()
    for index in accepted:
        if index < 0 or index >= len(built.instance.bids):
            raise AssertionError("an accepted bid index is invalid")
        if used.intersection(built.instance.bids[index].items):
            raise AssertionError("accepted bids overlap")
        used.update(built.instance.bids[index].items)
    if sum(built.instance.bids[index].value for index in accepted) != objective:
        raise AssertionError("the objective does not match accepted bid value")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON combinatorial auction instance")
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
        f"prob063 items={instance.item_count} bids={len(instance.bids)} status={solution.status}"
    )
    if not solution.is_sat():
        return 1
    accepted = decode(built, solution)
    validate(built, accepted, solution.objective)
    print(f"value={solution.objective} accepted={accepted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
