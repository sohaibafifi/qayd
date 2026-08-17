"""CSPLib prob081: Black Hole solitaire.

Specification: https://www.csplib.org/Problems/prob081/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values

DEFAULT_INSTANCE = {
    "rank_count": 4,
    "initial_rank": 0,
    "card_ranks": [1, 2, 3, 0, 1, 2, 3],
    "fans": [[0, 3], [1, 4], [2, 5], [6]],
}


@dataclass(frozen=True)
class BlackHoleInstance:
    rank_count: int
    initial_rank: int
    card_ranks: tuple[int, ...]
    fans: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class BlackHoleModel:
    model: cp.Model
    instance: BlackHoleInstance
    play_order: list[cp.IntVar]
    positions: list[cp.IntVar]


def parse_instance(data: str | bytes) -> BlackHoleInstance:
    raw = json.loads(data)
    try:
        return BlackHoleInstance(
            int(raw["rank_count"]),
            int(raw["initial_rank"]),
            tuple(int(value) for value in raw["card_ranks"]),
            tuple(tuple(int(card) for card in fan) for fan in raw["fans"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid Black Hole JSON instance") from error


def load_instance(path: str | Path) -> BlackHoleInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def _adjacent(left: int, right: int, rank_count: int) -> bool:
    return (left - right) % rank_count in (1, rank_count - 1)


def build_model(instance: BlackHoleInstance) -> BlackHoleModel:
    card_count = len(instance.card_ranks)
    if instance.rank_count < 3 or card_count < 1:
        raise ValueError("rank_count must be at least three and cards cannot be empty")
    if instance.initial_rank < 0 or instance.initial_rank >= instance.rank_count:
        raise ValueError("initial_rank is invalid")
    if any(rank < 0 or rank >= instance.rank_count for rank in instance.card_ranks):
        raise ValueError("a card rank is invalid")
    fan_cards = [card for fan in instance.fans for card in fan]
    if sorted(fan_cards) != list(range(card_count)):
        raise ValueError("fans must contain every remaining card exactly once")

    model = cp.Model()
    play_order = model.int_vars(card_count, 0, card_count - 1, name="play")
    positions = model.int_vars(card_count, 0, card_count - 1, name="position")
    model.all_different(play_order)
    model.channel(play_order, positions)
    for fan in instance.fans:
        for upper, lower in zip(fan, fan[1:]):
            model.add(positions[upper] < positions[lower])

    first_cards = [
        (card,)
        for card, rank in enumerate(instance.card_ranks)
        if _adjacent(instance.initial_rank, rank, instance.rank_count)
    ]
    if first_cards:
        model.table([play_order[0]], first_cards)
    else:
        model.add(play_order[0] != play_order[0])
    adjacent_pairs = [
        (left, right)
        for left, left_rank in enumerate(instance.card_ranks)
        for right, right_rank in enumerate(instance.card_ranks)
        if left != right and _adjacent(left_rank, right_rank, instance.rank_count)
    ]
    for index in range(card_count - 1):
        model.table([play_order[index], play_order[index + 1]], adjacent_pairs)
    return BlackHoleModel(model, instance, play_order, positions)


def decode(built: BlackHoleModel, solution: cp.Solution) -> list[int]:
    return values(solution, built.play_order)


def validate(built: BlackHoleModel, play_order: list[int]) -> None:
    if sorted(play_order) != list(range(len(built.instance.card_ranks))):
        raise AssertionError("every card must be played exactly once")
    ranks = [
        built.instance.initial_rank,
        *(built.instance.card_ranks[card] for card in play_order),
    ]
    if any(
        not _adjacent(left, right, built.instance.rank_count)
        for left, right in zip(ranks, ranks[1:])
    ):
        raise AssertionError("two consecutive cards are not adjacent in rank")
    positions = {card: position for position, card in enumerate(play_order)}
    for fan in built.instance.fans:
        if any(
            positions[upper] >= positions[lower] for upper, lower in zip(fan, fan[1:])
        ):
            raise AssertionError("a lower fan card was played before the card above it")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON Black Hole deal")
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
        f"prob081 cards={len(instance.card_ranks)} fans={len(instance.fans)} status={solution.status}"
    )
    if not solution.is_sat():
        return 1
    play_order = decode(built, solution)
    validate(built, play_order)
    print(f"play_order={play_order}")
    print(
        f"ranks={[instance.initial_rank, *(instance.card_ranks[card] for card in play_order)]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
