"""CSPLib prob039: rehearsal and talent scheduling.

Specification: https://www.csplib.org/Problems/prob039/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values

DEFAULT_INSTANCE = {
    "durations": [2, 4, 1, 3],
    "playing": [[1, 1, 0, 1], [1, 0, 1, 0], [0, 1, 1, 1]],
    "player_costs": [1, 1, 1],
}


@dataclass(frozen=True)
class RehearsalInstance:
    durations: tuple[int, ...]
    playing: tuple[tuple[int, ...], ...]
    player_costs: tuple[int, ...]


@dataclass(frozen=True)
class RehearsalModel:
    model: cp.Model
    instance: RehearsalInstance
    order: list[cp.IntVar]


def parse_instance(data: str | bytes) -> RehearsalInstance:
    raw = json.loads(data)
    try:
        playing = tuple(tuple(int(value) for value in row) for row in raw["playing"])
        costs = tuple(
            int(value) for value in raw.get("player_costs", [1] * len(playing))
        )
        return RehearsalInstance(
            tuple(int(value) for value in raw["durations"]),
            playing,
            costs,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid rehearsal JSON instance") from error


def load_instance(path: str | Path) -> RehearsalInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def build_model(instance: RehearsalInstance) -> RehearsalModel:
    piece_count = len(instance.durations)
    if (
        piece_count < 1
        or not instance.playing
        or len(instance.playing) != len(instance.player_costs)
    ):
        raise ValueError("pieces, players, and player costs must be non-empty")
    if any(duration < 1 for duration in instance.durations) or any(
        cost < 0 for cost in instance.player_costs
    ):
        raise ValueError("durations must be positive and costs non-negative")
    if any(
        len(row) != piece_count or any(value not in (0, 1) for value in row)
        for row in instance.playing
    ):
        raise ValueError("the playing matrix is invalid")
    if any(not any(row) for row in instance.playing):
        raise ValueError("every player must participate in at least one piece")

    model = cp.Model()
    order = model.int_vars(piece_count, 0, piece_count - 1, name="piece")
    model.all_different(order)
    durations_at_slot = model.int_vars(
        piece_count, min(instance.durations), max(instance.durations), name="duration"
    )
    for slot in range(piece_count):
        model.element_const(
            list(instance.durations), order[slot], durations_at_slot[slot]
        )

    waiting_costs = []
    for player, playing in enumerate(instance.playing):
        arrival = model.int_var(0, piece_count - 1, name=f"arrival_{player}")
        leaving = model.int_var(0, piece_count - 1, name=f"leaving_{player}")
        model.add(arrival <= leaving)
        for slot in range(piece_count):
            plays = model.bool_var(name=f"plays_{player}_{slot}")
            waits = model.bool_var(name=f"waits_{player}_{slot}")
            model.element_const(list(playing), order[slot], plays)
            attendance_table = [
                (play, start, end, int(not play and start <= slot <= end))
                for play in (0, 1)
                for start in range(piece_count)
                for end in range(piece_count)
                if start <= end and (not play or start <= slot <= end)
            ]
            model.table([plays, arrival, leaving, waits], attendance_table)
            cost = model.int_var(
                0,
                max(instance.durations) * instance.player_costs[player],
                name=f"wait_cost_{player}_{slot}",
            )
            model.table(
                [durations_at_slot[slot], waits, cost],
                [
                    (
                        duration,
                        waiting,
                        duration * waiting * instance.player_costs[player],
                    )
                    for duration in set(instance.durations)
                    for waiting in (0, 1)
                ],
            )
            waiting_costs.append(cost)
    model.minimize(sum(waiting_costs))
    return RehearsalModel(model, instance, order)


def decode(built: RehearsalModel, solution: cp.Solution) -> list[int]:
    return values(solution, built.order)


def waiting_cost(instance: RehearsalInstance, order: list[int]) -> int:
    total = 0
    for player, playing in enumerate(instance.playing):
        required_slots = [slot for slot, piece in enumerate(order) if playing[piece]]
        first, last = min(required_slots), max(required_slots)
        total += instance.player_costs[player] * sum(
            instance.durations[order[slot]]
            for slot in range(first, last + 1)
            if not playing[order[slot]]
        )
    return total


def validate(built: RehearsalModel, order: list[int], objective: int | None) -> None:
    if sorted(order) != list(range(len(built.instance.durations))):
        raise AssertionError("the rehearsal order is not a permutation")
    if objective is not None and waiting_cost(built.instance, order) != objective:
        raise AssertionError("the objective does not match the waiting cost")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON rehearsal instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob039 pieces={len(instance.durations)} status={solution.status}")
    if not solution.is_sat():
        return 1
    order = decode(built, solution)
    validate(built, order, solution.objective)
    print(f"waiting_cost={solution.objective} order={order}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
