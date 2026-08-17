"""CSPLib prob018: water bucket planning.

Specification: https://www.csplib.org/Problems/prob018/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product

import qayd as cp

from ..common import add_solver_arguments, solve_from_args


@dataclass(frozen=True)
class WaterBucketsModel:
    model: cp.Model
    capacities: tuple[int, ...]
    initial: tuple[int, ...]
    target: tuple[int, ...]
    states: list[list[cp.IntVar]]
    moved: list[cp.IntVar]


def _transitions(capacities: tuple[int, ...], total: int) -> list[tuple[int, ...]]:
    states = [
        state
        for state in product(*(range(capacity + 1) for capacity in capacities))
        if sum(state) == total
    ]
    transitions: set[tuple[int, ...]] = set()
    for state in states:
        transitions.add((*state, *state, 0))
        for source in range(len(capacities)):
            for target in range(len(capacities)):
                if source == target:
                    continue
                amount = min(state[source], capacities[target] - state[target])
                if amount <= 0:
                    continue
                next_state = list(state)
                next_state[source] -= amount
                next_state[target] += amount
                transitions.add((*state, *next_state, 1))
    return sorted(transitions)


def build_model(
    capacities: tuple[int, ...],
    initial: tuple[int, ...],
    target: tuple[int, ...],
    *,
    max_steps: int,
) -> WaterBucketsModel:
    if (
        len(capacities) < 2
        or len(initial) != len(capacities)
        or len(target) != len(capacities)
    ):
        raise ValueError(
            "capacities, initial, and target must describe at least two matching buckets"
        )
    if max_steps < 0 or any(capacity < 1 for capacity in capacities):
        raise ValueError("capacities must be positive and max_steps non-negative")
    if any(
        value < 0 or value > capacity for value, capacity in zip(initial, capacities)
    ):
        raise ValueError("an initial level is outside its bucket")
    if any(
        value < 0 or value > capacity for value, capacity in zip(target, capacities)
    ):
        raise ValueError("a target level is outside its bucket")
    if sum(initial) != sum(target):
        raise ValueError(
            "initial and target states must contain the same amount of water"
        )

    model = cp.Model()
    states = [
        [
            model.int_var(0, capacity, name=f"level_{step}_{bucket}")
            for bucket, capacity in enumerate(capacities)
        ]
        for step in range(max_steps + 1)
    ]
    moved = [model.bool_var(name=f"moved_{step}") for step in range(max_steps)]
    for bucket, value in enumerate(initial):
        model.add(states[0][bucket] == value)
    for bucket, value in enumerate(target):
        model.add(states[-1][bucket] == value)
    transition_table = _transitions(capacities, sum(initial))
    for step in range(max_steps):
        model.table([*states[step], *states[step + 1], moved[step]], transition_table)
    if moved:
        model.minimize(sum(moved))
    return WaterBucketsModel(model, capacities, initial, target, states, moved)


def decode(
    built: WaterBucketsModel, solution: cp.Solution
) -> tuple[list[tuple[int, ...]], list[int]]:
    if not solution.is_sat():
        raise RuntimeError(f"no solution available, solver status is {solution.status}")
    states = [tuple(solution.value(level) for level in state) for state in built.states]
    moved = [solution.value(variable) for variable in built.moved]
    return states, moved


def _is_pour(
    current: tuple[int, ...], following: tuple[int, ...], capacities: tuple[int, ...]
) -> bool:
    for source in range(len(capacities)):
        for target in range(len(capacities)):
            if source == target:
                continue
            amount = min(current[source], capacities[target] - current[target])
            candidate = list(current)
            candidate[source] -= amount
            candidate[target] += amount
            if amount > 0 and tuple(candidate) == following:
                return True
    return False


def validate(
    built: WaterBucketsModel,
    states: list[tuple[int, ...]],
    moved: list[int],
    objective: int | None,
) -> None:
    if states[0] != built.initial or states[-1] != built.target:
        raise AssertionError("the state path has the wrong endpoints")
    for index, (current, following) in enumerate(zip(states, states[1:])):
        if moved[index] == 0 and following != current:
            raise AssertionError("a no-op transition changed the state")
        if moved[index] == 1 and not _is_pour(current, following, built.capacities):
            raise AssertionError("a transition is not a complete legal pour")
    if sum(moved) != objective:
        raise AssertionError("the reported objective does not match the transfer count")


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(","))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacities", type=_parse_ints, default=(8, 5, 3))
    parser.add_argument("--initial", type=_parse_ints, default=(8, 0, 0))
    parser.add_argument("--target", type=_parse_ints, default=(4, 4, 0))
    parser.add_argument("--max-steps", type=int, default=8)
    add_solver_arguments(parser)
    args = parser.parse_args(argv)

    built = build_model(
        args.capacities, args.initial, args.target, max_steps=args.max_steps
    )
    solution = solve_from_args(built.model, args)
    print(f"prob018 capacities={args.capacities} status={solution.status}")
    if not solution.is_sat():
        return 1
    states, moved = decode(built, solution)
    validate(built, states, moved, solution.objective)
    print(f"transfers={solution.objective}")
    for state, did_move in zip(states, [0, *moved]):
        print(f"{'pour' if did_move else 'state'} {state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
