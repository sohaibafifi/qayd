"""CSPLib prob006: optimal Golomb rulers.

Specification: https://www.csplib.org/Problems/prob006/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values


@dataclass(frozen=True)
class GolombRulerModel:
    model: cp.Model
    marks: list[cp.IntVar]
    differences: list[cp.IntVar]


def build_model(mark_count: int, *, upper_bound: int | None = None) -> GolombRulerModel:
    """Build the minimum-length ruler model for ``mark_count`` marks."""

    if mark_count < 2:
        raise ValueError("a Golomb ruler needs at least two marks")
    if upper_bound is None:
        if mark_count > 20:
            raise ValueError(
                "provide an explicit upper bound when mark_count is above 20"
            )
        upper_bound = (1 << (mark_count - 1)) - 1
    if upper_bound < mark_count - 1:
        raise ValueError("upper_bound is too small for strictly ordered marks")

    model = cp.Model()
    marks = model.int_vars(mark_count, 0, upper_bound, name="mark")
    model.add(marks[0] == 0)
    model.ordered(marks, "<")

    differences: list[cp.IntVar] = []
    for left in range(mark_count):
        for right in range(left + 1, mark_count):
            distance = model.int_var(1, upper_bound, name=f"d_{left}_{right}")
            model.add(distance == marks[right] - marks[left])
            differences.append(distance)
    model.all_different(differences)

    if mark_count > 2:
        model.add(marks[1] - marks[0] < marks[-1] - marks[-2])
    model.minimize(marks[-1])
    return GolombRulerModel(model, marks, differences)


def decode(built: GolombRulerModel, solution: cp.Solution) -> list[int]:
    return values(solution, built.marks)


def validate(mark_positions: list[int]) -> None:
    if not mark_positions or mark_positions[0] != 0:
        raise AssertionError("the first mark must be zero")
    if any(left >= right for left, right in zip(mark_positions, mark_positions[1:])):
        raise AssertionError("marks must be strictly increasing")
    differences = {
        mark_positions[right] - mark_positions[left]
        for left in range(len(mark_positions))
        for right in range(left + 1, len(mark_positions))
    }
    expected = len(mark_positions) * (len(mark_positions) - 1) // 2
    if len(differences) != expected:
        raise AssertionError("all pairwise differences must be distinct")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marks", type=int, default=6)
    parser.add_argument("--upper-bound", type=int)
    add_solver_arguments(parser, time_limit=30)
    args = parser.parse_args(argv)

    built = build_model(args.marks, upper_bound=args.upper_bound)
    solution = solve_from_args(built.model, args)
    print(f"prob006 marks={args.marks} status={solution.status}")
    if not solution.is_sat():
        return 1
    ruler = decode(built, solution)
    validate(ruler)
    print(f"length={ruler[-1]} ruler={ruler}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
