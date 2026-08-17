"""CSPLib prob015: Schur's Lemma partitioning problem.

Specification: https://www.csplib.org/Problems/prob015/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import qayd as cp

from ..common import add_solver_arguments, solve_from_args


@dataclass(frozen=True)
class SchurModel:
    model: cp.Model
    assignment: list[list[cp.IntVar]]


def build_model(size: int, *, boxes: int = 3) -> SchurModel:
    if size < 1:
        raise ValueError("size must be positive")
    if boxes < 1:
        raise ValueError("boxes must be positive")

    model = cp.Model()
    assignment = [
        [model.bool_var(name=f"ball_{ball}_box_{box}") for box in range(boxes)]
        for ball in range(1, size + 1)
    ]
    for row in assignment:
        model.add(sum(row) == 1)

    for x in range(1, size + 1):
        for y in range(x, size + 1):
            z = x + y
            if z > size:
                break
            for box in range(boxes):
                model.add(
                    assignment[x - 1][box]
                    + assignment[y - 1][box]
                    + assignment[z - 1][box]
                    <= 2
                )

    model.add(assignment[0][0] == 1)
    return SchurModel(model, assignment)


def decode(built: SchurModel, solution: cp.Solution) -> list[int]:
    if not solution.is_sat():
        raise RuntimeError(f"no solution available, solver status is {solution.status}")
    return [
        next(box for box, variable in enumerate(row) if solution.value(variable) == 1)
        for row in built.assignment
    ]


def validate(box_for_ball: list[int], *, boxes: int) -> None:
    if any(box < 0 or box >= boxes for box in box_for_ball):
        raise AssertionError("every ball must be assigned to a valid box")
    size = len(box_for_ball)
    for x in range(1, size + 1):
        for y in range(x, size + 1):
            z = x + y
            if z > size:
                break
            if box_for_ball[x - 1] == box_for_ball[y - 1] == box_for_ball[z - 1]:
                raise AssertionError(f"monochromatic Schur triple: {x}+{y}={z}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=13)
    parser.add_argument("--boxes", type=int, default=3)
    add_solver_arguments(parser)
    args = parser.parse_args(argv)

    built = build_model(args.size, boxes=args.boxes)
    solution = solve_from_args(built.model, args)
    print(f"prob015 size={args.size} boxes={args.boxes} status={solution.status}")
    if not solution.is_sat():
        return 1
    assignment = decode(built, solution)
    validate(assignment, boxes=args.boxes)
    groups = [
        [ball for ball, box in enumerate(assignment, start=1) if box == target]
        for target in range(args.boxes)
    ]
    print(f"boxes={groups}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
