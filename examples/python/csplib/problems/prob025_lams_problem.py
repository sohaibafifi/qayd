"""CSPLib prob025: Lam's projective-plane problem.

Specification: https://www.csplib.org/Problems/prob025/
"""

from __future__ import annotations

import argparse

import qayd as cp

from ..common import add_solver_arguments, solve_from_args
from . import prob028_bibd


def build_model(order: int) -> prob028_bibd.BibdModel:
    if order < 1:
        raise ValueError("projective-plane order must be positive")
    point_count = order * order + order + 1
    line_size = order + 1
    return prob028_bibd.build_model(
        point_count,
        point_count,
        line_size,
        line_size,
        1,
    )


def decode(built: prob028_bibd.BibdModel, solution: cp.Solution) -> list[list[int]]:
    return prob028_bibd.decode(built, solution)


def validate(matrix: list[list[int]], order: int) -> None:
    parameters = (
        order * order + order + 1,
        order * order + order + 1,
        order + 1,
        order + 1,
        1,
    )
    prob028_bibd.validate(matrix, parameters)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=int, default=2)
    add_solver_arguments(parser, time_limit=30)
    args = parser.parse_args(argv)
    built = build_model(args.order)
    solution = solve_from_args(built.model, args)
    print(f"prob025 order={args.order} status={solution.status}")
    if not solution.is_sat():
        return 1
    matrix = decode(built, solution)
    validate(matrix, args.order)
    for row in matrix:
        print(" ".join(map(str, row)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
