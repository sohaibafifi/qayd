"""Shared CLI and validation helpers for the CSPLib examples."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import qayd as cp


def add_solver_arguments(
    parser: argparse.ArgumentParser, *, time_limit: int = 10
) -> None:
    parser.add_argument("--time-limit", type=int, default=time_limit)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--engine", choices=("auto", "exact", "ls"), default="exact")
    parser.add_argument("--verbose", action="store_true")


def solve_from_args(model: cp.Model, args: argparse.Namespace) -> cp.Solution:
    return model.solve(
        engine=args.engine,
        time_limit=args.time_limit,
        seed=args.seed,
        threads=args.threads,
        verbose=args.verbose,
    )


def require_solution(solution: cp.Solution) -> None:
    if not solution.is_sat():
        raise RuntimeError(f"no solution available, solver status is {solution.status}")


def values(solution: cp.Solution, variables: Sequence[cp.IntVar]) -> list[int]:
    require_solution(solution)
    return [solution.value(variable) for variable in variables]
