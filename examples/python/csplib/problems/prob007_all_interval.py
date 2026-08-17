"""CSPLib prob007: all-interval series.

Specification: https://www.csplib.org/Problems/prob007/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values


@dataclass(frozen=True)
class AllIntervalModel:
    model: cp.Model
    series: list[cp.IntVar]
    intervals: list[cp.IntVar]


def build_model(size: int) -> AllIntervalModel:
    if size < 2:
        raise ValueError("an all-interval series needs size at least two")

    model = cp.Model()
    series = model.int_vars(size, 0, size - 1, name="pitch")
    intervals = model.int_vars(size - 1, 1, size - 1, name="interval")
    model.all_different(series)
    model.all_different(intervals)
    for index, interval in enumerate(intervals):
        model.add(interval == abs(series[index + 1] - series[index]))

    model.add(series[0] < series[-1])
    if size > 2:
        model.add(intervals[0] > intervals[-1])
    return AllIntervalModel(model, series, intervals)


def decode(
    built: AllIntervalModel, solution: cp.Solution
) -> tuple[list[int], list[int]]:
    return values(solution, built.series), values(solution, built.intervals)


def validate(series: list[int], intervals: list[int]) -> None:
    size = len(series)
    if sorted(series) != list(range(size)):
        raise AssertionError("the series must be a permutation of 0..n-1")
    replayed = [abs(right - left) for left, right in zip(series, series[1:])]
    if replayed != intervals:
        raise AssertionError("reported intervals do not match the series")
    if sorted(intervals) != list(range(1, size)):
        raise AssertionError("the intervals must be a permutation of 1..n-1")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=12)
    add_solver_arguments(parser)
    args = parser.parse_args(argv)

    built = build_model(args.size)
    solution = solve_from_args(built.model, args)
    print(f"prob007 size={args.size} status={solution.status}")
    if not solution.is_sat():
        return 1
    series, intervals = decode(built, solution)
    validate(series, intervals)
    print(f"series={series}")
    print(f"intervals={intervals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
