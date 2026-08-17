"""CSPLib prob036: fixed-length error-correcting codes.

Specification: https://www.csplib.org/Problems/prob036/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations

import qayd as cp

from ..code_common import symbol_distance
from ..common import add_solver_arguments, solve_from_args


@dataclass(frozen=True)
class ErrorCorrectingCodeModel:
    model: cp.Model
    codeword_count: int
    length: int
    alphabet: int
    minimum_distance: int
    metric: str
    codewords: list[list[cp.IntVar]]


def build_model(
    codeword_count: int,
    length: int,
    alphabet: int,
    minimum_distance: int,
    *,
    metric: str = "hamming",
) -> ErrorCorrectingCodeModel:
    if min(codeword_count, length, alphabet) < 1:
        raise ValueError("codeword_count, length, and alphabet must be positive")
    if minimum_distance < 0:
        raise ValueError("minimum_distance must be non-negative")

    model = cp.Model()
    codewords = [
        model.int_vars(length, 0, alphabet - 1, name=f"word_{word}")
        for word in range(codeword_count)
    ]
    for first, second in combinations(range(codeword_count), 2):
        distances = [
            symbol_distance(
                model,
                codewords[first][position],
                codewords[second][position],
                alphabet=alphabet,
                metric=metric,
                name=f"distance_{first}_{second}_{position}",
            )
            for position in range(length)
        ]
        model.add(sum(distances) >= minimum_distance)
    for symbol in codewords[0]:
        model.add(symbol == 0)
    return ErrorCorrectingCodeModel(
        model,
        codeword_count,
        length,
        alphabet,
        minimum_distance,
        metric,
        codewords,
    )


def decode(built: ErrorCorrectingCodeModel, solution: cp.Solution) -> list[list[int]]:
    if not solution.is_sat():
        raise RuntimeError(f"no solution available, solver status is {solution.status}")
    return [[solution.value(symbol) for symbol in word] for word in built.codewords]


def _distance(left: list[int], right: list[int], alphabet: int, metric: str) -> int:
    if metric == "hamming":
        return sum(first != second for first, second in zip(left, right))
    return sum(
        min(abs(first - second), alphabet - abs(first - second))
        for first, second in zip(left, right)
    )


def validate(built: ErrorCorrectingCodeModel, codewords: list[list[int]]) -> None:
    if len(codewords) != built.codeword_count or any(
        len(word) != built.length for word in codewords
    ):
        raise AssertionError("the code has the wrong dimensions")
    if any(
        symbol < 0 or symbol >= built.alphabet for word in codewords for symbol in word
    ):
        raise AssertionError("a symbol lies outside the alphabet")
    for left, right in combinations(codewords, 2):
        if (
            _distance(left, right, built.alphabet, built.metric)
            < built.minimum_distance
        ):
            raise AssertionError("two codewords are too close")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codewords", type=int, default=4)
    parser.add_argument("--length", type=int, default=3)
    parser.add_argument("--alphabet", type=int, default=2)
    parser.add_argument("--distance", type=int, default=2)
    parser.add_argument("--metric", choices=("hamming", "lee"), default="hamming")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)

    built = build_model(
        args.codewords,
        args.length,
        args.alphabet,
        args.distance,
        metric=args.metric,
    )
    solution = solve_from_args(built.model, args)
    print(
        f"prob036 words={args.codewords} length={args.length} alphabet={args.alphabet} "
        f"distance={args.distance} metric={args.metric} status={solution.status}"
    )
    if not solution.is_sat():
        return 1
    codewords = decode(built, solution)
    validate(built, codewords)
    for word in codewords:
        print(" ".join(map(str, word)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
