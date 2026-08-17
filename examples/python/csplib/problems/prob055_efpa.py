"""CSPLib prob055: equidistant frequency permutation arrays.

Specification: https://www.csplib.org/Problems/prob055/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations

import qayd as cp

from ..code_common import symbol_distance
from ..common import add_solver_arguments, solve_from_args


@dataclass(frozen=True)
class EfpaModel:
    model: cp.Model
    codeword_count: int
    alphabet: int
    frequency: int
    distance: int
    codewords: list[list[cp.IntVar]]


def build_model(
    codeword_count: int, alphabet: int, frequency: int, distance: int
) -> EfpaModel:
    if min(codeword_count, alphabet, frequency) < 1:
        raise ValueError("codeword_count, alphabet, and frequency must be positive")
    length = alphabet * frequency
    if distance < 0 or distance > length:
        raise ValueError("distance must be between zero and the codeword length")

    model = cp.Model()
    codewords = [
        model.int_vars(length, 0, alphabet - 1, name=f"word_{word}")
        for word in range(codeword_count)
    ]
    for word in codewords:
        model.cardinality(
            word,
            list(range(alphabet)),
            [frequency] * alphabet,
            [frequency] * alphabet,
            closed=True,
        )
    for first, second in combinations(range(codeword_count), 2):
        differences = [
            symbol_distance(
                model,
                codewords[first][position],
                codewords[second][position],
                alphabet=alphabet,
                metric="hamming",
                name=f"different_{first}_{second}_{position}",
            )
            for position in range(length)
        ]
        model.add(sum(differences) == distance)

    for position in range(length):
        model.add(codewords[0][position] == position // frequency)
    return EfpaModel(model, codeword_count, alphabet, frequency, distance, codewords)


def decode(built: EfpaModel, solution: cp.Solution) -> list[list[int]]:
    if not solution.is_sat():
        raise RuntimeError(f"no solution available, solver status is {solution.status}")
    return [[solution.value(symbol) for symbol in word] for word in built.codewords]


def validate(built: EfpaModel, codewords: list[list[int]]) -> None:
    length = built.alphabet * built.frequency
    if len(codewords) != built.codeword_count or any(
        len(word) != length for word in codewords
    ):
        raise AssertionError("the array has the wrong dimensions")
    for word in codewords:
        if any(
            word.count(symbol) != built.frequency for symbol in range(built.alphabet)
        ):
            raise AssertionError("a symbol has the wrong frequency")
    for left, right in combinations(codewords, 2):
        if sum(first != second for first, second in zip(left, right)) != built.distance:
            raise AssertionError("two codewords have the wrong Hamming distance")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codewords", type=int, default=5)
    parser.add_argument("--alphabet", type=int, default=3)
    parser.add_argument("--frequency", type=int, default=2)
    parser.add_argument("--distance", type=int, default=4)
    add_solver_arguments(parser, time_limit=30)
    args = parser.parse_args(argv)

    built = build_model(args.codewords, args.alphabet, args.frequency, args.distance)
    solution = solve_from_args(built.model, args)
    print(
        f"prob055 words={args.codewords} alphabet={args.alphabet} frequency={args.frequency} "
        f"distance={args.distance} status={solution.status}"
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
