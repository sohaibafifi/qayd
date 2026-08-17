"""CSPLib prob033: word design for DNA computing on surfaces.

Specification: https://www.csplib.org/Problems/prob033/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values

ALPHABET = "ACGT"
COMPLEMENT = (3, 2, 1, 0)


@dataclass(frozen=True)
class DnaWordModel:
    model: cp.Model
    length: int
    minimum_distance: int
    candidates: tuple[tuple[int, ...], ...]
    selected: list[cp.IntVar]


def _distance(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    return sum(a != b for a, b in zip(left, right))


def _reverse_complement_distance(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    return _distance(tuple(reversed(left)), tuple(COMPLEMENT[value] for value in right))


def build_model(length: int = 2, minimum_distance: int = 1) -> DnaWordModel:
    if length < 2 or length % 2:
        raise ValueError("length must be positive and even")
    if minimum_distance < 1 or minimum_distance > length:
        raise ValueError("minimum_distance is invalid")
    candidates = tuple(
        word
        for word in product(range(4), repeat=length)
        if sum(value in (1, 2) for value in word) == length // 2
        and _reverse_complement_distance(word, word) >= minimum_distance
    )
    model = cp.Model()
    selected = [
        model.bool_var(name=f"word_{index}") for index in range(len(candidates))
    ]
    for first, left in enumerate(candidates):
        for second in range(first + 1, len(candidates)):
            right = candidates[second]
            if (
                _distance(left, right) < minimum_distance
                or _reverse_complement_distance(left, right) < minimum_distance
                or _reverse_complement_distance(right, left) < minimum_distance
            ):
                model.add(selected[first] + selected[second] <= 1)
    model.maximize(sum(selected))
    return DnaWordModel(model, length, minimum_distance, candidates, selected)


def decode(built: DnaWordModel, solution: cp.Solution) -> list[str]:
    membership = values(solution, built.selected)
    return [
        "".join(ALPHABET[value] for value in word)
        for word, selected in zip(built.candidates, membership)
        if selected
    ]


def validate(built: DnaWordModel, words: list[str], objective: int | None) -> None:
    encoded = [tuple(ALPHABET.index(symbol) for symbol in word) for word in words]
    if len(set(encoded)) != len(encoded):
        raise AssertionError("the selected DNA words are not distinct")
    for word in encoded:
        if (
            len(word) != built.length
            or sum(value in (1, 2) for value in word) != built.length // 2
        ):
            raise AssertionError("a word violates the GC-content constraint")
        if _reverse_complement_distance(word, word) < built.minimum_distance:
            raise AssertionError("a word violates its reverse-complement distance")
    for first, left in enumerate(encoded):
        for right in encoded[first + 1 :]:
            if _distance(left, right) < built.minimum_distance:
                raise AssertionError("two words are too close")
            if (
                min(
                    _reverse_complement_distance(left, right),
                    _reverse_complement_distance(right, left),
                )
                < built.minimum_distance
            ):
                raise AssertionError("two words violate reverse-complement distance")
    if objective is not None and len(words) != objective:
        raise AssertionError("the objective does not match the code size")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=2)
    parser.add_argument("--distance", type=int, default=1)
    add_solver_arguments(parser, time_limit=30)
    args = parser.parse_args(argv)
    built = build_model(args.length, args.distance)
    solution = solve_from_args(built.model, args)
    print(
        f"prob033 length={args.length} distance={args.distance} status={solution.status}"
    )
    if not solution.is_sat():
        return 1
    words = decode(built, solution)
    validate(built, words, solution.objective)
    print(f"size={solution.objective} words={words}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
