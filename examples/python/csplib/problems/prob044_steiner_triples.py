"""CSPLib prob044: Steiner triple systems.

Specification: https://www.csplib.org/Problems/prob044/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations

import qayd as cp

from ..common import add_solver_arguments, solve_from_args

Triple = tuple[int, int, int]


@dataclass(frozen=True)
class SteinerModel:
    model: cp.Model
    order: int
    selected: dict[Triple, cp.IntVar]


def build_model(order: int) -> SteinerModel:
    if order < 3:
        raise ValueError("order must be at least three")

    model = cp.Model()
    selected = {
        triple: model.bool_var(name=f"triple_{triple[0]}_{triple[1]}_{triple[2]}")
        for triple in combinations(range(order), 3)
    }
    if order % 6 not in (1, 3):
        witness = next(iter(selected.values()))
        model.add(witness == witness + 1)
        return SteinerModel(model, order, selected)

    for pair in combinations(range(order), 2):
        containing = [
            variable
            for triple, variable in selected.items()
            if set(pair).issubset(triple)
        ]
        model.add(sum(containing) == 1)
    model.add(selected[(0, 1, 2)] == 1)
    return SteinerModel(model, order, selected)


def decode(built: SteinerModel, solution: cp.Solution) -> list[Triple]:
    if not solution.is_sat():
        raise RuntimeError(f"no solution available, solver status is {solution.status}")
    return [
        triple
        for triple, variable in built.selected.items()
        if solution.value(variable) == 1
    ]


def validate(triples: list[Triple], *, order: int) -> None:
    if len(triples) != order * (order - 1) // 6:
        raise AssertionError("the Steiner system has the wrong number of triples")
    pair_counts = {pair: 0 for pair in combinations(range(order), 2)}
    for triple in triples:
        if len(set(triple)) != 3 or any(
            value < 0 or value >= order for value in triple
        ):
            raise AssertionError(
                "each block must contain three distinct valid elements"
            )
        for pair in combinations(sorted(triple), 2):
            pair_counts[pair] += 1
    if any(count != 1 for count in pair_counts.values()):
        raise AssertionError("every pair must occur in exactly one triple")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=int, default=7)
    add_solver_arguments(parser)
    args = parser.parse_args(argv)

    built = build_model(args.order)
    solution = solve_from_args(built.model, args)
    print(f"prob044 order={args.order} status={solution.status}")
    if not solution.is_sat():
        return 1
    triples = decode(built, solution)
    validate(triples, order=args.order)
    print(f"triples={triples}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
