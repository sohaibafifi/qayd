"""CSPLib prob005: low-autocorrelation binary sequences.

Specification: https://www.csplib.org/Problems/prob005/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values


@dataclass(frozen=True)
class LabsModel:
    model: cp.Model
    size: int
    periodic: bool
    spins: list[cp.IntVar]
    correlations: list[cp.IntVar]
    energies: list[cp.IntVar]


def build_model(size: int, *, periodic: bool = False) -> LabsModel:
    if size < 2:
        raise ValueError("size must be at least two")
    model = cp.Model()
    spins = [
        model.int_var(values=(-1, 1), name=f"spin_{index}") for index in range(size)
    ]
    correlations: list[cp.IntVar] = []
    energies: list[cp.IntVar] = []
    product_table = [(-1, -1, 1), (-1, 1, -1), (1, -1, -1), (1, 1, 1)]

    for shift in range(1, size):
        pair_count = size if periodic else size - shift
        products: list[cp.IntVar] = []
        for index in range(pair_count):
            product_var = model.int_var(values=(-1, 1), name=f"product_{shift}_{index}")
            other = (index + shift) % size
            model.table([spins[index], spins[other], product_var], product_table)
            products.append(product_var)
        correlation = model.int_var(
            -pair_count, pair_count, name=f"correlation_{shift}"
        )
        model.add(correlation == sum(products))
        energy = model.int_var(0, pair_count * pair_count, name=f"energy_{shift}")
        square_table = [
            (value, value * value) for value in range(-pair_count, pair_count + 1)
        ]
        model.table([correlation, energy], square_table)
        correlations.append(correlation)
        energies.append(energy)

    model.add(spins[0] == 1)
    model.minimize(sum(energies))
    return LabsModel(model, size, periodic, spins, correlations, energies)


def decode(built: LabsModel, solution: cp.Solution) -> list[int]:
    return values(solution, built.spins)


def energy(sequence: list[int], *, periodic: bool = False) -> int:
    size = len(sequence)
    total = 0
    for shift in range(1, size):
        pair_count = size if periodic else size - shift
        correlation = sum(
            sequence[index] * sequence[(index + shift) % size]
            for index in range(pair_count)
        )
        total += correlation * correlation
    return total


def validate(sequence: list[int], *, periodic: bool, objective: int | None) -> None:
    if any(value not in (-1, 1) for value in sequence):
        raise AssertionError("LABS values must be -1 or +1")
    if energy(sequence, periodic=periodic) != objective:
        raise AssertionError(
            "the reported objective does not match the sequence energy"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--periodic", action="store_true")
    add_solver_arguments(parser, time_limit=30)
    args = parser.parse_args(argv)

    built = build_model(args.size, periodic=args.periodic)
    solution = solve_from_args(built.model, args)
    print(f"prob005 size={args.size} periodic={args.periodic} status={solution.status}")
    if not solution.is_sat():
        return 1
    sequence = decode(built, solution)
    validate(sequence, periodic=args.periodic, objective=solution.objective)
    print(f"energy={solution.objective} sequence={sequence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
