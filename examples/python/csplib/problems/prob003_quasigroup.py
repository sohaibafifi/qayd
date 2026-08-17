"""CSPLib prob003: quasigroup existence for laws QG1 through QG7.

Specification: https://www.csplib.org/Problems/prob003/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values

SUPPORTED_LAWS = (1, 2, 3, 4, 5, 6, 7)


@dataclass(frozen=True)
class QuasigroupModel:
    model: cp.Model
    order: int
    law: int
    table: list[list[cp.IntVar]]


def _lookup(
    model: cp.Model,
    flat: list[cp.IntVar],
    order: int,
    row: int | cp.IntVar,
    column: int | cp.IntVar,
    *,
    name: str,
) -> cp.IntVar:
    if isinstance(row, int) and isinstance(column, int):
        return flat[row * order + column]
    index = model.int_var(0, order * order - 1, name=f"{name}_index")
    value = model.int_var(0, order - 1, name=name)
    model.add(index == row * order + column)
    model.element(flat, index, value)
    return value


def build_model(
    order: int, *, law: int = 3, idempotent: bool = False
) -> QuasigroupModel:
    if order < 1:
        raise ValueError("order must be positive")
    if law not in SUPPORTED_LAWS:
        raise ValueError(f"law must be one of {SUPPORTED_LAWS}")

    model = cp.Model()
    flat = model.int_vars(order * order, 0, order - 1, name="cell")
    table = [flat[row * order : (row + 1) * order] for row in range(order)]
    for row in table:
        model.all_different(row)
    for column in range(order):
        model.all_different([table[row][column] for row in range(order)])
    if idempotent:
        for value in range(order):
            model.add(table[value][value] == value)

    compound_codes = []
    for a in range(order):
        for b in range(order):
            ab = table[a][b]
            ba = table[b][a]
            if law in (1, 2):
                inverse = model.int_var(0, order - 1, name=f"qg{law}_inverse_{a}_{b}")
                if law == 1:
                    inverse_result = _lookup(
                        model,
                        flat,
                        order,
                        inverse,
                        b,
                        name=f"qg1_relation_{a}_{b}",
                    )
                else:
                    inverse_result = _lookup(
                        model,
                        flat,
                        order,
                        b,
                        inverse,
                        name=f"qg2_relation_{a}_{b}",
                    )
                model.add(inverse_result == a)
                code = model.int_var(0, order * order - 1, name=f"qg{law}_code_{a}_{b}")
                model.add(code == ab * order + inverse)
                compound_codes.append(code)
            elif law == 3:
                result = _lookup(model, flat, order, ab, ba, name=f"qg3_{a}_{b}")
                model.add(result == a)
            elif law == 4:
                result = _lookup(model, flat, order, ba, ab, name=f"qg4_{a}_{b}")
                model.add(result == a)
            elif law == 5:
                inner = _lookup(model, flat, order, ba, b, name=f"qg5_inner_{a}_{b}")
                result = _lookup(model, flat, order, inner, b, name=f"qg5_{a}_{b}")
                model.add(result == a)
            elif law == 6:
                left = _lookup(model, flat, order, ab, b, name=f"qg6_left_{a}_{b}")
                right = _lookup(model, flat, order, a, ab, name=f"qg6_right_{a}_{b}")
                model.add(left == right)
            elif law == 7:
                left = _lookup(model, flat, order, ba, b, name=f"qg7_left_{a}_{b}")
                right = _lookup(model, flat, order, a, ba, name=f"qg7_right_{a}_{b}")
                model.add(left == right)

    if compound_codes:
        model.all_different(compound_codes)

    return QuasigroupModel(model, order, law, table)


def decode(built: QuasigroupModel, solution: cp.Solution) -> list[list[int]]:
    return [values(solution, row) for row in built.table]


def _mul(table: list[list[int]], left: int, right: int) -> int:
    return table[left][right]


def validate(table: list[list[int]], *, law: int, idempotent: bool = False) -> None:
    order = len(table)
    expected = list(range(order))
    if order == 0 or any(sorted(row) != expected for row in table):
        raise AssertionError("every row must be a permutation")
    if any(
        sorted(table[row][column] for row in range(order)) != expected
        for column in range(order)
    ):
        raise AssertionError("every column must be a permutation")
    if idempotent and any(table[value][value] != value for value in range(order)):
        raise AssertionError("the quasigroup must be idempotent")

    codes: list[tuple[int, int]] = []
    for a in range(order):
        for b in range(order):
            ab = _mul(table, a, b)
            ba = _mul(table, b, a)
            if law in (1, 2):
                if law == 1:
                    inverse = next(
                        value for value in range(order) if table[value][b] == a
                    )
                else:
                    inverse = next(
                        value for value in range(order) if table[b][value] == a
                    )
                codes.append((ab, inverse))
            if law == 3 and _mul(table, ab, ba) != a:
                raise AssertionError("QG3 identity violated")
            if law == 4 and _mul(table, ba, ab) != a:
                raise AssertionError("QG4 identity violated")
            if law == 5 and _mul(table, _mul(table, ba, b), b) != a:
                raise AssertionError("QG5 identity violated")
            if law == 6 and _mul(table, ab, b) != _mul(table, a, ab):
                raise AssertionError("QG6 identity violated")
            if law == 7 and _mul(table, ba, b) != _mul(table, a, ba):
                raise AssertionError("QG7 identity violated")
    if law in (1, 2) and len(set(codes)) != order * order:
        raise AssertionError(f"QG{law} compound relation is not injective")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--law", type=int, choices=SUPPORTED_LAWS, default=3)
    parser.add_argument("--idempotent", action="store_true")
    add_solver_arguments(parser, time_limit=30)
    args = parser.parse_args(argv)

    built = build_model(args.order, law=args.law, idempotent=args.idempotent)
    solution = solve_from_args(built.model, args)
    print(
        f"prob003 order={args.order} law=QG{args.law} "
        f"idempotent={args.idempotent} status={solution.status}"
    )
    if not solution.is_sat():
        return 1
    table = decode(built, solution)
    validate(table, law=args.law, idempotent=args.idempotent)
    for row in table:
        print(" ".join(map(str, row)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
