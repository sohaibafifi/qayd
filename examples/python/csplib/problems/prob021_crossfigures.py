"""CSPLib prob021: crossfigures.

Specification: https://www.csplib.org/Problems/prob021/
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values

DEFAULT_INSTANCE = {
    "cells": 4,
    "entries": [
        {"name": "1A", "cells": [0, 1], "clue": {"type": "constant", "value": 23}},
        {"name": "1D", "cells": [0, 2], "clue": {"type": "prime"}},
        {
            "name": "2A",
            "cells": [2, 3],
            "clue": {"type": "multiple", "of": "1A", "factor": 4},
        },
        {"name": "2D", "cells": [1, 3], "clue": {"type": "constant", "value": 32}},
    ],
}


@dataclass(frozen=True)
class CrossfigureEntry:
    name: str
    cells: tuple[int, ...]
    clue: tuple[tuple[str, object], ...]

    def clue_dict(self) -> dict[str, object]:
        return dict(self.clue)


@dataclass(frozen=True)
class CrossfigureInstance:
    cells: int
    entries: tuple[CrossfigureEntry, ...]


@dataclass(frozen=True)
class CrossfigureModel:
    model: cp.Model
    instance: CrossfigureInstance
    digits: list[cp.IntVar]
    entry_values: list[cp.IntVar]


def parse_instance(data: str | bytes) -> CrossfigureInstance:
    raw = json.loads(data)
    try:
        entries = tuple(
            CrossfigureEntry(
                str(item["name"]),
                tuple(int(cell) for cell in item["cells"]),
                tuple(sorted((str(key), value) for key, value in item["clue"].items())),
            )
            for item in raw["entries"]
        )
        return CrossfigureInstance(int(raw["cells"]), entries)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid crossfigures JSON instance") from error


def load_instance(path: str | Path) -> CrossfigureInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def is_prime(value: int) -> bool:
    return value >= 2 and all(
        value % divisor for divisor in range(2, math.isqrt(value) + 1)
    )


def build_model(instance: CrossfigureInstance) -> CrossfigureModel:
    if instance.cells < 1 or not instance.entries:
        raise ValueError("cells and entries must be non-empty")
    if len({entry.name for entry in instance.entries}) != len(instance.entries):
        raise ValueError("entry names must be unique")
    model = cp.Model()
    digits = model.int_vars(instance.cells, 0, 9, name="digit")
    entry_values = []
    names = {entry.name: index for index, entry in enumerate(instance.entries)}
    for index, entry in enumerate(instance.entries):
        if not entry.cells or any(
            cell < 0 or cell >= instance.cells for cell in entry.cells
        ):
            raise ValueError("an entry cell is invalid")
        maximum = 10 ** len(entry.cells) - 1
        value = model.int_var(
            10 ** (len(entry.cells) - 1) if len(entry.cells) > 1 else 0,
            maximum,
            name=f"entry_{index}",
        )
        model.add(
            value
            == sum(
                10 ** (len(entry.cells) - offset - 1) * digits[cell]
                for offset, cell in enumerate(entry.cells)
            )
        )
        if len(entry.cells) > 1:
            model.table([digits[entry.cells[0]]], [(digit,) for digit in range(1, 10)])
        entry_values.append(value)
    for index, entry in enumerate(instance.entries):
        clue = entry.clue_dict()
        kind = str(clue.get("type"))
        if kind == "constant":
            model.add(entry_values[index] == int(clue["value"]))
        elif kind == "prime":
            maximum = 10 ** len(entry.cells) - 1
            model.table(
                [entry_values[index]],
                [(value,) for value in range(2, maximum + 1) if is_prime(value)],
            )
        elif kind == "square":
            maximum = 10 ** len(entry.cells) - 1
            model.table(
                [entry_values[index]],
                [(root * root,) for root in range(1, math.isqrt(maximum) + 1)],
            )
        elif kind == "multiple":
            other = names[str(clue["of"])]
            model.add(
                entry_values[index]
                == int(clue.get("factor", 1)) * entry_values[other]
                + int(clue.get("offset", 0))
            )
        elif kind == "sum":
            refs = [names[str(name)] for name in clue["of"]]
            model.add(entry_values[index] == sum(entry_values[other] for other in refs))
        elif kind == "product":
            refs = [names[str(name)] for name in clue["of"]]
            if len(refs) != 2:
                raise ValueError("product clues require exactly two entries")
            limits = [10 ** len(instance.entries[other].cells) for other in refs]
            if limits[0] * limits[1] > 200_000:
                raise ValueError(
                    "product clue domains are too large for extensional encoding"
                )
            maximum = 10 ** len(entry.cells) - 1
            model.table(
                [entry_values[refs[0]], entry_values[refs[1]], entry_values[index]],
                [
                    (left, right, left * right)
                    for left, right in itertools.product(
                        range(limits[0]), range(limits[1])
                    )
                    if left * right <= maximum
                ],
            )
        else:
            raise ValueError(f"unsupported clue type: {kind}")
    return CrossfigureModel(model, instance, digits, entry_values)


def decode(
    built: CrossfigureModel, solution: cp.Solution
) -> tuple[list[int], dict[str, int]]:
    return values(solution, built.digits), {
        entry.name: solution.value(value)
        for entry, value in zip(built.instance.entries, built.entry_values)
    }


def clue_holds(entry: CrossfigureEntry, values_by_name: dict[str, int]) -> bool:
    clue = entry.clue_dict()
    value = values_by_name[entry.name]
    kind = str(clue["type"])
    if kind == "constant":
        return value == int(clue["value"])
    if kind == "prime":
        return is_prime(value)
    if kind == "square":
        return math.isqrt(value) ** 2 == value
    if kind == "multiple":
        return value == int(clue.get("factor", 1)) * values_by_name[
            str(clue["of"])
        ] + int(clue.get("offset", 0))
    refs = [values_by_name[str(name)] for name in clue["of"]]
    return value == (sum(refs) if kind == "sum" else math.prod(refs))


def validate(built: CrossfigureModel, result: tuple[list[int], dict[str, int]]) -> None:
    digits, entries = result
    for entry in built.instance.entries:
        value = sum(
            10 ** (len(entry.cells) - offset - 1) * digits[cell]
            for offset, cell in enumerate(entry.cells)
        )
        if entries[entry.name] != value or not clue_holds(entry, entries):
            raise AssertionError("a crossfigure entry or clue is invalid")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON crossfigures instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob021 entries={len(instance.entries)} status={solution.status}")
    if not solution.is_sat():
        return 1
    result = decode(built, solution)
    validate(built, result)
    print(f"digits={result[0]} entries={result[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
