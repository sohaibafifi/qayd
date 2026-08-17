"""CSPLib prob001: car sequencing.

Specification: https://www.csplib.org/Problems/prob001/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values

SAMPLE_INSTANCE = """\
10 5 6
1 2 1 2 1
2 3 3 5 5
0 1 1 0 1 1 0
1 1 0 0 0 1 0
2 2 0 1 0 0 1
3 2 0 1 0 1 0
4 2 1 0 1 0 0
5 2 1 1 0 0 0
"""


@dataclass(frozen=True)
class CarSequencingInstance:
    car_count: int
    option_limits: tuple[int, ...]
    window_sizes: tuple[int, ...]
    class_demands: tuple[int, ...]
    class_options: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class CarSequencingModel:
    model: cp.Model
    instance: CarSequencingInstance
    sequence: list[cp.IntVar]


def parse_instance(text: str) -> CarSequencingInstance:
    rows = [
        line.split()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(rows) < 3:
        raise ValueError("car sequencing instance is incomplete")
    car_count, option_count, class_count = map(int, rows[0])
    option_limits = tuple(map(int, rows[1]))
    window_sizes = tuple(map(int, rows[2]))
    if len(option_limits) != option_count or len(window_sizes) != option_count:
        raise ValueError("option capacity rows have the wrong length")
    if len(rows[3:]) != class_count:
        raise ValueError("class row count does not match the header")

    demands = [0] * class_count
    options: list[tuple[int, ...] | None] = [None] * class_count
    for row in rows[3:]:
        if len(row) != option_count + 2:
            raise ValueError("a class row has the wrong length")
        class_id, demand, *flags = map(int, row)
        if class_id < 0 or class_id >= class_count or options[class_id] is not None:
            raise ValueError(
                "class identifiers must be unique values from 0..classes-1"
            )
        if demand < 0 or any(flag not in (0, 1) for flag in flags):
            raise ValueError(
                "class demands must be non-negative and option flags must be binary"
            )
        demands[class_id] = demand
        options[class_id] = tuple(flags)
    if sum(demands) != car_count:
        raise ValueError("class demands do not sum to the declared car count")
    return CarSequencingInstance(
        car_count,
        option_limits,
        window_sizes,
        tuple(demands),
        tuple(option for option in options if option is not None),
    )


def load_instance(path: str | Path) -> CarSequencingInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def build_model(instance: CarSequencingInstance) -> CarSequencingModel:
    class_count = len(instance.class_demands)
    option_count = len(instance.option_limits)
    if class_count < 1 or instance.car_count < 1:
        raise ValueError("the instance must contain cars and classes")
    if len(instance.class_options) != class_count:
        raise ValueError("every class must define its option flags")

    model = cp.Model()
    sequence = model.int_vars(instance.car_count, 0, class_count - 1, name="class")
    model.cardinality(
        sequence,
        list(range(class_count)),
        list(instance.class_demands),
        list(instance.class_demands),
        closed=True,
    )
    for option in range(option_count):
        flags = [
            model.bool_var(name=f"option_{option}_{position}")
            for position in range(instance.car_count)
        ]
        required = [
            instance.class_options[class_id][option] for class_id in range(class_count)
        ]
        for position in range(instance.car_count):
            model.element_const(required, sequence[position], flags[position])
        window = instance.window_sizes[option]
        limit = instance.option_limits[option]
        if window < 1 or limit < 0:
            raise ValueError("window sizes must be positive and limits non-negative")
        for start in range(instance.car_count - window + 1):
            model.add(sum(flags[start : start + window]) <= limit)
    return CarSequencingModel(model, instance, sequence)


def decode(built: CarSequencingModel, solution: cp.Solution) -> list[int]:
    return values(solution, built.sequence)


def validate(sequence: list[int], instance: CarSequencingInstance) -> None:
    if len(sequence) != instance.car_count:
        raise AssertionError("the sequence has the wrong length")
    if any(
        sequence.count(class_id) != demand
        for class_id, demand in enumerate(instance.class_demands)
    ):
        raise AssertionError("class demand counts are not respected")
    for option, (limit, window) in enumerate(
        zip(instance.option_limits, instance.window_sizes)
    ):
        for start in range(instance.car_count - window + 1):
            count = sum(
                instance.class_options[class_id][option]
                for class_id in sequence[start : start + window]
            )
            if count > limit:
                raise AssertionError("an option capacity window is overloaded")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="CSPLib car sequencing instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)

    instance = (
        load_instance(args.path) if args.path else parse_instance(SAMPLE_INSTANCE)
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(
        f"prob001 cars={instance.car_count} classes={len(instance.class_demands)} status={solution.status}"
    )
    if not solution.is_sat():
        return 1
    sequence = decode(built, solution)
    validate(sequence, instance)
    print(f"sequence={sequence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
