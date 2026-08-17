"""CSPLib prob073: test scheduling.

Specification: https://www.csplib.org/Problems/prob073/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values

DEFAULT_INSTANCE = {
    "machines": 2,
    "tests": [
        {"duration": 3, "eligible": [0, 1], "resources": [0]},
        {"duration": 2, "eligible": [0], "resources": []},
        {"duration": 2, "eligible": [1], "resources": [0]},
        {"duration": 1, "eligible": [0, 1], "resources": []},
    ],
}


@dataclass(frozen=True)
class Test:
    duration: int
    eligible: tuple[int, ...]
    resources: frozenset[int]


@dataclass(frozen=True)
class TestSchedulingInstance:
    machines: int
    tests: tuple[Test, ...]


@dataclass(frozen=True)
class TestSchedulingModel:
    model: cp.Model
    instance: TestSchedulingInstance
    horizon: int
    starts: list[cp.IntVar]
    machines: list[cp.IntVar]


def parse_instance(data: str | bytes) -> TestSchedulingInstance:
    raw = json.loads(data)
    try:
        tests = tuple(
            Test(
                int(item["duration"]),
                tuple(int(value) for value in item["eligible"]),
                frozenset(int(value) for value in item.get("resources", [])),
            )
            for item in raw["tests"]
        )
        return TestSchedulingInstance(int(raw["machines"]), tests)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid test-scheduling JSON instance") from error


def load_instance(path: str | Path) -> TestSchedulingInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def build_model(instance: TestSchedulingInstance) -> TestSchedulingModel:
    if instance.machines < 1 or not instance.tests:
        raise ValueError("machines and tests must be non-empty")
    if any(
        test.duration < 1
        or not test.eligible
        or min(test.eligible) < 0
        or max(test.eligible) >= instance.machines
        for test in instance.tests
    ):
        raise ValueError("a test duration or eligible-machine set is invalid")
    horizon = sum(test.duration for test in instance.tests)
    model = cp.Model()
    starts = [
        model.int_var(0, horizon - test.duration, name=f"start_{index}")
        for index, test in enumerate(instance.tests)
    ]
    machines = model.int_vars(
        len(instance.tests), 0, instance.machines - 1, name="machine"
    )
    for index, test in enumerate(instance.tests):
        model.table(
            [machines[index]], [(machine,) for machine in sorted(set(test.eligible))]
        )
    for first, left in enumerate(instance.tests):
        for second in range(first + 1, len(instance.tests)):
            separated = (starts[first] + left.duration <= starts[second]) | (
                starts[second] + instance.tests[second].duration <= starts[first]
            )
            if left.resources.intersection(instance.tests[second].resources):
                model.add(separated)
            else:
                model.add((machines[first] != machines[second]) | separated)
    makespan = model.int_var(0, horizon, name="makespan")
    for index, test in enumerate(instance.tests):
        model.add(makespan >= starts[index] + test.duration)
    model.minimize(makespan)
    return TestSchedulingModel(model, instance, horizon, starts, machines)


def decode(built: TestSchedulingModel, solution: cp.Solution) -> list[tuple[int, int]]:
    return list(zip(values(solution, built.starts), values(solution, built.machines)))


def validate(
    built: TestSchedulingModel,
    schedule: list[tuple[int, int]],
    objective: int | None,
) -> None:
    if len(schedule) != len(built.instance.tests):
        raise AssertionError("the number of scheduled tests is invalid")
    for index, ((start, machine), test) in enumerate(
        zip(schedule, built.instance.tests)
    ):
        if (
            start < 0
            or start + test.duration > built.horizon
            or machine not in test.eligible
        ):
            raise AssertionError("a test assignment is invalid")
        for second in range(index + 1, len(schedule)):
            other_start, other_machine = schedule[second]
            other = built.instance.tests[second]
            overlap = (
                start < other_start + other.duration
                and other_start < start + test.duration
            )
            if overlap and (
                machine == other_machine or test.resources.intersection(other.resources)
            ):
                raise AssertionError("two incompatible tests overlap")
    makespan = max(
        start + test.duration
        for (start, _), test in zip(schedule, built.instance.tests)
    )
    if objective is not None and makespan != objective:
        raise AssertionError("the objective does not match the makespan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON test-scheduling instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob073 tests={len(instance.tests)} status={solution.status}")
    if not solution.is_sat():
        return 1
    schedule = decode(built, solution)
    validate(built, schedule, solution.objective)
    print(f"makespan={solution.objective} schedule={schedule}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
