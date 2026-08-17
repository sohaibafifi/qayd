"""CSPLib prob030: balanced academic curriculum problem.

Specification: https://www.csplib.org/Problems/prob030/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values

DEFAULT_INSTANCE = {
    "credits": [3, 2, 4, 3, 2, 4],
    "periods": 3,
    "min_load": 4,
    "max_load": 8,
    "min_courses": 1,
    "max_courses": 3,
    "prerequisites": [[0, 3], [1, 4], [2, 5]],
}


@dataclass(frozen=True)
class BacpInstance:
    credits: tuple[int, ...]
    periods: int
    min_load: int
    max_load: int
    min_courses: int
    max_courses: int
    prerequisites: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class BacpModel:
    model: cp.Model
    instance: BacpInstance
    assigned_period: list[cp.IntVar]
    loads: list[cp.IntVar]


def parse_instance(data: str | bytes) -> BacpInstance:
    raw = json.loads(data)
    try:
        return BacpInstance(
            tuple(int(value) for value in raw["credits"]),
            int(raw["periods"]),
            int(raw["min_load"]),
            int(raw["max_load"]),
            int(raw["min_courses"]),
            int(raw["max_courses"]),
            tuple(
                (int(pair[0]), int(pair[1])) for pair in raw.get("prerequisites", [])
            ),
        )
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError("invalid BACP JSON instance") from error


def load_instance(path: str | Path) -> BacpInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def build_model(instance: BacpInstance) -> BacpModel:
    course_count = len(instance.credits)
    if (
        course_count < 1
        or instance.periods < 1
        or any(credit < 1 for credit in instance.credits)
    ):
        raise ValueError("credits and periods must be positive")
    if not (0 <= instance.min_load <= instance.max_load):
        raise ValueError("load bounds are invalid")
    if not (0 <= instance.min_courses <= instance.max_courses):
        raise ValueError("course-count bounds are invalid")
    if any(
        before < 0 or after < 0 or before >= course_count or after >= course_count
        for before, after in instance.prerequisites
    ):
        raise ValueError("a prerequisite pair is invalid")

    model = cp.Model()
    assigned_period = model.int_vars(
        course_count, 0, instance.periods - 1, name="period"
    )
    membership = [
        [
            model.bool_var(name=f"course_{course}_period_{period}")
            for period in range(instance.periods)
        ]
        for course in range(course_count)
    ]
    for course in range(course_count):
        for period in range(instance.periods):
            model.table(
                [assigned_period[course], membership[course][period]],
                [(p, int(p == period)) for p in range(instance.periods)],
            )

    loads = model.int_vars(
        instance.periods, instance.min_load, instance.max_load, name="load"
    )
    for period in range(instance.periods):
        count = sum(membership[course][period] for course in range(course_count))
        model.add(count >= instance.min_courses)
        model.add(count <= instance.max_courses)
        model.add(
            loads[period]
            == sum(
                instance.credits[course] * membership[course][period]
                for course in range(course_count)
            )
        )
    for before, after in instance.prerequisites:
        model.add(assigned_period[before] < assigned_period[after])
    maximum_load = model.int_var(
        instance.min_load, instance.max_load, name="maximum_load"
    )
    for load in loads:
        model.add(maximum_load >= load)
    model.minimize(maximum_load)
    return BacpModel(model, instance, assigned_period, loads)


def decode(built: BacpModel, solution: cp.Solution) -> list[list[int]]:
    periods = [[] for _ in range(built.instance.periods)]
    for course, period in enumerate(values(solution, built.assigned_period)):
        periods[period].append(course)
    return periods


def validate(built: BacpModel, periods: list[list[int]], objective: int | None) -> None:
    courses = [course for period in periods for course in period]
    if sorted(courses) != list(range(len(built.instance.credits))):
        raise AssertionError("courses are not assigned exactly once")
    course_period = {
        course: period
        for period, courses_in_period in enumerate(periods)
        for course in courses_in_period
    }
    loads = [
        sum(built.instance.credits[course] for course in courses_in_period)
        for courses_in_period in periods
    ]
    for courses_in_period, load in zip(periods, loads):
        if (
            not built.instance.min_courses
            <= len(courses_in_period)
            <= built.instance.max_courses
        ):
            raise AssertionError("a period has an invalid number of courses")
        if not built.instance.min_load <= load <= built.instance.max_load:
            raise AssertionError("a period has an invalid credit load")
    if any(
        course_period[before] >= course_period[after]
        for before, after in built.instance.prerequisites
    ):
        raise AssertionError("a prerequisite is not scheduled earlier")
    if objective is not None and max(loads) != objective:
        raise AssertionError("the objective does not match the maximum load")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON BACP instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob030 courses={len(instance.credits)} status={solution.status}")
    if not solution.is_sat():
        return 1
    periods = decode(built, solution)
    validate(built, periods, solution.objective)
    print(f"maximum_load={solution.objective} periods={periods}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
