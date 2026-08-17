"""CSPLib prob064: generalized balanced academic curriculum.

Specification: https://www.csplib.org/Problems/prob064/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values

DEFAULT_INSTANCE = {
    "credits": [3, 2, 4, 3, 2],
    "periods": 3,
    "min_courses": 1,
    "max_courses": 2,
    "prerequisites": [[0, 3], [1, 4]],
    "curricula": [[0, 1, 3], [1, 2, 4]],
    "undesired": [{"course": 2, "period": 0, "penalty": 3}],
}


@dataclass(frozen=True)
class UndesiredPeriod:
    course: int
    period: int
    penalty: int


@dataclass(frozen=True)
class GeneralizedBacpInstance:
    credits: tuple[int, ...]
    periods: int
    min_courses: int
    max_courses: int
    prerequisites: tuple[tuple[int, int], ...]
    curricula: tuple[tuple[int, ...], ...]
    undesired: tuple[UndesiredPeriod, ...]


@dataclass(frozen=True)
class GeneralizedBacpModel:
    model: cp.Model
    instance: GeneralizedBacpInstance
    assigned_period: list[cp.IntVar]


def parse_instance(data: str | bytes) -> GeneralizedBacpInstance:
    raw = json.loads(data)
    try:
        return GeneralizedBacpInstance(
            tuple(int(value) for value in raw["credits"]),
            int(raw["periods"]),
            int(raw["min_courses"]),
            int(raw["max_courses"]),
            tuple(
                (int(pair[0]), int(pair[1])) for pair in raw.get("prerequisites", [])
            ),
            tuple(
                tuple(int(course) for course in curriculum)
                for curriculum in raw["curricula"]
            ),
            tuple(
                UndesiredPeriod(
                    int(item["course"]), int(item["period"]), int(item["penalty"])
                )
                for item in raw.get("undesired", [])
            ),
        )
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError("invalid generalized-BACP JSON instance") from error


def load_instance(path: str | Path) -> GeneralizedBacpInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def build_model(instance: GeneralizedBacpInstance) -> GeneralizedBacpModel:
    course_count = len(instance.credits)
    if course_count < 1 or instance.periods < 1 or not instance.curricula:
        raise ValueError("courses, periods, and curricula must be non-empty")
    if not 0 <= instance.min_courses <= instance.max_courses:
        raise ValueError("course-count bounds are invalid")
    if any(credit < 1 for credit in instance.credits):
        raise ValueError("credits must be positive")
    if any(
        not curriculum or min(curriculum) < 0 or max(curriculum) >= course_count
        for curriculum in instance.curricula
    ):
        raise ValueError("a curriculum is invalid")
    if any(
        before < 0 or before >= course_count or after < 0 or after >= course_count
        for before, after in instance.prerequisites
    ):
        raise ValueError("a prerequisite is invalid")
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
                [(value, int(value == period)) for value in range(instance.periods)],
            )
    for period in range(instance.periods):
        count = sum(membership[course][period] for course in range(course_count))
        model.add(count >= instance.min_courses)
        model.add(count <= instance.max_courses)
    for before, after in instance.prerequisites:
        model.add(assigned_period[before] < assigned_period[after])

    maximum_curriculum_load = model.int_var(
        0, sum(instance.credits), name="maximum_curriculum_load"
    )
    for curriculum in instance.curricula:
        for period in range(instance.periods):
            load = sum(
                instance.credits[course] * membership[course][period]
                for course in curriculum
            )
            model.add(maximum_curriculum_load >= load)
    penalties = []
    for index, preference in enumerate(instance.undesired):
        if (
            preference.course < 0
            or preference.course >= course_count
            or preference.period < 0
            or preference.period >= instance.periods
            or preference.penalty < 0
        ):
            raise ValueError("an undesired-period preference is invalid")
        violated = model.bool_var(name=f"preference_{index}")
        model.table(
            [assigned_period[preference.course], violated],
            [
                (value, int(value == preference.period))
                for value in range(instance.periods)
            ],
        )
        penalties.append(preference.penalty * violated)
    total_penalty_bound = sum(preference.penalty for preference in instance.undesired)
    model.minimize(maximum_curriculum_load * (total_penalty_bound + 1) + sum(penalties))
    return GeneralizedBacpModel(model, instance, assigned_period)


def decode(built: GeneralizedBacpModel, solution: cp.Solution) -> list[list[int]]:
    periods = [[] for _ in range(built.instance.periods)]
    for course, period in enumerate(values(solution, built.assigned_period)):
        periods[period].append(course)
    return periods


def objective_value(instance: GeneralizedBacpInstance, periods: list[list[int]]) -> int:
    course_period = {
        course: period for period, courses in enumerate(periods) for course in courses
    }
    maximum_load = max(
        sum(
            instance.credits[course]
            for course in curriculum
            if course_period[course] == period
        )
        for curriculum in instance.curricula
        for period in range(instance.periods)
    )
    penalty = sum(
        preference.penalty
        for preference in instance.undesired
        if course_period[preference.course] == preference.period
    )
    return (
        maximum_load * (sum(item.penalty for item in instance.undesired) + 1) + penalty
    )


def validate(
    built: GeneralizedBacpModel, periods: list[list[int]], objective: int | None
) -> None:
    courses = [course for period in periods for course in period]
    if sorted(courses) != list(range(len(built.instance.credits))):
        raise AssertionError("courses are not assigned exactly once")
    if any(
        not built.instance.min_courses <= len(period) <= built.instance.max_courses
        for period in periods
    ):
        raise AssertionError("a period violates course-count bounds")
    course_period = {
        course: period for period, courses in enumerate(periods) for course in courses
    }
    if any(
        course_period[before] >= course_period[after]
        for before, after in built.instance.prerequisites
    ):
        raise AssertionError("a prerequisite is not scheduled earlier")
    if objective is not None and objective_value(built.instance, periods) != objective:
        raise AssertionError(
            "the objective does not match curriculum balance and penalties"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON generalized-BACP instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob064 courses={len(instance.credits)} status={solution.status}")
    if not solution.is_sat():
        return 1
    periods = decode(built, solution)
    validate(built, periods, solution.objective)
    print(f"objective={solution.objective} periods={periods}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
