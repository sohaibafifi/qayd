"""CSPLib prob062: interview assignment.

Specification: https://www.csplib.org/Problems/prob062/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args

DEFAULT_INSTANCE = {
    "interviews_per_student": 1,
    "company_capacities": [1, 1, 1],
    "preferences": [[1, 3, 2], [2, 1, 3], [3, 2, 1]],
}


@dataclass(frozen=True)
class InterviewInstance:
    interviews_per_student: int
    company_capacities: tuple[int, ...]
    preferences: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class InterviewModel:
    model: cp.Model
    instance: InterviewInstance
    assignments: list[list[cp.IntVar]]


def parse_instance(data: str | bytes) -> InterviewInstance:
    raw = json.loads(data)
    try:
        return InterviewInstance(
            int(raw["interviews_per_student"]),
            tuple(int(value) for value in raw["company_capacities"]),
            tuple(tuple(int(value) for value in row) for row in raw["preferences"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid interview-assignment JSON instance") from error


def load_instance(path: str | Path) -> InterviewInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def build_model(instance: InterviewInstance) -> InterviewModel:
    company_count = len(instance.company_capacities)
    student_count = len(instance.preferences)
    if (
        company_count < 1
        or student_count < 1
        or not 1 <= instance.interviews_per_student <= company_count
    ):
        raise ValueError("student, company, or interview counts are invalid")
    if any(capacity < 0 for capacity in instance.company_capacities):
        raise ValueError("company capacities must be non-negative")
    if any(
        len(row) != company_count or any(value < 0 for value in row)
        for row in instance.preferences
    ):
        raise ValueError("preference rows are invalid")
    if (
        sum(instance.company_capacities)
        < student_count * instance.interviews_per_student
    ):
        raise ValueError("company capacity is insufficient")
    model = cp.Model()
    assignments = [
        model.int_vars(
            instance.interviews_per_student,
            0,
            company_count - 1,
            name=f"student_{student}",
        )
        for student in range(student_count)
    ]
    membership = [
        [
            model.bool_var(name=f"student_{student}_company_{company}")
            for company in range(company_count)
        ]
        for student in range(student_count)
    ]
    for student, row in enumerate(assignments):
        model.all_different(row)
        for index in range(len(row) - 1):
            model.add(row[index] < row[index + 1])
        for company in range(company_count):
            matches = []
            for interview, assigned in enumerate(row):
                match = model.bool_var(name=f"match_{student}_{interview}_{company}")
                model.table(
                    [assigned, match],
                    [
                        (candidate, int(candidate == company))
                        for candidate in range(company_count)
                    ],
                )
                matches.append(match)
            model.add(membership[student][company] == sum(matches))
    for company, capacity in enumerate(instance.company_capacities):
        model.add(
            sum(membership[student][company] for student in range(student_count))
            <= capacity
        )
    student_costs = []
    maximum_preference = max(value for row in instance.preferences for value in row)
    maximum_student_cost = maximum_preference * instance.interviews_per_student
    for student in range(student_count):
        cost = model.int_var(0, maximum_student_cost, name=f"preference_cost_{student}")
        model.add(
            cost
            == sum(
                instance.preferences[student][company] * membership[student][company]
                for company in range(company_count)
            )
        )
        student_costs.append(cost)
    worst_cost = model.int_var(0, maximum_student_cost, name="worst_student_cost")
    for cost in student_costs:
        model.add(worst_cost >= cost)
    model.minimize(sum(student_costs))
    model.then_minimize(worst_cost)
    return InterviewModel(model, instance, assignments)


def decode(built: InterviewModel, solution: cp.Solution) -> list[list[int]]:
    if not solution.is_sat():
        raise RuntimeError(f"no solution available, solver status is {solution.status}")
    return [[solution.value(variable) for variable in row] for row in built.assignments]


def validate(
    built: InterviewModel,
    assignments: list[list[int]],
    objectives: list[int] | None,
) -> None:
    if len(assignments) != len(built.instance.preferences):
        raise AssertionError("the number of student assignments is invalid")
    company_load = [0] * len(built.instance.company_capacities)
    student_costs = []
    for student, companies in enumerate(assignments):
        if len(companies) != built.instance.interviews_per_student or len(
            set(companies)
        ) != len(companies):
            raise AssertionError("a student interview list is invalid")
        for company in companies:
            company_load[company] += 1
        student_costs.append(
            sum(built.instance.preferences[student][company] for company in companies)
        )
    if any(
        load > capacity
        for load, capacity in zip(company_load, built.instance.company_capacities)
    ):
        raise AssertionError("a company interview capacity is exceeded")
    expected = [sum(student_costs), max(student_costs)]
    if objectives is not None and objectives != expected:
        raise AssertionError("lexicographic objectives do not match preference costs")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON interview-assignment instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob062 students={len(instance.preferences)} status={solution.status}")
    if not solution.is_sat():
        return 1
    assignments = decode(built, solution)
    validate(built, assignments, solution.objectives)
    print(f"objectives={solution.objectives} assignments={assignments}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
