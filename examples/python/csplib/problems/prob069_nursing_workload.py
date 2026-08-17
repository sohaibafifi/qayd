"""CSPLib prob069: balanced nursing workload.

Specification: https://www.csplib.org/Problems/prob069/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values

DEFAULT_INSTANCE = {
    "nurses": 2,
    "min_patients": 2,
    "max_patients": 2,
    "max_workload": 8,
    "zones": [[2, 3], [1, 4]],
}


@dataclass(frozen=True)
class NursingInstance:
    nurses: int
    min_patients: int
    max_patients: int
    max_workload: int
    zones: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class NursingModel:
    model: cp.Model
    instance: NursingInstance
    patients: tuple[tuple[int, int], ...]
    assignment: list[cp.IntVar]


def parse_instance(data: str | bytes) -> NursingInstance:
    raw = json.loads(data)
    try:
        return NursingInstance(
            int(raw["nurses"]),
            int(raw["min_patients"]),
            int(raw["max_patients"]),
            int(raw["max_workload"]),
            tuple(tuple(int(value) for value in zone) for zone in raw["zones"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid nursing-workload JSON instance") from error


def load_instance(path: str | Path) -> NursingInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def build_model(instance: NursingInstance) -> NursingModel:
    patients = tuple(
        (zone, workload)
        for zone, workloads in enumerate(instance.zones)
        for workload in workloads
    )
    if instance.nurses < 1 or not patients or any(not zone for zone in instance.zones):
        raise ValueError("nurses, zones, and patients must be non-empty")
    if not 0 <= instance.min_patients <= instance.max_patients:
        raise ValueError("patient-count bounds are invalid")
    if any(
        workload < 0 or workload > instance.max_workload for _, workload in patients
    ):
        raise ValueError("a patient workload is invalid")
    model = cp.Model()
    assignment = model.int_vars(len(patients), 0, instance.nurses - 1, name="nurse")
    model.cardinality(
        assignment,
        list(range(instance.nurses)),
        [instance.min_patients] * instance.nurses,
        [instance.max_patients] * instance.nurses,
        closed=True,
    )
    for first, (left_zone, _) in enumerate(patients):
        for second in range(first + 1, len(patients)):
            if left_zone != patients[second][0]:
                model.add(assignment[first] != assignment[second])

    workload_variables = []
    squared_variables = []
    for nurse in range(instance.nurses):
        membership = []
        for patient in range(len(patients)):
            belongs = model.bool_var(name=f"patient_{patient}_nurse_{nurse}")
            model.table(
                [assignment[patient], belongs],
                [(value, int(value == nurse)) for value in range(instance.nurses)],
            )
            membership.append(belongs)
        workload = model.int_var(0, instance.max_workload, name=f"workload_{nurse}")
        squared = model.int_var(0, instance.max_workload**2, name=f"squared_{nurse}")
        model.add(
            workload
            == sum(
                patients[index][1] * membership[index] for index in range(len(patients))
            )
        )
        model.table(
            [workload, squared],
            [(value, value * value) for value in range(instance.max_workload + 1)],
        )
        workload_variables.append(workload)
        squared_variables.append(squared)
    for nurse in range(instance.nurses - 1):
        model.add(workload_variables[nurse] <= workload_variables[nurse + 1])
    model.minimize(sum(squared_variables))
    return NursingModel(model, instance, patients, assignment)


def decode(built: NursingModel, solution: cp.Solution) -> list[list[int]]:
    nurses = [[] for _ in range(built.instance.nurses)]
    for patient, nurse in enumerate(values(solution, built.assignment)):
        nurses[nurse].append(patient)
    return nurses


def validate(
    built: NursingModel, nurses: list[list[int]], objective: int | None
) -> None:
    assigned = [patient for nurse in nurses for patient in nurse]
    if sorted(assigned) != list(range(len(built.patients))):
        raise AssertionError("patients are not assigned exactly once")
    workloads = []
    for patients in nurses:
        if (
            not built.instance.min_patients
            <= len(patients)
            <= built.instance.max_patients
        ):
            raise AssertionError("a nurse has an invalid number of patients")
        if len({built.patients[patient][0] for patient in patients}) > 1:
            raise AssertionError("a nurse is assigned patients from different zones")
        workload = sum(built.patients[patient][1] for patient in patients)
        if workload > built.instance.max_workload:
            raise AssertionError("a nurse exceeds maximum workload")
        workloads.append(workload)
    if (
        objective is not None
        and sum(workload * workload for workload in workloads) != objective
    ):
        raise AssertionError("the objective does not match squared workloads")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON nursing-workload instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob069 patients={len(built.patients)} status={solution.status}")
    if not solution.is_sat():
        return 1
    nurses = decode(built, solution)
    validate(built, nurses, solution.objective)
    print(f"objective={solution.objective} assignments={nurses}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
