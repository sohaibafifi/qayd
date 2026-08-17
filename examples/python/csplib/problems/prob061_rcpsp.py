"""CSPLib prob061: resource-constrained project scheduling.

Specification: https://www.csplib.org/Problems/prob061/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values

DEFAULT_INSTANCE = {
    "capacities": [3],
    "jobs": [
        {"duration": 0, "demands": [0], "successors": [1, 2]},
        {"duration": 2, "demands": [2], "successors": [3]},
        {"duration": 3, "demands": [1], "successors": [3]},
        {"duration": 0, "demands": [0], "successors": []},
    ],
}


@dataclass(frozen=True)
class RcpspJob:
    duration: int
    demands: tuple[int, ...]
    successors: tuple[int, ...]


@dataclass(frozen=True)
class RcpspInstance:
    capacities: tuple[int, ...]
    jobs: tuple[RcpspJob, ...]


@dataclass(frozen=True)
class RcpspModel:
    model: cp.Model
    instance: RcpspInstance
    horizon: int
    starts: list[cp.IntVar]


def parse_instance(data: str | bytes) -> RcpspInstance:
    raw = json.loads(data)
    try:
        jobs = tuple(
            RcpspJob(
                int(job["duration"]),
                tuple(int(value) for value in job["demands"]),
                tuple(int(value) for value in job.get("successors", [])),
            )
            for job in raw["jobs"]
        )
        return RcpspInstance(tuple(int(value) for value in raw["capacities"]), jobs)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid RCPSP JSON instance") from error


def load_instance(path: str | Path) -> RcpspInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def build_model(instance: RcpspInstance) -> RcpspModel:
    resource_count = len(instance.capacities)
    job_count = len(instance.jobs)
    if (
        resource_count < 1
        or job_count < 1
        or any(capacity < 1 for capacity in instance.capacities)
    ):
        raise ValueError("resources and jobs must be non-empty")
    if any(
        job.duration < 0 or len(job.demands) != resource_count for job in instance.jobs
    ):
        raise ValueError("a job has invalid duration or resource demands")
    if any(
        demand < 0 or demand > instance.capacities[resource]
        for job in instance.jobs
        for resource, demand in enumerate(job.demands)
    ):
        raise ValueError("a job resource demand is invalid")
    if any(
        successor < 0 or successor >= job_count
        for job in instance.jobs
        for successor in job.successors
    ):
        raise ValueError("a successor index is invalid")
    horizon = max(1, sum(job.duration for job in instance.jobs))
    model = cp.Model()
    starts = [
        model.int_var(0, horizon - job.duration, name=f"start_{index}")
        for index, job in enumerate(instance.jobs)
    ]
    for index, job in enumerate(instance.jobs):
        for successor in job.successors:
            model.add(starts[index] + job.duration <= starts[successor])

    for time in range(horizon):
        active = []
        for index, job in enumerate(instance.jobs):
            variable = model.bool_var(name=f"active_{index}_{time}")
            model.table(
                [starts[index], variable],
                [
                    (start, int(start <= time < start + job.duration))
                    for start in range(horizon - job.duration + 1)
                ],
            )
            active.append(variable)
        for resource, capacity in enumerate(instance.capacities):
            model.add(
                sum(
                    instance.jobs[index].demands[resource] * active[index]
                    for index in range(job_count)
                )
                <= capacity
            )
    makespan = model.int_var(0, horizon, name="makespan")
    for index, job in enumerate(instance.jobs):
        model.add(makespan >= starts[index] + job.duration)
    model.minimize(makespan)
    return RcpspModel(model, instance, horizon, starts)


def decode(built: RcpspModel, solution: cp.Solution) -> list[int]:
    return values(solution, built.starts)


def validate(built: RcpspModel, starts: list[int], objective: int | None) -> None:
    if len(starts) != len(built.instance.jobs):
        raise AssertionError("the number of start times is invalid")
    for index, job in enumerate(built.instance.jobs):
        if starts[index] < 0 or starts[index] + job.duration > built.horizon:
            raise AssertionError("a job lies outside the scheduling horizon")
        if any(
            starts[index] + job.duration > starts[successor]
            for successor in job.successors
        ):
            raise AssertionError("a precedence constraint is violated")
    for time in range(built.horizon):
        for resource, capacity in enumerate(built.instance.capacities):
            usage = sum(
                job.demands[resource]
                for start, job in zip(starts, built.instance.jobs)
                if start <= time < start + job.duration
            )
            if usage > capacity:
                raise AssertionError("a resource capacity is exceeded")
    makespan = max(
        start + job.duration for start, job in zip(starts, built.instance.jobs)
    )
    if objective is not None and makespan != objective:
        raise AssertionError("the objective does not match the makespan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON RCPSP instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob061 jobs={len(instance.jobs)} status={solution.status}")
    if not solution.is_sat():
        return 1
    starts = decode(built, solution)
    validate(built, starts, solution.objective)
    print(f"makespan={solution.objective} starts={starts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
