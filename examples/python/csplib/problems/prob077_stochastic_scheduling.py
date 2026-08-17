"""CSPLib prob077: stochastic assignment and scheduling.

Specification: https://www.csplib.org/Problems/prob077/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values

DEFAULT_INSTANCE = {
    "scenario_weights": [1, 1],
    "durations": [
        [[2, 3], [3, 2]],
        [[3, 2], [2, 4]],
        [[2, 2], [4, 3]],
    ],
    "precedences": [[0, 2], [1, 2]],
}


@dataclass(frozen=True)
class StochasticInstance:
    scenario_weights: tuple[int, ...]
    durations: tuple[tuple[tuple[int, ...], ...], ...]
    precedences: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class StochasticModel:
    model: cp.Model
    instance: StochasticInstance
    assignments: list[cp.IntVar]
    starts: list[list[cp.IntVar]]


def parse_instance(data: str | bytes) -> StochasticInstance:
    raw = json.loads(data)
    try:
        return StochasticInstance(
            tuple(int(value) for value in raw["scenario_weights"]),
            tuple(
                tuple(
                    tuple(int(value) for value in scenario_values)
                    for scenario_values in machine_values
                )
                for machine_values in raw["durations"]
            ),
            tuple((int(pair[0]), int(pair[1])) for pair in raw.get("precedences", [])),
        )
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError("invalid stochastic-scheduling JSON instance") from error


def load_instance(path: str | Path) -> StochasticInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def build_model(instance: StochasticInstance) -> StochasticModel:
    task_count = len(instance.durations)
    scenario_count = len(instance.scenario_weights)
    machine_count = len(instance.durations[0]) if task_count else 0
    if (
        task_count < 1
        or scenario_count < 1
        or machine_count < 1
        or any(weight < 1 for weight in instance.scenario_weights)
    ):
        raise ValueError("tasks, machines, scenarios, and weights must be non-empty")
    if any(
        len(task) != machine_count
        or any(len(machine) != scenario_count for machine in task)
        for task in instance.durations
    ):
        raise ValueError("duration dimensions are inconsistent")
    if any(
        before < 0 or before >= task_count or after < 0 or after >= task_count
        for before, after in instance.precedences
    ):
        raise ValueError("a precedence index is invalid")
    compatible = [
        [
            machine
            for machine in range(machine_count)
            if all(value > 0 for value in instance.durations[task][machine])
        ]
        for task in range(task_count)
    ]
    if any(not machines for machines in compatible):
        raise ValueError("a task has no compatible machine")
    horizon = sum(
        max(instance.durations[task][machine][scenario] for machine in compatible[task])
        for task in range(task_count)
        for scenario in range(scenario_count)
    )
    model = cp.Model()
    assignments = model.int_vars(task_count, 0, machine_count - 1, name="machine")
    starts = [
        [
            model.int_var(0, horizon, name=f"start_{scenario}_{task}")
            for task in range(task_count)
        ]
        for scenario in range(scenario_count)
    ]
    duration_variables = [
        [None for _ in range(task_count)] for _ in range(scenario_count)
    ]
    for task in range(task_count):
        model.table([assignments[task]], [(machine,) for machine in compatible[task]])
        for scenario in range(scenario_count):
            duration = model.int_var(
                1,
                max(
                    instance.durations[task][machine][scenario]
                    for machine in compatible[task]
                ),
                name=f"duration_{scenario}_{task}",
            )
            model.table(
                [assignments[task], duration],
                [
                    (machine, instance.durations[task][machine][scenario])
                    for machine in compatible[task]
                ],
            )
            duration_variables[scenario][task] = duration
    makespans = []
    for scenario in range(scenario_count):
        for before, after in instance.precedences:
            model.add(
                starts[scenario][before] + duration_variables[scenario][before]
                <= starts[scenario][after]
            )
        for first in range(task_count):
            for second in range(first + 1, task_count):
                model.add(
                    (assignments[first] == assignments[second]).implies(
                        (
                            starts[scenario][first]
                            + duration_variables[scenario][first]
                            <= starts[scenario][second]
                        )
                        | (
                            starts[scenario][second]
                            + duration_variables[scenario][second]
                            <= starts[scenario][first]
                        )
                    )
                )
        makespan = model.int_var(0, horizon, name=f"makespan_{scenario}")
        for task in range(task_count):
            model.add(
                makespan >= starts[scenario][task] + duration_variables[scenario][task]
            )
        makespans.append(makespan)
    model.minimize(
        sum(
            weight * makespan
            for weight, makespan in zip(instance.scenario_weights, makespans)
        )
    )
    return StochasticModel(model, instance, assignments, starts)


def decode(
    built: StochasticModel, solution: cp.Solution
) -> tuple[list[int], list[list[int]]]:
    return values(solution, built.assignments), [
        values(solution, row) for row in built.starts
    ]


def validate(
    built: StochasticModel,
    result: tuple[list[int], list[list[int]]],
    objective: int | None,
) -> None:
    assignments, starts = result
    makespans = []
    for scenario, scenario_starts in enumerate(starts):
        durations = [
            built.instance.durations[task][assignments[task]][scenario]
            for task in range(len(assignments))
        ]
        for before, after in built.instance.precedences:
            if scenario_starts[before] + durations[before] > scenario_starts[after]:
                raise AssertionError("a scenario precedence is violated")
        for first in range(len(assignments)):
            for second in range(first + 1, len(assignments)):
                if assignments[first] == assignments[second] and not (
                    scenario_starts[first] + durations[first] <= scenario_starts[second]
                    or scenario_starts[second] + durations[second]
                    <= scenario_starts[first]
                ):
                    raise AssertionError("two scenario tasks overlap on a machine")
        makespans.append(
            max(start + duration for start, duration in zip(scenario_starts, durations))
        )
    expected = sum(
        weight * makespan
        for weight, makespan in zip(built.instance.scenario_weights, makespans)
    )
    if objective is not None and expected != objective:
        raise AssertionError("the objective does not match weighted scenario makespans")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON stochastic-scheduling instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob077 tasks={len(instance.durations)} status={solution.status}")
    if not solution.is_sat():
        return 1
    result = decode(built, solution)
    validate(built, result, solution.objective)
    print(
        f"weighted_makespan={solution.objective} machines={result[0]} starts={result[1]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
