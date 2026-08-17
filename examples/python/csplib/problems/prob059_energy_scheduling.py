"""CSPLib prob059: energy-cost aware scheduling.

Specification: https://www.csplib.org/Problems/prob059/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values

DEFAULT_INSTANCE = {
    "forecast_prices": [5, 2, 1, 3, 6, 2],
    "actual_prices": [4, 3, 1, 4, 5, 2],
    "servers": [
        {"capacities": [3, 3, 2], "idle_power": 1, "startup": 2, "shutdown": 1},
        {"capacities": [2, 2, 2], "idle_power": 1, "startup": 2, "shutdown": 1},
    ],
    "tasks": [
        {"duration": 2, "release": 0, "deadline": 5, "demands": [2, 1, 1], "power": 2},
        {"duration": 2, "release": 1, "deadline": 6, "demands": [1, 2, 1], "power": 1},
        {"duration": 1, "release": 0, "deadline": 6, "demands": [2, 1, 1], "power": 2},
    ],
}


@dataclass(frozen=True)
class EnergyServer:
    capacities: tuple[int, ...]
    idle_power: int
    startup: int
    shutdown: int


@dataclass(frozen=True)
class EnergyTask:
    duration: int
    release: int
    deadline: int
    demands: tuple[int, ...]
    power: int


@dataclass(frozen=True)
class EnergyInstance:
    forecast_prices: tuple[int, ...]
    actual_prices: tuple[int, ...]
    servers: tuple[EnergyServer, ...]
    tasks: tuple[EnergyTask, ...]


@dataclass(frozen=True)
class EnergyModel:
    model: cp.Model
    instance: EnergyInstance
    starts: list[cp.IntVar]
    assignments: list[cp.IntVar]
    on: list[list[cp.IntVar]]


def parse_instance(data: str | bytes) -> EnergyInstance:
    raw = json.loads(data)
    try:
        servers = tuple(
            EnergyServer(
                tuple(int(value) for value in item["capacities"]),
                int(item["idle_power"]),
                int(item["startup"]),
                int(item["shutdown"]),
            )
            for item in raw["servers"]
        )
        tasks = tuple(
            EnergyTask(
                int(item["duration"]),
                int(item["release"]),
                int(item["deadline"]),
                tuple(int(value) for value in item["demands"]),
                int(item["power"]),
            )
            for item in raw["tasks"]
        )
        forecast = tuple(int(value) for value in raw["forecast_prices"])
        return EnergyInstance(
            forecast,
            tuple(int(value) for value in raw.get("actual_prices", forecast)),
            servers,
            tasks,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid energy-scheduling JSON instance") from error


def load_instance(path: str | Path) -> EnergyInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def build_model(instance: EnergyInstance) -> EnergyModel:
    horizon = len(instance.forecast_prices)
    if (
        horizon < 1
        or len(instance.actual_prices) != horizon
        or not instance.servers
        or not instance.tasks
    ):
        raise ValueError("prices, servers, and tasks must be non-empty")
    resource_count = len(instance.servers[0].capacities)
    if resource_count < 1 or any(
        len(server.capacities) != resource_count for server in instance.servers
    ):
        raise ValueError("server capacity vectors are inconsistent")
    if any(price < 0 for price in (*instance.forecast_prices, *instance.actual_prices)):
        raise ValueError("energy prices must be non-negative")
    if any(
        task.duration < 1
        or task.release < 0
        or task.deadline > horizon
        or task.release + task.duration > task.deadline
        or len(task.demands) != resource_count
        for task in instance.tasks
    ):
        raise ValueError("a task is invalid")
    model = cp.Model()
    starts = [
        model.int_var(
            task.release, task.deadline - task.duration, name=f"start_{index}"
        )
        for index, task in enumerate(instance.tasks)
    ]
    assignments = model.int_vars(
        len(instance.tasks), 0, len(instance.servers) - 1, name="server"
    )
    active = [
        [
            [
                model.bool_var(name=f"active_{task}_{server}_{time}")
                for time in range(horizon)
            ]
            for server in range(len(instance.servers))
        ]
        for task in range(len(instance.tasks))
    ]
    on = [
        [model.bool_var(name=f"on_{server}_{time}") for time in range(horizon)]
        for server in range(len(instance.servers))
    ]
    for task, item in enumerate(instance.tasks):
        compatible = [
            server
            for server, machine in enumerate(instance.servers)
            if all(
                demand <= capacity
                for demand, capacity in zip(item.demands, machine.capacities)
            )
        ]
        if not compatible:
            raise ValueError(f"task {task} fits no server")
        model.table([assignments[task]], [(server,) for server in compatible])
        for server in range(len(instance.servers)):
            for time in range(horizon):
                model.table(
                    [starts[task], assignments[task], active[task][server][time]],
                    [
                        (
                            start,
                            candidate,
                            int(
                                candidate == server
                                and start <= time < start + item.duration
                            ),
                        )
                        for start in range(
                            item.release, item.deadline - item.duration + 1
                        )
                        for candidate in compatible
                    ],
                )
    objective_terms = []
    transition_table = [
        (
            previous,
            current,
            int(previous == 0 and current == 1),
            int(previous == 1 and current == 0),
        )
        for previous in range(2)
        for current in range(2)
    ]
    for server, machine in enumerate(instance.servers):
        for time in range(horizon):
            running = [
                active[task][server][time] for task in range(len(instance.tasks))
            ]
            model.add(sum(running) <= len(instance.tasks) * on[server][time])
            model.add(on[server][time] <= sum(running))
            for resource, capacity in enumerate(machine.capacities):
                model.add(
                    sum(
                        instance.tasks[task].demands[resource] * running[task]
                        for task in range(len(running))
                    )
                    <= capacity
                )
            objective_terms.append(
                instance.forecast_prices[time] * machine.idle_power * on[server][time]
            )
            objective_terms.extend(
                instance.forecast_prices[time]
                * instance.tasks[task].power
                * running[task]
                for task in range(len(instance.tasks))
            )
        boundaries = [None, *on[server], None]
        for boundary in range(horizon + 1):
            startup = model.bool_var(name=f"startup_{server}_{boundary}")
            shutdown = model.bool_var(name=f"shutdown_{server}_{boundary}")
            previous = boundaries[boundary]
            current = boundaries[boundary + 1]
            if previous is None:
                model.table(
                    [current, startup, shutdown],
                    [(value, int(value == 1), 0) for value in range(2)],
                )
            elif current is None:
                model.table(
                    [previous, startup, shutdown],
                    [(value, 0, int(value == 1)) for value in range(2)],
                )
            else:
                model.table([previous, current, startup, shutdown], transition_table)
            objective_terms.append(
                machine.startup * startup + machine.shutdown * shutdown
            )
    model.minimize(sum(objective_terms))
    return EnergyModel(model, instance, starts, assignments, on)


def decode(
    built: EnergyModel, solution: cp.Solution
) -> tuple[list[int], list[int], list[list[int]]]:
    return (
        values(solution, built.starts),
        values(solution, built.assignments),
        [values(solution, row) for row in built.on],
    )


def schedule_cost(
    instance: EnergyInstance,
    result: tuple[list[int], list[int], list[list[int]]],
    actual: bool = False,
) -> int:
    starts, assignments, on = result
    prices = instance.actual_prices if actual else instance.forecast_prices
    total = 0
    for server, machine in enumerate(instance.servers):
        for time in range(len(prices)):
            total += prices[time] * machine.idle_power * on[server][time]
            total += sum(
                prices[time] * task.power
                for start, assigned, task in zip(starts, assignments, instance.tasks)
                if assigned == server and start <= time < start + task.duration
            )
        previous = 0
        for current in [*on[server], 0]:
            total += machine.startup if previous == 0 and current == 1 else 0
            total += machine.shutdown if previous == 1 and current == 0 else 0
            previous = current
    return total


def validate(
    built: EnergyModel,
    result: tuple[list[int], list[int], list[list[int]]],
    objective: int | None,
) -> None:
    starts, assignments, on = result
    horizon = len(built.instance.forecast_prices)
    for time in range(horizon):
        for server, machine in enumerate(built.instance.servers):
            active = [
                task
                for task, (start, assigned) in enumerate(zip(starts, assignments))
                if assigned == server
                and start <= time < start + built.instance.tasks[task].duration
            ]
            if on[server][time] != int(bool(active)):
                raise AssertionError("a server power state is inconsistent")
            for resource, capacity in enumerate(machine.capacities):
                if (
                    sum(built.instance.tasks[task].demands[resource] for task in active)
                    > capacity
                ):
                    raise AssertionError("a server capacity is exceeded")
    if objective is not None and schedule_cost(built.instance, result) != objective:
        raise AssertionError("the objective does not match forecast energy cost")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON energy-scheduling instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob059 tasks={len(instance.tasks)} status={solution.status}")
    if not solution.is_sat():
        return 1
    result = decode(built, solution)
    validate(built, result, solution.objective)
    print(
        f"forecast_cost={solution.objective} actual_cost={schedule_cost(instance, result, actual=True)} starts={result[0]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
