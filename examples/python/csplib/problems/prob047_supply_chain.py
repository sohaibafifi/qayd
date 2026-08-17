"""CSPLib prob047: supply chain coordination.

Specification: https://www.csplib.org/Problems/prob047/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args

DEFAULT_INSTANCE = {
    "horizon": 3,
    "items": ["raw", "component", "finished"],
    "agents": [
        {
            "capacity": [3, 3, 3],
            "initial_inventory": [0, 0, 0],
            "external_supply": [[3, 3, 0], [0, 0, 0], [0, 0, 0]],
            "holding_costs": [0, 1, 1],
            "processes": [
                {
                    "output": 1,
                    "quantity": 1,
                    "inputs": {"0": 1},
                    "capacity": 1,
                    "setup_cost": 1,
                    "unit_cost": 1,
                }
            ],
        },
        {
            "capacity": [2, 2, 2],
            "initial_inventory": [0, 0, 0],
            "external_supply": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            "holding_costs": [0, 1, 1],
            "processes": [
                {
                    "output": 2,
                    "quantity": 1,
                    "inputs": {"1": 1},
                    "capacity": 1,
                    "setup_cost": 2,
                    "unit_cost": 1,
                }
            ],
        },
    ],
    "arcs": [
        {"source": 0, "target": 1, "item": 1, "capacity": 3, "lead": 0, "unit_cost": 1}
    ],
    "demands": [{"agent": 1, "item": 2, "quantities": [0, 1, 2], "penalty": 10}],
}


@dataclass(frozen=True)
class Process:
    output: int
    quantity: int
    inputs: tuple[tuple[int, int], ...]
    capacity: int
    setup_cost: int
    unit_cost: int


@dataclass(frozen=True)
class Agent:
    capacity: tuple[int, ...]
    initial_inventory: tuple[int, ...]
    external_supply: tuple[tuple[int, ...], ...]
    holding_costs: tuple[int, ...]
    processes: tuple[Process, ...]


@dataclass(frozen=True)
class Arc:
    source: int
    target: int
    item: int
    capacity: int
    lead: int
    unit_cost: int


@dataclass(frozen=True)
class Demand:
    agent: int
    item: int
    quantities: tuple[int, ...]
    penalty: int


@dataclass(frozen=True)
class SupplyChainInstance:
    horizon: int
    items: tuple[str, ...]
    agents: tuple[Agent, ...]
    arcs: tuple[Arc, ...]
    demands: tuple[Demand, ...]


@dataclass(frozen=True)
class SupplyChainModel:
    model: cp.Model
    instance: SupplyChainInstance
    production: list[list[list[cp.IntVar]]]
    transfers: list[list[cp.IntVar]]
    deliveries: list[list[cp.IntVar]]
    inventories: list[list[list[cp.IntVar]]]


def parse_instance(data: str | bytes) -> SupplyChainInstance:
    raw = json.loads(data)
    try:
        agents = []
        for item in raw["agents"]:
            processes = tuple(
                Process(
                    int(process["output"]),
                    int(process.get("quantity", 1)),
                    tuple(
                        sorted(
                            (int(key), int(value))
                            for key, value in process.get("inputs", {}).items()
                        )
                    ),
                    int(process.get("capacity", 1)),
                    int(process.get("setup_cost", 0)),
                    int(process.get("unit_cost", 0)),
                )
                for process in item.get("processes", [])
            )
            agents.append(
                Agent(
                    tuple(int(value) for value in item["capacity"]),
                    tuple(int(value) for value in item["initial_inventory"]),
                    tuple(
                        tuple(int(value) for value in row)
                        for row in item["external_supply"]
                    ),
                    tuple(int(value) for value in item["holding_costs"]),
                    processes,
                )
            )
        arcs = tuple(
            Arc(
                int(item["source"]),
                int(item["target"]),
                int(item["item"]),
                int(item["capacity"]),
                int(item.get("lead", 0)),
                int(item.get("unit_cost", 0)),
            )
            for item in raw.get("arcs", [])
        )
        demands = tuple(
            Demand(
                int(item["agent"]),
                int(item["item"]),
                tuple(int(value) for value in item["quantities"]),
                int(item["penalty"]),
            )
            for item in raw.get("demands", [])
        )
        return SupplyChainInstance(
            int(raw["horizon"]),
            tuple(str(value) for value in raw["items"]),
            tuple(agents),
            arcs,
            demands,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid supply-chain JSON instance") from error


def load_instance(path: str | Path) -> SupplyChainInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def build_model(instance: SupplyChainInstance) -> SupplyChainModel:
    item_count = len(instance.items)
    agent_count = len(instance.agents)
    if instance.horizon < 1 or item_count < 1 or agent_count < 1:
        raise ValueError("horizon, items, and agents must be non-empty")
    for agent in instance.agents:
        if (
            len(agent.capacity) != instance.horizon
            or len(agent.initial_inventory) != item_count
        ):
            raise ValueError(
                "an agent has invalid capacity or initial inventory dimensions"
            )
        if len(agent.external_supply) != item_count or any(
            len(row) != instance.horizon for row in agent.external_supply
        ):
            raise ValueError("an external-supply matrix is invalid")
        if len(agent.holding_costs) != item_count:
            raise ValueError("an agent holding-cost vector is invalid")
    if any(len(demand.quantities) != instance.horizon for demand in instance.demands):
        raise ValueError("a demand horizon is invalid")
    bound = sum(
        sum(agent.initial_inventory) + sum(sum(row) for row in agent.external_supply)
        for agent in instance.agents
    ) + sum(sum(demand.quantities) for demand in instance.demands)
    bound = max(1, bound)
    model = cp.Model()
    production = []
    objective_terms = []
    for agent_index, agent in enumerate(instance.agents):
        agent_rows = []
        for process_index, process in enumerate(agent.processes):
            if (
                process.output < 0
                or process.output >= item_count
                or process.quantity < 1
                or process.capacity < 1
            ):
                raise ValueError("a production process is invalid")
            row = model.int_vars(
                instance.horizon,
                0,
                process.capacity,
                name=f"production_{agent_index}_{process_index}",
            )
            for time, amount in enumerate(row):
                setup = model.bool_var(
                    name=f"setup_{agent_index}_{process_index}_{time}"
                )
                model.table(
                    [amount, setup],
                    [(value, int(value > 0)) for value in range(process.capacity + 1)],
                )
                objective_terms.append(
                    process.setup_cost * setup + process.unit_cost * amount
                )
            agent_rows.append(row)
        for time in range(instance.horizon):
            model.add(
                sum(
                    process.capacity * agent_rows[index][time]
                    for index, process in enumerate(agent.processes)
                )
                <= agent.capacity[time]
            )
        production.append(agent_rows)
    transfers = [
        model.int_vars(instance.horizon, 0, arc.capacity, name=f"transfer_{index}")
        for index, arc in enumerate(instance.arcs)
    ]
    for index, arc in enumerate(instance.arcs):
        if not (
            0 <= arc.source < agent_count
            and 0 <= arc.target < agent_count
            and 0 <= arc.item < item_count
        ):
            raise ValueError("a supply-chain arc is invalid")
        objective_terms.extend(arc.unit_cost * amount for amount in transfers[index])
    deliveries = []
    for demand_index, demand in enumerate(instance.demands):
        if not (
            0 <= demand.agent < agent_count
            and 0 <= demand.item < item_count
            and demand.penalty >= 0
        ):
            raise ValueError("a customer demand is invalid")
        row = [
            model.int_var(0, quantity, name=f"delivery_{demand_index}_{time}")
            for time, quantity in enumerate(demand.quantities)
        ]
        objective_terms.extend(
            demand.penalty * (quantity - delivered)
            for quantity, delivered in zip(demand.quantities, row)
        )
        deliveries.append(row)
    inventories = [
        [
            model.int_vars(instance.horizon, 0, bound, name=f"inventory_{agent}_{item}")
            for item in range(item_count)
        ]
        for agent in range(agent_count)
    ]
    for agent_index, agent in enumerate(instance.agents):
        for item in range(item_count):
            for time in range(instance.horizon):
                previous = (
                    agent.initial_inventory[item]
                    if time == 0
                    else inventories[agent_index][item][time - 1]
                )
                produced = sum(
                    process.quantity * production[agent_index][process_index][time]
                    for process_index, process in enumerate(agent.processes)
                    if process.output == item
                )
                consumed = sum(
                    quantity * production[agent_index][process_index][time]
                    for process_index, process in enumerate(agent.processes)
                    for input_item, quantity in process.inputs
                    if input_item == item
                )
                incoming = sum(
                    transfers[arc_index][time - arc.lead]
                    for arc_index, arc in enumerate(instance.arcs)
                    if arc.target == agent_index
                    and arc.item == item
                    and time >= arc.lead
                )
                outgoing = sum(
                    transfers[arc_index][time]
                    for arc_index, arc in enumerate(instance.arcs)
                    if arc.source == agent_index and arc.item == item
                )
                delivered = sum(
                    deliveries[demand_index][time]
                    for demand_index, demand in enumerate(instance.demands)
                    if demand.agent == agent_index and demand.item == item
                )
                model.add(
                    inventories[agent_index][item][time]
                    == previous
                    + agent.external_supply[item][time]
                    + produced
                    + incoming
                    - consumed
                    - outgoing
                    - delivered
                )
                objective_terms.append(
                    agent.holding_costs[item] * inventories[agent_index][item][time]
                )
    model.minimize(sum(objective_terms))
    return SupplyChainModel(
        model, instance, production, transfers, deliveries, inventories
    )


def decode(built: SupplyChainModel, solution: cp.Solution) -> dict[str, object]:
    return {
        "production": [
            [[solution.value(value) for value in row] for row in agent]
            for agent in built.production
        ],
        "transfers": [
            [solution.value(value) for value in row] for row in built.transfers
        ],
        "deliveries": [
            [solution.value(value) for value in row] for row in built.deliveries
        ],
        "inventories": [
            [[solution.value(value) for value in row] for row in agent]
            for agent in built.inventories
        ],
    }


def objective_value(built: SupplyChainModel, result: dict[str, object]) -> int:
    production = result["production"]
    transfers = result["transfers"]
    deliveries = result["deliveries"]
    inventories = result["inventories"]
    total = 0
    for agent_index, agent in enumerate(built.instance.agents):
        for process_index, process in enumerate(agent.processes):
            total += sum(
                process.unit_cost * amount + process.setup_cost * int(amount > 0)
                for amount in production[agent_index][process_index]
            )
        total += sum(
            agent.holding_costs[item] * inventories[agent_index][item][time]
            for item in range(len(built.instance.items))
            for time in range(built.instance.horizon)
        )
    total += sum(
        arc.unit_cost * amount
        for arc, row in zip(built.instance.arcs, transfers)
        for amount in row
    )
    total += sum(
        demand.penalty * (quantity - delivered)
        for demand, row in zip(built.instance.demands, deliveries)
        for quantity, delivered in zip(demand.quantities, row)
    )
    return total


def validate(
    built: SupplyChainModel,
    result: dict[str, object],
    objective: int | None = None,
) -> None:
    production = result["production"]
    transfers = result["transfers"]
    deliveries = result["deliveries"]
    inventories = result["inventories"]
    for agent_index, agent in enumerate(built.instance.agents):
        for time in range(built.instance.horizon):
            if (
                sum(
                    process.capacity * production[agent_index][index][time]
                    for index, process in enumerate(agent.processes)
                )
                > agent.capacity[time]
            ):
                raise AssertionError("an agent production capacity is exceeded")
        for item in range(len(built.instance.items)):
            for time in range(built.instance.horizon):
                previous = (
                    agent.initial_inventory[item]
                    if time == 0
                    else inventories[agent_index][item][time - 1]
                )
                produced = sum(
                    process.quantity * production[agent_index][index][time]
                    for index, process in enumerate(agent.processes)
                    if process.output == item
                )
                consumed = sum(
                    quantity * production[agent_index][index][time]
                    for index, process in enumerate(agent.processes)
                    for input_item, quantity in process.inputs
                    if input_item == item
                )
                incoming = sum(
                    transfers[index][time - arc.lead]
                    for index, arc in enumerate(built.instance.arcs)
                    if arc.target == agent_index
                    and arc.item == item
                    and time >= arc.lead
                )
                outgoing = sum(
                    transfers[index][time]
                    for index, arc in enumerate(built.instance.arcs)
                    if arc.source == agent_index and arc.item == item
                )
                delivered = sum(
                    deliveries[index][time]
                    for index, demand in enumerate(built.instance.demands)
                    if demand.agent == agent_index and demand.item == item
                )
                expected = (
                    previous
                    + agent.external_supply[item][time]
                    + produced
                    + incoming
                    - consumed
                    - outgoing
                    - delivered
                )
                if inventories[agent_index][item][time] != expected or expected < 0:
                    raise AssertionError("an inventory balance is violated")
    if objective is not None and objective_value(built, result) != objective:
        raise AssertionError("the objective does not match supply-chain costs")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON supply-chain instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob047 agents={len(instance.agents)} status={solution.status}")
    if not solution.is_sat():
        return 1
    result = decode(built, solution)
    validate(built, result, solution.objective)
    print(
        f"cost={solution.objective} transfers={result['transfers']} deliveries={result['deliveries']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
