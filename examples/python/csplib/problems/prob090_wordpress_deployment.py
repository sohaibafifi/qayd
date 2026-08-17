"""CSPLib prob090: WordPress application deployment in the cloud.

Specification: https://www.csplib.org/Problems/prob090/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import qayd as cp

from ..common import add_solver_arguments, solve_from_args, values

DEFAULT_INSTANCE = {
    "vm_slots": 5,
    "vm_types": [
        {"name": "small", "cpu": 3, "memory": 4, "cost": 4},
        {"name": "large", "cpu": 6, "memory": 8, "cost": 7},
    ],
    "components": [
        {"name": "wordpress", "kind": "wordpress", "cpu": 2, "memory": 2},
        {"name": "mysql-1", "kind": "mysql", "cpu": 1, "memory": 2},
        {"name": "mysql-2", "kind": "mysql", "cpu": 1, "memory": 2},
        {"name": "mysql-3", "kind": "mysql", "cpu": 1, "memory": 2},
        {"name": "varnish-1", "kind": "varnish", "cpu": 1, "memory": 1},
        {"name": "varnish-2", "kind": "varnish", "cpu": 1, "memory": 1},
        {"name": "http", "kind": "http_balancer", "cpu": 1, "memory": 1},
    ],
    "conflicting_kinds": [["varnish", "mysql"], ["varnish", "http_balancer"]],
}


@dataclass(frozen=True)
class VmType:
    name: str
    cpu: int
    memory: int
    cost: int


@dataclass(frozen=True)
class Component:
    name: str
    kind: str
    cpu: int
    memory: int


@dataclass(frozen=True)
class DeploymentInstance:
    vm_slots: int
    vm_types: tuple[VmType, ...]
    components: tuple[Component, ...]
    conflicting_kinds: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class DeploymentModel:
    model: cp.Model
    instance: DeploymentInstance
    assignment: list[cp.IntVar]
    vm_type: list[cp.IntVar]


def parse_instance(data: str | bytes) -> DeploymentInstance:
    raw = json.loads(data)
    try:
        vm_types = tuple(
            VmType(
                str(item["name"]),
                int(item["cpu"]),
                int(item["memory"]),
                int(item["cost"]),
            )
            for item in raw["vm_types"]
        )
        components = tuple(
            Component(
                str(item["name"]),
                str(item["kind"]),
                int(item["cpu"]),
                int(item["memory"]),
            )
            for item in raw["components"]
        )
        return DeploymentInstance(
            int(raw["vm_slots"]),
            vm_types,
            components,
            tuple(
                (str(pair[0]), str(pair[1]))
                for pair in raw.get("conflicting_kinds", [])
            ),
        )
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError("invalid WordPress-deployment JSON instance") from error


def load_instance(path: str | Path) -> DeploymentInstance:
    return parse_instance(Path(path).read_text(encoding="utf-8"))


def validate_structure(instance: DeploymentInstance) -> None:
    counts = {
        kind: sum(component.kind == kind for component in instance.components)
        for kind in {component.kind for component in instance.components}
    }
    wordpress = counts.get("wordpress", 0)
    mysql = counts.get("mysql", 0)
    varnish = counts.get("varnish", 0)
    dns = counts.get("dns_balancer", 0)
    http = counts.get("http_balancer", 0)
    if wordpress < 1 or mysql < max(3, (wordpress + 1) // 2) or varnish < 2:
        raise ValueError("WordPress, MySQL, and Varnish deployment bounds are violated")
    if (
        (dns > 0) == (http > 0)
        or dns > 1
        or (dns and wordpress > 7)
        or (http and wordpress > 3 * http)
    ):
        raise ValueError(
            "load-balancer require/provide or exclusivity constraints are violated"
        )


def build_model(instance: DeploymentInstance) -> DeploymentModel:
    if instance.vm_slots < 1 or not instance.vm_types or not instance.components:
        raise ValueError("VM slots, VM types, and components must be non-empty")
    if any(
        item.cpu < 0 or item.memory < 0 or item.cost < 0 for item in instance.vm_types
    ):
        raise ValueError("a VM type is invalid")
    if any(item.cpu < 0 or item.memory < 0 for item in instance.components):
        raise ValueError("a component requirement is invalid")
    validate_structure(instance)
    model = cp.Model()
    assignment = model.int_vars(
        len(instance.components), 0, instance.vm_slots - 1, name="vm"
    )
    vm_type = model.int_vars(
        instance.vm_slots, 0, len(instance.vm_types), name="vm_type"
    )
    max_cpu = sum(component.cpu for component in instance.components)
    max_memory = sum(component.memory for component in instance.components)
    costs = []
    for slot in range(instance.vm_slots):
        cpu_load = model.int_var(0, max_cpu, name=f"cpu_{slot}")
        memory_load = model.int_var(0, max_memory, name=f"memory_{slot}")
        cost = model.int_var(
            0, max(item.cost for item in instance.vm_types), name=f"cost_{slot}"
        )
        memberships = []
        for component in range(len(instance.components)):
            member = model.bool_var(name=f"component_{component}_vm_{slot}")
            model.table(
                [assignment[component], member],
                [
                    (candidate, int(candidate == slot))
                    for candidate in range(instance.vm_slots)
                ],
            )
            memberships.append(member)
        model.add(
            cpu_load
            == sum(
                item.cpu * member
                for item, member in zip(instance.components, memberships)
            )
        )
        model.add(
            memory_load
            == sum(
                item.memory * member
                for item, member in zip(instance.components, memberships)
            )
        )
        model.table(
            [vm_type[slot], cpu_load, memory_load, cost],
            [
                (0, 0, 0, 0),
                *[
                    (index, cpu, memory, item.cost)
                    for index, item in enumerate(instance.vm_types, start=1)
                    for cpu in range(item.cpu + 1)
                    for memory in range(item.memory + 1)
                ],
            ],
        )
        costs.append(cost)
    for first, left in enumerate(instance.components):
        for second in range(first + 1, len(instance.components)):
            right = instance.components[second]
            if any(
                {left.kind, right.kind} == set(pair)
                for pair in instance.conflicting_kinds
            ):
                model.add(assignment[first] != assignment[second])
    model.minimize(sum(costs))
    return DeploymentModel(model, instance, assignment, vm_type)


def decode(
    built: DeploymentModel, solution: cp.Solution
) -> tuple[list[int], list[int]]:
    return values(solution, built.assignment), values(solution, built.vm_type)


def deployment_cost(instance: DeploymentInstance, types: list[int]) -> int:
    return sum(0 if kind == 0 else instance.vm_types[kind - 1].cost for kind in types)


def validate(
    built: DeploymentModel, result: tuple[list[int], list[int]], objective: int | None
) -> None:
    assignment, types = result
    for slot, kind in enumerate(types):
        members = [
            item
            for item, assigned in zip(built.instance.components, assignment)
            if assigned == slot
        ]
        cpu = sum(item.cpu for item in members)
        memory = sum(item.memory for item in members)
        if kind == 0 and members:
            raise AssertionError("a component is assigned to an inactive VM")
        if kind and (
            cpu > built.instance.vm_types[kind - 1].cpu
            or memory > built.instance.vm_types[kind - 1].memory
        ):
            raise AssertionError("a VM capacity is exceeded")
        for first, left in enumerate(members):
            for right in members[first + 1 :]:
                if any(
                    {left.kind, right.kind} == set(pair)
                    for pair in built.instance.conflicting_kinds
                ):
                    raise AssertionError("conflicting components share a VM")
    if objective is not None and deployment_cost(built.instance, types) != objective:
        raise AssertionError("the objective does not match VM leasing costs")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="JSON WordPress deployment instance")
    add_solver_arguments(parser)
    args = parser.parse_args(argv)
    instance = (
        load_instance(args.path)
        if args.path
        else parse_instance(json.dumps(DEFAULT_INSTANCE))
    )
    built = build_model(instance)
    solution = solve_from_args(built.model, args)
    print(f"prob090 components={len(instance.components)} status={solution.status}")
    if not solution.is_sat():
        return 1
    result = decode(built, solution)
    validate(built, result, solution.objective)
    print(f"cost={solution.objective} assignment={result[0]} vm_types={result[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
