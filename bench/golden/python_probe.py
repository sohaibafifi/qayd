#!/usr/bin/env python3
"""Run one structured Python golden fixture and emit one normalized record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import qayd as cp


RESULT_PREFIX = "QAYD_GOLDEN_RESULT="
FEASIBLE_STATUSES = {"SATISFIABLE", "OPTIMAL"}


def linear_expr(variables: Dict[str, Any], terms: Dict[str, int], constant: int = 0) -> Any:
    value: Any = constant
    for name, coefficient in terms.items():
        value = value + coefficient * variables[name]
    return value


def add_linear_constraints(model: Any, variables: Dict[str, Any], constraints: Iterable[Dict[str, Any]]) -> None:
    for constraint in constraints:
        lhs = linear_expr(variables, constraint["terms"], constraint.get("constant", 0))
        rhs = constraint["rhs"]
        relation = constraint["relation"]
        if relation == "eq":
            model.add(lhs == rhs)
        elif relation == "le":
            model.add(lhs <= rhs)
        elif relation == "ge":
            model.add(lhs >= rhs)
        elif relation == "lt":
            model.add(lhs < rhs)
        elif relation == "gt":
            model.add(lhs > rhs)
        elif relation == "ne":
            model.add(lhs != rhs)
        else:
            raise ValueError("unsupported relation {!r}".format(relation))


def add_objectives(model: Any, variables: Dict[str, Any], objectives: List[Dict[str, Any]]) -> None:
    for index, objective in enumerate(objectives):
        expression = linear_expr(variables, objective["terms"], objective.get("constant", 0))
        sense = objective["sense"]
        if index == 0 and sense == "minimize":
            model.minimize(expression)
        elif index == 0 and sense == "maximize":
            model.maximize(expression)
        elif sense == "minimize":
            model.then_minimize(expression)
        elif sense == "maximize":
            model.then_maximize(expression)
        else:
            raise ValueError("unsupported objective sense {!r}".format(sense))


def solve_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"seed": args.seed, "threads": args.threads, "engine": args.engine}
    if args.time_limit is not None:
        kwargs["time_limit"] = args.time_limit
    if args.max_iterations is not None:
        kwargs["max_iterations"] = args.max_iterations
    return kwargs


def proof_for(status: str) -> Any:
    if status == "OPTIMAL":
        return {"claim": "optimality", "kind": "solver-status", "verified": False}
    if status == "UNSATISFIABLE":
        return {"claim": "unsatisfiability", "kind": "solver-status", "verified": False}
    return None


def bound_for(solution: Any) -> Any:
    if solution.dual_bound is None:
        return None
    return {"values": [solution.dual_bound], "source": "solver"}


def solve_integer(spec: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    model = cp.Model()
    variables = {
        entry["name"]: model.int_var(values=entry["domain"], name=entry["name"])
        for entry in spec["variables"]
    }
    add_linear_constraints(model, variables, spec.get("constraints", []))
    add_objectives(model, variables, spec.get("objectives", []))
    solution = model.solve(**solve_kwargs(args))
    assignment = None
    if solution.status in FEASIBLE_STATUSES:
        assignment = {name: solution.value(variable) for name, variable in variables.items()}
    return {
        "status": solution.status,
        "senses": [objective["sense"] for objective in spec.get("objectives", [])],
        "objectives": list(solution.objectives),
        "solution": None if assignment is None else {"assignments": assignment},
        "bound": bound_for(solution),
        "proof": proof_for(solution.status),
    }


def distance_objective(model: Any, lists: Iterable[Any], travel: List[List[int]], depot: int) -> None:
    matrix = cp.matrix(travel)
    model.minimize(
        cp.sum(cp.sum_edges(route, lambda left, right: matrix[left][right], start=depot, end=depot) for route in lists)
    )


def solve_lists(spec: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    model = cp.Model()
    lists = model.list_vars(
        spec["items"], count=spec["list_count"], optional=spec.get("optional", False)
    )
    distance_objective(model, lists, spec["travel"], spec["depot"])
    solution = model.solve(**solve_kwargs(args))
    return {
        "status": solution.status,
        "senses": [spec["objective"]["sense"]],
        "objectives": list(solution.objectives),
        "solution": None if solution.lists is None else {"lists": solution.lists},
        "bound": bound_for(solution),
        "proof": proof_for(solution.status),
    }


def solve_routing(spec: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    model = cp.Model()
    customers = model.customers(spec["customers"])
    routes = model.routes(
        customers,
        vehicles=spec["vehicles"],
        depot=spec["depot"],
        travel=spec["travel"],
        optional=spec.get("optional", False),
    )
    model.minimize(routes.total_distance())
    solution = model.solve(**solve_kwargs(args))
    return {
        "status": solution.status,
        "senses": [spec["objective"]["sense"]],
        "objectives": list(solution.objectives),
        "solution": None if solution.lists is None else {"routes": solution.lists},
        "bound": bound_for(solution),
        "proof": proof_for(solution.status),
    }


def solve_scheduling(spec: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    model = cp.Model()
    task_ids = [entry["id"] for entry in spec["tasks"]]
    tasks = model.tasks(task_ids)
    task_spec = {entry["id"]: entry for entry in spec["tasks"]}
    for task in tasks:
        entry = task_spec[task.id]
        task.duration = entry["duration"]
        task.demands = entry.get("demands", [])
    schedule = model.schedule(tasks, horizon=spec["horizon"])
    for before, after in spec.get("precedences", []):
        model.add(schedule[before].end <= schedule[after].start)
    for resource in spec.get("resources", []):
        index = resource["demand_index"]
        model.add(schedule.resource(lambda task, i=index: task.demands[i]) <= resource["capacity"])
    model.minimize(schedule.makespan())
    solution = model.solve(**solve_kwargs(args))
    starts = None
    if solution.status in FEASIBLE_STATUSES:
        starts = {str(task_id): solution.starts[index] for index, task_id in enumerate(task_ids)}
    return {
        "status": solution.status,
        "senses": [spec["objective"]["sense"]],
        "objectives": list(solution.objectives),
        "solution": None if starts is None else {"starts": starts},
        "bound": bound_for(solution),
        "proof": proof_for(solution.status),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--engine", choices=("auto", "exact", "ls"), default="exact")
    parser.add_argument("--time-limit", type=int)
    parser.add_argument("--max-iterations", type=int)
    args = parser.parse_args()

    spec = json.loads(args.instance.read_text(encoding="utf-8"))
    if spec.get("schema") != "qayd.python-golden/v1":
        raise ValueError("unsupported Python golden fixture schema")
    runners = {
        "integer": solve_integer,
        "lists": solve_lists,
        "routing": solve_routing,
        "scheduling": solve_scheduling,
    }
    try:
        runner = runners[spec["kind"]]
    except KeyError as error:
        raise ValueError("unsupported Python golden kind {!r}".format(spec.get("kind"))) from error
    result = runner(spec, args)
    print(RESULT_PREFIX + json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
