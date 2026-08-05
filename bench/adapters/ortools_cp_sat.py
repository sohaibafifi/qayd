#!/usr/bin/env python3
"""OR-Tools CP-SAT adapter for JSPLIB and PSPLIB instances."""

from __future__ import annotations

import argparse
import json
import math
import time

from ortools import __version__ as ortools_version
from ortools.sat.python import cp_model

from qayd.datasets import read_jsplib, read_psplib


class Trace(cp_model.CpSolverSolutionCallback):
    def __init__(self) -> None:
        super().__init__()
        self.points: list[dict[str, float]] = []

    def on_solution_callback(self) -> None:
        self.points.append({
            "time": self.wall_time,
            "primal": self.objective_value,
            "dual": self.best_objective_bound,
        })


def configured_solver(args: argparse.Namespace) -> tuple[cp_model.CpSolver, Trace]:
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = args.time_limit
    solver.parameters.num_workers = args.threads
    solver.parameters.random_seed = args.seed
    trace = Trace()
    return solver, trace


def status_name(status: cp_model.CpSolverStatus) -> str:
    if status == cp_model.OPTIMAL:
        return "OPTIMAL"
    if status == cp_model.FEASIBLE:
        return "SATISFIABLE"
    if status == cp_model.INFEASIBLE:
        return "UNSAT"
    if status == cp_model.MODEL_INVALID:
        return "ERROR"
    return "UNKNOWN"


def bound_fields(solver: cp_model.CpSolver, has_primal: bool) -> dict[str, object]:
    dual = float(solver.best_objective_bound)
    if not math.isfinite(dual):
        dual = None
    primal = float(solver.objective_value) if has_primal else None
    absolute = max(0.0, primal - dual) if primal is not None and dual is not None else None
    relative = absolute / max(1.0, abs(primal), abs(dual)) if absolute is not None else None
    return {
        "dual_bound": dual,
        "absolute_gap": absolute,
        "relative_gap": relative,
        "bound_method": "OR-Tools CP-SAT certified objective bound" if dual is not None else None,
    }


def solve_jssp(args: argparse.Namespace) -> dict[str, object]:
    instance = read_jsplib(args.instance)
    model = cp_model.CpModel()
    horizon = max(1, instance.horizon)
    starts: list[list[cp_model.IntVar]] = []
    ends: list[list[cp_model.IntVar]] = []
    by_machine: list[list[cp_model.IntervalVar]] = [[] for _ in range(instance.num_machines)]
    for job_index, job in enumerate(instance.jobs):
        job_starts = []
        job_ends = []
        for operation_index, operation in enumerate(job):
            start = model.new_int_var(0, horizon, f"start_{job_index}_{operation_index}")
            end = model.new_int_var(0, horizon, f"end_{job_index}_{operation_index}")
            interval = model.new_interval_var(start, operation.duration, end, f"op_{job_index}_{operation_index}")
            job_starts.append(start)
            job_ends.append(end)
            by_machine[operation.machine].append(interval)
        starts.append(job_starts)
        ends.append(job_ends)
        for before in range(len(job) - 1):
            model.add(job_ends[before] <= job_starts[before + 1])
    for intervals in by_machine:
        model.add_no_overlap(intervals)
    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, [job_ends[-1] for job_ends in ends])
    model.minimize(makespan)

    solver, trace = configured_solver(args)
    started = time.perf_counter()
    status = solver.solve(model, trace)
    elapsed = time.perf_counter() - started
    has_primal = status in {cp_model.FEASIBLE, cp_model.OPTIMAL}
    record: dict[str, object] = {
        "instance": instance.name,
        "status": status_name(status),
        "objectives": [],
        "elapsed_seconds": elapsed,
        "solver_engine": "OR-Tools CP-SAT",
        "solver_library_version": ortools_version,
        "objective_convention": "makespan",
        "anytime": trace.points,
        "verified": False,
    }
    record.update(bound_fields(solver, has_primal))
    if not has_primal:
        return record

    schedule = []
    occupied: list[list[tuple[int, int]]] = [[] for _ in range(instance.num_machines)]
    makespan_value = 0
    for job_index, job in enumerate(instance.jobs):
        row = []
        previous_end = 0
        for operation_index, operation in enumerate(job):
            start = solver.value(starts[job_index][operation_index])
            end = start + operation.duration
            assert start >= previous_end
            previous_end = end
            makespan_value = max(makespan_value, end)
            occupied[operation.machine].append((start, end))
            row.append({"machine": operation.machine, "start": start, "duration": operation.duration})
        schedule.append(row)
    for intervals in occupied:
        intervals.sort()
        assert all(before[1] <= after[0] for before, after in zip(intervals, intervals[1:]))
    assert makespan_value == round(solver.objective_value)
    record.update({"objectives": [makespan_value], "schedule": schedule, "verified": True})
    return record


def solve_rcpsp(args: argparse.Namespace) -> dict[str, object]:
    instance = read_psplib(args.instance)
    model = cp_model.CpModel()
    horizon = max(
        instance.horizon or 0,
        sum(max(mode.duration for mode in job.modes) for job in instance.jobs),
        1,
    )
    starts: dict[int, cp_model.IntVar] = {}
    ends: dict[int, cp_model.IntVar] = {}
    selected: dict[tuple[int, int], cp_model.BoolVar] = {}
    intervals: dict[tuple[int, int], cp_model.IntervalVar] = {}
    mode_by_key = {}
    for job in instance.jobs:
        starts[job.job] = model.new_int_var(0, horizon, f"start_{job.job}")
        ends[job.job] = model.new_int_var(0, horizon, f"end_{job.job}")
        choices = []
        for mode in job.modes:
            key = (job.job, mode.mode)
            present = model.new_bool_var(f"present_{job.job}_{mode.mode}")
            selected[key] = present
            intervals[key] = model.new_optional_interval_var(
                starts[job.job], mode.duration, ends[job.job], present,
                f"mode_{job.job}_{mode.mode}",
            )
            mode_by_key[key] = mode
            choices.append(present)
        model.add_exactly_one(choices)
    for job in instance.jobs:
        for successor in job.successors:
            model.add(ends[job.job] <= starts[successor])
    for resource, (kind, capacity) in enumerate(zip(instance.resource_kinds, instance.capacities)):
        if kind in {"renewable", "doubly_constrained"}:
            resource_intervals = []
            demands = []
            for key, interval in intervals.items():
                demand = mode_by_key[key].demands[resource]
                if demand:
                    resource_intervals.append(interval)
                    demands.append(demand)
            if resource_intervals:
                model.add_cumulative(resource_intervals, demands, capacity)
        if kind in {"nonrenewable", "doubly_constrained"}:
            model.add(sum(
                mode_by_key[key].demands[resource] * present
                for key, present in selected.items()
            ) <= capacity)
    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, list(ends.values()))
    model.minimize(makespan)

    solver, trace = configured_solver(args)
    started = time.perf_counter()
    status = solver.solve(model, trace)
    elapsed = time.perf_counter() - started
    has_primal = status in {cp_model.FEASIBLE, cp_model.OPTIMAL}
    record: dict[str, object] = {
        "instance": instance.name,
        "status": status_name(status),
        "objectives": [],
        "elapsed_seconds": elapsed,
        "solver_engine": "OR-Tools CP-SAT",
        "solver_library_version": ortools_version,
        "objective_convention": "makespan",
        "anytime": trace.points,
        "verified": False,
    }
    record.update(bound_fields(solver, has_primal))
    if not has_primal:
        return record

    schedule = []
    for job in instance.jobs:
        mode = next(mode for mode in job.modes if solver.boolean_value(selected[(job.job, mode.mode)]))
        start = solver.value(starts[job.job])
        schedule.append({
            "job": job.job,
            "mode": mode.mode,
            "start": start,
            "duration": mode.duration,
            "demands": mode.demands,
        })
    by_job = {job["job"]: job for job in schedule}
    for job in instance.jobs:
        for successor in job.successors:
            assert by_job[job.job]["start"] + by_job[job.job]["duration"] <= by_job[successor]["start"]
    for resource, (kind, capacity) in enumerate(zip(instance.resource_kinds, instance.capacities)):
        if kind in {"renewable", "doubly_constrained"}:
            points = sorted({job["start"] for job in schedule} | {job["start"] + job["duration"] for job in schedule})
            for point in points:
                usage = sum(
                    job["demands"][resource] for job in schedule
                    if job["start"] <= point < job["start"] + job["duration"]
                )
                assert usage <= capacity
        if kind in {"nonrenewable", "doubly_constrained"}:
            assert sum(job["demands"][resource] for job in schedule) <= capacity
    makespan_value = max(job["start"] + job["duration"] for job in schedule)
    assert makespan_value == round(solver.objective_value)
    for job in schedule:
        job["demands"] = list(job["demands"])
    record.update({"objectives": [makespan_value], "schedule": schedule, "verified": True})
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem", choices=("jssp", "rcpsp"))
    parser.add_argument("instance")
    parser.add_argument("--time-limit", type=int, default=60)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.time_limit < 0 or args.threads <= 0 or args.seed < 0:
        raise SystemExit("time limit and seed must be non-negative; threads must be positive")
    record = solve_jssp(args) if args.problem == "jssp" else solve_rcpsp(args)
    if args.json:
        print(json.dumps(record, sort_keys=True))
    else:
        print(record)


if __name__ == "__main__":
    main()
