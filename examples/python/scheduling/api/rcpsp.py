"""RCPSP/MRCPSP through the scheduling convenience API.

The command is identical to ``scheduling/native/rcpsp.py``. Without a
positional file it generates the same deterministic RCPSP. PSPLIB ``.sm`` and
``.mm`` files can be passed directly:

    uv run examples/python/scheduling/api/rcpsp.py j301_1.sm --threads 4
    uv run examples/python/scheduling/api/rcpsp.py c151_1.mm --engine auto
"""

import argparse
import contextlib
import json
import sys
import time
from random import Random

import qayd as cp
from qayd.datasets import read_psplib


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("instance", nargs="?")
parser.add_argument("--tasks", type=int, default=12)
parser.add_argument("--resources", type=int, default=2)
parser.add_argument("--time-limit", type=int, default=8)
parser.add_argument("--threads", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--engine", choices=("auto", "exact", "ls"))
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--json", action="store_true", help="emit one machine-readable result")
args = parser.parse_args()

if args.threads <= 0 or args.time_limit < 0 or args.seed < 0:
    raise SystemExit("threads must be positive; time limit and seed must be non-negative")

instance = read_psplib(args.instance) if args.instance else None
if instance is None:
    rng = Random(args.seed)
    job_ids = list(range(args.tasks))
    successors = {job: [] for job in job_ids}
    for task in range(1, args.tasks):
        if rng.random() < 0.4:
            successors[rng.randint(0, task - 1)].append(task)
    resource_kinds = ["renewable"] * args.resources
    resource_names = [f"R{resource + 1}" for resource in range(args.resources)]
    capacities = [10] * args.resources
    modes = {
        job: [
            {
                "mode": 1,
                "duration": rng.randint(2, 6),
                "demands": tuple(rng.randint(0, 5) for _ in range(args.resources)),
            }
        ]
        for job in job_ids
    }
    horizon = sum(options[0]["duration"] for options in modes.values())
    name = f"generated-rcpsp-n{args.tasks}"
    multi_mode = False
else:
    job_ids = [job.job for job in instance.jobs]
    successors = {job.job: list(job.successors) for job in instance.jobs}
    resource_kinds = list(instance.resource_kinds)
    resource_names = list(instance.resource_names)
    capacities = list(instance.capacities)
    modes = {
        job.job: [
            {"mode": mode.mode, "duration": mode.duration, "demands": mode.demands}
            for mode in job.modes
        ]
        for job in instance.jobs
    }
    horizon = instance.horizon or sum(max(mode["duration"] for mode in modes[job]) for job in job_ids)
    name = instance.name
    multi_mode = instance.multi_mode
horizon = max(horizon, 1)

if multi_mode and (args.threads != 1 or args.engine == "ls"):
    raise SystemExit("PSPLIB multi-mode currently requires --threads 1 and --engine auto or exact")
engine = args.engine or ("auto" if multi_mode or args.threads == 1 else "ls")

model = cp.Model()
member_records = []
if multi_mode:
    masters = {}
    members = []
    for job in job_ids:
        alternatives = []
        for mode in modes[job]:
            interval = model.interval(mode["duration"], horizon, optional=True, name=f"job{job}.mode{mode['mode']}")
            alternatives.append(interval)
            members.append(interval)
            member_records.append((job, mode, interval))
        masters[job] = model.alternative(alternatives, name=f"job{job}")
    for job in job_ids:
        for successor in successors[job]:
            model.add(masters[job].end <= masters[successor].start)
    for resource, (kind, capacity) in enumerate(zip(resource_kinds, capacities)):
        if kind in {"renewable", "doubly_constrained"}:
            demands = [
                (interval, mode["demands"][resource])
                for _, mode, interval in member_records
                if mode["demands"][resource] > 0
            ]
            if demands:
                model.resource(demands, capacity)
        if kind in {"nonrenewable", "doubly_constrained"}:
            terms = [
                interval.presence * mode["demands"][resource]
                for _, mode, interval in member_records
                if mode["demands"][resource] > 0
            ]
            if terms:
                model.add(cp.sum(terms) <= capacity)
    model.minimize_makespan(members)
else:
    chosen_modes = [modes[job][0] for job in job_ids]
    index_of = {job: index for index, job in enumerate(job_ids)}
    tasks = model.tasks(range(len(job_ids)))
    for task in tasks:
        task.duration = chosen_modes[task.id]["duration"]
        task.demand = chosen_modes[task.id]["demands"]
    schedule_model = model.schedule(tasks, horizon=horizon)
    for job in job_ids:
        for successor in successors[job]:
            model.add(schedule_model[index_of[job]].end <= schedule_model[index_of[successor]].start)
    for resource, (kind, capacity) in enumerate(zip(resource_kinds, capacities)):
        if kind in {"renewable", "doubly_constrained"}:
            model.add(schedule_model.resource(lambda task, resource=resource: task.demand[resource]) <= capacity)
        if kind in {"nonrenewable", "doubly_constrained"}:
            total = sum(mode["demands"][resource] for mode in chosen_modes)
            if total > capacity:
                raise SystemExit(f"fixed demand exceeds {resource_names[resource]} capacity")
    model.minimize(schedule_model.makespan())

started = time.perf_counter()
output = contextlib.redirect_stdout(sys.stderr) if args.json else contextlib.nullcontext()
with output:
    solution = model.solve(
        engine=engine,
        threads=args.threads,
        time_limit=args.time_limit,
        seed=args.seed,
        verbose=args.verbose,
    )
elapsed = time.perf_counter() - started

if not solution.starts:
    record = {
        "instance": name,
        "status": solution.status,
        "elapsed_seconds": elapsed,
        "objectives": [],
        "objective_convention": "makespan",
        "dual_bound": solution.dual_bound,
        "absolute_gap": solution.absolute_gap,
        "relative_gap": solution.relative_gap,
        "bound_method": solution.bound_method,
    }
    print(json.dumps(record, sort_keys=True) if args.json else f"instance: {name}  status: {solution.status}")
    raise SystemExit(0)

if multi_mode:
    selected_by_job = {}
    for index, (job, mode, _) in enumerate(member_records):
        if solution.presences[index]:
            assert job not in selected_by_job, "one mode per job"
            selected_by_job[job] = {
                "job": job,
                "mode": mode["mode"],
                "start": int(solution.starts[index]),
                "duration": mode["duration"],
                "demands": mode["demands"],
            }
    assert set(selected_by_job) == set(job_ids), "every job selects one mode"
    schedule = [selected_by_job[job] for job in job_ids]
else:
    assert len(solution.starts) == len(job_ids), "every job has a start"
    schedule = [
        {
            "job": job,
            "mode": chosen_modes[index]["mode"],
            "start": int(solution.starts[index]),
            "duration": chosen_modes[index]["duration"],
            "demands": chosen_modes[index]["demands"],
        }
        for index, job in enumerate(job_ids)
    ]

by_job = {job["job"]: job for job in schedule}
for job in job_ids:
    for successor in successors[job]:
        assert by_job[job]["start"] + by_job[job]["duration"] <= by_job[successor]["start"], "precedence respected"
for resource, (kind, capacity) in enumerate(zip(resource_kinds, capacities)):
    if kind in {"renewable", "doubly_constrained"}:
        event_times = sorted({job["start"] for job in schedule} | {job["start"] + job["duration"] for job in schedule})
        for point in event_times:
            used = sum(
                job["demands"][resource]
                for job in schedule
                if job["start"] <= point < job["start"] + job["duration"]
            )
            assert used <= capacity, f"{resource_names[resource]} capacity respected"
    if kind in {"nonrenewable", "doubly_constrained"}:
        assert sum(job["demands"][resource] for job in schedule) <= capacity, f"{resource_names[resource]} total respected"
makespan = max(job["start"] + job["duration"] for job in schedule)
assert list(solution.objectives) == [makespan], "reported makespan matches replay"

public_schedule = [{key: value for key, value in job.items() if key != "demands"} for job in schedule]
record = {
    "instance": name,
    "status": solution.status,
    "problem": "mrcpsp" if multi_mode else "rcpsp",
    "jobs": len(job_ids),
    "resources": len(resource_names),
    "resource_kinds": resource_kinds,
    "objectives": [makespan],
    "objective_convention": "makespan",
    "dual_bound": solution.dual_bound,
    "absolute_gap": solution.absolute_gap,
    "relative_gap": solution.relative_gap,
    "bound_method": solution.bound_method,
    "elapsed_seconds": elapsed,
    "seed": args.seed,
    "threads": args.threads,
    "engine": engine,
    "schedule": public_schedule,
    "verified": True,
}
if args.json:
    print(json.dumps(record, sort_keys=True))
else:
    certified = (
        f"  dual: {solution.dual_bound}  gap: {100 * solution.relative_gap:.2f}%  bound: {solution.bound_method}"
        if solution.dual_bound is not None
        else "  dual: unavailable"
    )
    print(f"instance: {name}  problem: {record['problem']}  jobs: {len(job_ids)}  status: {solution.status}")
    print(f"makespan: {makespan}{certified}  elapsed: {elapsed:.3f}s")
