"""Job-shop scheduling through the scheduling convenience API.

The command is identical to ``scheduling/native/jssp.py``. Without a positional
file it generates the same deterministic JSSP. A JSPLIB pair-format file can be
passed directly:

    uv run examples/python/scheduling/api/jssp.py abz5.txt --threads 4
"""

import argparse
import contextlib
import json
import sys
import time
from random import Random

import qayd as cp
from qayd.datasets import read_jsplib


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("instance", nargs="?")
parser.add_argument("--jobs", type=int, default=4)
parser.add_argument("--machines", type=int, default=3)
parser.add_argument("--time-limit", type=int, default=8)
parser.add_argument("--memory-limit-mb", type=int, default=256)
parser.add_argument("--threads", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max-iterations", type=int)
parser.add_argument("--engine", choices=("auto", "exact", "ls"))
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--profile", action="store_true")
parser.add_argument("--json", action="store_true", help="emit one machine-readable result")
args = parser.parse_args()

if (
    args.threads <= 0
    or args.time_limit < 0
    or args.seed < 0
    or args.memory_limit_mb < 0
    or (args.max_iterations is not None and args.max_iterations < 0)
):
    raise SystemExit("threads must be positive; time, memory and iteration limits and seed must be non-negative")
engine = args.engine or ("ls" if args.threads > 1 else "auto")

instance = read_jsplib(args.instance) if args.instance else None
if instance is None:
    jobs, machines = args.jobs, args.machines
    rng = Random(args.seed)
    machine_order = [rng.sample(range(machines), machines) for _ in range(jobs)]
    processing = [[rng.randint(2, 9) for _ in range(machines)] for _ in range(jobs)]
    name = f"generated-jssp-{jobs}x{machines}"
else:
    jobs, machines = instance.num_jobs, instance.num_machines
    machine_order = [list(row) for row in instance.machines]
    processing = [list(row) for row in instance.durations]
    name = instance.name

durations = [processing[job][operation] for job in range(jobs) for operation in range(len(processing[job]))]
machine_of = [machine_order[job][operation] for job in range(jobs) for operation in range(len(machine_order[job]))]
job_operations = []
offset = 0
for job in range(jobs):
    count = len(processing[job])
    job_operations.append(list(range(offset, offset + count)))
    offset += count
horizon = sum(durations)

model = cp.Model()
task_data = model.tasks(range(len(durations)))
for task in task_data:
    task.duration = durations[task.id]
    task.machine = machine_of[task.id]
schedule_model = model.schedule(task_data, horizon=horizon)
for operations in job_operations:
    for before, after in zip(operations, operations[1:]):
        model.add(schedule_model[before].end <= schedule_model[after].start)
model.add(schedule_model.no_overlap(lambda task: task.machine))
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
        profile=args.profile,
        memory_limit_mb=args.memory_limit_mb or None,
        max_iterations=args.max_iterations,
    )
elapsed = time.perf_counter() - started
profile_record = {
    "backend_build_seconds": solution.backend_build_seconds,
    "construction_seconds": solution.construction_seconds,
    "time_to_first_feasible": solution.time_to_first_feasible,
    "construction_candidates": solution.construction_candidates,
    "candidates_evaluated": solution.candidates_evaluated,
    "candidates_per_second": solution.candidates_per_second,
    "schedule_moves_considered": solution.schedule_moves_considered,
    "schedule_moves_accepted": solution.schedule_moves_accepted,
    "schedule_moves_rejected": solution.schedule_moves_rejected,
    "schedule_work_steps": solution.schedule_work_steps,
    "schedule_local_improvements": solution.schedule_local_improvements,
    "schedule_global_improvements": solution.schedule_global_improvements,
    "schedule_progress_publications": solution.schedule_progress_publications,
    "schedule_incumbent_publication_attempts": solution.schedule_incumbent_publication_attempts,
    "schedule_incumbent_publications": solution.schedule_incumbent_publications,
    "schedule_incumbent_injection_attempts": solution.schedule_incumbent_injection_attempts,
    "schedule_incumbent_injections": solution.schedule_incumbent_injections,
    "schedule_incumbent_rejections": solution.schedule_incumbent_rejections,
    "schedule_incumbent_verifications": solution.schedule_incumbent_verifications,
    "schedule_incumbent_verification_rejections": solution.schedule_incumbent_verification_rejections,
    "schedule_incumbent_verification_interruptions": solution.schedule_incumbent_verification_interruptions,
    "schedule_incumbent_incomplete_rejections": solution.schedule_incumbent_incomplete_rejections,
    "schedule_restart_boundaries": solution.schedule_restart_boundaries,
    "schedule_peak_buffered_candidates": solution.schedule_peak_buffered_candidates,
    "schedule_stalled_workers": solution.schedule_stalled_workers,
    "schedule_stalled_unused_work_steps": solution.schedule_stalled_unused_work_steps,
    "schedule_worker_work_min": solution.schedule_worker_work_min,
    "schedule_worker_work_max": solution.schedule_worker_work_max,
    "schedule_work_budget_overruns": solution.schedule_work_budget_overruns,
    "schedule_elite_pool_size": solution.schedule_elite_pool_size,
    "schedule_unused_work_steps": solution.schedule_unused_work_steps,
    "schedule_incumbent_source_worker": solution.schedule_incumbent_source_worker,
    "schedule_incumbent_source_round": solution.schedule_incumbent_source_round,
    "schedule_cycle_rejections": solution.schedule_cycle_rejections,
    "schedule_reconstructions": solution.schedule_reconstructions,
    "critical_path_updates": solution.critical_path_updates,
    "full_recompute_percentage": solution.full_recompute_percentage,
    "schedule_window_rejections": solution.schedule_window_rejections,
    "schedule_objective_rejections": solution.schedule_objective_rejections,
    "resource_profile_checks": solution.resource_profile_checks,
    "resource_candidate_scheduling_attempts": solution.resource_candidate_scheduling_attempts,
    "resource_event_visits": solution.resource_event_visits,
    "resource_peak_profile_events": solution.resource_peak_profile_events,
    "schedule_precedence_rejections": solution.schedule_precedence_rejections,
    "schedule_infeasible_rejections": solution.schedule_infeasible_rejections,
    "schedule_justification_attempts": solution.schedule_justification_attempts,
    "schedule_delta_evaluations": solution.schedule_delta_evaluations,
    "schedule_full_evaluations": solution.schedule_full_evaluations,
    "schedule_full_fallbacks": solution.schedule_full_fallbacks,
    "schedule_topological_rebuilds": solution.schedule_topological_rebuilds,
    "schedule_oracle_validations": solution.schedule_oracle_validations,
    "schedule_oracle_mismatches": solution.schedule_oracle_mismatches,
    "schedule_dirty_cone_operations": solution.schedule_dirty_cone_operations,
    "schedule_max_dirty_cone": solution.schedule_max_dirty_cone,
    "schedule_workspace_growths": solution.schedule_workspace_growths,
    "schedule_workspace_rollbacks": solution.schedule_workspace_rollbacks,
    "schedule_alns_generation_attempts": solution.schedule_alns_generation_attempts,
    "schedule_alns_moves_generated": solution.schedule_alns_moves_generated,
    "estimated_backend_bytes": solution.estimated_backend_bytes,
    "constructor": solution.constructor,
    "memory_limit_mb": args.memory_limit_mb,
}

if not solution.starts:
    record = {
        **profile_record,
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

starts = [int(start) for start in solution.starts]
ends = [start + duration for start, duration in zip(starts, durations)]
for operations in job_operations:
    for before, after in zip(operations, operations[1:]):
        assert ends[before] <= starts[after], "job order respected"
for machine in range(machines):
    operations = sorted((index for index, owner in enumerate(machine_of) if owner == machine), key=starts.__getitem__)
    for before, after in zip(operations, operations[1:]):
        assert ends[before] <= starts[after], f"machine {machine} has no overlap"
makespan = max(ends, default=0)
assert list(solution.objectives) == [makespan], "reported makespan matches replay"

schedule = [
    [
        {"machine": machine_of[index], "start": starts[index], "duration": durations[index]}
        for index in operations
    ]
    for operations in job_operations
]
record = {
    **profile_record,
    "instance": name,
    "status": solution.status,
    "jobs": jobs,
    "machines": machines,
    "operations": len(durations),
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
    "schedule": schedule,
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
    print(f"instance: {name}  jobs: {jobs}  machines: {machines}  status: {solution.status}")
    print(f"makespan: {makespan}{certified}  elapsed: {elapsed:.3f}s")
