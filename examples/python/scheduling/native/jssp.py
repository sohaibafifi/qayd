"""Job-shop scheduling with an optional JSPLIB instance.

Without a positional file this keeps the deterministic generated example. Pass
a standard JSPLIB pair-format file to solve it:

    uv run examples/python/scheduling/native/jssp.py abz5.txt --threads 4
"""

import argparse
import contextlib
import hashlib
import json
import sys
import time
from random import Random

import qayd as cp
from qayd.datasets import read_jsplib


def start_vector_sha256(starts):
    """Hash the canonical compact-JSON encoding without materializing it."""
    digest = hashlib.sha256()
    digest.update(b"[")
    chunk_size = 4096
    for begin in range(0, len(starts), chunk_size):
        if begin:
            digest.update(b",")
        chunk = ",".join(str(value) for value in starts[begin : begin + chunk_size])
        digest.update(chunk.encode("ascii"))
    digest.update(b"]")
    return digest.hexdigest()


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
parser.add_argument(
    "--compact-json",
    action="store_true",
    help="omit the detailed schedule and emit a SHA-256 start-vector commitment",
)
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

parse_started = time.perf_counter()
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
parse_seconds = time.perf_counter() - parse_started

model_build_started = time.perf_counter()
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
intervals = model.alternatives([[(machine, duration)] for machine, duration in zip(machine_of, durations)], horizon)
for operations in job_operations:
    for before, after in zip(operations, operations[1:]):
        model.precedence(intervals[before], intervals[after])
model.no_overlap_by_machine()
model.minimize_makespan(intervals)
model_build_seconds = time.perf_counter() - model_build_started

solve_started = time.perf_counter()
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
solve_seconds = time.perf_counter() - solve_started
elapsed = solve_seconds
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
    "schedule_restart_work": solution.schedule_restart_work,
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
    "schedule_tabu_steps": solution.schedule_tabu_steps,
    "schedule_tabu_hits": solution.schedule_tabu_hits,
    "schedule_tabu_aspirations": solution.schedule_tabu_aspirations,
    "schedule_tabu_forced_moves": solution.schedule_tabu_forced_moves,
    "schedule_session_initializations": solution.schedule_session_initializations,
    "schedule_session_resumes": solution.schedule_session_resumes,
    "schedule_session_rebases": solution.schedule_session_rebases,
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
        "elapsed_seconds_scope": "model.solve",
        "parse_seconds": parse_seconds,
        "model_build_seconds": model_build_seconds,
        "solve_seconds": solve_seconds,
        "replay_seconds": 0.0,
        "verification_counts": {
            "starts": 0,
            "job_precedence_pairs": 0,
            "machine_non_overlap_pairs": 0,
            "objective_checks": 0,
        },
        "objectives": [],
        "objective_convention": "makespan",
        "dual_bound": solution.dual_bound,
        "absolute_gap": solution.absolute_gap,
        "relative_gap": solution.relative_gap,
        "bound_method": solution.bound_method,
    }
    if args.compact_json:
        record.update({
            "start_vector_length": 0,
            "start_vector_sha256": start_vector_sha256([]),
            "start_vector_sha256_encoding": "canonical-json-array-v1",
        })
    print(json.dumps(record, sort_keys=True) if args.json else f"instance: {name}  status: {solution.status}")
    raise SystemExit(0)

replay_started = time.perf_counter()
starts = [int(start) for start in solution.starts]
if len(starts) != len(durations):
    raise AssertionError("solver returned an incomplete start vector")
if any(start < 0 for start in starts):
    raise AssertionError("solver returned a negative start time")
ends = [start + duration for start, duration in zip(starts, durations)]
job_precedence_pairs = 0
for operations in job_operations:
    for before, after in zip(operations, operations[1:]):
        job_precedence_pairs += 1
        if ends[before] > starts[after]:
            raise AssertionError(
                f"job precedence violated: operation {before} ends at {ends[before]} "
                f"after operation {after} starts at {starts[after]}"
            )
machine_operations = [[] for _ in range(machines)]
for index, machine in enumerate(machine_of):
    machine_operations[machine].append(index)
machine_non_overlap_pairs = 0
for machine, operations in enumerate(machine_operations):
    operations.sort(key=starts.__getitem__)
    for before, after in zip(operations, operations[1:]):
        machine_non_overlap_pairs += 1
        if ends[before] > starts[after]:
            raise AssertionError(
                f"machine {machine} overlap: operation {before} ends at {ends[before]} "
                f"after operation {after} starts at {starts[after]}"
            )
makespan = max(ends, default=0)
reported_objectives = list(solution.objectives)
if reported_objectives != [makespan]:
    raise AssertionError(
        f"reported objectives {reported_objectives} do not match replayed makespan {makespan}"
    )
replay_seconds = time.perf_counter() - replay_started
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
    "elapsed_seconds_scope": "model.solve",
    "parse_seconds": parse_seconds,
    "model_build_seconds": model_build_seconds,
    "solve_seconds": solve_seconds,
    "replay_seconds": replay_seconds,
    "seed": args.seed,
    "threads": args.threads,
    "engine": engine,
    "verified": True,
    "verification_counts": {
        "starts": len(starts),
        "job_precedence_pairs": job_precedence_pairs,
        "machine_non_overlap_pairs": machine_non_overlap_pairs,
        "objective_checks": 1,
    },
}
if args.compact_json:
    commitment_started = time.perf_counter()
    record.update({
        "start_vector_length": len(starts),
        "start_vector_sha256": start_vector_sha256(starts),
        "start_vector_sha256_encoding": "canonical-json-array-v1",
        "commitment_seconds": time.perf_counter() - commitment_started,
    })
else:
    record["schedule"] = [
        [
            {"machine": machine_of[index], "start": starts[index], "duration": durations[index]}
            for index in operations
        ]
        for operations in job_operations
    ]
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
