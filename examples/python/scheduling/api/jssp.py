"""Job-shop scheduling through the scheduling convenience API.

The command is identical to ``scheduling/native/jssp.py``. Without a positional
file it generates the same deterministic JSSP. A JSPLIB pair-format file can be
passed directly:

    uv run examples/python/scheduling/api/jssp.py abz5.txt --threads 4
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
parser.add_argument("--schedule-path-relink", action="store_true", help="enable bounded guide-directed JSSP path relinking")
parser.add_argument("--schedule-lns-shadow", action="store_true", help="measure bounded exact JSSP repairs without publishing them")
parser.add_argument(
    "--schedule-constructor-workers",
    type=int,
    default=0,
    help="reserve one worker for deterministic JSSP multi-start construction (C1 requires 1 with 7 threads)",
)
parser.add_argument(
    "--schedule-jssp-search",
    choices=("legacy", "tsab-candidate", "tsab-multi-candidate"),
    default="legacy",
    help="select a single-owner or all-worker TSAB-inspired candidate-ranked N5 pilot, or legacy JSSP search",
)
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
    or args.schedule_constructor_workers < 0
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
        schedule_path_relink=args.schedule_path_relink,
        schedule_lns_shadow=args.schedule_lns_shadow,
        schedule_constructor_workers=args.schedule_constructor_workers,
        schedule_jssp_search=args.schedule_jssp_search,
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
    "schedule_incumbent_verification_seconds": solution.schedule_incumbent_verification_seconds,
    "schedule_incumbent_verification_max_seconds": solution.schedule_incumbent_verification_max_seconds,
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
    "schedule_tsab_owner_worker_mask": solution.schedule_tsab_owner_worker_mask,
    "schedule_tsab_n5_generated": solution.schedule_tsab_n5_generated,
    "schedule_tsab_ranked": solution.schedule_tsab_ranked,
    "schedule_tsab_shortlists": solution.schedule_tsab_shortlists,
    "schedule_tsab_delta_probes": solution.schedule_tsab_delta_probes,
    "schedule_tsab_additional_delta_probes": solution.schedule_tsab_additional_delta_probes,
    "schedule_tsab_full_oracle_commits": solution.schedule_tsab_full_oracle_commits,
    "schedule_tsab_selected_shortlist_rank_sum": solution.schedule_tsab_selected_shortlist_rank_sum,
    "schedule_tsab_selections": solution.schedule_tsab_selections,
    "schedule_tsab_aspirations": solution.schedule_tsab_aspirations,
    "schedule_tsab_tabu_rejections": solution.schedule_tsab_tabu_rejections,
    "schedule_tsab_tabu_resets": solution.schedule_tsab_tabu_resets,
    "schedule_tsab_fingerprint_repeats": solution.schedule_tsab_fingerprint_repeats,
    "schedule_tsab_escape_signals": solution.schedule_tsab_escape_signals,
    "schedule_tsab_n1_kicks": solution.schedule_tsab_n1_kicks,
    "schedule_tsab_kick_moves": solution.schedule_tsab_kick_moves,
    "schedule_tsab_elite_restarts": solution.schedule_tsab_elite_restarts,
    "schedule_tsab_n6_kicks": solution.schedule_tsab_n6_kicks,
    "schedule_tsab_restart_attempts": solution.schedule_tsab_restart_attempts,
    "schedule_tsab_restart_global_rebases": solution.schedule_tsab_restart_global_rebases,
    "schedule_tsab_restart_n6_generated": solution.schedule_tsab_restart_n6_generated,
    "schedule_tsab_restart_delta_probes": solution.schedule_tsab_restart_delta_probes,
    "schedule_tsab_restart_oracle_commits": solution.schedule_tsab_restart_oracle_commits,
    "schedule_tsab_restart_relink_attempts": solution.schedule_tsab_restart_relink_attempts,
    "schedule_tsab_restart_relink_commits": solution.schedule_tsab_restart_relink_commits,
    "schedule_tsab_restart_relink_components_shortlisted": solution.schedule_tsab_restart_relink_components_shortlisted,
    "schedule_tsab_restart_relink_components_committed": solution.schedule_tsab_restart_relink_components_committed,
    "schedule_tsab_restart_relink_guide_arc_gain_shortlisted": solution.schedule_tsab_restart_relink_guide_arc_gain_shortlisted,
    "schedule_tsab_restart_relink_guide_arc_gain_accepted": solution.schedule_tsab_restart_relink_guide_arc_gain_accepted,
    "schedule_tsab_restart_rejections": solution.schedule_tsab_restart_rejections,
    "schedule_tsab_restart_interruptions": solution.schedule_tsab_restart_interruptions,
    "schedule_tsab_restart_work_units": solution.schedule_tsab_restart_work_units,
    "schedule_tsab_restart_best_base_objective": solution.schedule_tsab_restart_best_base_objective,
    "schedule_tsab_restart_best_kicked_objective": solution.schedule_tsab_restart_best_kicked_objective,
    "schedule_tsab_post_restart_improvements": solution.schedule_tsab_post_restart_improvements,
    "schedule_tsab_restart_shortlist_peak_bytes": solution.schedule_tsab_restart_shortlist_peak_bytes,
    "schedule_tsab_ranking_audits": solution.schedule_tsab_ranking_audits,
    "schedule_tsab_exact_best_matches": solution.schedule_tsab_exact_best_matches,
    "schedule_tsab_regret_sum": solution.schedule_tsab_regret_sum,
    "schedule_tsab_regret_max": solution.schedule_tsab_regret_max,
    "schedule_tsab_workspace_peak_bytes": solution.schedule_tsab_workspace_peak_bytes,
    "schedule_tsab_activations": solution.schedule_tsab_activations,
    "schedule_tsab_activation_boundary": solution.schedule_tsab_activation_boundary,
    "schedule_tsab_legacy_warmup_work_steps": solution.schedule_tsab_legacy_warmup_work_steps,
    "schedule_tsab_activation_rebases": solution.schedule_tsab_activation_rebases,
    "schedule_tsab_activation_objective": solution.schedule_tsab_activation_objective,
    "schedule_tsab_active_boundaries": solution.schedule_tsab_active_boundaries,
    "schedule_tsab_burst_work_limit": solution.schedule_tsab_burst_work_limit,
    "schedule_tsab_burst_work_units": solution.schedule_tsab_burst_work_units,
    "schedule_tsab_improving_commits": solution.schedule_tsab_improving_commits,
    "schedule_tsab_best_committed_objective": solution.schedule_tsab_best_committed_objective,
    "schedule_tsab_fast_enabled": solution.schedule_tsab_fast_enabled,
    "schedule_tsab_fast_eligible": solution.schedule_tsab_fast_eligible,
    "schedule_tsab_fast_disabled": solution.schedule_tsab_fast_disabled,
    "schedule_tsab_fast_attempts": solution.schedule_tsab_fast_attempts,
    "schedule_tsab_fast_commits": solution.schedule_tsab_fast_commits,
    "schedule_tsab_fast_fallbacks": solution.schedule_tsab_fast_fallbacks,
    "schedule_tsab_fast_work_cap_recoveries": solution.schedule_tsab_fast_work_cap_recoveries,
    "schedule_tsab_fast_date_changes": solution.schedule_tsab_fast_date_changes,
    "schedule_tsab_fast_queue_pops": solution.schedule_tsab_fast_queue_pops,
    "schedule_tsab_fast_full_validations": solution.schedule_tsab_fast_full_validations,
    "schedule_tsab_fast_oracle_mismatches": solution.schedule_tsab_fast_oracle_mismatches,
    "schedule_tsab_fast_pending_promotions": solution.schedule_tsab_fast_pending_promotions,
    "schedule_tsab_fast_pending_discards": solution.schedule_tsab_fast_pending_discards,
    "schedule_tsab_fast_transitions": solution.schedule_tsab_fast_transitions,
    "schedule_tsab_fast_work_units": solution.schedule_tsab_fast_work_units,
    "schedule_tsab_fast_elapsed_seconds": solution.schedule_tsab_fast_elapsed_seconds,
    "schedule_tsab_fast_workspace_peak_bytes": solution.schedule_tsab_fast_workspace_peak_bytes,
    "schedule_session_initializations": solution.schedule_session_initializations,
    "schedule_session_resumes": solution.schedule_session_resumes,
    "schedule_session_rebases": solution.schedule_session_rebases,
    "schedule_island_profile_mask": solution.schedule_island_profile_mask,
    "schedule_baseline_island_profile_mask": solution.schedule_baseline_island_profile_mask,
    "schedule_scored_island_profile_mask": solution.schedule_scored_island_profile_mask,
    "schedule_island_profile_count": solution.schedule_island_profile_count,
    "schedule_profile_construction_seconds": solution.schedule_profile_construction_seconds,
    "schedule_profile_initial_objectives": solution.schedule_profile_initial_objectives,
    "schedule_profile_initial_dispatch_rules": solution.schedule_profile_initial_dispatch_rules,
    "schedule_profile_best_objectives": solution.schedule_profile_best_objectives,
    "schedule_profile_work_steps": solution.schedule_profile_work_steps,
    "schedule_construction_bucket_visits": solution.schedule_construction_bucket_visits,
    "schedule_construction_heap_pushes": solution.schedule_construction_heap_pushes,
    "schedule_construction_stale_pops": solution.schedule_construction_stale_pops,
    "schedule_construction_heap_rebuilds": solution.schedule_construction_heap_rebuilds,
    "schedule_construction_heap_peak": solution.schedule_construction_heap_peak,
    "schedule_reactive_restarts": solution.schedule_reactive_restarts,
    "schedule_reactive_restart_dispatches": solution.schedule_reactive_restart_dispatches,
    "schedule_reactive_restart_perturbations": solution.schedule_reactive_restart_perturbations,
    "schedule_reactive_restart_rebuild_failures": solution.schedule_reactive_restart_rebuild_failures,
    "schedule_island_scored_candidates": solution.schedule_island_scored_candidates,
    "schedule_island_shortlisted_candidates": solution.schedule_island_shortlisted_candidates,
    "schedule_approximate_candidates_generated": solution.schedule_approximate_candidates_generated,
    "schedule_approximate_candidates_refined": solution.schedule_approximate_candidates_refined,
    "schedule_approximate_candidates_certified": solution.schedule_approximate_candidates_certified,
    "schedule_approximate_candidates_unknown": solution.schedule_approximate_candidates_unknown,
    "schedule_approximation_score_items": solution.schedule_approximation_score_items,
    "schedule_approximation_sort_items": solution.schedule_approximation_sort_items,
    "schedule_approximation_local_span_items": solution.schedule_approximation_local_span_items,
    "schedule_approximation_elapsed_seconds": solution.schedule_approximation_elapsed_seconds,
    "schedule_approximation_work_units": solution.schedule_approximation_work_units,
    "schedule_direct_oracle_attempts": solution.schedule_direct_oracle_attempts,
    "schedule_direct_oracle_accepts": solution.schedule_direct_oracle_accepts,
    "schedule_direct_oracle_cycles": solution.schedule_direct_oracle_cycles,
    "schedule_direct_oracle_windows": solution.schedule_direct_oracle_windows,
    "schedule_direct_oracle_objective_rejections": solution.schedule_direct_oracle_objective_rejections,
    "schedule_exact_probes_avoided": solution.schedule_exact_probes_avoided,
    "schedule_search_elite_pool_size": solution.schedule_search_elite_pool_size,
    "schedule_search_elite_batches": solution.schedule_search_elite_batches,
    "schedule_search_elite_batches_skipped_after_stop": solution.schedule_search_elite_batches_skipped_after_stop,
    "schedule_search_elite_candidates": solution.schedule_search_elite_candidates,
    "schedule_search_elite_insertions": solution.schedule_search_elite_insertions,
    "schedule_search_elite_duplicates": solution.schedule_search_elite_duplicates,
    "schedule_search_elite_dominated": solution.schedule_search_elite_dominated,
    "schedule_search_elite_evictions": solution.schedule_search_elite_evictions,
    "schedule_search_elite_interruptions": solution.schedule_search_elite_interruptions,
    "schedule_search_elite_merge_errors": solution.schedule_search_elite_merge_errors,
    "schedule_search_elite_snapshot_captures": solution.schedule_search_elite_snapshot_captures,
    "schedule_search_elite_snapshot_interruptions": solution.schedule_search_elite_snapshot_interruptions,
    "schedule_search_elite_snapshot_errors": solution.schedule_search_elite_snapshot_errors,
    "schedule_search_elite_objectives": solution.schedule_search_elite_objectives,
    "schedule_search_elite_pairwise_distances_ppm": solution.schedule_search_elite_pairwise_distances_ppm,
    "schedule_search_elite_min_distance_ppm": solution.schedule_search_elite_min_distance_ppm,
    "schedule_search_elite_mean_distance_ppm": solution.schedule_search_elite_mean_distance_ppm,
    "schedule_search_elite_max_distance_ppm": solution.schedule_search_elite_max_distance_ppm,
    "schedule_search_elite_capture_worker_seconds_sum": solution.schedule_search_elite_capture_worker_seconds_sum,
    "schedule_search_elite_merge_wall_seconds": solution.schedule_search_elite_merge_wall_seconds,
    "schedule_search_elite_heap_lower_bound_bytes": solution.schedule_search_elite_heap_lower_bound_bytes,
    "schedule_search_elite_peak_heap_lower_bound_bytes": solution.schedule_search_elite_peak_heap_lower_bound_bytes,
    "schedule_lns_shadow_enabled": solution.schedule_lns_shadow_enabled,
    "schedule_lns_shadow_active": solution.schedule_lns_shadow_active,
    "schedule_lns_shadow_owner_worker_mask": solution.schedule_lns_shadow_owner_worker_mask,
    "schedule_lns_attempts": solution.schedule_lns_attempts,
    "schedule_lns_selected_operations": solution.schedule_lns_selected_operations,
    "schedule_lns_feasible": solution.schedule_lns_feasible,
    "schedule_lns_reconstructed": solution.schedule_lns_reconstructed,
    "schedule_lns_improvements": solution.schedule_lns_improvements,
    "schedule_lns_shadow_improvements": solution.schedule_lns_shadow_improvements,
    "schedule_lns_shadow_improvement_sum": solution.schedule_lns_shadow_improvement_sum,
    "schedule_lns_shadow_best_improvement": solution.schedule_lns_shadow_best_improvement,
    "schedule_lns_timeouts": solution.schedule_lns_timeouts,
    "schedule_lns_interruptions": solution.schedule_lns_interruptions,
    "schedule_lns_infeasible": solution.schedule_lns_infeasible,
    "schedule_lns_non_improving": solution.schedule_lns_non_improving,
    "schedule_lns_reconstruction_rejections": solution.schedule_lns_reconstruction_rejections,
    "schedule_lns_oracle_rejections": solution.schedule_lns_oracle_rejections,
    "schedule_lns_exact_rejections": solution.schedule_lns_exact_rejections,
    "schedule_lns_worker_seconds_sum": solution.schedule_lns_worker_seconds_sum,
    "schedule_lns_workspace_peak_bytes": solution.schedule_lns_workspace_peak_bytes,
    "schedule_constructor_workers_requested": solution.schedule_constructor_workers_requested,
    "schedule_constructor_multistart_enabled": solution.schedule_constructor_multistart_enabled,
    "schedule_constructor_multistart_active": solution.schedule_constructor_multistart_active,
    "schedule_constructor_multistart_owner_worker_mask": solution.schedule_constructor_multistart_owner_worker_mask,
    "schedule_constructor_multistart_attempts": solution.schedule_constructor_multistart_attempts,
    "schedule_constructor_multistart_constructions": solution.schedule_constructor_multistart_constructions,
    "schedule_constructor_multistart_interruptions": solution.schedule_constructor_multistart_interruptions,
    "schedule_constructor_multistart_failures": solution.schedule_constructor_multistart_failures,
    "schedule_constructor_multistart_feasible": solution.schedule_constructor_multistart_feasible,
    "schedule_constructor_multistart_distinct_fingerprints": solution.schedule_constructor_multistart_distinct_fingerprints,
    "schedule_constructor_multistart_other_fingerprint_observations": solution.schedule_constructor_multistart_other_fingerprint_observations,
    "schedule_constructor_multistart_initial_objective": solution.schedule_constructor_multistart_initial_objective,
    "schedule_constructor_multistart_best_objective": solution.schedule_constructor_multistart_best_objective,
    "schedule_constructor_multistart_improvements": solution.schedule_constructor_multistart_improvements,
    "schedule_constructor_multistart_work_units": solution.schedule_constructor_multistart_work_units,
    "schedule_constructor_multistart_worker_seconds_sum": solution.schedule_constructor_multistart_worker_seconds_sum,
    "schedule_constructor_multistart_workspace_peak_bytes": solution.schedule_constructor_multistart_workspace_peak_bytes,
    "schedule_constructor_multistart_best_ordinal": solution.schedule_constructor_multistart_best_ordinal,
    "schedule_constructor_multistart_best_seed": solution.schedule_constructor_multistart_best_seed,
    "schedule_constructor_multistart_best_fingerprint": solution.schedule_constructor_multistart_best_fingerprint,
    "schedule_constructor_multistart_next_ordinal": solution.schedule_constructor_multistart_next_ordinal,
    "schedule_path_relink_enabled": solution.schedule_path_relink_enabled,
    "schedule_path_relink_active": solution.schedule_path_relink_active,
    "schedule_path_relink_guide_requests": solution.schedule_path_relink_guide_requests,
    "schedule_path_relink_best_guides": solution.schedule_path_relink_best_guides,
    "schedule_path_relink_diverse_guides": solution.schedule_path_relink_diverse_guides,
    "schedule_path_relink_guide_loads": solution.schedule_path_relink_guide_loads,
    "schedule_path_relink_guide_incompatible": solution.schedule_path_relink_guide_incompatible,
    "schedule_path_relink_guide_interruptions": solution.schedule_path_relink_guide_interruptions,
    "schedule_path_relink_critical_operations_scanned": solution.schedule_path_relink_critical_operations_scanned,
    "schedule_path_relink_candidates_generated": solution.schedule_path_relink_candidates_generated,
    "schedule_path_relink_candidates_positive_gain": solution.schedule_path_relink_candidates_positive_gain,
    "schedule_path_relink_acyclicity_certified": solution.schedule_path_relink_acyclicity_certified,
    "schedule_path_relink_acyclicity_unknown": solution.schedule_path_relink_acyclicity_unknown,
    "schedule_path_relink_prefilter_rejections": solution.schedule_path_relink_prefilter_rejections,
    "schedule_path_relink_candidates_retained": solution.schedule_path_relink_candidates_retained,
    "schedule_path_relink_candidates_refined": solution.schedule_path_relink_candidates_refined,
    "schedule_path_relink_candidates_shortlisted": solution.schedule_path_relink_candidates_shortlisted,
    "schedule_path_relink_no_move": solution.schedule_path_relink_no_move,
    "schedule_path_relink_guide_arc_gain_shortlisted": solution.schedule_path_relink_guide_arc_gain_shortlisted,
    "schedule_path_relink_oracle_attempts": solution.schedule_path_relink_oracle_attempts,
    "schedule_path_relink_oracle_accepts": solution.schedule_path_relink_oracle_accepts,
    "schedule_path_relink_cycle_rejections": solution.schedule_path_relink_cycle_rejections,
    "schedule_path_relink_window_rejections": solution.schedule_path_relink_window_rejections,
    "schedule_path_relink_other_rejections": solution.schedule_path_relink_other_rejections,
    "schedule_path_relink_rollbacks": solution.schedule_path_relink_rollbacks,
    "schedule_path_relink_elite_improvements": solution.schedule_path_relink_elite_improvements,
    "schedule_path_relink_guide_arc_gain_accepted": solution.schedule_path_relink_guide_arc_gain_accepted,
    "schedule_path_relink_worker_seconds_sum": solution.schedule_path_relink_worker_seconds_sum,
    "schedule_path_relink_workspace_peak_bytes": solution.schedule_path_relink_workspace_peak_bytes,
    "estimated_backend_bytes": solution.estimated_backend_bytes,
    "constructor": solution.constructor,
    "memory_limit_mb": args.memory_limit_mb,
    "schedule_jssp_search": args.schedule_jssp_search,
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
