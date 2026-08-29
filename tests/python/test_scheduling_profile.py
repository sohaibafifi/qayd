"""Public scheduling local-search profile coverage."""

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

cp = pytest.importorskip("qayd")


ROOT = Path(__file__).resolve().parents[2]
SCHEDULE_COUNTERS = (
    "schedule_moves_considered",
    "schedule_moves_accepted",
    "schedule_moves_rejected",
    "schedule_work_steps",
    "schedule_local_improvements",
    "schedule_global_improvements",
    "schedule_progress_publications",
    "schedule_incumbent_publication_attempts",
    "schedule_incumbent_publications",
    "schedule_incumbent_injection_attempts",
    "schedule_incumbent_injections",
    "schedule_incumbent_rejections",
    "schedule_incumbent_verifications",
    "schedule_incumbent_verification_rejections",
    "schedule_incumbent_verification_interruptions",
    "schedule_incumbent_incomplete_rejections",
    "schedule_restart_boundaries",
    "schedule_peak_buffered_candidates",
    "schedule_stalled_workers",
    "schedule_stalled_unused_work_steps",
    "schedule_worker_work_min",
    "schedule_worker_work_max",
    "schedule_work_budget_overruns",
    "schedule_elite_pool_size",
    "schedule_unused_work_steps",
    "schedule_incumbent_source_worker",
    "schedule_incumbent_source_round",
    "schedule_cycle_rejections",
    "schedule_reconstructions",
    "critical_path_updates",
    "schedule_window_rejections",
    "schedule_objective_rejections",
    "resource_profile_checks",
    "resource_candidate_scheduling_attempts",
    "resource_event_visits",
    "resource_peak_profile_events",
    "schedule_precedence_rejections",
    "schedule_infeasible_rejections",
    "schedule_justification_attempts",
    "schedule_delta_evaluations",
    "schedule_full_evaluations",
    "schedule_full_fallbacks",
    "schedule_topological_rebuilds",
    "schedule_oracle_validations",
    "schedule_oracle_mismatches",
    "schedule_dirty_cone_operations",
    "schedule_max_dirty_cone",
    "schedule_workspace_growths",
    "schedule_workspace_rollbacks",
    "schedule_alns_generation_attempts",
    "schedule_alns_moves_generated",
)
JSSP_VERIFICATION_TIMINGS = (
    "schedule_incumbent_verification_seconds",
    "schedule_incumbent_verification_max_seconds",
)
JSSP_STRATEGY_COUNTERS = (
    "schedule_tabu_steps",
    "schedule_tabu_hits",
    "schedule_tabu_aspirations",
    "schedule_tabu_forced_moves",
    "schedule_session_initializations",
    "schedule_session_resumes",
    "schedule_session_rebases",
)
JSSP_TSAB_COUNTERS = (
    "schedule_tsab_owner_worker_mask",
    "schedule_tsab_n5_generated",
    "schedule_tsab_ranked",
    "schedule_tsab_shortlists",
    "schedule_tsab_delta_probes",
    "schedule_tsab_additional_delta_probes",
    "schedule_tsab_full_oracle_commits",
    "schedule_tsab_selected_shortlist_rank_sum",
    "schedule_tsab_selections",
    "schedule_tsab_aspirations",
    "schedule_tsab_tabu_rejections",
    "schedule_tsab_tabu_resets",
    "schedule_tsab_fingerprint_repeats",
    "schedule_tsab_escape_signals",
    "schedule_tsab_n1_kicks",
    "schedule_tsab_kick_moves",
    "schedule_tsab_elite_restarts",
    "schedule_tsab_n6_kicks",
    "schedule_tsab_restart_attempts",
    "schedule_tsab_restart_global_rebases",
    "schedule_tsab_restart_n6_generated",
    "schedule_tsab_restart_delta_probes",
    "schedule_tsab_restart_oracle_commits",
    "schedule_tsab_restart_rejections",
    "schedule_tsab_restart_interruptions",
    "schedule_tsab_restart_work_units",
    "schedule_tsab_post_restart_improvements",
    "schedule_tsab_restart_shortlist_peak_bytes",
    "schedule_tsab_ranking_audits",
    "schedule_tsab_exact_best_matches",
    "schedule_tsab_regret_sum",
    "schedule_tsab_regret_max",
    "schedule_tsab_workspace_peak_bytes",
    "schedule_tsab_activations",
    "schedule_tsab_legacy_warmup_work_steps",
    "schedule_tsab_activation_rebases",
    "schedule_tsab_active_boundaries",
    "schedule_tsab_burst_work_limit",
    "schedule_tsab_burst_work_units",
    "schedule_tsab_improving_commits",
    "schedule_tsab_fast_enabled",
    "schedule_tsab_fast_eligible",
    "schedule_tsab_fast_disabled",
    "schedule_tsab_fast_attempts",
    "schedule_tsab_fast_commits",
    "schedule_tsab_fast_fallbacks",
    "schedule_tsab_fast_date_changes",
    "schedule_tsab_fast_queue_pops",
    "schedule_tsab_fast_full_validations",
    "schedule_tsab_fast_oracle_mismatches",
    "schedule_tsab_fast_pending_promotions",
    "schedule_tsab_fast_pending_discards",
    "schedule_tsab_fast_transitions",
    "schedule_tsab_fast_work_units",
    "schedule_tsab_fast_workspace_peak_bytes",
)
JSSP_TSAB_TIMINGS = ("schedule_tsab_fast_elapsed_seconds",)
JSSP_TSAB_OPTIONAL_COUNTERS = (
    "schedule_tsab_activation_boundary",
    "schedule_tsab_activation_objective",
    "schedule_tsab_best_committed_objective",
    "schedule_tsab_restart_best_base_objective",
    "schedule_tsab_restart_best_kicked_objective",
)
JSSP_ISLAND_COUNTERS = (
    "schedule_island_profile_mask",
    "schedule_baseline_island_profile_mask",
    "schedule_scored_island_profile_mask",
    "schedule_island_profile_count",
    "schedule_reactive_restarts",
    "schedule_reactive_restart_dispatches",
    "schedule_reactive_restart_perturbations",
    "schedule_reactive_restart_rebuild_failures",
    "schedule_island_scored_candidates",
    "schedule_island_shortlisted_candidates",
)
JSSP_GT_COUNTERS = (
    "schedule_construction_bucket_visits",
    "schedule_construction_heap_pushes",
    "schedule_construction_stale_pops",
    "schedule_construction_heap_rebuilds",
    "schedule_construction_heap_peak",
)
JSSP_ORACLE_COUNTERS = (
    "schedule_approximate_candidates_generated",
    "schedule_approximate_candidates_refined",
    "schedule_approximate_candidates_certified",
    "schedule_approximate_candidates_unknown",
    "schedule_approximation_score_items",
    "schedule_approximation_sort_items",
    "schedule_approximation_local_span_items",
    "schedule_approximation_elapsed_seconds",
    "schedule_approximation_work_units",
    "schedule_direct_oracle_attempts",
    "schedule_direct_oracle_accepts",
    "schedule_direct_oracle_cycles",
    "schedule_direct_oracle_windows",
    "schedule_direct_oracle_objective_rejections",
    "schedule_exact_probes_avoided",
)
JSSP_SEARCH_ELITE_COUNTERS = (
    "schedule_search_elite_pool_size",
    "schedule_search_elite_batches",
    "schedule_search_elite_batches_skipped_after_stop",
    "schedule_search_elite_candidates",
    "schedule_search_elite_insertions",
    "schedule_search_elite_duplicates",
    "schedule_search_elite_dominated",
    "schedule_search_elite_evictions",
    "schedule_search_elite_interruptions",
    "schedule_search_elite_merge_errors",
    "schedule_search_elite_snapshot_captures",
    "schedule_search_elite_snapshot_interruptions",
    "schedule_search_elite_snapshot_errors",
    "schedule_search_elite_heap_lower_bound_bytes",
    "schedule_search_elite_peak_heap_lower_bound_bytes",
)
JSSP_SEARCH_ELITE_TIMINGS = (
    "schedule_search_elite_capture_worker_seconds_sum",
    "schedule_search_elite_merge_wall_seconds",
)
JSSP_SEARCH_ELITE_FIELDS = (
    *JSSP_SEARCH_ELITE_COUNTERS,
    *JSSP_SEARCH_ELITE_TIMINGS,
    "schedule_search_elite_objectives",
    "schedule_search_elite_pairwise_distances_ppm",
    "schedule_search_elite_min_distance_ppm",
    "schedule_search_elite_mean_distance_ppm",
    "schedule_search_elite_max_distance_ppm",
)
JSSP_PATH_RELINK_COUNTERS = (
    "schedule_path_relink_enabled",
    "schedule_path_relink_active",
    "schedule_path_relink_guide_requests",
    "schedule_path_relink_best_guides",
    "schedule_path_relink_diverse_guides",
    "schedule_path_relink_guide_loads",
    "schedule_path_relink_guide_incompatible",
    "schedule_path_relink_guide_interruptions",
    "schedule_path_relink_critical_operations_scanned",
    "schedule_path_relink_candidates_generated",
    "schedule_path_relink_candidates_positive_gain",
    "schedule_path_relink_acyclicity_certified",
    "schedule_path_relink_acyclicity_unknown",
    "schedule_path_relink_prefilter_rejections",
    "schedule_path_relink_candidates_retained",
    "schedule_path_relink_candidates_refined",
    "schedule_path_relink_candidates_shortlisted",
    "schedule_path_relink_no_move",
    "schedule_path_relink_guide_arc_gain_shortlisted",
    "schedule_path_relink_oracle_attempts",
    "schedule_path_relink_oracle_accepts",
    "schedule_path_relink_cycle_rejections",
    "schedule_path_relink_window_rejections",
    "schedule_path_relink_other_rejections",
    "schedule_path_relink_rollbacks",
    "schedule_path_relink_elite_improvements",
    "schedule_path_relink_guide_arc_gain_accepted",
    "schedule_path_relink_workspace_peak_bytes",
)
JSSP_PATH_RELINK_FIELDS = (
    *JSSP_PATH_RELINK_COUNTERS,
    "schedule_path_relink_worker_seconds_sum",
)
JSSP_LNS_SHADOW_COUNTERS = (
    "schedule_lns_shadow_enabled",
    "schedule_lns_shadow_active",
    "schedule_lns_shadow_owner_worker_mask",
    "schedule_lns_attempts",
    "schedule_lns_selected_operations",
    "schedule_lns_feasible",
    "schedule_lns_reconstructed",
    "schedule_lns_improvements",
    "schedule_lns_shadow_improvements",
    "schedule_lns_shadow_improvement_sum",
    "schedule_lns_shadow_best_improvement",
    "schedule_lns_timeouts",
    "schedule_lns_interruptions",
    "schedule_lns_infeasible",
    "schedule_lns_non_improving",
    "schedule_lns_reconstruction_rejections",
    "schedule_lns_oracle_rejections",
    "schedule_lns_exact_rejections",
    "schedule_lns_workspace_peak_bytes",
)
JSSP_LNS_SHADOW_FIELDS = (
    *JSSP_LNS_SHADOW_COUNTERS,
    "schedule_lns_worker_seconds_sum",
)
JSSP_CONSTRUCTOR_COUNTERS = (
    "schedule_constructor_workers_requested",
    "schedule_constructor_multistart_enabled",
    "schedule_constructor_multistart_active",
    "schedule_constructor_multistart_owner_worker_mask",
    "schedule_constructor_multistart_attempts",
    "schedule_constructor_multistart_constructions",
    "schedule_constructor_multistart_interruptions",
    "schedule_constructor_multistart_failures",
    "schedule_constructor_multistart_feasible",
    "schedule_constructor_multistart_distinct_fingerprints",
    "schedule_constructor_multistart_other_fingerprint_observations",
    "schedule_constructor_multistart_initial_objective",
    "schedule_constructor_multistart_best_objective",
    "schedule_constructor_multistart_improvements",
    "schedule_constructor_multistart_work_units",
    "schedule_constructor_multistart_workspace_peak_bytes",
    "schedule_constructor_multistart_best_ordinal",
    "schedule_constructor_multistart_best_seed",
    "schedule_constructor_multistart_best_fingerprint",
    "schedule_constructor_multistart_next_ordinal",
)
JSSP_CONSTRUCTOR_FIELDS = (
    *JSSP_CONSTRUCTOR_COUNTERS,
    "schedule_constructor_multistart_worker_seconds_sum",
)


def _profile_construction_times(encoded):
    return {
        int(profile): float(seconds)
        for item in encoded.split(",")
        for profile, seconds in (item.split("=", 1),)
    }


def _profile_best_objectives(encoded):
    return {
        int(profile): int(objective)
        for item in encoded.split(",")
        for profile, objective in (item.split("=", 1),)
    }


def _profile_strings(encoded):
    return {
        int(profile): value
        for item in encoded.split(",")
        for profile, value in (item.split("=", 1),)
    }


def _field(record, name):
    return record[name] if isinstance(record, dict) else getattr(record, name)


def _assert_search_elite_profile(record):
    pool_size = _field(record, "schedule_search_elite_pool_size")
    batches = _field(record, "schedule_search_elite_batches")
    candidates = _field(record, "schedule_search_elite_candidates")
    insertions = _field(record, "schedule_search_elite_insertions")
    duplicates = _field(record, "schedule_search_elite_duplicates")
    dominated = _field(record, "schedule_search_elite_dominated")
    evictions = _field(record, "schedule_search_elite_evictions")

    assert 1 <= pool_size <= 4
    assert batches > 0
    assert _field(record, "schedule_search_elite_batches_skipped_after_stop") == 0
    assert candidates >= pool_size
    assert candidates == insertions + duplicates + dominated
    assert evictions <= insertions
    assert _field(record, "schedule_search_elite_interruptions") >= 0
    assert _field(record, "schedule_search_elite_merge_errors") == 0
    assert _field(record, "schedule_search_elite_snapshot_interruptions") >= 0
    assert _field(record, "schedule_search_elite_snapshot_errors") == 0
    assert _field(record, "schedule_search_elite_snapshot_captures") >= candidates

    objectives = [int(value) for value in _field(record, "schedule_search_elite_objectives").split(",")]
    assert len(objectives) == pool_size
    assert objectives[0] == min(objectives)

    encoded_distances = _field(record, "schedule_search_elite_pairwise_distances_ppm")
    distance_fields = (
        "schedule_search_elite_min_distance_ppm",
        "schedule_search_elite_mean_distance_ppm",
        "schedule_search_elite_max_distance_ppm",
    )
    if pool_size == 1:
        assert encoded_distances == "none"
        assert all(_field(record, field) is None for field in distance_fields)
    else:
        distances = []
        for encoded in encoded_distances.split(","):
            pair, distance = encoded.split("=", 1)
            assert pair.count("-") == 1
            distances.append(int(distance))
        assert len(distances) == pool_size * (pool_size - 1) // 2
        assert all(0 <= distance <= 1_000_000 for distance in distances)
        assert _field(record, distance_fields[0]) == min(distances)
        assert _field(record, distance_fields[1]) == sum(distances) // len(distances)
        assert _field(record, distance_fields[2]) == max(distances)

    capture_worker_seconds_sum = _field(record, "schedule_search_elite_capture_worker_seconds_sum")
    merge_wall_seconds = _field(record, "schedule_search_elite_merge_wall_seconds")
    assert all(math.isfinite(value) and value >= 0.0 for value in (capture_worker_seconds_sum, merge_wall_seconds))
    current_bytes = _field(record, "schedule_search_elite_heap_lower_bound_bytes")
    peak_bytes = _field(record, "schedule_search_elite_peak_heap_lower_bound_bytes")
    assert current_bytes > 0
    assert peak_bytes >= current_bytes


def _solve_job_shop(
    *,
    profile,
    threads=2,
    max_iterations=16,
    schedule_path_relink=False,
    schedule_lns_shadow=False,
    schedule_constructor_workers=0,
    schedule_jssp_search="legacy",
):
    model = cp.Model()
    tasks = model.tasks(range(6))
    durations = (3, 2, 2, 2, 1, 4)
    machines = (0, 1, 2, 1, 2, 0)
    for task in tasks:
        task.duration = durations[task.id]
        task.machine = machines[task.id]

    schedule = model.schedule(tasks, horizon=sum(durations))
    model.add(schedule[0].end <= schedule[1].start)
    model.add(schedule[1].end <= schedule[2].start)
    model.add(schedule[3].end <= schedule[4].start)
    model.add(schedule[4].end <= schedule[5].start)
    model.add(schedule.no_overlap(lambda task: task.machine))
    model.minimize(schedule.makespan())
    return model.solve(
        engine="ls",
        seed=11,
        threads=threads,
        time_limit=1,
        max_iterations=max_iterations,
        profile=profile,
        schedule_path_relink=schedule_path_relink,
        schedule_lns_shadow=schedule_lns_shadow,
        schedule_constructor_workers=schedule_constructor_workers,
        schedule_jssp_search=schedule_jssp_search,
    )


def _solve_diversified_job_shop_with_path_relink(*, schedule_path_relink=True, schedule_jssp_search="legacy"):
    model = cp.Model()
    tasks = model.tasks(range(12))
    durations = (3, 2, 2, 2, 1, 4, 4, 3, 1, 2, 3, 3)
    machines = (0, 1, 2, 0, 2, 1, 1, 0, 2, 2, 1, 0)
    for task in tasks:
        task.duration = durations[task.id]
        task.machine = machines[task.id]

    schedule = model.schedule(tasks, horizon=sum(durations))
    for before, after in ((0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8), (9, 10), (10, 11)):
        model.add(schedule[before].end <= schedule[after].start)
    model.add(schedule.no_overlap(lambda task: task.machine))
    model.minimize(schedule.makespan())
    return model.solve(
        engine="ls",
        seed=11,
        threads=7,
        time_limit=1,
        max_iterations=50_000,
        profile=True,
        schedule_path_relink=schedule_path_relink,
        schedule_jssp_search=schedule_jssp_search,
    )


def _solve_job_shop_with_lns_shadow():
    model = cp.Model()
    tasks = model.tasks(range(12))
    durations = (3, 2, 2, 2, 1, 4, 4, 3, 1, 2, 3, 3)
    machines = (0, 1, 2, 0, 2, 1, 1, 0, 2, 2, 1, 0)
    for task in tasks:
        task.duration = durations[task.id]
        task.machine = machines[task.id]

    schedule = model.schedule(tasks, horizon=sum(durations))
    for before, after in ((0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8), (9, 10), (10, 11)):
        model.add(schedule[before].end <= schedule[after].start)
    model.add(schedule.no_overlap(lambda task: task.machine))
    model.minimize(schedule.makespan())
    return model.solve(
        engine="ls",
        seed=17,
        threads=1,
        time_limit=1,
        profile=True,
        schedule_lns_shadow=True,
    )


def _solve_job_shop_with_constructor_multistart():
    model = cp.Model()
    tasks = model.tasks(range(12))
    durations = (3, 2, 2, 2, 1, 4, 4, 3, 1, 2, 3, 3)
    machines = (0, 1, 2, 0, 2, 1, 1, 0, 2, 2, 1, 0)
    for task in tasks:
        task.duration = durations[task.id]
        task.machine = machines[task.id]
    schedule = model.schedule(tasks, horizon=sum(durations))
    for before, after in ((0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8), (9, 10), (10, 11)):
        model.add(schedule[before].end <= schedule[after].start)
    model.add(schedule.no_overlap(lambda task: task.machine))
    model.minimize(schedule.makespan())
    return model.solve(
        engine="ls",
        seed=19,
        threads=7,
        time_limit=1,
        profile=True,
        schedule_constructor_workers=1,
    )


def test_schedule_profile_is_present_only_when_requested():
    plain = _solve_job_shop(profile=False)
    profiled = _solve_job_shop(profile=True)

    assert plain.status == profiled.status == "SATISFIABLE"
    assert plain.candidates_evaluated is None
    assert plain.candidates_per_second is None
    assert plain.full_recompute_percentage is None
    assert plain.schedule_restart_work is None
    assert plain.schedule_profile_construction_seconds is None
    assert plain.schedule_profile_initial_objectives is None
    assert plain.schedule_profile_initial_dispatch_rules is None
    assert plain.schedule_profile_best_objectives is None
    assert plain.schedule_profile_work_steps is None
    assert all(getattr(plain, field) is None for field in SCHEDULE_COUNTERS)
    assert all(getattr(plain, field) is None for field in JSSP_VERIFICATION_TIMINGS)
    assert all(getattr(plain, field) is None for field in JSSP_STRATEGY_COUNTERS)
    assert all(getattr(plain, field) is None for field in (*JSSP_TSAB_COUNTERS, *JSSP_TSAB_TIMINGS, *JSSP_TSAB_OPTIONAL_COUNTERS))
    assert all(getattr(plain, field) is None for field in JSSP_ISLAND_COUNTERS)
    assert all(getattr(plain, field) is None for field in JSSP_GT_COUNTERS)
    assert all(getattr(plain, field) is None for field in JSSP_ORACLE_COUNTERS)
    assert all(getattr(plain, field) is None for field in JSSP_SEARCH_ELITE_FIELDS)
    assert all(getattr(plain, field) is None for field in JSSP_LNS_SHADOW_FIELDS)
    assert all(getattr(plain, field) is None for field in JSSP_CONSTRUCTOR_FIELDS)
    assert all(getattr(plain, field) is None for field in JSSP_PATH_RELINK_FIELDS)

    assert profiled.candidates_evaluated == profiled.schedule_moves_considered
    assert profiled.schedule_moves_considered >= 0
    assert profiled.schedule_moves_accepted >= 0
    assert profiled.schedule_moves_rejected >= 0
    assert profiled.schedule_moves_accepted + profiled.schedule_moves_rejected == profiled.schedule_moves_considered
    assert profiled.schedule_cycle_rejections >= 0
    assert profiled.schedule_reconstructions >= 0
    assert profiled.critical_path_updates >= 0
    assert all(getattr(profiled, field) is not None and getattr(profiled, field) >= 0 for field in SCHEDULE_COUNTERS)
    assert all(getattr(profiled, field) is not None and getattr(profiled, field) >= 0 for field in JSSP_STRATEGY_COUNTERS)
    assert all(getattr(profiled, field) == 0 for field in JSSP_TSAB_COUNTERS)
    assert all(getattr(profiled, field) == 0.0 for field in JSSP_TSAB_TIMINGS)
    assert all(getattr(profiled, field) is None for field in JSSP_TSAB_OPTIONAL_COUNTERS)
    assert all(getattr(profiled, field) is not None and getattr(profiled, field) >= 0 for field in JSSP_ISLAND_COUNTERS)
    assert all(getattr(profiled, field) is not None and getattr(profiled, field) >= 0 for field in JSSP_GT_COUNTERS)
    assert all(getattr(profiled, field) is not None and getattr(profiled, field) >= 0 for field in JSSP_ORACLE_COUNTERS)
    assert all(getattr(profiled, field) is not None and getattr(profiled, field) >= 0 for field in JSSP_SEARCH_ELITE_COUNTERS)
    assert all(getattr(profiled, field) is not None for field in JSSP_SEARCH_ELITE_TIMINGS)
    assert all(getattr(profiled, field) is not None and getattr(profiled, field) >= 0 for field in JSSP_LNS_SHADOW_COUNTERS)
    assert math.isfinite(profiled.schedule_lns_worker_seconds_sum)
    assert profiled.schedule_lns_worker_seconds_sum >= 0.0
    assert profiled.schedule_lns_shadow_enabled == 0
    assert profiled.schedule_lns_shadow_active == 0
    assert all(getattr(profiled, field) is not None and getattr(profiled, field) >= 0 for field in JSSP_PATH_RELINK_COUNTERS)
    assert math.isfinite(profiled.schedule_path_relink_worker_seconds_sum)
    assert profiled.schedule_path_relink_worker_seconds_sum >= 0.0
    assert profiled.schedule_path_relink_enabled == 0
    assert profiled.schedule_path_relink_active == 0
    _assert_search_elite_profile(profiled)
    assert profiled.schedule_session_initializations == 2
    if profiled.schedule_restart_boundaries > 1:
        assert profiled.schedule_session_resumes > 0
    assert profiled.schedule_tabu_steps > 0
    assert profiled.schedule_island_profile_mask.bit_count() == profiled.schedule_island_profile_count == 2
    assert profiled.schedule_baseline_island_profile_mask | profiled.schedule_scored_island_profile_mask == (
        profiled.schedule_island_profile_mask
    )
    assert profiled.schedule_baseline_island_profile_mask & profiled.schedule_scored_island_profile_mask == 0
    assert profiled.schedule_approximate_candidates_refined == (
        profiled.schedule_approximate_candidates_certified + profiled.schedule_approximate_candidates_unknown
    )
    direct_oracle_outcomes = (
        profiled.schedule_direct_oracle_accepts
        + profiled.schedule_direct_oracle_cycles
        + profiled.schedule_direct_oracle_windows
        + profiled.schedule_direct_oracle_objective_rejections
    )
    assert direct_oracle_outcomes <= profiled.schedule_direct_oracle_attempts
    assert profiled.schedule_direct_oracle_attempts + profiled.schedule_exact_probes_avoided <= (
        profiled.schedule_approximate_candidates_refined
    )
    assert profiled.schedule_island_scored_candidates == profiled.schedule_approximate_candidates_generated
    assert profiled.schedule_approximation_score_items == profiled.schedule_approximate_candidates_generated
    assert profiled.schedule_approximation_sort_items >= profiled.schedule_approximation_score_items
    assert math.isfinite(profiled.schedule_approximation_elapsed_seconds)
    profile_times = _profile_construction_times(profiled.schedule_profile_construction_seconds)
    assert set(profile_times) == {0, 1}
    assert all(math.isfinite(seconds) and seconds >= 0.0 for seconds in profile_times.values())
    profile_objectives = _profile_best_objectives(profiled.schedule_profile_best_objectives)
    assert set(profile_objectives) == {0, 1}
    assert all(objective > 0 for objective in profile_objectives.values())
    assert min(profile_objectives.values()) == profiled.objective
    profile_initial_objectives = _profile_best_objectives(profiled.schedule_profile_initial_objectives)
    assert set(profile_initial_objectives) == {0, 1}
    assert all(objective > 0 for objective in profile_initial_objectives.values())
    assert _profile_strings(profiled.schedule_profile_initial_dispatch_rules) == {
        0: "earliest-start",
        1: "earliest-start",
    }
    assert profiled.schedule_construction_heap_pushes > 0
    assert 0 < profiled.schedule_construction_heap_peak <= profiled.schedule_construction_heap_pushes
    assert profiled.schedule_global_improvements == profiled.schedule_progress_publications
    assert profiled.schedule_local_improvements >= profiled.schedule_global_improvements
    assert profiled.schedule_incumbent_publication_attempts == (
        profiled.schedule_incumbent_publications + profiled.schedule_incumbent_rejections
    )
    assert profiled.schedule_incumbent_injections <= profiled.schedule_incumbent_injection_attempts
    assert profiled.schedule_incumbent_verifications <= profiled.schedule_incumbent_publication_attempts
    assert profiled.schedule_incumbent_verification_rejections <= profiled.schedule_incumbent_rejections
    assert profiled.schedule_incumbent_verification_interruptions <= profiled.schedule_incumbent_rejections
    assert math.isfinite(profiled.schedule_incumbent_verification_seconds)
    assert math.isfinite(profiled.schedule_incumbent_verification_max_seconds)
    assert 0.0 <= profiled.schedule_incumbent_verification_max_seconds <= profiled.schedule_incumbent_verification_seconds
    assert profiled.schedule_incumbent_incomplete_rejections <= profiled.schedule_incumbent_rejections
    assert profiled.schedule_restart_boundaries > 0
    assert profiled.schedule_restart_work == 2_048
    assert 1 <= profiled.schedule_peak_buffered_candidates <= 2
    assert profiled.schedule_worker_work_min <= profiled.schedule_worker_work_max
    assert profiled.schedule_work_budget_overruns == 0
    assert profiled.schedule_elite_pool_size == 1
    assert profiled.schedule_unused_work_steps + profiled.schedule_work_steps == 16
    assert profiled.schedule_incumbent_source_worker < 2
    assert math.isfinite(profiled.candidates_per_second)
    assert profiled.candidates_per_second >= 0.0
    assert math.isfinite(profiled.full_recompute_percentage)
    assert 0.0 <= profiled.full_recompute_percentage <= 100.0


def test_scored_island_exposes_direct_oracle_accounting():
    profiled = _solve_job_shop(profile=True, threads=7, max_iterations=896)

    assert profiled.status == "SATISFIABLE"
    assert profiled.schedule_scored_island_profile_mask != 0
    assert profiled.schedule_direct_oracle_attempts > 0
    assert profiled.schedule_approximation_score_items > 0
    assert profiled.schedule_approximation_work_units >= profiled.schedule_approximation_score_items
    assert profiled.schedule_approximate_candidates_refined == (
        profiled.schedule_approximate_candidates_certified + profiled.schedule_approximate_candidates_unknown
    )
    classified = (
        profiled.schedule_direct_oracle_accepts
        + profiled.schedule_direct_oracle_cycles
        + profiled.schedule_direct_oracle_windows
        + profiled.schedule_direct_oracle_objective_rejections
    )
    assert classified <= profiled.schedule_direct_oracle_attempts
    assert set(_profile_best_objectives(profiled.schedule_profile_initial_objectives)) == set(range(7))
    assert _profile_strings(profiled.schedule_profile_initial_dispatch_rules) == {
        0: "earliest-start",
        1: "earliest-start",
        2: "earliest-start",
        3: "earliest-start",
        4: "earliest-start",
        5: "earliest-start",
        6: "earliest-start",
    }


def test_path_relink_control_exercises_a_profile_six_island():
    profiled = _solve_diversified_job_shop_with_path_relink()

    assert profiled.status == "SATISFIABLE"
    assert profiled.schedule_path_relink_enabled == 1
    assert profiled.schedule_path_relink_active == 1
    assert profiled.schedule_path_relink_guide_requests > 0
    assert profiled.schedule_path_relink_best_guides + profiled.schedule_path_relink_diverse_guides == (
        profiled.schedule_path_relink_guide_requests
    )
    assert profiled.schedule_path_relink_guide_incompatible == 0
    assert profiled.schedule_path_relink_guide_interruptions == 0
    assert profiled.schedule_path_relink_candidates_positive_gain == (
        profiled.schedule_path_relink_acyclicity_certified
        + profiled.schedule_path_relink_acyclicity_unknown
    )
    assert profiled.schedule_path_relink_acyclicity_unknown == profiled.schedule_path_relink_prefilter_rejections
    assert profiled.schedule_path_relink_candidates_retained <= profiled.schedule_path_relink_acyclicity_certified
    assert profiled.schedule_path_relink_cycle_rejections == 0
    assert profiled.schedule_path_relink_candidates_shortlisted <= profiled.schedule_path_relink_candidates_refined
    assert profiled.schedule_path_relink_oracle_accepts <= profiled.schedule_path_relink_oracle_attempts
    assert profiled.schedule_path_relink_elite_improvements <= profiled.schedule_path_relink_oracle_accepts
    assert math.isfinite(profiled.schedule_path_relink_worker_seconds_sum)
    assert profiled.schedule_path_relink_worker_seconds_sum >= 0.0


def test_lns_shadow_requires_wall_clock_search_and_never_publishes_repair():
    ineligible = _solve_job_shop(
        profile=True,
        max_iterations=4_096,
        schedule_lns_shadow=True,
    )
    assert ineligible.schedule_lns_shadow_enabled == 1
    assert ineligible.schedule_lns_shadow_active == 0
    assert ineligible.schedule_lns_shadow_owner_worker_mask == 0
    assert ineligible.schedule_lns_attempts == 0

    profiled = _solve_job_shop_with_lns_shadow()
    assert profiled.status == "SATISFIABLE"
    assert profiled.schedule_lns_shadow_enabled == 1
    assert profiled.schedule_lns_shadow_active == 1
    assert profiled.schedule_lns_shadow_owner_worker_mask == 1
    assert profiled.schedule_lns_attempts > 0
    assert profiled.schedule_lns_improvements == profiled.schedule_lns_shadow_improvements
    assert profiled.schedule_lns_interruptions == 0
    assert profiled.schedule_lns_oracle_rejections == 0
    assert profiled.schedule_lns_shadow_improvements <= profiled.schedule_lns_reconstructed
    assert profiled.schedule_lns_shadow_best_improvement <= profiled.schedule_lns_shadow_improvement_sum
    assert math.isfinite(profiled.schedule_lns_worker_seconds_sum)
    assert profiled.schedule_lns_worker_seconds_sum >= 0.0
    assert 0 < profiled.schedule_lns_workspace_peak_bytes < 512 * 1024 * 1024


def test_tsab_candidate_is_phased_bounded_and_reserves_only_worker_six():
    profiled = _solve_diversified_job_shop_with_path_relink(
        schedule_path_relink=False,
        schedule_jssp_search="tsab-candidate",
    )

    assert profiled.status == "SATISFIABLE"
    assert profiled.schedule_tsab_owner_worker_mask == 0x40
    assert profiled.schedule_island_profile_mask == 0x7F
    assert profiled.schedule_baseline_island_profile_mask == 0x3F
    assert profiled.schedule_scored_island_profile_mask == 0x40
    assert profiled.schedule_tsab_activations == 1
    assert profiled.schedule_tsab_activation_boundary == 2
    assert profiled.schedule_tsab_legacy_warmup_work_steps == 512
    assert profiled.schedule_tsab_activation_rebases <= profiled.schedule_tsab_activations
    assert profiled.schedule_tsab_activation_objective is not None
    assert profiled.schedule_tsab_burst_work_limit == 128
    assert profiled.schedule_tsab_burst_work_units <= 128 * profiled.schedule_tsab_active_boundaries
    assert profiled.schedule_tsab_fast_enabled > 0
    assert profiled.schedule_tsab_fast_eligible > 0
    assert profiled.schedule_tsab_fast_disabled == 0
    assert profiled.schedule_tsab_fast_attempts >= 0
    assert profiled.schedule_tsab_fast_work_units >= 0
    assert profiled.schedule_tsab_fast_transitions >= 0
    assert profiled.schedule_tsab_fast_full_validations >= 0
    assert profiled.schedule_tsab_fast_commits == profiled.schedule_tsab_fast_transitions
    assert profiled.schedule_tsab_fast_transitions <= profiled.schedule_tsab_fast_attempts
    assert profiled.schedule_tsab_fast_transitions <= 4 * profiled.schedule_tsab_fast_work_units
    if profiled.schedule_tsab_fast_transitions > 0:
        assert profiled.schedule_tsab_fast_full_validations > 0
    assert profiled.schedule_tsab_fast_oracle_mismatches == 0
    assert profiled.schedule_tsab_fast_pending_discards == 0
    assert profiled.schedule_tsab_fast_fallbacks >= 0
    assert profiled.schedule_tsab_fast_date_changes >= 0
    assert profiled.schedule_tsab_fast_queue_pops >= 0
    assert profiled.schedule_tsab_fast_pending_promotions >= 0
    assert math.isfinite(profiled.schedule_tsab_fast_elapsed_seconds)
    assert profiled.schedule_tsab_fast_elapsed_seconds >= 0.0
    assert profiled.schedule_tsab_fast_workspace_peak_bytes >= 0
    assert profiled.schedule_tsab_n5_generated == profiled.schedule_tsab_ranked
    assert profiled.schedule_tsab_delta_probes <= 4 * profiled.schedule_tsab_shortlists
    assert profiled.schedule_tsab_additional_delta_probes <= profiled.schedule_tsab_delta_probes
    assert profiled.schedule_tsab_full_oracle_commits <= profiled.schedule_tsab_selections
    assert profiled.schedule_tsab_selections <= profiled.schedule_tsab_shortlists
    assert profiled.schedule_tsab_aspirations <= profiled.schedule_tsab_full_oracle_commits
    assert profiled.schedule_tsab_tabu_resets <= profiled.schedule_tsab_shortlists
    assert profiled.schedule_tsab_selected_shortlist_rank_sum <= 3 * profiled.schedule_tsab_full_oracle_commits
    deferred = (
        "schedule_tsab_fingerprint_repeats",
        "schedule_tsab_n1_kicks",
        "schedule_tsab_ranking_audits",
        "schedule_tsab_exact_best_matches",
        "schedule_tsab_regret_sum",
        "schedule_tsab_regret_max",
    )
    assert all(getattr(profiled, field) == 0 for field in deferred)
    assert profiled.schedule_tsab_restart_delta_probes <= 4 * profiled.schedule_tsab_restart_attempts
    assert profiled.schedule_tsab_restart_oracle_commits == profiled.schedule_tsab_n6_kicks
    assert profiled.schedule_tsab_n6_kicks == profiled.schedule_tsab_kick_moves
    assert profiled.schedule_tsab_restart_oracle_commits <= profiled.schedule_tsab_elite_restarts
    assert profiled.schedule_tsab_elite_restarts <= profiled.schedule_tsab_restart_attempts
    assert profiled.schedule_tsab_restart_global_rebases <= profiled.schedule_tsab_elite_restarts
    assert (
        profiled.schedule_tsab_restart_oracle_commits + profiled.schedule_tsab_restart_rejections
        <= profiled.schedule_tsab_restart_attempts
    )
    assert profiled.schedule_tsab_restart_work_units <= profiled.schedule_tsab_burst_work_units
    assert profiled.schedule_tsab_post_restart_improvements <= profiled.schedule_tsab_improving_commits
    if profiled.schedule_tsab_restart_oracle_commits == 0:
        assert profiled.schedule_tsab_restart_best_kicked_objective is None
    else:
        assert profiled.schedule_tsab_restart_best_kicked_objective is not None
    if profiled.schedule_tsab_restart_best_base_objective is not None:
        assert profiled.schedule_tsab_restart_attempts > 0
    assert profiled.schedule_tsab_improving_commits <= (
        profiled.schedule_tsab_full_oracle_commits + profiled.schedule_tsab_fast_commits
    )
    if profiled.schedule_tsab_improving_commits == 0:
        assert profiled.schedule_tsab_best_committed_objective is None
    else:
        assert profiled.schedule_tsab_best_committed_objective < profiled.schedule_tsab_activation_objective


def test_tsab_candidate_rejects_invalid_public_value_and_experimental_combinations():
    with pytest.raises(ValueError, match="must be 'legacy' or 'tsab-candidate'"):
        _solve_job_shop(profile=True, schedule_jssp_search="unknown")
    with pytest.raises(ValueError, match="cannot yet be combined"):
        _solve_job_shop(
            profile=True,
            threads=7,
            schedule_jssp_search="tsab-candidate",
            schedule_lns_shadow=True,
        )
    with pytest.raises(ValueError, match="currently requires threads=7"):
        _solve_job_shop(profile=True, threads=6, schedule_jssp_search="tsab-candidate")


def test_constructor_multistart_reserves_only_worker_six_and_reports_exact_accounting():
    profiled = _solve_job_shop_with_constructor_multistart()

    assert profiled.status == "SATISFIABLE"
    assert profiled.schedule_constructor_workers_requested == 1
    assert profiled.schedule_constructor_multistart_enabled == 1
    assert profiled.schedule_constructor_multistart_active == 1
    assert profiled.schedule_constructor_multistart_owner_worker_mask == 64
    assert profiled.schedule_constructor_multistart_attempts > 0
    assert profiled.schedule_constructor_multistart_attempts == (
        profiled.schedule_constructor_multistart_constructions
        + profiled.schedule_constructor_multistart_interruptions
        + profiled.schedule_constructor_multistart_failures
    )
    assert profiled.schedule_constructor_multistart_constructions == profiled.schedule_constructor_multistart_feasible
    assert profiled.schedule_constructor_multistart_feasible == (
        profiled.schedule_constructor_multistart_distinct_fingerprints
        + profiled.schedule_constructor_multistart_other_fingerprint_observations
    )
    assert profiled.schedule_constructor_multistart_work_units == 64 * profiled.schedule_constructor_multistart_attempts
    assert profiled.schedule_constructor_multistart_next_ordinal == (
        6
        + profiled.schedule_constructor_multistart_constructions
        + profiled.schedule_constructor_multistart_failures
    )
    assert profiled.schedule_constructor_multistart_best_ordinal >= 6
    assert profiled.schedule_constructor_multistart_best_seed is not None
    assert profiled.schedule_constructor_multistart_best_fingerprint is not None
    assert profiled.schedule_constructor_multistart_best_objective <= profiled.schedule_constructor_multistart_initial_objective
    assert profiled.schedule_constructor_multistart_attempts <= 4 * profiled.schedule_restart_boundaries
    assert 0 < profiled.schedule_constructor_multistart_workspace_peak_bytes < 512 * 1024 * 1024
    assert math.isfinite(profiled.schedule_constructor_multistart_worker_seconds_sum)
    assert profiled.schedule_island_profile_mask == 63
    assert profiled.schedule_baseline_island_profile_mask == 63
    assert profiled.schedule_scored_island_profile_mask == 0
    assert profiled.schedule_island_profile_count == 6
    assert set(_profile_best_objectives(profiled.schedule_profile_work_steps)) == set(range(6))


def test_constructor_multistart_rejects_invalid_public_combinations():
    with pytest.raises(ValueError, match="requires threads=7"):
        _solve_job_shop(profile=True, threads=6, schedule_constructor_workers=1)

    model = cp.Model()
    tasks = model.tasks(range(2))
    for task in tasks:
        task.duration = 1
        task.machine = task.id
    schedule = model.schedule(tasks, horizon=2)
    model.add(schedule.no_overlap(lambda task: task.machine))
    model.minimize(schedule.makespan())
    with pytest.raises(ValueError, match="cannot yet be combined"):
        model.solve(
            engine="ls",
            threads=7,
            time_limit=1,
            schedule_constructor_workers=1,
            schedule_lns_shadow=True,
        )


@pytest.mark.parametrize(
    "launcher",
    (
        "examples/python/scheduling/api/jssp.py",
        "examples/python/scheduling/native/jssp.py",
    ),
)
def test_tsab_launchers_emit_non_vacuous_bounded_candidate_metrics(launcher):
    completed = subprocess.run(
        (
            sys.executable,
            str(ROOT / launcher),
            "--jobs",
            "8",
            "--machines",
            "6",
            "--seed",
            "1",
            "--engine",
            "ls",
            "--threads",
            "7",
            "--time-limit",
            "1",
            "--max-iterations",
            "16384",
            "--profile",
            "--schedule-jssp-search",
            "tsab-candidate",
            "--json",
            "--compact-json",
        ),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=20,
    )
    record = json.loads(completed.stdout)

    assert record["verified"] is True
    assert record["schedule_jssp_search"] == "tsab-candidate"
    assert record["schedule_tsab_owner_worker_mask"] == 0x40
    assert record["schedule_tsab_activations"] == 1
    assert record["schedule_tsab_activation_boundary"] == 2
    assert record["schedule_tsab_legacy_warmup_work_steps"] == 512
    assert record["schedule_tsab_activation_rebases"] == 1
    assert record["schedule_tsab_activation_objective"] is not None
    assert record["schedule_tsab_active_boundaries"] > 0
    assert record["schedule_tsab_burst_work_limit"] == 128
    assert record["schedule_tsab_burst_work_units"] <= 128 * record["schedule_tsab_active_boundaries"]
    assert record["schedule_tsab_fast_enabled"] > 0
    assert record["schedule_tsab_fast_eligible"] > 0
    assert record["schedule_tsab_fast_disabled"] == 0
    assert record["schedule_tsab_fast_attempts"] > 0
    assert record["schedule_tsab_fast_work_units"] > 0
    assert record["schedule_tsab_fast_transitions"] > 0
    assert record["schedule_tsab_fast_full_validations"] > 0
    assert record["schedule_tsab_fast_commits"] == record["schedule_tsab_fast_transitions"]
    assert record["schedule_tsab_fast_transitions"] <= record["schedule_tsab_fast_attempts"]
    assert record["schedule_tsab_fast_transitions"] <= 4 * record["schedule_tsab_fast_work_units"]
    assert record["schedule_tsab_fast_oracle_mismatches"] == 0
    assert record["schedule_tsab_fast_pending_discards"] == 0
    assert record["schedule_tsab_fast_fallbacks"] >= 0
    assert record["schedule_tsab_fast_date_changes"] >= 0
    assert record["schedule_tsab_fast_queue_pops"] >= 0
    assert record["schedule_tsab_fast_pending_promotions"] >= 0
    assert math.isfinite(record["schedule_tsab_fast_elapsed_seconds"])
    assert record["schedule_tsab_fast_elapsed_seconds"] >= 0.0
    assert record["schedule_tsab_fast_workspace_peak_bytes"] > 0
    assert record["schedule_tsab_n5_generated"] == record["schedule_tsab_ranked"] > 0
    assert 0 < record["schedule_tsab_delta_probes"] <= 4 * record["schedule_tsab_shortlists"]
    assert record["schedule_tsab_full_oracle_commits"] <= record["schedule_tsab_selections"]
    assert record["schedule_tsab_selections"] <= record["schedule_tsab_shortlists"]
    assert record["schedule_tsab_restart_attempts"] > 0
    assert record["schedule_tsab_restart_n6_generated"] > 0
    assert 0 < record["schedule_tsab_restart_delta_probes"] <= 4 * record["schedule_tsab_restart_attempts"]
    assert record["schedule_tsab_restart_oracle_commits"] == record["schedule_tsab_n6_kicks"]
    assert record["schedule_tsab_n6_kicks"] == record["schedule_tsab_kick_moves"]
    assert record["schedule_tsab_restart_oracle_commits"] <= record["schedule_tsab_elite_restarts"]
    assert record["schedule_tsab_elite_restarts"] <= record["schedule_tsab_restart_attempts"]
    assert record["schedule_tsab_restart_global_rebases"] <= record["schedule_tsab_elite_restarts"]
    assert 0 < record["schedule_tsab_restart_work_units"] <= record["schedule_tsab_burst_work_units"]
    assert record["schedule_tsab_restart_best_base_objective"] is not None
    if record["schedule_tsab_restart_oracle_commits"] == 0:
        assert record["schedule_tsab_restart_best_kicked_objective"] is None
    else:
        assert record["schedule_tsab_restart_best_kicked_objective"] is not None
    assert record["schedule_tsab_restart_shortlist_peak_bytes"] > 0
    assert math.isfinite(record["schedule_incumbent_verification_seconds"])
    assert math.isfinite(record["schedule_incumbent_verification_max_seconds"])
    assert 0.0 <= record["schedule_incumbent_verification_max_seconds"] <= record["schedule_incumbent_verification_seconds"]


@pytest.mark.parametrize(
    ("launcher", "shape_args"),
    (
        ("examples/python/scheduling/api/jssp.py", ("--jobs", "2", "--machines", "2")),
        ("examples/python/scheduling/native/jssp.py", ("--jobs", "2", "--machines", "2")),
        ("examples/python/scheduling/api/rcpsp.py", ("--tasks", "4", "--resources", "1")),
        ("examples/python/scheduling/native/rcpsp.py", ("--tasks", "4", "--resources", "1")),
    ),
)
def test_scheduling_launchers_emit_search_profile(launcher, shape_args):
    command = (
        sys.executable,
        str(ROOT / launcher),
        *shape_args,
        "--engine",
        "ls",
        "--seed",
        "7",
        "--time-limit",
        "1",
        "--threads",
        "2",
        "--max-iterations",
        "8",
        "--memory-limit-mb",
        "0",
        "--profile",
        "--json",
    )
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True, timeout=20)
    record = json.loads(completed.stdout)

    assert record["verified"] is True
    assert record["candidates_evaluated"] == record["schedule_moves_considered"]
    assert record["schedule_moves_accepted"] + record["schedule_moves_rejected"] == record["schedule_moves_considered"]
    assert record["schedule_global_improvements"] == record["schedule_progress_publications"]
    assert record["schedule_incumbent_publication_attempts"] == (
        record["schedule_incumbent_publications"] + record["schedule_incumbent_rejections"]
    )
    assert record["schedule_incumbent_injections"] <= record["schedule_incumbent_injection_attempts"]
    if launcher.endswith("/jssp.py"):
        assert record["schedule_jssp_search"] == "legacy"
        assert all(record[field] == 0 for field in JSSP_TSAB_COUNTERS)
        assert all(record[field] == 0.0 for field in JSSP_TSAB_TIMINGS)
        assert all(record[field] is None for field in JSSP_TSAB_OPTIONAL_COUNTERS)
        assert record["schedule_restart_work"] == 2_048
        assert all(record[field] is not None and record[field] >= 0 for field in JSSP_STRATEGY_COUNTERS)
        assert record["schedule_session_initializations"] == 2
        if record["schedule_restart_boundaries"] > 1:
            assert record["schedule_session_resumes"] > 0
        assert record["schedule_tabu_steps"] > 0
        assert all(record[field] is not None and record[field] >= 0 for field in JSSP_ISLAND_COUNTERS)
        assert all(record[field] is not None and record[field] >= 0 for field in JSSP_GT_COUNTERS)
        assert all(record[field] is not None and record[field] >= 0 for field in JSSP_ORACLE_COUNTERS)
        assert all(record[field] is not None and record[field] >= 0 for field in JSSP_SEARCH_ELITE_COUNTERS)
        assert all(record[field] is not None for field in JSSP_SEARCH_ELITE_TIMINGS)
        assert all(record[field] is not None and record[field] >= 0 for field in JSSP_LNS_SHADOW_COUNTERS)
        assert math.isfinite(record["schedule_lns_worker_seconds_sum"])
        assert record["schedule_lns_shadow_enabled"] == 0
        assert record["schedule_lns_shadow_active"] == 0
        assert record["schedule_constructor_workers_requested"] == 0
        assert record["schedule_constructor_multistart_enabled"] == 0
        assert record["schedule_constructor_multistart_active"] == 0
        assert all(math.isfinite(record[field]) and record[field] >= 0.0 for field in JSSP_VERIFICATION_TIMINGS)
        assert record["schedule_incumbent_verification_max_seconds"] <= record["schedule_incumbent_verification_seconds"]
        assert set(_profile_best_objectives(record["schedule_profile_work_steps"])) == {0, 1}
        assert all(record[field] is not None and record[field] >= 0 for field in JSSP_PATH_RELINK_COUNTERS)
        assert math.isfinite(record["schedule_path_relink_worker_seconds_sum"])
        assert record["schedule_path_relink_enabled"] == 0
        assert record["schedule_path_relink_active"] == 0
        _assert_search_elite_profile(record)
        assert record["schedule_island_profile_mask"].bit_count() == record["schedule_island_profile_count"] == 2
        assert record["schedule_baseline_island_profile_mask"] | record["schedule_scored_island_profile_mask"] == (
            record["schedule_island_profile_mask"]
        )
        assert record["schedule_baseline_island_profile_mask"] & record["schedule_scored_island_profile_mask"] == 0
        assert record["schedule_approximate_candidates_refined"] == (
            record["schedule_approximate_candidates_certified"] + record["schedule_approximate_candidates_unknown"]
        )
        classified = (
            record["schedule_direct_oracle_accepts"]
            + record["schedule_direct_oracle_cycles"]
            + record["schedule_direct_oracle_windows"]
            + record["schedule_direct_oracle_objective_rejections"]
        )
        assert classified <= record["schedule_direct_oracle_attempts"]
        assert record["schedule_direct_oracle_attempts"] + record["schedule_exact_probes_avoided"] <= (
            record["schedule_approximate_candidates_refined"]
        )
        assert record["schedule_island_scored_candidates"] == record["schedule_approximate_candidates_generated"]
        assert record["schedule_approximation_score_items"] == record["schedule_approximate_candidates_generated"]
        assert record["schedule_approximation_sort_items"] >= record["schedule_approximation_score_items"]
        assert math.isfinite(record["schedule_approximation_elapsed_seconds"])
        profile_times = _profile_construction_times(record["schedule_profile_construction_seconds"])
        assert set(profile_times) == {0, 1}
        assert all(math.isfinite(seconds) and seconds >= 0.0 for seconds in profile_times.values())
        profile_objectives = _profile_best_objectives(record["schedule_profile_best_objectives"])
        assert set(profile_objectives) == {0, 1}
        assert all(objective > 0 for objective in profile_objectives.values())
        assert min(profile_objectives.values()) == record["objectives"][0]
        assert set(_profile_best_objectives(record["schedule_profile_initial_objectives"])) == {0, 1}
        assert _profile_strings(record["schedule_profile_initial_dispatch_rules"]) == {
            0: "earliest-start",
            1: "earliest-start",
        }
        assert record["schedule_construction_heap_pushes"] > 0
        assert 0 < record["schedule_construction_heap_peak"] <= record["schedule_construction_heap_pushes"]
    else:
        assert all(field not in record for field in JSSP_ISLAND_COUNTERS)
        assert all(field not in record for field in JSSP_GT_COUNTERS)
        assert all(field not in record for field in JSSP_ORACLE_COUNTERS)
        assert all(field not in record for field in JSSP_SEARCH_ELITE_FIELDS)
        assert all(field not in record for field in JSSP_PATH_RELINK_FIELDS)
        assert "schedule_profile_construction_seconds" not in record
        assert "schedule_profile_initial_objectives" not in record
        assert "schedule_profile_initial_dispatch_rules" not in record
        assert "schedule_profile_best_objectives" not in record
    assert record["schedule_worker_work_min"] <= record["schedule_worker_work_max"]
    assert record["schedule_work_budget_overruns"] == 0
    assert record["schedule_elite_pool_size"] == 1
    assert record["schedule_unused_work_steps"] + record["schedule_work_steps"] == 8
    assert record["schedule_incumbent_source_worker"] < 2
    assert math.isfinite(record["candidates_per_second"])
    assert record["candidates_per_second"] >= 0.0
    assert math.isfinite(record["full_recompute_percentage"])
    assert 0.0 <= record["full_recompute_percentage"] <= 100.0
    assert all(record[field] is not None for field in SCHEDULE_COUNTERS)


def test_scheduling_verbose_output_includes_search_profile():
    command = (
        sys.executable,
        str(ROOT / "examples/python/scheduling/api/jssp.py"),
        "--jobs",
        "2",
        "--machines",
        "2",
        "--engine",
        "ls",
        "--time-limit",
        "1",
        "--max-iterations",
        "4",
        "--profile",
        "--verbose",
        "--json",
    )
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True, timeout=20)

    assert "schedule moves considered:" in completed.stdout
    assert "schedule moves accepted:" in completed.stdout
    assert "critical path updates:" in completed.stdout
    assert "schedule window rejections:" in completed.stdout
    assert "resource profile checks:" in completed.stdout
    assert "schedule justification attempts:" in completed.stdout
    assert "schedule delta evaluations:" in completed.stdout
    assert "schedule oracle mismatches:" in completed.stdout
    assert "schedule max dirty cone:" in completed.stdout
    assert "schedule restart work: 2048" in completed.stdout
    assert "schedule tabu steps:" in completed.stdout
    assert "schedule tabu hits:" in completed.stdout
    assert "schedule tabu aspirations:" in completed.stdout
    assert "schedule tabu forced moves:" in completed.stdout
    assert "schedule session initializations:" in completed.stdout
    assert "schedule session resumes:" in completed.stdout
    assert "schedule session rebases:" in completed.stdout
    assert "schedule island profile mask:" in completed.stdout
    assert "schedule baseline island profile mask:" in completed.stdout
    assert "schedule scored island profile mask:" in completed.stdout
    assert "schedule island profile count:" in completed.stdout
    assert "schedule profile construction seconds:" in completed.stdout
    assert "schedule profile initial objectives:" in completed.stdout
    assert "schedule profile initial dispatch rules:" in completed.stdout
    assert "schedule profile best objectives:" in completed.stdout
    assert "schedule construction bucket visits:" in completed.stdout
    assert "schedule construction heap pushes:" in completed.stdout
    assert "schedule construction stale pops:" in completed.stdout
    assert "schedule construction heap rebuilds:" in completed.stdout
    assert "schedule construction heap peak:" in completed.stdout
    assert "schedule reactive restarts:" in completed.stdout
    assert "schedule reactive restart dispatches:" in completed.stdout
    assert "schedule reactive restart perturbations:" in completed.stdout
    assert "schedule reactive restart rebuild failures:" in completed.stdout
    assert "schedule island scored candidates:" in completed.stdout
    assert "schedule island shortlisted candidates:" in completed.stdout
    assert "schedule approximate candidates generated:" in completed.stdout
    assert "schedule approximate candidates refined:" in completed.stdout
    assert "schedule approximate candidates certified:" in completed.stdout
    assert "schedule approximate candidates unknown:" in completed.stdout
    assert "schedule approximation score items:" in completed.stdout
    assert "schedule approximation sort items:" in completed.stdout
    assert "schedule approximation local span items:" in completed.stdout
    assert "schedule approximation elapsed:" in completed.stdout
    assert "schedule approximation work units:" in completed.stdout
    assert "schedule direct oracle attempts:" in completed.stdout
    assert "schedule direct oracle accepts:" in completed.stdout
    assert "schedule direct oracle cycles:" in completed.stdout
    assert "schedule direct oracle windows:" in completed.stdout
    assert "schedule direct oracle objective rejections:" in completed.stdout
    assert "schedule exact probes avoided:" in completed.stdout
    assert "schedule search elite pool size:" in completed.stdout
    assert "schedule search elite batches:" in completed.stdout
    assert "schedule search elite batches skipped after stop:" in completed.stdout
    assert "schedule search elite candidates:" in completed.stdout
    assert "schedule search elite insertions:" in completed.stdout
    assert "schedule search elite duplicates:" in completed.stdout
    assert "schedule search elite dominated:" in completed.stdout
    assert "schedule search elite evictions:" in completed.stdout
    assert "schedule search elite interruptions:" in completed.stdout
    assert "schedule search elite merge errors:" in completed.stdout
    assert "schedule search elite snapshot captures:" in completed.stdout
    assert "schedule search elite snapshot interruptions:" in completed.stdout
    assert "schedule search elite snapshot errors:" in completed.stdout
    assert "schedule search elite objectives:" in completed.stdout
    assert "schedule search elite pairwise distances ppm:" in completed.stdout
    assert "schedule search elite min distance ppm:" in completed.stdout
    assert "schedule search elite mean distance ppm:" in completed.stdout
    assert "schedule search elite max distance ppm:" in completed.stdout
    assert "schedule search elite capture worker seconds sum:" in completed.stdout
    assert "schedule search elite merge wall seconds:" in completed.stdout
    assert "schedule search elite heap lower bound bytes:" in completed.stdout
    assert "schedule search elite peak heap lower bound bytes:" in completed.stdout
    assert "schedule LNS shadow enabled:" in completed.stdout
    assert "schedule LNS shadow active:" in completed.stdout
    assert "schedule LNS shadow owner worker mask:" in completed.stdout
    assert "schedule LNS attempts:" in completed.stdout
    assert "schedule LNS selected operations:" in completed.stdout
    assert "schedule LNS feasible:" in completed.stdout
    assert "schedule LNS reconstructed:" in completed.stdout
    assert "schedule LNS improvements:" in completed.stdout
    assert "schedule LNS shadow improvements:" in completed.stdout
    assert "schedule LNS shadow improvement sum:" in completed.stdout
    assert "schedule LNS shadow best improvement:" in completed.stdout
    assert "schedule LNS timeouts:" in completed.stdout
    assert "schedule LNS interruptions:" in completed.stdout
    assert "schedule LNS infeasible:" in completed.stdout
    assert "schedule LNS non-improving:" in completed.stdout
    assert "schedule LNS reconstruction rejections:" in completed.stdout
    assert "schedule LNS oracle rejections:" in completed.stdout
    assert "schedule LNS exact rejections:" in completed.stdout
    assert "schedule LNS worker seconds sum:" in completed.stdout
    assert "schedule LNS workspace peak bytes:" in completed.stdout
    assert "schedule path relink enabled:" in completed.stdout
    assert "schedule path relink active:" in completed.stdout
    assert "schedule path relink guide requests:" in completed.stdout
    assert "schedule path relink best guides:" in completed.stdout
    assert "schedule path relink diverse guides:" in completed.stdout
    assert "schedule path relink guide loads:" in completed.stdout
    assert "schedule path relink guide incompatible:" in completed.stdout
    assert "schedule path relink guide interruptions:" in completed.stdout
    assert "schedule path relink critical operations scanned:" in completed.stdout
    assert "schedule path relink candidates generated:" in completed.stdout
    assert "schedule path relink candidates positive gain:" in completed.stdout
    assert "schedule path relink acyclicity certified:" in completed.stdout
    assert "schedule path relink acyclicity unknown:" in completed.stdout
    assert "schedule path relink prefilter rejections:" in completed.stdout
    assert "schedule path relink candidates retained:" in completed.stdout
    assert "schedule path relink candidates refined:" in completed.stdout
    assert "schedule path relink candidates shortlisted:" in completed.stdout
    assert "schedule path relink no move:" in completed.stdout
    assert "schedule path relink guide arc gain shortlisted:" in completed.stdout
    assert "schedule path relink oracle attempts:" in completed.stdout
    assert "schedule path relink oracle accepts:" in completed.stdout
    assert "schedule path relink cycle rejections:" in completed.stdout
    assert "schedule path relink window rejections:" in completed.stdout
    assert "schedule path relink other rejections:" in completed.stdout
    assert "schedule path relink rollbacks:" in completed.stdout
    assert "schedule path relink elite improvements:" in completed.stdout
    assert "schedule path relink guide arc gain accepted:" in completed.stdout
    assert "schedule path relink worker seconds sum:" in completed.stdout
    assert "schedule path relink workspace peak bytes:" in completed.stdout
