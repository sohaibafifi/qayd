#!/usr/bin/env python3
"""Fail-closed gate for staged Large-TA JSSP search experiments."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable


FEASIBLE = {"SAT", "SATISFIABLE", "FEASIBLE", "OPTIMAL", "OPTIMUM"}
EXPECTED_COUNTS = {
    "starts": 1_000_000,
    "job_precedence_pairs": 999_000,
    "machine_non_overlap_pairs": 999_000,
    "objective_checks": 1,
}
STAGE_THRESHOLDS = {
    "first-signal": 936_343,
    "portfolio": 926_341,
    "lns": 909_672,
    "hybrid-strong": 893_003,
    "pre-target": 883_001,
    "target": 876_334,
}
SHADOW_STAGE = "shadow"
STAGE_ORDER = (SHADOW_STAGE, *STAGE_THRESHOLDS)
DEFAULT_MAX_PEAK_MEMORY_MB = 12_288.0
METRIC_DEFAULTS = {
    "schedule_global_improvements_after_round4": 0,
    "schedule_objective_gain_after_round4": 0,
    "schedule_last_global_improvement_round": 0,
    "schedule_reactive_restarts": 0,
    "schedule_lns_attempts": 0,
    "schedule_lns_feasible": 0,
    "schedule_lns_improvements": 0,
}
SEARCH_ELITE_INTEGER_FIELDS = (
    "schedule_search_elite_pool_size",
    "schedule_search_elite_batches",
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
SEARCH_ELITE_TIMING_FIELDS = (
    "schedule_search_elite_capture_worker_seconds_sum",
    "schedule_search_elite_merge_wall_seconds",
)
SEARCH_ELITE_OPTIONAL_INTEGER_FIELDS = (
    "schedule_search_elite_batches_skipped_after_stop",
)
PATH_RELINK_INTEGER_FIELDS = (
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
PATH_RELINK_TIMING_FIELDS = (
    "schedule_path_relink_worker_seconds_sum",
)
PATH_RELINK_MAX_WORKSPACE_BYTES = 16 * 1024 * 1024
LNS_SHADOW_INTEGER_FIELDS = (
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
LNS_SHADOW_TIMING_FIELDS = ("schedule_lns_worker_seconds_sum",)
LNS_SHADOW_MAX_WORKSPACE_BYTES = 512 * 1024 * 1024
C1_MAX_PEAK_MEMORY_MB = 11_264.0
C1_MAX_WORKSPACE_BYTES = 512 * 1024 * 1024
C1_FINAL_OBJECTIVE_LIMIT = 943_065
C1_CONSTRUCTOR_OBJECTIVE_EXCLUSIVE = 943_228
C1_LS_PROFILE_RETENTION = 0.90
C1_CONSTRUCTOR_INTEGER_FIELDS = (
    "schedule_constructor_workers_requested",
    "schedule_constructor_multistart_enabled",
    "schedule_constructor_multistart_active",
    "schedule_constructor_multistart_owner_worker_mask",
    "schedule_constructor_multistart_attempts",
    "schedule_constructor_multistart_constructions",
    "schedule_constructor_multistart_feasible",
    "schedule_constructor_multistart_distinct_fingerprints",
    "schedule_constructor_multistart_other_fingerprint_observations",
    "schedule_constructor_multistart_initial_objective",
    "schedule_constructor_multistart_best_objective",
    "schedule_constructor_multistart_improvements",
    "schedule_constructor_multistart_work_units",
    "schedule_constructor_multistart_interruptions",
    "schedule_constructor_multistart_failures",
    "schedule_constructor_multistart_workspace_peak_bytes",
    "schedule_constructor_multistart_next_ordinal",
    "schedule_constructor_multistart_best_ordinal",
    "schedule_constructor_multistart_best_seed",
    "schedule_constructor_multistart_best_fingerprint",
    "schedule_island_profile_mask",
    "schedule_baseline_island_profile_mask",
    "schedule_scored_island_profile_mask",
    "schedule_island_profile_count",
    "schedule_restart_boundaries",
)
C1_CONSTRUCTOR_TIMING_FIELDS = ("schedule_constructor_multistart_worker_seconds_sum",)


class GateError(ValueError):
    """Raised when an experiment cannot support a qualified comparison."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise GateError(f"{path} must contain a JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise GateError(f"{path}:{line_number}: invalid JSON: {error}") from error
                if not isinstance(record, dict):
                    raise GateError(f"{path}:{line_number}: record must be an object")
                records.append(record)
    except OSError as error:
        raise GateError(f"cannot read campaign {path}: {error}") from error
    if not records:
        raise GateError(f"campaign is empty: {path}")
    return records


def provenance_path(campaign: Path) -> Path:
    return campaign.with_suffix(campaign.suffix + ".provenance.json")


def _number(value: object, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise GateError(f"{label} must be finite and >= {minimum:g}")
    return number


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GateError(f"{label} must be an integer >= {minimum}")
    return value


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdefABCDEF" for character in value)


def _search_elite_objectives(value: object, label: str, *, pool_size: int) -> list[int]:
    if pool_size == 0:
        if value != "none":
            raise GateError(f"{label} must be 'none' for an empty pool")
        return []
    if not isinstance(value, str) or not value:
        raise GateError(f"{label} must be a comma-separated objective summary")
    try:
        objectives = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise GateError(f"{label} must contain integer objectives") from error
    if len(objectives) != pool_size or any(objective <= 0 for objective in objectives):
        raise GateError(f"{label} must contain one positive objective per pool entry")
    return objectives


def _search_elite_distances(value: object, label: str, *, pool_size: int) -> list[int]:
    expected_pairs = pool_size * (pool_size - 1) // 2
    if expected_pairs == 0:
        if value != "none":
            raise GateError(f"{label} must be 'none' when fewer than two entries are retained")
        return []
    if not isinstance(value, str) or not value:
        raise GateError(f"{label} must encode every retained pair")
    parsed: dict[tuple[int, int], int] = {}
    for item in value.split(","):
        try:
            pair, raw_distance = item.split("=", 1)
            raw_left, raw_right = pair.split("-", 1)
            left = int(raw_left)
            right = int(raw_right)
            distance = int(raw_distance)
        except (TypeError, ValueError) as error:
            raise GateError(f"{label} contains an invalid pair encoding") from error
        key = (left, right)
        if not 0 <= left < right < pool_size or key in parsed or not 0 <= distance <= 1_000_000:
            raise GateError(f"{label} contains an invalid or duplicate retained pair")
        parsed[key] = distance
    expected = {(left, right) for left in range(pool_size) for right in range(left + 1, pool_size)}
    if set(parsed) != expected or len(parsed) != expected_pairs:
        raise GateError(f"{label} must encode all {expected_pairs} retained pairs exactly once")
    return [parsed[key] for key in sorted(parsed)]


def _validate_search_elite_metrics(
    record: dict[str, Any],
    *,
    label: str,
    wall_seconds: float,
    peak_memory_mb: float,
    threads: int,
    max_peak_memory_mb: float,
) -> tuple[dict[str, Any], list[str]]:
    integers = {field: _integer(record.get(field), f"{label}.{field}") for field in SEARCH_ELITE_INTEGER_FIELDS}
    timings = {field: _number(record.get(field), f"{label}.{field}") for field in SEARCH_ELITE_TIMING_FIELDS}
    optional = {
        field: _integer(record[field], f"{label}.{field}")
        for field in SEARCH_ELITE_OPTIONAL_INTEGER_FIELDS
        if field in record
    }
    pool_size = integers["schedule_search_elite_pool_size"]
    batches = integers["schedule_search_elite_batches"]
    candidates = integers["schedule_search_elite_candidates"]
    insertions = integers["schedule_search_elite_insertions"]
    captures = integers["schedule_search_elite_snapshot_captures"]
    heap_bytes = integers["schedule_search_elite_heap_lower_bound_bytes"]
    peak_heap_bytes = integers["schedule_search_elite_peak_heap_lower_bound_bytes"]

    if insertions < pool_size:
        raise GateError(f"{label}: search elite insertions cannot be smaller than the retained pool")
    if candidates < insertions:
        raise GateError(f"{label}: search elite insertions cannot exceed candidates")
    if captures < candidates:
        raise GateError(f"{label}: search elite candidates cannot exceed captured snapshots")
    if candidates > 0 and batches == 0:
        raise GateError(f"{label}: a non-empty search elite candidate stream requires a batch")
    if heap_bytes > peak_heap_bytes:
        raise GateError(f"{label}: current search elite heap lower bound exceeds its peak lower bound")
    if pool_size > 0 and heap_bytes == 0:
        raise GateError(f"{label}: a non-empty search elite pool requires a positive heap lower bound")

    capture_seconds = timings["schedule_search_elite_capture_worker_seconds_sum"]
    merge_seconds = timings["schedule_search_elite_merge_wall_seconds"]
    timing_tolerance = 1e-6
    if captures > 0 and capture_seconds == 0:
        raise GateError(f"{label}: captured snapshots require a positive worker-time sum")
    if batches > 0 and merge_seconds == 0:
        raise GateError(f"{label}: merged search elite batches require positive merge wall time")
    if merge_seconds > wall_seconds + timing_tolerance:
        raise GateError(f"{label}: search elite merge wall time exceeds process wall time")
    if capture_seconds > wall_seconds * threads + timing_tolerance:
        raise GateError(f"{label}: search elite worker-time sum exceeds the parallel wall envelope")

    objectives = _search_elite_objectives(
        record.get("schedule_search_elite_objectives"),
        f"{label}.schedule_search_elite_objectives",
        pool_size=pool_size,
    )
    distances = _search_elite_distances(
        record.get("schedule_search_elite_pairwise_distances_ppm"),
        f"{label}.schedule_search_elite_pairwise_distances_ppm",
        pool_size=pool_size,
    )
    distance_fields = (
        "schedule_search_elite_min_distance_ppm",
        "schedule_search_elite_mean_distance_ppm",
        "schedule_search_elite_max_distance_ppm",
    )
    if distances:
        minimum, mean, maximum = (
            _integer(record.get(field), f"{label}.{field}")
            for field in distance_fields
        )
        expected_mean = sum(distances) // len(distances)
        if (minimum, mean, maximum) != (min(distances), expected_mean, max(distances)):
            raise GateError(f"{label}: search elite distance statistics do not match the pair summary")
    else:
        if any(record.get(field) is not None for field in distance_fields):
            raise GateError(f"{label}: search elite distance statistics require at least two entries")
        minimum = mean = maximum = None

    rss_bytes = peak_memory_mb * 1024 * 1024
    failures = []
    if pool_size < 2:
        failures.append("pool_size_lt_2")
    if integers["schedule_search_elite_snapshot_errors"] != 0:
        failures.append("snapshot_errors")
    if integers["schedule_search_elite_merge_errors"] != 0:
        failures.append("merge_errors")
    if integers["schedule_search_elite_interruptions"] != 0:
        failures.append("merge_interruptions")
    if integers["schedule_search_elite_snapshot_interruptions"] != 0:
        failures.append("snapshot_interruptions")
    if peak_memory_mb >= max_peak_memory_mb:
        failures.append("rss_limit")
    if peak_heap_bytes > rss_bytes:
        failures.append("peak_heap_lower_bound_exceeds_rss")

    metrics: dict[str, Any] = {
        **integers,
        **timings,
        **optional,
        "schedule_search_elite_objectives": objectives,
        "schedule_search_elite_pairwise_distances_ppm": distances,
        "schedule_search_elite_min_distance_ppm": minimum,
        "schedule_search_elite_mean_distance_ppm": mean,
        "schedule_search_elite_max_distance_ppm": maximum,
        "schedule_search_elite_heap_lower_bound_mb": heap_bytes / (1024 * 1024),
        "schedule_search_elite_peak_heap_lower_bound_mb": peak_heap_bytes / (1024 * 1024),
    }
    return metrics, failures


def _validate_path_relink_metrics(
    record: dict[str, Any],
    *,
    label: str,
    wall_seconds: float,
) -> tuple[dict[str, Any], list[str]]:
    integers = {field: _integer(record.get(field), f"{label}.{field}") for field in PATH_RELINK_INTEGER_FIELDS}
    timings = {field: _number(record.get(field), f"{label}.{field}") for field in PATH_RELINK_TIMING_FIELDS}
    failures: list[str] = []

    enabled = integers["schedule_path_relink_enabled"]
    active = integers["schedule_path_relink_active"]
    requests = integers["schedule_path_relink_guide_requests"]
    best = integers["schedule_path_relink_best_guides"]
    diverse = integers["schedule_path_relink_diverse_guides"]
    loads = integers["schedule_path_relink_guide_loads"]
    generated = integers["schedule_path_relink_candidates_generated"]
    positive = integers["schedule_path_relink_candidates_positive_gain"]
    certified = integers["schedule_path_relink_acyclicity_certified"]
    unknown = integers["schedule_path_relink_acyclicity_unknown"]
    prefilter_rejections = integers["schedule_path_relink_prefilter_rejections"]
    retained = integers["schedule_path_relink_candidates_retained"]
    refined = integers["schedule_path_relink_candidates_refined"]
    shortlisted = integers["schedule_path_relink_candidates_shortlisted"]
    attempts = integers["schedule_path_relink_oracle_attempts"]
    accepts = integers["schedule_path_relink_oracle_accepts"]
    classified_rejections = (
        integers["schedule_path_relink_cycle_rejections"]
        + integers["schedule_path_relink_window_rejections"]
        + integers["schedule_path_relink_other_rejections"]
    )
    elapsed = timings["schedule_path_relink_worker_seconds_sum"]
    workspace = integers["schedule_path_relink_workspace_peak_bytes"]

    if enabled != 1:
        failures.append("path_relink_not_enabled")
    if active != 1 or requests == 0:
        failures.append("path_relink_not_active")
    if best + diverse != requests:
        raise GateError(f"{label}: path relink guide kinds must partition requests")
    if loads > requests:
        raise GateError(f"{label}: path relink guide loads cannot exceed requests")
    if positive != certified + unknown:
        raise GateError(f"{label}: positive path relink candidates must partition into certified and unknown acyclicity")
    if unknown != prefilter_rejections:
        raise GateError(f"{label}: unknown path relink acyclicity must be rejected by the prefilter")
    if positive > generated or retained > certified or refined > retained or shortlisted > refined:
        raise GateError(f"{label}: path relink candidate funnel is inconsistent")
    if integers["schedule_path_relink_no_move"] > requests:
        raise GateError(f"{label}: path relink no-move count cannot exceed requests")
    if integers["schedule_path_relink_guide_arc_gain_shortlisted"] < shortlisted:
        raise GateError(f"{label}: every shortlisted path relink move must have positive guide-arc gain")
    if attempts > shortlisted or accepts + classified_rejections > attempts:
        raise GateError(f"{label}: path relink oracle accounting is inconsistent")
    if integers["schedule_path_relink_rollbacks"] < classified_rejections:
        raise GateError(f"{label}: path relink classified rejections require rollback accounting")
    if integers["schedule_path_relink_elite_improvements"] > accepts:
        raise GateError(f"{label}: path relink elite improvements cannot exceed accepted moves")
    if integers["schedule_path_relink_guide_arc_gain_accepted"] < accepts:
        raise GateError(f"{label}: every accepted path relink move must have positive guide-arc gain")
    if elapsed > wall_seconds + 1e-6:
        raise GateError(f"{label}: path relink worker time exceeds process wall time")
    if requests > 0 and workspace == 0:
        raise GateError(f"{label}: active path relinking requires a positive workspace lower bound")
    if workspace >= PATH_RELINK_MAX_WORKSPACE_BYTES:
        failures.append("path_relink_workspace_limit")
    if integers["schedule_path_relink_guide_incompatible"] != 0:
        failures.append("path_relink_incompatible_guide")
    if integers["schedule_path_relink_guide_interruptions"] != 0:
        failures.append("path_relink_guide_interruption")
    if integers["schedule_path_relink_cycle_rejections"] != 0:
        failures.append("path_relink_cycle_rejection")
    if attempts == 0:
        failures.append("path_relink_no_oracle_attempt")

    return {**integers, **timings}, failures


def _validate_lns_shadow_metrics(
    record: dict[str, Any],
    *,
    label: str,
    wall_seconds: float,
) -> tuple[dict[str, Any], list[str], bool]:
    integers = {field: _integer(record.get(field), f"{label}.{field}") for field in LNS_SHADOW_INTEGER_FIELDS}
    timings = {field: _number(record.get(field), f"{label}.{field}") for field in LNS_SHADOW_TIMING_FIELDS}
    failures: list[str] = []

    enabled = integers["schedule_lns_shadow_enabled"]
    active = integers["schedule_lns_shadow_active"]
    owner_mask = integers["schedule_lns_shadow_owner_worker_mask"]
    attempts = integers["schedule_lns_attempts"]
    selected = integers["schedule_lns_selected_operations"]
    feasible = integers["schedule_lns_feasible"]
    reconstructed = integers["schedule_lns_reconstructed"]
    improvements = integers["schedule_lns_improvements"]
    shadow_improvements = integers["schedule_lns_shadow_improvements"]
    improvement_sum = integers["schedule_lns_shadow_improvement_sum"]
    best_improvement = integers["schedule_lns_shadow_best_improvement"]
    timeouts = integers["schedule_lns_timeouts"]
    elapsed = timings["schedule_lns_worker_seconds_sum"]
    workspace = integers["schedule_lns_workspace_peak_bytes"]

    if enabled != 1:
        failures.append("lns_shadow_not_enabled")
    if active != 1:
        failures.append("lns_shadow_not_active")
    if owner_mask != 1:
        failures.append("lns_shadow_wrong_owner")
    if attempts == 0:
        failures.append("lns_shadow_no_attempt")
    if selected == 0:
        failures.append("lns_shadow_no_selected_operation")
    if feasible > attempts or reconstructed > feasible:
        raise GateError(f"{label}: LNS feasible/reconstructed accounting is inconsistent")
    if improvements != shadow_improvements or shadow_improvements > reconstructed:
        raise GateError(f"{label}: every LNS improvement must remain a reconstructed shadow improvement")
    if shadow_improvements == 0:
        if improvement_sum != 0 or best_improvement != 0:
            raise GateError(f"{label}: empty LNS improvement stream must have zero gain")
    elif best_improvement == 0 or best_improvement > improvement_sum:
        raise GateError(f"{label}: LNS shadow gain accounting is inconsistent")
    if timeouts > attempts:
        raise GateError(f"{label}: LNS timeouts cannot exceed attempts")
    if shadow_improvements > attempts - timeouts:
        raise GateError(f"{label}: timed-out LNS attempts cannot report shadow improvements")
    for field in (
        "schedule_lns_infeasible",
        "schedule_lns_non_improving",
        "schedule_lns_reconstruction_rejections",
        "schedule_lns_oracle_rejections",
        "schedule_lns_exact_rejections",
    ):
        if integers[field] > attempts:
            raise GateError(f"{label}: {field} cannot exceed LNS attempts")
    if elapsed > wall_seconds + 1e-6:
        raise GateError(f"{label}: LNS worker time exceeds process wall time")
    if attempts > 0 and workspace == 0:
        raise GateError(f"{label}: attempted LNS shadow requires a positive workspace estimate")
    if workspace >= LNS_SHADOW_MAX_WORKSPACE_BYTES:
        failures.append("lns_shadow_workspace_limit")
    if integers["schedule_lns_interruptions"] != 0:
        failures.append("lns_shadow_interruption")
    if integers["schedule_lns_oracle_rejections"] != 0:
        failures.append("lns_shadow_oracle_rejection")

    effective = shadow_improvements > 0
    return {**integers, **timings}, failures, effective


def _indexed_integer_summary(value: object, label: str) -> dict[int, int] | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise GateError(f"{label} must be a non-empty profile=value summary")
    parsed: dict[int, int] = {}
    for item in value.split(","):
        try:
            raw_profile, raw_work = item.split("=", 1)
            profile = int(raw_profile)
            work = int(raw_work)
        except (TypeError, ValueError) as error:
            raise GateError(f"{label} contains an invalid profile=value item") from error
        if not 0 <= profile < 8 or work < 0 or profile in parsed:
            raise GateError(f"{label} contains an invalid or duplicate profile")
        parsed[profile] = work
    return parsed


def _validate_constructor_multistart_metrics(
    record: dict[str, Any],
    *,
    label: str,
    wall_seconds: float,
    peak_memory_mb: float,
) -> tuple[dict[str, Any], list[str]]:
    integers = {field: _integer(record.get(field), f"{label}.{field}") for field in C1_CONSTRUCTOR_INTEGER_FIELDS}
    timings = {field: _number(record.get(field), f"{label}.{field}") for field in C1_CONSTRUCTOR_TIMING_FIELDS}
    failures: list[str] = []

    requested = integers["schedule_constructor_workers_requested"]
    enabled = integers["schedule_constructor_multistart_enabled"]
    active = integers["schedule_constructor_multistart_active"]
    owner_mask = integers["schedule_constructor_multistart_owner_worker_mask"]
    attempts = integers["schedule_constructor_multistart_attempts"]
    constructions = integers["schedule_constructor_multistart_constructions"]
    feasible = integers["schedule_constructor_multistart_feasible"]
    distinct_fingerprints = integers["schedule_constructor_multistart_distinct_fingerprints"]
    other_fingerprint_observations = integers["schedule_constructor_multistart_other_fingerprint_observations"]
    initial_objective = integers["schedule_constructor_multistart_initial_objective"]
    best_objective = integers["schedule_constructor_multistart_best_objective"]
    improvements = integers["schedule_constructor_multistart_improvements"]
    work_units = integers["schedule_constructor_multistart_work_units"]
    interruptions = integers["schedule_constructor_multistart_interruptions"]
    failures_count = integers["schedule_constructor_multistart_failures"]
    workspace = integers["schedule_constructor_multistart_workspace_peak_bytes"]
    next_ordinal = integers["schedule_constructor_multistart_next_ordinal"]
    best_ordinal = integers["schedule_constructor_multistart_best_ordinal"]
    elapsed = timings["schedule_constructor_multistart_worker_seconds_sum"]

    if requested != 1 or enabled != 1:
        failures.append("constructor_multistart_not_enabled")
    if active != 1 or owner_mask != 1 << 6:
        failures.append("constructor_multistart_wrong_owner")
    if (
        integers["schedule_island_profile_mask"] != 63
        or integers["schedule_baseline_island_profile_mask"] != 63
        or integers["schedule_scored_island_profile_mask"] != 0
        or integers["schedule_island_profile_count"] != 6
    ):
        failures.append("constructor_multistart_wrong_ls_portfolio")
    if attempts == 0:
        failures.append("constructor_multistart_no_attempt")
    if attempts != constructions + interruptions + failures_count:
        raise GateError(f"{label}: constructor attempts must partition into completed constructions, interruptions, and failures")
    if constructions != feasible:
        raise GateError(f"{label}: every completed constructor run must be feasible")
    if feasible != distinct_fingerprints + other_fingerprint_observations:
        raise GateError(f"{label}: feasible constructor runs must partition into distinct and other fingerprint observations")
    if distinct_fingerprints > 2:
        raise GateError(f"{label}: constructor distinct fingerprint lower bound is saturated at two")
    if work_units != 64 * attempts:
        raise GateError(f"{label}: every constructor attempt must charge exactly 64 work units")
    if improvements > feasible:
        raise GateError(f"{label}: constructor improvements cannot exceed feasible constructions")
    if best_objective > initial_objective:
        raise GateError(f"{label}: constructor best objective cannot exceed its initial objective")
    if (best_objective < initial_objective) != (improvements > 0):
        raise GateError(f"{label}: constructor improvement count must agree with initial and best objectives")
    if not 6 <= best_ordinal < next_ordinal:
        raise GateError(f"{label}: constructor best ordinal must belong to the consumed stream")
    if next_ordinal != 6 + constructions + failures_count:
        raise GateError(f"{label}: constructor ordinal stream must advance only for completed or failed attempts")
    if attempts > 4 * integers["schedule_restart_boundaries"]:
        raise GateError(f"{label}: constructor attempts exceed the four-per-boundary cap")
    if elapsed > wall_seconds + 1e-6:
        raise GateError(f"{label}: constructor worker time exceeds process wall time")
    if attempts > 0 and workspace == 0:
        raise GateError(f"{label}: active constructor multi-start requires a positive workspace estimate")
    if workspace >= C1_MAX_WORKSPACE_BYTES:
        failures.append("constructor_multistart_workspace_limit")
    if distinct_fingerprints < 2:
        failures.append("constructor_multistart_lt_2_distinct_fingerprints")
    if failures_count != 0:
        failures.append("constructor_multistart_failure")
    if interruptions != 0:
        failures.append("constructor_multistart_interruption")
    if peak_memory_mb >= C1_MAX_PEAK_MEMORY_MB:
        failures.append("constructor_multistart_rss_limit")
    for field in (
        "schedule_incumbent_verification_rejections",
        "schedule_incumbent_incomplete_rejections",
        "schedule_work_budget_overruns",
    ):
        if record.get(field) != 0:
            failures.append(f"constructor_multistart_nonzero_{field}")

    profile_work = _indexed_integer_summary(
        record.get("schedule_profile_work_steps"),
        f"{label}.schedule_profile_work_steps",
    )
    if profile_work is None or set(profile_work) != set(range(6)):
        failures.append("constructor_multistart_wrong_profile_work_keys")
    return {**integers, **timings, "schedule_profile_work_steps": profile_work}, failures


def validate_provenance(path: Path, *, threads: int, checkpoints: set[float], seeds: set[int]) -> dict[str, Any]:
    sidecar = provenance_path(path)
    if not sidecar.is_file():
        raise GateError(f"provenance sidecar is required: {sidecar}")
    value = _read_json(sidecar)
    if value.get("schema_version") != 1:
        raise GateError(f"{sidecar}: schema_version must be 1")
    if value.get("threads") != threads:
        raise GateError(f"{sidecar}: threads do not match the gate")
    raw_budgets = value.get("budgets")
    raw_seeds = value.get("seeds")
    if not isinstance(raw_budgets, list) or not checkpoints.issubset({float(item) for item in raw_budgets if isinstance(item, (int, float)) and not isinstance(item, bool)}):
        raise GateError(f"{sidecar}: budgets do not cover the gate checkpoints")
    if not isinstance(raw_seeds, list) or not seeds.issubset({item for item in raw_seeds if isinstance(item, int) and not isinstance(item, bool)}):
        raise GateError(f"{sidecar}: seeds do not cover the gate")
    host = value.get("host")
    artifact = value.get("qayd_artifact")
    if not isinstance(host, dict) or not _valid_sha256(host.get("source_tree_sha256")):
        raise GateError(f"{sidecar}: source tree fingerprint is required")
    if not isinstance(artifact, dict) or not _valid_sha256(artifact.get("sha256")):
        raise GateError(f"{sidecar}: Qayd artifact fingerprint is required")
    if value.get("qayd_prepared") is not True:
        raise GateError(f"{sidecar}: qayd_prepared must be true")
    return value


def _instance_name(record: dict[str, Any]) -> str:
    value = record.get("instance")
    if isinstance(value, str) and value:
        return Path(value).stem
    value = record.get("instance_path")
    if isinstance(value, str) and value:
        return Path(value).stem
    raise GateError("record must identify its instance")


def validate_record(
    record: dict[str, Any],
    *,
    label: str,
    instance: str,
    checkpoint: float,
    seed: int,
    threads: int,
    require_search_elite: bool,
    require_path_relink: bool,
    require_lns_shadow: bool,
    require_constructor_multistart: bool,
    max_peak_memory_mb: float,
) -> dict[str, Any]:
    if _instance_name(record) != instance:
        raise GateError(f"{label}: wrong instance")
    if float(_number(record.get("checkpoint_seconds"), f"{label}.checkpoint_seconds")) != checkpoint:
        raise GateError(f"{label}: wrong checkpoint")
    if record.get("seed") != seed:
        raise GateError(f"{label}: wrong seed")
    if record.get("threads") != threads or record.get("requested_threads") != threads:
        raise GateError(f"{label}: threads and requested_threads must match the gate")
    solver = record.get("solver")
    if not isinstance(solver, str) or not solver.startswith("qayd-"):
        raise GateError(f"{label}: solver must be a Qayd variant")
    if record.get("problem") != "jssp" or record.get("engine") != "ls":
        raise GateError(f"{label}: record must be a JSSP LS run")
    if record.get("status") not in FEASIBLE:
        raise GateError(f"{label}: record must contain a feasible status")
    if record.get("verified") is not True or record.get("return_code") != 0 or record.get("timed_out") is not False:
        raise GateError(f"{label}: record must be verified, successful, and not externally timed out")
    objectives = record.get("objectives")
    if not isinstance(objectives, list) or len(objectives) != 1:
        raise GateError(f"{label}: objectives must contain one makespan")
    objective = _integer(objectives[0], f"{label}.objectives[0]", minimum=1)
    digest = record.get("instance_sha256")
    if not _valid_sha256(digest):
        raise GateError(f"{label}: instance SHA-256 is required")
    if record.get("start_vector_length") != 1_000_000 or not _valid_sha256(record.get("start_vector_sha256")):
        raise GateError(f"{label}: compact one-million-start commitment is required")
    if record.get("verification_counts") != EXPECTED_COUNTS:
        raise GateError(f"{label}: independent replay counts do not match Large-TA")
    for field in (
        "schedule_oracle_mismatches",
        "schedule_work_budget_overruns",
        "schedule_incumbent_verification_rejections",
        "schedule_incumbent_incomplete_rejections",
    ):
        if record.get(field) != 0:
            raise GateError(f"{label}: {field} must be exactly 0")
    verification_interruptions = _integer(
        record.get("schedule_incumbent_verification_interruptions"),
        f"{label}.schedule_incumbent_verification_interruptions",
    )
    if _integer(record.get("schedule_incumbent_verifications"), f"{label}.schedule_incumbent_verifications", minimum=1) < 1:
        raise GateError(f"{label}: at least one verified incumbent is required")
    probes = _integer(record.get("schedule_moves_considered"), f"{label}.schedule_moves_considered", minimum=1)
    dirty = _integer(record.get("schedule_dirty_cone_operations"), f"{label}.schedule_dirty_cone_operations")
    accepted = _integer(record.get("schedule_moves_accepted"), f"{label}.schedule_moves_accepted")
    wall_seconds = _number(record.get("wall_seconds"), f"{label}.wall_seconds")
    peak_memory_mb = _number(record.get("peak_memory_mb"), f"{label}.peak_memory_mb")
    metrics: dict[str, Any] = {
        field: _integer(record.get(field, default), f"{label}.{field}")
        for field, default in METRIC_DEFAULTS.items()
    }
    metrics["schedule_profile_work_steps"] = _indexed_integer_summary(
        record.get("schedule_profile_work_steps"),
        f"{label}.schedule_profile_work_steps",
    )
    shadow_failures: list[str] = []
    if require_search_elite:
        search_elite_metrics, shadow_failures = _validate_search_elite_metrics(
            record,
            label=label,
            wall_seconds=wall_seconds,
            peak_memory_mb=peak_memory_mb,
            threads=threads,
            max_peak_memory_mb=max_peak_memory_mb,
        )
        metrics.update(search_elite_metrics)
    path_relink_failures: list[str] = []
    if require_path_relink:
        path_relink_metrics, path_relink_failures = _validate_path_relink_metrics(
            record,
            label=label,
            wall_seconds=wall_seconds,
        )
        metrics.update(path_relink_metrics)
    lns_shadow_failures: list[str] = []
    lns_effective = False
    if require_lns_shadow:
        lns_shadow_metrics, lns_shadow_failures, lns_effective = _validate_lns_shadow_metrics(
            record,
            label=label,
            wall_seconds=wall_seconds,
        )
        metrics.update(lns_shadow_metrics)
    constructor_multistart_failures: list[str] = []
    if require_constructor_multistart:
        constructor_metrics, constructor_multistart_failures = _validate_constructor_multistart_metrics(
            record,
            label=label,
            wall_seconds=wall_seconds,
            peak_memory_mb=peak_memory_mb,
        )
        metrics.update(constructor_metrics)
    return {
        "objective": objective,
        "instance_sha256": str(digest).lower(),
        "wall_seconds": wall_seconds,
        "peak_memory_mb": peak_memory_mb,
        "probes": probes,
        "accepted": accepted,
        "verification_interruptions": verification_interruptions,
        "dirty_cone_operations": dirty,
        "dirty_operations_per_probe": dirty / probes,
        "probes_per_accepted_move": probes / accepted if accepted else None,
        "metrics": metrics,
        "shadow_ready": require_search_elite and not shadow_failures,
        "shadow_failures": shadow_failures,
        "path_relink_ready": require_path_relink and not path_relink_failures,
        "path_relink_failures": path_relink_failures,
        "lns_shadow_ready": require_lns_shadow and not lns_shadow_failures,
        "lns_shadow_failures": lns_shadow_failures,
        "lns_effective": require_lns_shadow and lns_effective,
        "constructor_multistart_ready": require_constructor_multistart and not constructor_multistart_failures,
        "constructor_multistart_failures": constructor_multistart_failures,
    }


def _select(
    records: Iterable[dict[str, Any]],
    *,
    instance: str,
    checkpoints: set[float],
    seeds: set[int],
) -> dict[tuple[float, int], dict[str, Any]]:
    selected: dict[tuple[float, int], dict[str, Any]] = {}
    for record in records:
        try:
            name = _instance_name(record)
        except GateError:
            continue
        checkpoint = record.get("checkpoint_seconds")
        seed = record.get("seed")
        if name != instance or not isinstance(checkpoint, (int, float)) or isinstance(checkpoint, bool) or not isinstance(seed, int) or isinstance(seed, bool):
            continue
        key = (float(checkpoint), seed)
        if key not in {(budget, item_seed) for budget in checkpoints for item_seed in seeds}:
            continue
        if key in selected:
            raise GateError(f"duplicate record for checkpoint={key[0]:g}, seed={key[1]}")
        selected[key] = record
    expected = {(budget, seed) for budget in checkpoints for seed in seeds}
    missing = sorted(expected - set(selected))
    if missing:
        raise GateError(f"missing records: {missing}")
    return selected


def build_summary(
    candidate_path: Path,
    *,
    baseline_path: Path | None,
    instance: str,
    checkpoints: set[float],
    seeds: set[int],
    threads: int,
    require_path_relink: bool = False,
    require_lns_shadow: bool = False,
    require_constructor_multistart: bool = False,
    max_peak_memory_mb: float = DEFAULT_MAX_PEAK_MEMORY_MB,
) -> dict[str, Any]:
    max_peak_memory_mb = _number(max_peak_memory_mb, "max_peak_memory_mb", minimum=1.0)
    candidate_provenance = validate_provenance(candidate_path, threads=threads, checkpoints=checkpoints, seeds=seeds)
    if require_path_relink and candidate_provenance.get("qayd_schedule_path_relink") is not True:
        raise GateError("candidate provenance must enable qayd_schedule_path_relink")
    if require_lns_shadow and candidate_provenance.get("qayd_schedule_lns_shadow") is not True:
        raise GateError("candidate provenance must enable qayd_schedule_lns_shadow")
    if require_constructor_multistart:
        if baseline_path is None:
            raise GateError("constructor C1 requires a fresh baseline campaign with per-profile work accounting")
        if candidate_provenance.get("qayd_schedule_constructor_workers") != 1:
            raise GateError("candidate provenance must set qayd_schedule_constructor_workers to 1")
        if candidate_provenance.get("qayd_schedule_path_relink") is not False:
            raise GateError("constructor C1 provenance must disable qayd_schedule_path_relink")
        if candidate_provenance.get("qayd_schedule_lns_shadow") is not False:
            raise GateError("constructor C1 provenance must disable qayd_schedule_lns_shadow")
        if candidate_provenance.get("qayd_engine") != "ls" or candidate_provenance.get("profile_qayd") is not True:
            raise GateError("constructor C1 provenance must select profiled Qayd local search")
    candidates = _select(load_jsonl(candidate_path), instance=instance, checkpoints=checkpoints, seeds=seeds)
    baseline_records = None
    baseline_provenance = None
    if baseline_path is not None:
        baseline_provenance = validate_provenance(baseline_path, threads=threads, checkpoints=checkpoints, seeds=seeds)
        baseline_records = _select(load_jsonl(baseline_path), instance=instance, checkpoints=checkpoints, seeds=seeds)
        if require_constructor_multistart:
            if baseline_provenance.get("qayd_schedule_constructor_workers", 0) != 0:
                raise GateError("constructor C1 baseline must disable schedule constructor workers")
            if baseline_provenance.get("qayd_schedule_path_relink") is not False or baseline_provenance.get("qayd_schedule_lns_shadow") is not False:
                raise GateError("constructor C1 baseline must disable path relinking and LNS shadow")
            if baseline_provenance.get("qayd_engine") != "ls" or baseline_provenance.get("profile_qayd") is not True:
                raise GateError("constructor C1 baseline must select profiled Qayd local search")
            candidate_identity = (
                candidate_provenance["host"]["source_tree_sha256"],
                candidate_provenance["qayd_artifact"]["sha256"],
            )
            baseline_identity = (
                baseline_provenance["host"]["source_tree_sha256"],
                baseline_provenance["qayd_artifact"]["sha256"],
            )
            if candidate_identity != baseline_identity:
                raise GateError("constructor C1 candidate and baseline must use identical source and artifact fingerprints")

    rows = []
    digests = set()
    for checkpoint, seed in sorted(candidates):
        key = (checkpoint, seed)
        candidate = validate_record(
            candidates[key],
            label=f"candidate[{checkpoint:g},{seed}]",
            instance=instance,
            checkpoint=checkpoint,
            seed=seed,
            threads=threads,
            require_search_elite=True,
            require_path_relink=require_path_relink,
            require_lns_shadow=require_lns_shadow,
            require_constructor_multistart=require_constructor_multistart,
            max_peak_memory_mb=max_peak_memory_mb,
        )
        digests.add(candidate["instance_sha256"])
        baseline = None
        if baseline_records is not None:
            baseline = validate_record(
                baseline_records[key],
                label=f"baseline[{checkpoint:g},{seed}]",
                instance=instance,
                checkpoint=checkpoint,
                seed=seed,
                threads=threads,
                require_search_elite=False,
                require_path_relink=False,
                require_lns_shadow=False,
                require_constructor_multistart=False,
                max_peak_memory_mb=max_peak_memory_mb,
            )
            if baseline["instance_sha256"] != candidate["instance_sha256"]:
                raise GateError(f"instance SHA-256 differs for checkpoint={checkpoint:g}, seed={seed}")
        rows.append(
            {
                "checkpoint_seconds": checkpoint,
                "seed": seed,
                "candidate": candidate,
                "baseline_objective": baseline["objective"] if baseline else None,
                "objective_gain": baseline["objective"] - candidate["objective"] if baseline else None,
            }
        )
    if len(digests) != 1:
        raise GateError("candidate records do not use one normalized instance")

    checkpoints_summary: list[dict[str, Any]] = []
    for checkpoint in sorted(checkpoints):
        checkpoint_rows = [row for row in rows if row["checkpoint_seconds"] == checkpoint]
        objectives = [row["candidate"]["objective"] for row in checkpoint_rows]
        median_objective = statistics.median(objectives)
        checkpoint_shadow_ready = all(row["candidate"]["shadow_ready"] for row in checkpoint_rows)
        checkpoint_path_relink_ready = require_path_relink and all(
            row["candidate"]["path_relink_ready"] for row in checkpoint_rows
        )
        checkpoint_lns_shadow_ready = require_lns_shadow and all(
            row["candidate"]["lns_shadow_ready"] for row in checkpoint_rows
        )
        checkpoint_lns_effective = require_lns_shadow and all(
            row["candidate"]["lns_effective"] for row in checkpoint_rows
        )
        checkpoint_constructor_raw_ready = require_constructor_multistart and all(
            row["candidate"]["constructor_multistart_ready"] for row in checkpoint_rows
        )
        retention_checks: list[bool] = []
        for row in checkpoint_rows:
            candidate_work = row["candidate"]["metrics"].get("schedule_profile_work_steps")
            baseline = baseline_records.get((checkpoint, row["seed"])) if baseline_records is not None else None
            baseline_work = (
                _indexed_integer_summary(
                    baseline.get("schedule_profile_work_steps"),
                    f"baseline[{checkpoint:g},{row['seed']}].schedule_profile_work_steps",
                )
                if baseline is not None
                else None
            )
            if candidate_work is not None and baseline_work is not None and all(
                profile in candidate_work and profile in baseline_work for profile in range(6)
            ):
                retention_checks.append(
                    all(candidate_work[profile] >= C1_LS_PROFILE_RETENTION * baseline_work[profile] for profile in range(6))
                )
            elif require_constructor_multistart:
                retention_checks.append(False)
        checkpoint_ls_profile_retention = all(retention_checks) if retention_checks else None
        checkpoint_constructor_ready = (
            checkpoint_constructor_raw_ready and checkpoint_ls_profile_retention is not False
        )
        checkpoint_constructor_final_ok = require_constructor_multistart and all(
            row["candidate"]["objective"] <= C1_FINAL_OBJECTIVE_LIMIT for row in checkpoint_rows
        )
        checkpoint_constructor_best_ok = require_constructor_multistart and all(
            row["candidate"]["metrics"]["schedule_constructor_multistart_best_objective"]
            < C1_CONSTRUCTOR_OBJECTIVE_EXCLUSIVE
            for row in checkpoint_rows
        )
        checkpoint_constructor_signal = (
            checkpoint_constructor_ready
            and checkpoint_constructor_final_ok
            and checkpoint_constructor_best_ok
        )
        quality_protocol_ready = (
            checkpoint_shadow_ready
            and (not require_path_relink or checkpoint_path_relink_ready)
            and (not require_lns_shadow or checkpoint_lns_shadow_ready)
            and (not require_constructor_multistart or checkpoint_constructor_ready)
        )
        stage_pass = {SHADOW_STAGE: checkpoint_shadow_ready}
        stage_pass.update({
            name: quality_protocol_ready and checkpoint == 600.0 and median_objective <= threshold
            for name, threshold in STAGE_THRESHOLDS.items()
        })
        search_elite_metrics = [row["candidate"]["metrics"] for row in checkpoint_rows]
        minimum_distances = [
            metrics["schedule_search_elite_min_distance_ppm"]
            for metrics in search_elite_metrics
            if metrics["schedule_search_elite_min_distance_ppm"] is not None
        ]
        shadow_failure_counts = Counter(
            failure
            for row in checkpoint_rows
            for failure in row["candidate"]["shadow_failures"]
        )
        path_relink_failure_counts = Counter(
            failure
            for row in checkpoint_rows
            for failure in row["candidate"]["path_relink_failures"]
        )
        lns_shadow_failure_counts = Counter(
            failure
            for row in checkpoint_rows
            for failure in row["candidate"]["lns_shadow_failures"]
        )
        constructor_failure_counts = Counter(
            failure
            for row in checkpoint_rows
            for failure in row["candidate"]["constructor_multistart_failures"]
        )
        checkpoints_summary.append(
            {
                "checkpoint_seconds": checkpoint,
                "seed_count": len(checkpoint_rows),
                "median_objective": median_objective,
                "minimum_objective": min(objectives),
                "maximum_objective": max(objectives),
                "median_dirty_operations_per_probe": statistics.median(
                    row["candidate"]["dirty_operations_per_probe"] for row in checkpoint_rows
                ),
                "median_peak_memory_mb": statistics.median(
                    row["candidate"]["peak_memory_mb"] for row in checkpoint_rows
                ),
                "minimum_search_elite_pool_size": min(
                    metrics["schedule_search_elite_pool_size"] for metrics in search_elite_metrics
                ),
                "maximum_search_elite_pool_size": max(
                    metrics["schedule_search_elite_pool_size"] for metrics in search_elite_metrics
                ),
                "minimum_search_elite_distance_ppm": min(minimum_distances) if minimum_distances else None,
                "median_search_elite_heap_lower_bound_mb": statistics.median(
                    metrics["schedule_search_elite_heap_lower_bound_mb"] for metrics in search_elite_metrics
                ),
                "maximum_search_elite_peak_heap_lower_bound_mb": max(
                    metrics["schedule_search_elite_peak_heap_lower_bound_mb"] for metrics in search_elite_metrics
                ),
                "median_search_elite_capture_worker_seconds_sum": statistics.median(
                    metrics["schedule_search_elite_capture_worker_seconds_sum"] for metrics in search_elite_metrics
                ),
                "median_search_elite_merge_wall_seconds": statistics.median(
                    metrics["schedule_search_elite_merge_wall_seconds"] for metrics in search_elite_metrics
                ),
                "shadow_ready": checkpoint_shadow_ready,
                "shadow_failure_counts": dict(sorted(shadow_failure_counts.items())),
                "path_relink_required": require_path_relink,
                "path_relink_ready": checkpoint_path_relink_ready,
                "path_relink_failure_counts": dict(sorted(path_relink_failure_counts.items())),
                "lns_shadow_required": require_lns_shadow,
                "lns_shadow_ready": checkpoint_lns_shadow_ready,
                "lns_effective": checkpoint_lns_effective,
                "lns_shadow_failure_counts": dict(sorted(lns_shadow_failure_counts.items())),
                "constructor_multistart_required": require_constructor_multistart,
                "constructor_multistart_ready": checkpoint_constructor_ready,
                "constructor_multistart_failure_counts": dict(sorted(constructor_failure_counts.items())),
                "constructor_multistart_final_objective_ok": checkpoint_constructor_final_ok,
                "constructor_multistart_best_objective_ok": checkpoint_constructor_best_ok,
                "constructor_multistart_quality_signal": checkpoint_constructor_signal,
                "constructor_multistart_ls_profile_retention": checkpoint_ls_profile_retention,
                "stage_pass": stage_pass,
            }
        )
    shadow_ready = all(row["candidate"]["shadow_ready"] for row in rows)
    path_relink_ready = require_path_relink and all(row["candidate"]["path_relink_ready"] for row in rows)
    lns_shadow_ready = require_lns_shadow and all(row["candidate"]["lns_shadow_ready"] for row in rows)
    lns_effective = require_lns_shadow and all(row["candidate"]["lns_effective"] for row in rows)
    constructor_multistart_ready = require_constructor_multistart and all(
        item["constructor_multistart_ready"] for item in checkpoints_summary
    )
    constructor_multistart_quality_signal = require_constructor_multistart and all(
        item["constructor_multistart_quality_signal"] for item in checkpoints_summary
    )
    quality_achieved = [
        name
        for name, threshold in STAGE_THRESHOLDS.items()
        if any(item["stage_pass"][name] for item in checkpoints_summary)
    ]
    stage_pass = {SHADOW_STAGE: shadow_ready}
    stage_pass.update({
        name: (
            shadow_ready
            and (not require_path_relink or path_relink_ready)
            and (not require_lns_shadow or lns_shadow_ready)
            and (not require_constructor_multistart or constructor_multistart_ready)
            and name in quality_achieved
        )
        for name in STAGE_THRESHOLDS
    })
    achieved = [name for name in STAGE_ORDER if stage_pass[name]]
    qualified_quality_achieved = [name for name in STAGE_THRESHOLDS if stage_pass[name]]
    return {
        "schema_version": 1,
        "instance": instance,
        "instance_sha256": next(iter(digests)),
        "threads": threads,
        "seeds": sorted(seeds),
        "checkpoints_seconds": sorted(checkpoints),
        "stage_thresholds": STAGE_THRESHOLDS,
        "stage_order": list(STAGE_ORDER),
        "stage_contract": {
            SHADOW_STAGE: {
                "quality_neutral": True,
                "minimum_pool_size": 2,
                "maximum_peak_memory_mb_exclusive": max_peak_memory_mb,
                "required_zero_fields": [
                    "schedule_oracle_mismatches",
                    "schedule_work_budget_overruns",
                    "schedule_incumbent_verification_rejections",
                    "schedule_incumbent_incomplete_rejections",
                    "schedule_search_elite_snapshot_errors",
                    "schedule_search_elite_merge_errors",
                    "schedule_search_elite_interruptions",
                    "schedule_search_elite_snapshot_interruptions",
                ],
                "informational_stop_fields": [
                    "schedule_search_elite_batches_skipped_after_stop",
                ],
            },
            "path_relink": {
                "required": require_path_relink,
                "maximum_workspace_bytes_exclusive": PATH_RELINK_MAX_WORKSPACE_BYTES,
                "requires_oracle_attempt": True,
                "required_zero_fields": [
                    "schedule_path_relink_guide_incompatible",
                    "schedule_path_relink_guide_interruptions",
                ],
            },
            "lns_shadow": {
                "required": require_lns_shadow,
                "quality_neutral": True,
                "owner_worker_mask": 1,
                "maximum_workspace_bytes_exclusive": LNS_SHADOW_MAX_WORKSPACE_BYTES,
                "requires_attempt": True,
                "requires_selected_operation": True,
                "required_zero_fields": [
                    "schedule_lns_interruptions",
                    "schedule_lns_oracle_rejections",
                ],
                "diagnostic_fields": [
                    "schedule_lns_timeouts",
                    "schedule_lns_reconstruction_rejections",
                    "schedule_lns_exact_rejections",
                ],
            },
            "constructor_multistart_c1": {
                "required": require_constructor_multistart,
                "requested_workers": 1,
                "owner_worker_mask": 64,
                "work_units_per_attempt": 64,
                "maximum_peak_memory_mb_exclusive": C1_MAX_PEAK_MEMORY_MB,
                "maximum_workspace_bytes_exclusive": C1_MAX_WORKSPACE_BYTES,
                "minimum_ls_profile_retention_when_measurable": C1_LS_PROFILE_RETENTION,
                "final_objective_limit_inclusive": C1_FINAL_OBJECTIVE_LIMIT,
                "constructor_best_objective_limit_exclusive": C1_CONSTRUCTOR_OBJECTIVE_EXCLUSIVE,
                "required_zero_fields": [
                    "schedule_constructor_multistart_interruptions",
                    "schedule_constructor_multistart_failures",
                    "schedule_incumbent_verification_rejections",
                    "schedule_incumbent_incomplete_rejections",
                    "schedule_work_budget_overruns",
                ],
            },
        },
        "stage_pass": stage_pass,
        "shadow_ready": shadow_ready,
        "path_relink_required": require_path_relink,
        "path_relink_ready": path_relink_ready,
        "lns_shadow_required": require_lns_shadow,
        "lns_shadow_ready": lns_shadow_ready,
        "lns_effective": lns_effective,
        "constructor_multistart_required": require_constructor_multistart,
        "constructor_multistart_ready": constructor_multistart_ready,
        "constructor_multistart_quality_signal": constructor_multistart_quality_signal,
        "highest_quality_stage": qualified_quality_achieved[-1] if qualified_quality_achieved else None,
        "highest_stage": achieved[-1] if achieved else None,
        "target_ready": stage_pass["target"],
        "rows": rows,
        "checkpoints": checkpoints_summary,
        "candidate_provenance": {
            "source_tree_sha256": candidate_provenance["host"]["source_tree_sha256"],
            "artifact_sha256": candidate_provenance["qayd_artifact"]["sha256"],
        },
        "baseline_provenance": None
        if baseline_provenance is None
        else {
            "source_tree_sha256": baseline_provenance["host"]["source_tree_sha256"],
            "artifact_sha256": baseline_provenance["qayd_artifact"]["sha256"],
        },
        "status_counts": dict(Counter(record.get("status", "MISSING") for record in candidates.values())),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Qayd JSSP hybrid gate",
        "",
        f"Instance: `{summary['instance']}`",
        "",
        "SHADOW is an observational archive/correctness gate. Passing it does not claim an objective improvement.",
        "",
        "| Checkpoint | Seeds | SHADOW | Path relink | LNS shadow | LNS effective | C1 ready | C1 signal | Median objective | Best | Worst | Min pool | Min distance ppm | Elite heap LB MiB | Peak RSS MiB |",
        "|---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary["checkpoints"]:
        lines.append(
            "| {checkpoint:g} | {seeds} | {shadow} | {path_relink} | {lns_shadow} | {lns_effective} | {constructor_ready} | {constructor_signal} | {median:g} | {best} | {worst} | {pool} | {distance} | {heap:.1f} | {rss:.1f} |".format(
                checkpoint=item["checkpoint_seconds"],
                seeds=item["seed_count"],
                shadow="pass" if item["shadow_ready"] else "fail",
                path_relink=("pass" if item["path_relink_ready"] else "fail") if item["path_relink_required"] else "not required",
                lns_shadow=("pass" if item["lns_shadow_ready"] else "fail") if item["lns_shadow_required"] else "not required",
                lns_effective=("yes" if item["lns_effective"] else "no") if item["lns_shadow_required"] else "not required",
                constructor_ready=("pass" if item["constructor_multistart_ready"] else "fail")
                if item["constructor_multistart_required"]
                else "not required",
                constructor_signal=("yes" if item["constructor_multistart_quality_signal"] else "no")
                if item["constructor_multistart_required"]
                else "not required",
                median=item["median_objective"],
                best=item["minimum_objective"],
                worst=item["maximum_objective"],
                pool=item["minimum_search_elite_pool_size"],
                distance=item["minimum_search_elite_distance_ppm"] if item["minimum_search_elite_distance_ppm"] is not None else "none",
                heap=item["median_search_elite_heap_lower_bound_mb"],
                rss=item["median_peak_memory_mb"],
            )
        )
    lines.extend(
        [
            "",
            f"SHADOW ready: `{'yes' if summary['shadow_ready'] else 'no'}`",
            "",
            f"Path relink ready: `{'yes' if summary['path_relink_ready'] else 'no'}`"
            if summary["path_relink_required"]
            else "Path relink ready: `not required`",
            "",
            f"LNS shadow ready: `{'yes' if summary['lns_shadow_ready'] else 'no'}`"
            if summary["lns_shadow_required"]
            else "LNS shadow ready: `not required`",
            "",
            f"LNS effective: `{'yes' if summary['lns_effective'] else 'no'}`"
            if summary["lns_shadow_required"]
            else "LNS effective: `not required`",
            "",
            f"Constructor C1 ready: `{'yes' if summary['constructor_multistart_ready'] else 'no'}`"
            if summary["constructor_multistart_required"]
            else "Constructor C1 ready: `not required`",
            "",
            f"Constructor C1 quality signal: `{'yes' if summary['constructor_multistart_quality_signal'] else 'no'}`"
            if summary["constructor_multistart_required"]
            else "Constructor C1 quality signal: `not required`",
            "",
            f"Highest stage: `{summary['highest_stage'] or 'none'}`",
            "",
            f"Highest quality stage: `{summary['highest_quality_stage'] or 'none'}`",
            "",
            f"Target ready: `{'yes' if summary['target_ready'] else 'no'}`",
            "",
        ]
    )
    return "\n".join(lines)


def _csv_ints(value: str) -> set[int]:
    try:
        values = {int(item) for item in value.split(",") if item}
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required")
    return values


def _csv_floats(value: str) -> set[float]:
    try:
        values = {float(item) for item in value.split(",") if item}
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from error
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--instance", default="tai_j1000_m1000_1")
    parser.add_argument("--checkpoints", type=_csv_floats, default={600.0})
    parser.add_argument("--seeds", type=_csv_ints, default={0})
    parser.add_argument("--threads", type=int, default=7)
    parser.add_argument(
        "--max-peak-memory-mb",
        type=float,
        default=DEFAULT_MAX_PEAK_MEMORY_MB,
        help="exclusive SHADOW RSS limit in MiB (default: 12288)",
    )
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--require-path-relink", action="store_true")
    parser.add_argument("--require-lns-shadow", action="store_true")
    parser.add_argument("--require-constructor-multistart", action="store_true")
    parser.add_argument("--require-constructor-signal", action="store_true")
    parser.add_argument("--require-stage", choices=STAGE_ORDER)
    args = parser.parse_args()
    try:
        summary = build_summary(
            args.candidate,
            baseline_path=args.baseline,
            instance=args.instance,
            checkpoints=args.checkpoints,
            seeds=args.seeds,
            threads=args.threads,
            require_path_relink=args.require_path_relink,
            require_lns_shadow=args.require_lns_shadow,
            require_constructor_multistart=args.require_constructor_multistart or args.require_constructor_signal,
            max_peak_memory_mb=args.max_peak_memory_mb,
        )
    except GateError as error:
        parser.error(str(error))
    if args.json:
        args.json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = render_markdown(summary)
    if args.markdown:
        args.markdown.write_text(markdown, encoding="utf-8")
    else:
        print(markdown)
    if args.require_stage and not summary["stage_pass"][args.require_stage]:
        return 2
    if args.require_path_relink and not summary["path_relink_ready"]:
        return 2
    if args.require_lns_shadow and not summary["lns_shadow_ready"]:
        return 2
    if (args.require_constructor_multistart or args.require_constructor_signal) and not summary["constructor_multistart_ready"]:
        return 2
    if args.require_constructor_signal and not summary["constructor_multistart_quality_signal"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
