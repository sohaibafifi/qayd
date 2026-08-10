#!/usr/bin/env python3
"""Audit the large-scale routing-search acceptance criteria.

The report combines a positioning campaign with a clean-room behavior-probe
summary. It is deliberately strict about paired runs, verified incumbents and
profiling coverage so a fast isolated run cannot make the gate pass.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Sequence


FEASIBLE_STATUSES = {"SAT", "SATISFIABLE", "FEASIBLE", "OPTIMAL", "OPTIMUM"}
ROOT = Path(__file__).resolve().parents[1]
TARGETS = {
    "X-n401-k29.vrp": {
        "path": "data/vrplib/CVRP/X-n401-k29.vrp",
        "sha256": "11ba49745cc7080cd8d67a6c508ba9447f059709b347f1eb81fcd41fbefc55a0",
        "best_known": 66_154.0,
        "gap_limit": 10.0,
        "hgs_median_limit": 1.35,
        "hgs_max_limit": 2.0,
        "hgs_mad_limit": 0.55,
    },
    "X-n701-k44.vrp": {
        "path": "data/vrplib/CVRP/X-n701-k44.vrp",
        "sha256": "3ee7f05bb239d893db8f6b5959ad151f2ef400881e63847fbdddd5b2ff3a7742",
        "best_known": 81_923.0,
        "gap_limit": 10.0,
        "hgs_median_limit": 3.40,
        "hgs_max_limit": 4.0,
        "hgs_mad_limit": 0.60,
    },
    "X-n1001-k43.vrp": {
        "path": "data/vrplib/CVRP/X-n1001-k43.vrp",
        "sha256": "97c950e592d8a63d4f81669f1fda9762de2a79e4a213f9e7401c699a754dc492",
        "best_known": 72_355.0,
        "gap_limit": 15.0,
        "hgs_median_limit": 5.15,
        "hgs_max_limit": 5.7,
        "hgs_mad_limit": 0.55,
    },
}
TARGET_GAPS = {name: float(spec["gap_limit"]) for name, spec in TARGETS.items()}
REQUIRED_SEED_IDS = (0, 1, 2, 3, 4)
REQUIRED_SEEDS = len(REQUIRED_SEED_IDS)
REQUIRED_BEHAVIOR_PAIRS = 40
MAX_RSS_MB = 256.0
REQUIRED_BEHAVIOR_GENERATOR = {
    "instance_seed": 20260806,
    "sizes": [200],
    "distance_scale": 7,
    "edge_delta": 25,
}
REQUIRED_BEHAVIOR_FACTORS = {
    "distance-scale", "index-permutation", "single-edge", "capacity-threshold",
}
REQUIRED_CHECKPOINTS_NS = {250_000_000, 1_000_000_000, 5_000_000_000}
REQUIRED_PROFILE_NAMES = {
    "relocate", "swap", "or-opt", "two-opt-star", "cross-exchange", "reverse",
    "route-elimination", "ejection-chain", "chain-relocate", "guided-segment-exchange",
    "path-relink", "elite-archive", "elite-selection",
    "alns_destroy/shaw", "alns_destroy/segments", "alns_destroy/worst", "alns_destroy/route",
    "alns_repair/greedy", "alns_repair/regret-2", "alns_repair/regret-3", "alns_repair/blink-regret-2",
}
REQUIRED_EVALUATED_PROFILE_NAMES = {
    "relocate", "swap", "or-opt", "two-opt-star", "cross-exchange", "reverse",
    "route-elimination", "ejection-chain", "chain-relocate", "guided-segment-exchange",
    "path-relink",
    "alns_repair/greedy", "alns_repair/regret-2", "alns_repair/regret-3",
    "alns_repair/blink-regret-2",
}
REQUIRED_ROUTING_COUNTERS = {
    "slices", "descent_slices", "alns_slices", "relink_slices", "global_scan_slices",
    "route_elimination_attempts", "ejection_chain_attempts", "chain_relocate_attempts",
    "guided_segment_exchange_attempts", "macro_candidates_built", "macro_budget_exhaustions",
    "elite_insertions", "elite_rejections", "path_relink_attempts", "path_relink_steps",
    "path_relink_budget_exhaustions",
}


def finite(value: object) -> float | None:
    if not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def median(values: Iterable[object]) -> float | None:
    numbers = [number for value in values if (number := finite(value)) is not None]
    return statistics.median(numbers) if numbers else None


def median_absolute_deviation(values: Iterable[object]) -> float | None:
    numbers = [number for value in values if (number := finite(value)) is not None]
    if not numbers:
        return None
    center = statistics.median(numbers)
    return statistics.median(abs(value - center) for value in numbers)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if isinstance(value, dict):
                records.append(value)
    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def campaign_provenance_path(campaign: Path) -> Path:
    return campaign.with_suffix(campaign.suffix + ".provenance.json")


def load_campaign_provenance(campaign: Path) -> dict[str, Any]:
    path = campaign_provenance_path(campaign)
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def campaign_provenance_errors(provenance: dict[str, Any], solver: str) -> list[str]:
    errors = []
    if provenance.get("schema_version") != 1:
        errors.append("campaign provenance sidecar is missing or invalid")
    if provenance.get("budgets") != [10]:
        errors.append("campaign provenance must declare only the 10-second budget")
    if provenance.get("seeds") != list(REQUIRED_SEED_IDS):
        errors.append("campaign provenance must declare seeds 0 through 4 exactly")
    if provenance.get("threads") != 1:
        errors.append("campaign provenance must declare single-thread execution")
    if provenance.get("qayd_engine") != "ls" or provenance.get("profile_qayd") is not True:
        errors.append("campaign provenance must declare profiled qayd local search")
    if provenance.get("qayd_prepared") is not True:
        errors.append("campaign provenance must declare a source-matched qayd build")
    versions = provenance.get("solvers")
    if not isinstance(versions, dict) or not all(isinstance(versions.get(name), str) and versions[name] for name in (solver, "hgs")):
        errors.append("campaign provenance must identify qayd and HGS versions")
    host = provenance.get("host")
    if (
        not isinstance(host, dict)
        or not isinstance(host.get("commit"), str)
        or len(host["commit"]) != 40
        or not isinstance(host.get("dirty"), bool)
        or not isinstance(host.get("source_tree_sha256"), str)
        or len(host["source_tree_sha256"]) != 64
    ):
        errors.append("campaign provenance must identify the exact qayd source tree")
    artifact = provenance.get("qayd_artifact")
    if (
        not isinstance(artifact, dict)
        or not isinstance(artifact.get("path"), str)
        or not artifact["path"]
        or not isinstance(artifact.get("sha256"), str)
        or len(artifact["sha256"]) != 64
    ):
        errors.append("campaign provenance must identify the exact qayd extension artifact")
    return errors


def canonical_target_record(
    record: dict[str, Any], instance: str, solver: str, provenance: dict[str, Any],
) -> bool:
    spec = TARGETS[instance]
    versions = provenance.get("solvers") if isinstance(provenance.get("solvers"), dict) else {}
    return bool(
        record.get("solver") == solver
        and record.get("problem") == "cvrp"
        and record.get("instance_path") == spec["path"]
        and record.get("instance_sha256") == spec["sha256"]
        and finite(record.get("checkpoint_seconds")) == 10.0
        and record.get("requested_threads") == 1
        and record.get("threads") == 1
        and finite(record.get("best_known")) == spec["best_known"]
        and isinstance(record.get("run_id"), str)
        and bool(record["run_id"])
        and isinstance(record.get("solver_version"), str)
        and record.get("solver_version") == versions.get(solver)
    )


def verified(record: dict[str, Any]) -> bool:
    objectives = record.get("objectives")
    return bool(
        record.get("status") in FEASIBLE_STATUSES
        and record.get("verified") is True
        and isinstance(objectives, list)
        and objectives
        and finite(objectives[0]) is not None
    )


def bks_gap(record: dict[str, Any]) -> float | None:
    if not verified(record):
        return None
    objective = finite(record["objectives"][0])
    best_known = finite(record.get("best_known"))
    if objective is None or best_known is None or best_known == 0:
        return None
    return 100.0 * (objective - best_known) / abs(best_known)


def seed_value(record: dict[str, Any]) -> int | None:
    seed = record.get("seed")
    return seed if isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0 else None


def target_records_by_seed(
    records: Sequence[dict[str, Any]], solver: str, provenance: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[int, dict[str, Any]]]]:
    grouped = {instance: [] for instance in TARGET_GAPS}
    selected = {instance: {} for instance in TARGET_GAPS}
    for record in records:
        instance = next(
            (name for name, spec in TARGETS.items() if record.get("instance_path") == spec["path"]),
            None,
        )
        if instance is None or record.get("solver") != solver or finite(record.get("checkpoint_seconds")) != 10.0:
            continue
        grouped[instance].append(record)
        seed = seed_value(record)
        if seed is not None and canonical_target_record(record, instance, solver, provenance):
            # JSONL order is chronological, so a restarted duplicate cannot
            # hide a later failed run for the same seed.
            selected[instance][seed] = record
    return grouped, selected


def target_rows(records: Sequence[dict[str, Any]], solver: str, provenance: dict[str, Any]) -> list[dict[str, Any]]:
    grouped, selected = target_records_by_seed(records, solver, provenance)

    rows = []
    for instance, threshold in TARGET_GAPS.items():
        runs = grouped[instance]
        by_seed = selected[instance]
        valid = {seed: record for seed, record in by_seed.items() if verified(record)}
        with_gap = {
            seed: (record, gap)
            for seed, record in valid.items()
            if (gap := bks_gap(record)) is not None
        }
        complete = {
            seed: (record, gap, rss)
            for seed, (record, gap) in with_gap.items()
            if (rss := finite(record.get("peak_memory_mb"))) is not None and rss >= 0
        }
        gaps = [gap for _record, gap, _rss in complete.values()]
        rss = [rss for _record, _gap, rss in complete.values()]
        run_ids = [record.get("run_id") for record in runs]
        rows.append({
            "solver": solver,
            "instance": instance,
            "runs": len(runs),
            "distinct_seeds": len(by_seed),
            "verified_incumbents": len(valid),
            "verified_seed_ids": sorted(valid),
            "bks_seed_count": len(with_gap),
            "complete_seed_count": len(complete),
            "complete_seed_ids": sorted(complete),
            "median_bks_gap_percent": median(gaps),
            "gap_mad_percent": median_absolute_deviation(gaps),
            "max_bks_gap_percent": max(gaps) if gaps else None,
            "gap_range_percent": max(gaps) - min(gaps) if gaps else None,
            "gap_threshold_percent": threshold,
            "max_peak_memory_mb": max(rss) if rss else None,
            "unique_run_ids": len(set(run_ids)),
        })
    return rows


def behavior_rows(document: dict[str, Any], solver: str, reference: str) -> list[dict[str, Any]]:
    return [
        row for row in document.get("solver_lag_rows", [])
        if row.get("solver") == solver and row.get("reference") == reference
        and row.get("size") == 200 and finite(row.get("checkpoint_seconds")) in {1.0, 5.0}
    ]


def behavior_evidence(
    document: dict[str, Any], solver: str, reference: str,
    expected_provenance: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    errors = []
    campaign = document.get("campaign")
    behavior_provenance = document.get("campaign_provenance")
    generator = document.get("generator")
    probes = document.get("probes")
    pairs = document.get("solver_lag_pairs")
    if document.get("schema_version") != 2 or document.get("method") != "clean-room-black-box":
        errors.append("behavior report schema or method is invalid")
    if not isinstance(campaign, dict):
        campaign = {}
    if (
        set(campaign.get("solvers", [])) != {solver, reference}
        or campaign.get("budgets") != [1, 5]
        or campaign.get("seeds") != list(REQUIRED_SEED_IDS)
        or campaign.get("threads") != 1
    ):
        errors.append("behavior campaign axes or solver provenance are invalid")
    if generator != REQUIRED_BEHAVIOR_GENERATOR:
        errors.append("behavior generator parameters are invalid")
    expected_provenance = expected_provenance or {}
    expected_versions = expected_provenance.get("solvers")
    expected_host = expected_provenance.get("host")
    expected_artifact = expected_provenance.get("qayd_artifact")
    behavior_versions = behavior_provenance.get("solvers") if isinstance(behavior_provenance, dict) else None
    behavior_host = behavior_provenance.get("host") if isinstance(behavior_provenance, dict) else None
    behavior_artifact = behavior_provenance.get("qayd_artifact") if isinstance(behavior_provenance, dict) else None
    if (
        not isinstance(behavior_provenance, dict)
        or behavior_provenance.get("schema_version") != 1
        or behavior_provenance.get("budgets") != [1, 5]
        or behavior_provenance.get("seeds") != list(REQUIRED_SEED_IDS)
        or behavior_provenance.get("threads") != 1
        or behavior_provenance.get("qayd_engine") != "ls"
        or behavior_provenance.get("profile_qayd") is not True
        or behavior_provenance.get("qayd_prepared") is not True
        or not isinstance(behavior_versions, dict)
        or behavior_versions.get(solver) != (expected_versions or {}).get(solver)
        or not isinstance(behavior_versions.get(reference), str)
        or not behavior_versions[reference]
        or not isinstance(behavior_host, dict)
        or behavior_host.get("commit") != (expected_host or {}).get("commit")
        or behavior_host.get("source_tree_sha256") != (expected_host or {}).get("source_tree_sha256")
        or behavior_artifact != expected_artifact
    ):
        errors.append("behavior campaign provenance does not match the positioning qayd build")
    if not isinstance(probes, list):
        probes = []
    probe_by_path = {
        probe.get("instance_path"): probe
        for probe in probes
        if isinstance(probe, dict) and isinstance(probe.get("instance_path"), str)
    }
    if len(probes) != 8 or len(probe_by_path) != 8 or any(probe.get("size") != 200 for probe in probes):
        errors.append("behavior report must contain exactly eight distinct n=200 probes")
    factor_roles: dict[str, set[str]] = defaultdict(set)
    pair_ids: dict[str, set[str]] = defaultdict(set)
    for probe in probes:
        if not isinstance(probe, dict):
            continue
        factor = probe.get("factor")
        role = probe.get("role")
        pair_id = probe.get("pair_id")
        if isinstance(factor, str) and isinstance(role, str):
            factor_roles[factor].add(role)
        if isinstance(factor, str) and isinstance(pair_id, str):
            pair_ids[factor].add(pair_id)
    if (
        set(factor_roles) != REQUIRED_BEHAVIOR_FACTORS
        or any(roles != {"control", "treatment"} for roles in factor_roles.values())
        or any(len(ids) != 1 for ids in pair_ids.values())
        or len({next(iter(ids)) for ids in pair_ids.values() if ids}) != len(REQUIRED_BEHAVIOR_FACTORS)
    ):
        errors.append("behavior probes do not contain the exact controlled factor pairs")
    for path, probe in probe_by_path.items():
        expected_hash = probe.get("instance_sha256")
        local = ROOT / path
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or not local.is_file()
            or sha256_file(local) != expected_hash
        ):
            errors.append(f"behavior probe hash is invalid: {path}")
    if not isinstance(pairs, list):
        pairs = []
    seen = set()
    lags: dict[float, list[float]] = defaultdict(list)
    solver_versions = set()
    reference_versions = set()
    solver_run_ids = set()
    reference_run_ids = set()
    for pair in pairs:
        if not isinstance(pair, dict):
            errors.append("behavior pair matrix contains a non-object entry")
            continue
        budget = finite(pair.get("checkpoint_seconds"))
        seed = seed_value(pair)
        path = pair.get("instance_path")
        key = (path, budget, seed)
        if key in seen:
            errors.append("behavior pair matrix contains duplicate coordinates")
        seen.add(key)
        probe = probe_by_path.get(path)
        expected_hash = probe.get("instance_sha256") if isinstance(probe, dict) else None
        left = finite(pair.get("solver_normalized_objective"))
        right = finite(pair.get("reference_normalized_objective"))
        lag = finite(pair.get("objective_lag_percent"))
        expected_lag = None if left is None or right is None else 100.0 * (left - right) / max(1.0, abs(right))
        valid = bool(
            pair.get("solver") == solver
            and pair.get("reference") == reference
            and pair.get("size") == 200
            and budget in {1.0, 5.0}
            and seed in REQUIRED_SEED_IDS
            and pair.get("campaign_threads") == 1
            and expected_hash is not None
            and pair.get("instance_sha256") == expected_hash
            and pair.get("solver_instance_path") == path
            and pair.get("reference_instance_path") == path
            and pair.get("solver_instance_sha256") == expected_hash
            and pair.get("reference_instance_sha256") == expected_hash
            and pair.get("solver_requested_threads") == 1
            and pair.get("solver_threads") == 1
            and pair.get("reference_requested_threads") == 1
            and pair.get("reference_threads") == 1
            and pair.get("solver_status") in FEASIBLE_STATUSES
            and pair.get("reference_status") in FEASIBLE_STATUSES
            and pair.get("solver_verified") is True
            and pair.get("reference_verified") is True
            and isinstance(pair.get("solver_run_id"), str)
            and bool(pair["solver_run_id"])
            and isinstance(pair.get("reference_run_id"), str)
            and bool(pair["reference_run_id"])
            and isinstance(pair.get("solver_version"), str)
            and bool(pair["solver_version"])
            and isinstance(pair.get("reference_version"), str)
            and bool(pair["reference_version"])
            and pair.get("missing_reason") is None
            and expected_lag is not None
            and lag is not None
            and math.isclose(lag, expected_lag, rel_tol=1e-12, abs_tol=1e-12)
        )
        if not valid:
            errors.append(f"behavior pair evidence is incomplete or inconsistent: {path}, seed={seed}, budget={budget}")
            continue
        solver_versions.add(pair["solver_version"])
        reference_versions.add(pair["reference_version"])
        solver_run_ids.add(pair["solver_run_id"])
        reference_run_ids.add(pair["reference_run_id"])
        lags[budget].append(lag)
    expected_coordinates = {
        (path, float(budget), seed)
        for path in probe_by_path
        for budget in (1, 5)
        for seed in REQUIRED_SEED_IDS
    }
    if seen != expected_coordinates or len(pairs) != REQUIRED_BEHAVIOR_PAIRS * 2:
        errors.append("behavior pair matrix is not the exact expected 8 x 2 x 5 matrix")
    if len(solver_versions) != 1 or len(reference_versions) != 1:
        errors.append("behavior solver versions are missing or inconsistent")
    expected_solver_version = (expected_versions or {}).get(solver)
    if expected_solver_version is not None and solver_versions != {expected_solver_version}:
        errors.append("behavior and positioning campaigns use different qayd source trees")
    if len(solver_run_ids) != len(pairs) or len(reference_run_ids) != len(pairs):
        errors.append("behavior run IDs are missing or reused")
    rows = [
        {
            "solver": solver,
            "reference": reference,
            "size": 200,
            "checkpoint_seconds": int(budget),
            "expected_pairs": REQUIRED_BEHAVIOR_PAIRS,
            "verified_pairs": len(lags[budget]),
            "missing_pairs": REQUIRED_BEHAVIOR_PAIRS - len(lags[budget]),
            "median_objective_lag_percent": median(lags[budget]),
        }
        for budget in (1.0, 5.0)
    ]
    declared = behavior_rows(document, solver, reference)
    if declared != rows:
        errors.append("behavior lag aggregates do not match their underlying pair matrix")
    return rows, errors


def nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def valid_profile_entry(entry: object) -> bool:
    if not isinstance(entry, dict) or not isinstance(entry.get("name"), str) or not entry["name"]:
        return False
    counters = (
        "uses", "generated", "evaluated", "cpu_nanos", "improvements",
        "global_bests", "positive_rewards",
    )
    weight = finite(entry.get("weight"))
    if not all(nonnegative_int(entry.get(field)) for field in counters) or weight is None or weight <= 0:
        return False
    uses = entry["uses"]
    if entry["improvements"] > uses or entry["global_bests"] > entry["improvements"] or entry["positive_rewards"] > uses:
        return False
    if uses == 0 and any(entry[field] != 0 for field in counters[1:]):
        return False
    return True


def valid_checkpoint(entry: object) -> bool:
    if not isinstance(entry, dict):
        return False
    target = entry.get("target_nanos")
    observed = entry.get("observed_nanos")
    candidates = entry.get("candidates")
    feasible = entry.get("feasible")
    objectives = entry.get("objectives")
    fleet = entry.get("fleet")
    if (
        not nonnegative_int(target)
        or not nonnegative_int(observed)
        or observed < target
        or not nonnegative_int(candidates)
        or not isinstance(feasible, bool)
        or not isinstance(objectives, list)
        or any(not isinstance(value, int) or isinstance(value, bool) for value in objectives)
    ):
        return False
    if feasible:
        return bool(objectives) and nonnegative_int(fleet) and fleet > 0
    return not objectives and fleet is None


def valid_checkpoint_timeline(checkpoints: object, record: dict[str, Any]) -> bool:
    if not isinstance(checkpoints, list) or not checkpoints or not all(valid_checkpoint(entry) for entry in checkpoints):
        return False
    targets = [entry["target_nanos"] for entry in checkpoints]
    observed = [entry["observed_nanos"] for entry in checkpoints]
    candidates = [entry["candidates"] for entry in checkpoints]
    if (
        any(left >= right for left, right in zip(targets, targets[1:]))
        or any(left > right for left, right in zip(observed, observed[1:]))
        or any(left > right for left, right in zip(candidates, candidates[1:]))
        or not REQUIRED_CHECKPOINTS_NS.issubset(targets)
    ):
        return False
    elapsed = finite(record.get("elapsed_seconds"))
    if elapsed is None or observed[-1] > int((elapsed + 0.1) * 1_000_000_000):
        return False
    feasible_seen = False
    incumbent: list[int] | None = None
    for entry in checkpoints:
        if feasible_seen and not entry["feasible"]:
            return False
        if not entry["feasible"]:
            continue
        feasible_seen = True
        objective = entry["objectives"]
        if incumbent is not None and (len(objective) != len(incumbent) or objective > incumbent):
            return False
        incumbent = objective
    final = record.get("objectives")
    if incumbent is not None and (not isinstance(final, list) or len(final) != len(incumbent) or final > incumbent):
        return False
    return True


def valid_routing_counters(counters: object, profile: list[dict[str, Any]]) -> bool:
    if not isinstance(counters, dict) or set(counters) != REQUIRED_ROUTING_COUNTERS:
        return False
    if not all(nonnegative_int(value) for value in counters.values()):
        return False
    macro_names = {
        "route-elimination": "route_elimination_attempts",
        "ejection-chain": "ejection_chain_attempts",
        "chain-relocate": "chain_relocate_attempts",
        "guided-segment-exchange": "guided_segment_exchange_attempts",
    }
    by_name = {entry["name"]: entry for entry in profile}
    macro_slices = sum(counters[field] for field in macro_names.values())
    if counters["slices"] != (
        counters["descent_slices"]
        + counters["alns_slices"]
        + counters["relink_slices"]
        + counters["global_scan_slices"]
        + macro_slices
    ):
        return False
    if counters["path_relink_attempts"] > counters["relink_slices"]:
        return False
    if counters["macro_candidates_built"] + counters["macro_budget_exhaustions"] > macro_slices:
        return False
    if counters["path_relink_budget_exhaustions"] > counters["path_relink_attempts"]:
        return False
    if counters["path_relink_steps"] > by_name["path-relink"]["evaluated"]:
        return False
    if by_name["elite-archive"]["uses"] < counters["elite_insertions"] + counters["elite_rejections"]:
        return False
    if by_name["elite-selection"]["uses"] != counters["relink_slices"]:
        return False
    if sum(by_name[name]["uses"] for name in REQUIRED_PROFILE_NAMES if name in {
        "relocate", "swap", "or-opt", "two-opt-star", "cross-exchange", "reverse",
    }) != counters["descent_slices"] + counters["global_scan_slices"]:
        return False
    if sum(by_name[name]["uses"] for name in REQUIRED_PROFILE_NAMES if name.startswith("alns_destroy/")) \
            != counters["alns_slices"]:
        return False
    if sum(by_name[name]["uses"] for name in REQUIRED_PROFILE_NAMES if name.startswith("alns_repair/")) \
            != counters["alns_slices"]:
        return False
    return all(by_name[name]["uses"] == counters[field] for name, field in macro_names.items()) \
        and by_name["path-relink"]["uses"] == counters["path_relink_attempts"]


def profile_audit(records: Sequence[dict[str, Any]], solver: str, provenance: dict[str, Any]) -> dict[str, Any]:
    _grouped, selected = target_records_by_seed(records, solver, provenance)
    relevant = [
        (instance, seed, record)
        for instance, by_seed in selected.items()
        for seed, record in by_seed.items()
        if verified(record)
        and bks_gap(record) is not None
        and (rss := finite(record.get("peak_memory_mb"))) is not None
        and rss >= 0
    ]
    missing_profiles = 0
    missing_checkpoints = 0
    missing_counters = 0
    zero_cpu_profiles = 0
    inactive_runs = 0
    unproductive_dominant_runs = []
    operators: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "uses": 0,
            "generated": 0,
            "evaluated": 0,
            "cpu_nanos": 0,
            "improvements": 0,
            "positive_rewards": 0,
        }
    )
    aggregate_counters = {name: 0 for name in REQUIRED_ROUTING_COUNTERS}
    for instance, seed, record in relevant:
        profile = record.get("neighborhood_profile")
        entries_valid = isinstance(profile, list) and bool(profile) and all(valid_profile_entry(entry) for entry in profile)
        names = [entry.get("name") for entry in profile] if isinstance(profile, list) else []
        profile_valid = entries_valid and len(names) == len(set(names)) and set(names) == REQUIRED_PROFILE_NAMES
        if not profile_valid:
            missing_profiles += 1
        else:
            run_cpu = sum(entry["cpu_nanos"] for entry in profile)
            if run_cpu <= 0 or sum(entry["uses"] for entry in profile) <= 0:
                zero_cpu_profiles += 1
            for entry in profile:
                aggregate = operators[entry["name"]]
                for field in aggregate:
                    aggregate[field] += entry[field]
                if (
                    run_cpu > 0
                    and 100.0 * entry["cpu_nanos"] / run_cpu > 70.0
                    and entry["improvements"] == 0
                    and entry["positive_rewards"] == 0
                ):
                    unproductive_dominant_runs.append(
                        f"{instance}:seed={seed}:{entry['name']}"
                    )
        counters = record.get("routing_counters")
        if not profile_valid or not valid_routing_counters(counters, profile):
            missing_counters += 1
        else:
            for name in aggregate_counters:
                aggregate_counters[name] += counters[name]
            if (
                counters["slices"] == 0
                or counters["descent_slices"] == 0
                or counters["alns_slices"] == 0
                or counters["elite_insertions"] == 0
            ):
                inactive_runs += 1
        if not valid_checkpoint_timeline(record.get("anytime_checkpoints"), record):
            missing_checkpoints += 1

    total_cpu = sum(operator["cpu_nanos"] for operator in operators.values())
    rows = []
    for name, operator in sorted(operators.items()):
        share = 100.0 * operator["cpu_nanos"] / total_cpu if total_cpu else 0.0
        rows.append({
            "name": name,
            **operator,
            "cpu_share_percent": share,
            "productive": operator["improvements"] > 0 or operator["positive_rewards"] > 0,
        })
    unproductive_dominant = [
        row["name"] for row in rows
        if row["cpu_share_percent"] > 70.0 and not row["productive"]
    ]
    operator_uses = {row["name"]: row["uses"] for row in rows}
    inactive_operators = sorted(name for name in REQUIRED_PROFILE_NAMES if operator_uses.get(name, 0) == 0)
    unevaluated_operators = sorted(
        name
        for name in REQUIRED_EVALUATED_PROFILE_NAMES
        if operators[name]["uses"] > 0 and operators[name]["evaluated"] == 0
    )
    inactive_mechanisms = sorted(
        name
        for name in (
            "relink_slices", "global_scan_slices", "route_elimination_attempts",
            "ejection_chain_attempts", "chain_relocate_attempts",
            "guided_segment_exchange_attempts", "path_relink_attempts", "path_relink_steps",
        )
        if aggregate_counters[name] == 0
    )
    return {
        "runs": len(relevant),
        "missing_profiles": missing_profiles,
        "zero_cpu_profiles": zero_cpu_profiles,
        "missing_required_checkpoints": missing_checkpoints,
        "missing_or_inconsistent_counters": missing_counters,
        "inactive_runs": inactive_runs,
        "total_cpu_nanos": total_cpu,
        "operators": rows,
        "routing_counters": aggregate_counters,
        "inactive_required_operators": inactive_operators,
        "unevaluated_required_operators": unevaluated_operators,
        "inactive_required_mechanisms": inactive_mechanisms,
        "unproductive_dominant_operators": unproductive_dominant,
        "unproductive_dominant_runs": sorted(set(unproductive_dominant_runs)),
    }


def build_summary(
    campaign: Path, behavior_probe: Path, *, solver: str = "qayd-api", reference: str = "hexaly",
) -> dict[str, Any]:
    records = load_jsonl(campaign)
    behavior = json.loads(behavior_probe.read_text(encoding="utf-8"))
    provenance = load_campaign_provenance(campaign)
    qayd = target_rows(records, solver, provenance)
    hgs = target_rows(records, "hgs", provenance)
    lag, behavior_errors = behavior_evidence(behavior, solver, reference, provenance)
    profile = profile_audit(records, solver, provenance)
    errors = [*campaign_provenance_errors(provenance, solver), *behavior_errors]

    for instance, spec in TARGETS.items():
        path = ROOT / str(spec["path"])
        if not path.is_file() or sha256_file(path) != spec["sha256"]:
            errors.append(f"{instance}: canonical instance file or SHA-256 is invalid")

    for row in qayd:
        if (
            row["runs"] != REQUIRED_SEEDS
            or row["unique_run_ids"] != REQUIRED_SEEDS
            or row["complete_seed_ids"] != list(REQUIRED_SEED_IDS)
        ):
            errors.append(
                f"{row['instance']}: requires exactly verified seeds 0 through 4 with canonical provenance, BKS and RSS"
            )
        gap = finite(row["median_bks_gap_percent"])
        if gap is None or gap >= row["gap_threshold_percent"]:
            errors.append(f"{row['instance']}: median BKS gap must be below {row['gap_threshold_percent']:.0f}%")
        max_gap = finite(row["max_bks_gap_percent"])
        spread = finite(row["gap_range_percent"])
        mad = finite(row["gap_mad_percent"])
        if (
            max_gap is None
            or max_gap >= row["gap_threshold_percent"] * 1.5
            or spread is None
            or spread >= row["gap_threshold_percent"]
            or mad is None
            or mad >= row["gap_threshold_percent"] / 2.0
        ):
            errors.append(f"{row['instance']}: Qayd seed variance or tail gap is too high")
        rss = finite(row["max_peak_memory_mb"])
        if rss is None or rss >= MAX_RSS_MB:
            errors.append(f"{row['instance']}: peak RSS must be below {MAX_RSS_MB:.0f} MB")

    for row in hgs:
        spec = TARGETS[row["instance"]]
        if (
            row["runs"] != REQUIRED_SEEDS
            or row["unique_run_ids"] != REQUIRED_SEEDS
            or row["complete_seed_ids"] != list(REQUIRED_SEED_IDS)
        ):
            errors.append(f"{row['instance']}: HGS reference is incomplete")
        if (
            finite(row["median_bks_gap_percent"]) is None
            or row["median_bks_gap_percent"] >= spec["hgs_median_limit"]
            or finite(row["max_bks_gap_percent"]) is None
            or row["max_bks_gap_percent"] >= spec["hgs_max_limit"]
            or finite(row["gap_mad_percent"]) is None
            or row["gap_mad_percent"] >= spec["hgs_mad_limit"]
        ):
            errors.append(f"{row['instance']}: HGS gap or seed variance regressed beyond the archived reference")

    lag_by_budget = {finite(row.get("checkpoint_seconds")): row for row in lag}
    for budget, threshold in ((1.0, 20.0), (5.0, 5.0)):
        if sum(finite(candidate.get("checkpoint_seconds")) == budget for candidate in lag) != 1:
            errors.append(f"n=200: expected exactly one paired {budget:g}s lag row against {reference}")
            continue
        row = lag_by_budget.get(budget)
        if row is None:
            errors.append(f"n=200: missing paired {budget:g}s lag against {reference}")
            continue
        expected_pairs = row.get("expected_pairs")
        if (
            not nonnegative_int(expected_pairs)
            or expected_pairs < REQUIRED_BEHAVIOR_PAIRS
            or row.get("missing_pairs") != 0
            or row.get("verified_pairs") != expected_pairs
        ):
            errors.append(f"n=200 at {budget:g}s: paired incumbent matrix is incomplete")
        value = finite(row.get("median_objective_lag_percent"))
        if value is None or value >= threshold:
            errors.append(f"n=200 at {budget:g}s: median lag must be below {threshold:.0f}%")

    if profile["runs"] != len(TARGET_GAPS) * REQUIRED_SEEDS:
        errors.append("routing profile does not cover every target run")
    if profile["missing_profiles"]:
        errors.append(f"{profile['missing_profiles']} target run(s) have no neighborhood profile")
    if profile["zero_cpu_profiles"]:
        errors.append(f"{profile['zero_cpu_profiles']} target run(s) have no measured neighborhood CPU")
    if profile["total_cpu_nanos"] <= 0:
        errors.append("routing profile has no measured neighborhood CPU")
    if profile["missing_required_checkpoints"]:
        errors.append(f"{profile['missing_required_checkpoints']} target run(s) miss required anytime checkpoints")
    if profile["missing_or_inconsistent_counters"]:
        errors.append(f"{profile['missing_or_inconsistent_counters']} target run(s) have invalid routing counters")
    if profile["inactive_runs"]:
        errors.append(f"{profile['inactive_runs']} target run(s) did not exercise sliced descent, ALNS and the elite archive")
    if profile["inactive_required_operators"]:
        errors.append("required routing operators were not exercised: " + ", ".join(profile["inactive_required_operators"]))
    if profile["unevaluated_required_operators"]:
        errors.append(
            "required routing operators attempted no candidate evaluation: "
            + ", ".join(profile["unevaluated_required_operators"])
        )
    if profile["inactive_required_mechanisms"]:
        errors.append("required routing mechanisms were not exercised: " + ", ".join(profile["inactive_required_mechanisms"]))
    if profile["unproductive_dominant_operators"]:
        errors.append(
            "unproductive operator above 70% CPU: "
            + ", ".join(profile["unproductive_dominant_operators"])
        )
    if profile["unproductive_dominant_runs"]:
        errors.append(
            "unproductive operator above 70% CPU in a target run: "
            + ", ".join(profile["unproductive_dominant_runs"])
        )

    return {
        "schema_version": 1,
        "campaign": str(campaign.resolve()),
        "behavior_probe": str(behavior_probe.resolve()),
        "campaign_provenance": provenance,
        "solver": solver,
        "reference": reference,
        "ready": not errors,
        "errors": sorted(set(errors)),
        "qayd_targets": qayd,
        "hgs_reference": hgs,
        "behavior_lag": lag,
        "profile": profile,
    }


def shown(value: object, suffix: str = "") -> str:
    number = finite(value)
    return "n/a" if number is None else f"{number:.2f}{suffix}"


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Routing search qualification",
        "",
        f"Gate: **{'READY' if summary['ready'] else 'NOT READY'}**.",
        "",
        "## Large CVRP targets",
        "",
        "| Solver | Instance | Incumbents | Gap BKS median | MAD | Gap max | RSS max |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in [*summary["qayd_targets"], *summary["hgs_reference"]]:
        lines.append(
            f"| {row['solver']} | {row['instance']} | {row['verified_incumbents']}/{row['runs']} | "
            f"{shown(row['median_bks_gap_percent'], '%')} | {shown(row['gap_mad_percent'], '%')} | "
            f"{shown(row['max_bks_gap_percent'], '%')} | "
            f"{shown(row['max_peak_memory_mb'], ' MB')} |"
        )
    lines.extend((
        "",
        "## Paired n=200 behavior",
        "",
        "| Budget | Verified pairs | Missing | Median lag |",
        "|---:|---:|---:|---:|",
    ))
    for row in summary["behavior_lag"]:
        lines.append(
            f"| {row['checkpoint_seconds']}s | {row['verified_pairs']}/{row['expected_pairs']} | "
            f"{row['missing_pairs']} | {shown(row['median_objective_lag_percent'], '%')} |"
        )
    lines.extend((
        "",
        "## Neighborhood CPU profile",
        "",
        "| Operator | Uses | Evaluated | CPU share | Improvements | Positive rewards |",
        "|---|---:|---:|---:|---:|---:|",
    ))
    for row in summary["profile"]["operators"]:
        lines.append(
            f"| {row['name']} | {row['uses']} | {row['evaluated']} | {row['cpu_share_percent']:.2f}% | "
            f"{row['improvements']} | {row['positive_rewards']} |"
        )
    lines.extend(("", "## Gate diagnostics", ""))
    lines.extend(f"- {error}" for error in summary["errors"])
    if not summary["errors"]:
        lines.append("- All acceptance checks passed.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--behavior-probe", type=Path, required=True)
    parser.add_argument("--solver", default="qayd-api")
    parser.add_argument("--reference", default="hexaly")
    parser.add_argument("--json", type=Path, required=True, dest="json_output")
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()

    summary = build_summary(args.campaign, args.behavior_probe, solver=args.solver, reference=args.reference)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.write_text(markdown(summary), encoding="utf-8")
    print(f"routing search gate: {'READY' if summary['ready'] else 'NOT READY'}")
    if args.require_ready and not summary["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
