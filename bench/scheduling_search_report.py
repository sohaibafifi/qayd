#!/usr/bin/env python3
"""Build the Phase 13 scheduling-search qualification report.

The gate audits four source-disjoint Qayd campaigns: automatic and forced-LS
JSSP, then automatic and forced-LS RCPSP.  An optional previous forced-LS
campaign can be supplied for paired regression checks.  Complete acceptance
also requires matched CP-SAT or Hexaly JSSP references and incumbent-complete
j60, j90, j120 and MMLIB campaigns.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Sequence


DEFAULT_JSSP_INSTANCES = ("ft06", "la38", "orb10", "swv10", "ta40", "ta80")
DEFAULT_RCPSP_INSTANCES = (
    "j3021_1",
    "j304_9",
    "j3032_8",
    "j3048_9",
    "j3015_3",
    "j307_10",
)
DEFAULT_SEEDS = tuple(range(5))
FEASIBLE_STATUSES = {"SAT", "SATISFIABLE", "FEASIBLE", "OPTIMAL", "OPTIMUM"}
FT06_TARGET = 55.0
ORB10_MEDIAN_LIMIT = 1_038.0
MAX_RSS_MB = 256.0
JSSP_LS_CONSTRUCTOR = "giffler-thompson-critical-path"
RCPSP_LS_CONSTRUCTOR = "resource-priority-sgs"
COMMON_LS_COUNTERS = (
    "candidates_evaluated",
    "schedule_moves_considered",
    "schedule_moves_accepted",
    "schedule_moves_rejected",
    "schedule_work_steps",
    "schedule_delta_evaluations",
    "schedule_full_evaluations",
    "schedule_full_fallbacks",
    "schedule_oracle_validations",
    "schedule_oracle_mismatches",
    "schedule_workspace_growths",
    "schedule_workspace_rollbacks",
)
RCPSP_LS_POSITIVE_COUNTERS = (
    "resource_candidate_scheduling_attempts",
    "resource_profile_checks",
    "resource_event_visits",
    "resource_peak_profile_events",
    "schedule_alns_generation_attempts",
    "schedule_alns_moves_generated",
    "schedule_workspace_rollbacks",
)
JSSP_REFERENCE_INSTANCES = ("la38", "swv10", "ta40", "ta80")
JSSP_REFERENCE_REGRESSION_LIMIT_PERCENT = 15.0
RCPSP_SCALE_GROUPS = ("j60", "j90", "j120", "mmlib")
REFERENCE_SOLVERS = {"ortools-cp-sat", "hexaly"}


def finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rcpsp_best_known(
    path: Path, expected_instances: Sequence[str],
) -> tuple[dict[str, Any], dict[str, float]]:
    resolved = path.resolve()
    errors = []
    values: dict[str, float] = {}
    metadata: dict[str, Any] = {
        "path": str(resolved),
        "sha256": None,
        "schema_version": None,
        "problem": None,
        "entries": {},
        "missing": [],
        "unexpected": [],
        "errors": errors,
        "pass": False,
    }
    if not path.is_file():
        errors.append(f"RCPSP best-known manifest does not exist: {resolved}")
        return metadata, values

    metadata["sha256"] = sha256_file(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"invalid RCPSP best-known manifest {resolved}: {error}")
        return metadata, values
    if not isinstance(payload, dict):
        errors.append("RCPSP best-known manifest must be a JSON object")
        return metadata, values

    metadata["schema_version"] = payload.get("schema_version")
    metadata["problem"] = payload.get("problem")
    if payload.get("schema_version") != 1:
        errors.append("RCPSP best-known manifest schema_version must be 1")
    if payload.get("problem") != "rcpsp":
        errors.append("RCPSP best-known manifest problem must be 'rcpsp'")
    entries = payload.get("best_known")
    if not isinstance(entries, dict):
        errors.append("RCPSP best-known manifest best_known must be a JSON object")
        return metadata, values

    normalized_keys: dict[str, str] = {}
    for key, value in entries.items():
        if not isinstance(key, str) or not key or key != key.strip():
            errors.append(f"RCPSP best-known manifest contains an invalid instance key: {key!r}")
            continue
        normalized = key.casefold()
        if normalized in normalized_keys:
            errors.append(
                "RCPSP best-known manifest contains case-insensitive duplicate keys: "
                f"{normalized_keys[normalized]!r} and {key!r}"
            )
            continue
        normalized_keys[normalized] = key
        number = finite(value)
        if number is None or number < 0 or not number.is_integer():
            errors.append(f"RCPSP best-known value for {key!r} must be a finite non-negative integer")
            continue
        values[normalized] = number

    expected = {instance.casefold() for instance in expected_instances}
    actual = set(normalized_keys)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    metadata["missing"] = missing
    metadata["unexpected"] = unexpected
    if missing:
        errors.append(f"RCPSP best-known manifest is missing {len(missing)} required instance keys")
    if unexpected:
        errors.append(f"RCPSP best-known manifest has {len(unexpected)} unexpected instance keys")
    metadata["entries"] = {key: values[key] for key in sorted(values)}
    metadata["pass"] = not errors and set(values) == expected
    return metadata, values


def canonical_instance(record: dict[str, Any]) -> str:
    value = record.get("instance")
    if isinstance(value, str) and value.strip():
        return value.strip().casefold()
    value = record.get("instance_path")
    if not isinstance(value, str) or not value:
        return ""
    return Path(value).stem.casefold()


def seed_value(record: dict[str, Any]) -> int | None:
    value = record.get("seed")
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def scalar_objective(record: dict[str, Any]) -> float | None:
    values = record.get("objectives")
    if not isinstance(values, list) or len(values) != 1:
        return None
    return finite(values[0])


def provenance_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".provenance.json")


def load_source(path: Path, label: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    resolved = path.resolve()
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    if not path.is_file():
        return ({
            "label": label,
            "path": str(resolved),
            "sha256": None,
            "records": 0,
            "provenance_path": str(provenance_path(resolved)),
            "provenance_sha256": None,
            "provenance": None,
            "errors": [f"{label}: campaign file does not exist: {resolved}"],
        }, records)

    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                errors.append(f"{label}: {resolved}:{line_number}: invalid JSON: {error}")
                continue
            if not isinstance(value, dict):
                errors.append(f"{label}: {resolved}:{line_number}: record is not a JSON object")
                continue
            record = dict(value)
            record["_source_path"] = str(resolved)
            record["_source_line"] = line_number
            records.append(record)
    if not records:
        errors.append(f"{label}: campaign contains no JSON records")

    sidecar = provenance_path(path)
    provenance: dict[str, Any] | None = None
    provenance_hash = None
    if not sidecar.is_file():
        errors.append(f"{label}: missing provenance sidecar {sidecar.resolve()}")
    else:
        provenance_hash = sha256_file(sidecar)
        try:
            value = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"{label}: invalid provenance sidecar {sidecar.resolve()}: {error}")
        else:
            if isinstance(value, dict):
                provenance = value
            else:
                errors.append(f"{label}: provenance sidecar is not a JSON object")

    expected_threads = provenance.get("threads") if provenance is not None else None
    provenance_solvers = provenance.get("solvers") if provenance is not None else None
    for record in records:
        record["_provenance_threads"] = expected_threads
        record["_provenance_solvers"] = provenance_solvers

    metadata = {
        "label": label,
        "path": str(resolved),
        "sha256": sha256_file(path),
        "records": len(records),
        "provenance_path": str(sidecar.resolve()),
        "provenance_sha256": provenance_hash,
        "provenance": provenance,
        "errors": errors,
    }
    return metadata, records


def record_location(record: dict[str, Any]) -> str:
    return f"{record.get('_source_path', '?')}:{record.get('_source_line', '?')}"


def non_negative_integer(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def local_search_profile_issues(record: dict[str, Any], problem: str) -> list[str]:
    issues = []
    expected_constructor = {
        "jssp": JSSP_LS_CONSTRUCTOR,
        "rcpsp": RCPSP_LS_CONSTRUCTOR,
    }[problem]
    if record.get("constructor") != expected_constructor:
        issues.append(
            f"constructor is {record.get('constructor')!r}, expected {expected_constructor!r}"
        )

    counters = {field: non_negative_integer(record.get(field)) for field in COMMON_LS_COUNTERS}
    for field, value in counters.items():
        if value is None:
            issues.append(f"{field} is missing or is not a non-negative integer")
    if any(value is None for value in counters.values()):
        return issues

    candidates = counters["candidates_evaluated"]
    considered = counters["schedule_moves_considered"]
    accepted = counters["schedule_moves_accepted"]
    rejected = counters["schedule_moves_rejected"]
    work_steps = counters["schedule_work_steps"]
    delta_evaluations = counters["schedule_delta_evaluations"]
    oracle_validations = counters["schedule_oracle_validations"]
    oracle_mismatches = counters["schedule_oracle_mismatches"]
    full_evaluations = counters["schedule_full_evaluations"]
    full_fallbacks = counters["schedule_full_fallbacks"]

    if candidates != considered:
        issues.append("candidates_evaluated must equal schedule_moves_considered")
    if accepted + rejected != considered:
        issues.append(
            "schedule_moves_accepted plus schedule_moves_rejected must equal schedule_moves_considered"
        )
    if work_steps < considered:
        issues.append("schedule_work_steps must be at least schedule_moves_considered")
    if delta_evaluations <= 0:
        issues.append("schedule_delta_evaluations must be positive")
    if oracle_mismatches != 0:
        issues.append("schedule_oracle_mismatches must be zero")
    if oracle_mismatches > oracle_validations:
        issues.append("schedule_oracle_mismatches cannot exceed schedule_oracle_validations")
    if full_fallbacks > full_evaluations:
        issues.append("schedule_full_fallbacks cannot exceed schedule_full_evaluations")

    throughput = finite(record.get("candidates_per_second"))
    if throughput is None or throughput < 0:
        issues.append("candidates_per_second is missing or is not finite and non-negative")
    full_percentage = finite(record.get("full_recompute_percentage"))
    if full_percentage is None or not 0 <= full_percentage <= 100:
        issues.append("full_recompute_percentage must be finite and between 0 and 100")

    if problem == "rcpsp":
        for field in RCPSP_LS_POSITIVE_COUNTERS:
            value = non_negative_integer(record.get(field))
            if value is None or value <= 0:
                issues.append(f"{field} must be a positive integer")
    return issues


def record_issues(
    record: dict[str, Any], problem: str, expected_engine: str | None, max_rss_mb: float,
) -> list[str]:
    issues = []
    if expected_engine is not None and record.get("engine") != expected_engine:
        issues.append(f"engine is {record.get('engine')!r}, expected {expected_engine!r}")
    status = record.get("status")
    if status not in FEASIBLE_STATUSES:
        issues.append(f"status is {status!r}, not a feasible status")
    if record.get("verified") is not True:
        issues.append("verified is not true")
    if scalar_objective(record) is None:
        issues.append("missing or non-finite scalar incumbent objective")
    if record.get("return_code") not in (None, 0):
        issues.append(f"process return code is {record.get('return_code')!r}")
    for field in ("timed_out", "external_timeout", "external_timed_out"):
        if record.get(field) is True:
            issues.append(f"{field} is true")
    if record.get("invalid") is True or record.get("valid") is False:
        issues.append("record is marked invalid")
    if record.get("error") not in (None, ""):
        issues.append(f"record carries an error: {record.get('error')}")
    rss = finite(record.get("peak_memory_mb"))
    if rss is None or rss <= 0:
        issues.append("peak_memory_mb is required and must be a finite positive RSS value")
    elif rss >= max_rss_mb:
        issues.append(f"peak_memory_mb is {rss:g} MB, limit is strictly below {max_rss_mb:g} MB")
    threads = non_negative_integer(record.get("threads"))
    expected_threads = non_negative_integer(record.get("_provenance_threads"))
    if threads is None or threads <= 0:
        issues.append("threads is required and must be a positive integer")
    if expected_threads is None or expected_threads <= 0:
        issues.append("the provenance sidecar does not provide a valid expected thread count")
    elif threads is not None and threads != expected_threads:
        issues.append(f"threads is {threads}, expected {expected_threads} from provenance")
    digest = record.get("instance_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        issues.append("instance_sha256 is missing or malformed")
    if expected_engine == "ls":
        issues.extend(local_search_profile_issues(record, problem))
    return issues


def audit_matrix(
    name: str,
    records: Sequence[dict[str, Any]],
    problem: str,
    expected_engine: str | None,
    instances: Sequence[str],
    seeds: Sequence[int],
    solver: str,
    max_rss_mb: float,
) -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    expected_instances = tuple(instance.casefold() for instance in instances)
    expected_keys = {(instance, seed) for instance in expected_instances for seed in seeds}
    relevant = [
        record for record in records
        if record.get("problem") == problem and record.get("solver") == solver
    ]
    groups: dict[tuple[str, int | None], list[dict[str, Any]]] = defaultdict(list)
    for record in relevant:
        groups[(canonical_instance(record), seed_value(record))].append(record)

    selected: dict[tuple[str, int], dict[str, Any]] = {}
    duplicates = []
    unexpected = []
    invalid = []
    for (instance, seed), runs in sorted(groups.items(), key=lambda item: (item[0][0], -1 if item[0][1] is None else item[0][1])):
        key_is_expected = seed is not None and (instance, seed) in expected_keys
        if len(runs) > 1:
            duplicates.append({
                "instance": instance or None,
                "seed": seed,
                "count": len(runs),
                "locations": [record_location(record) for record in runs],
            })
        if not key_is_expected:
            unexpected.append({
                "instance": instance or None,
                "seed": seed,
                "locations": [record_location(record) for record in runs],
            })
            continue
        record = runs[0]
        selected[(instance, seed)] = record
        issues = record_issues(record, problem, expected_engine, max_rss_mb)
        if issues:
            invalid.append({
                "instance": instance,
                "seed": seed,
                "location": record_location(record),
                "issues": issues,
            })

    missing = sorted(expected_keys - set(selected))
    rows = []
    for instance in expected_instances:
        instance_records = [selected.get((instance, seed)) for seed in seeds]
        objectives = [scalar_objective(record) if record is not None else None for record in instance_records]
        rss_values = [
            rss
            for record in instance_records if record is not None
            for rss in [finite(record.get("peak_memory_mb"))]
            if rss is not None
        ]
        rows.append({
            "instance": instance,
            "runs": sum(record is not None for record in instance_records),
            "objectives": objectives,
            "median_objective": statistics.median(value for value in objectives if value is not None) if any(value is not None for value in objectives) else None,
            "max_rss_mb": max(rss_values) if rss_values else None,
        })

    errors = []
    if not relevant:
        errors.append(f"{name}: no records for solver={solver!r}, problem={problem!r}")
    if missing:
        errors.append(f"{name}: missing {len(missing)} expected instance/seed runs")
    if unexpected:
        errors.append(f"{name}: found {len(unexpected)} unexpected instance/seed keys")
    if duplicates:
        errors.append(f"{name}: found {len(duplicates)} duplicate instance/seed keys")
    if invalid:
        errors.append(f"{name}: found {len(invalid)} invalid runs")
    summary = {
        "name": name,
        "problem": problem,
        "expected_engine": expected_engine,
        "expected_runs": len(expected_keys),
        "selected_runs": len(selected),
        "missing": [{"instance": instance, "seed": seed} for instance, seed in missing],
        "unexpected": unexpected,
        "duplicates": duplicates,
        "invalid": invalid,
        "instances": rows,
        "errors": errors,
        "pass": not errors,
    }
    return summary, selected


def valid_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdefABCDEF" for character in value)


def provenance_fingerprint(provenance: dict[str, Any] | None) -> tuple[str, str, str] | None:
    if provenance is None:
        return None
    host = provenance.get("host")
    artifact = provenance.get("qayd_artifact")
    if not isinstance(host, dict) or not isinstance(artifact, dict):
        return None
    commit = host.get("commit")
    source_hash = host.get("source_tree_sha256")
    artifact_hash = artifact.get("sha256")
    if not isinstance(commit, str) or len(commit) != 40 or not valid_sha256(source_hash) or not valid_sha256(artifact_hash):
        return None
    return commit, source_hash, artifact_hash


def audit_provenance(
    sources: Sequence[dict[str, Any]],
    expected_engines: dict[str, str | None],
    seeds: Sequence[int],
    current_labels: Sequence[str],
    solver: str,
) -> dict[str, Any]:
    errors = []
    fingerprints: dict[str, tuple[str, str, str]] = {}
    configurations: dict[str, tuple[tuple[float, ...], int, int, int | None, bool]] = {}
    rows = []
    for source in sources:
        label = source["label"]
        provenance = source["provenance"]
        row_errors = list(source["errors"])
        fingerprint = provenance_fingerprint(provenance)
        if provenance is not None:
            if provenance.get("schema_version") != 1:
                row_errors.append(f"{label}: provenance schema_version must be 1")
            if provenance.get("seeds") != list(seeds):
                row_errors.append(f"{label}: provenance seeds must be exactly {list(seeds)}")
            budgets = provenance.get("budgets")
            normalized_budgets = tuple(
                number for value in budgets if (number := finite(value)) is not None and number > 0
            ) if isinstance(budgets, list) else ()
            threads = provenance.get("threads")
            memory_limit = provenance.get("memory_limit_mb")
            max_iterations = provenance.get("max_iterations")
            profile_qayd = provenance.get("profile_qayd")
            if not normalized_budgets or len(normalized_budgets) != len(budgets):
                row_errors.append(f"{label}: provenance must declare finite positive budgets")
            if isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0:
                row_errors.append(f"{label}: provenance must declare a positive thread count")
            if isinstance(memory_limit, bool) or not isinstance(memory_limit, int) or memory_limit < 0:
                row_errors.append(f"{label}: provenance must declare a non-negative memory limit")
            if "max_iterations" not in provenance:
                row_errors.append(f"{label}: provenance must declare max_iterations")
            elif max_iterations is not None and (
                isinstance(max_iterations, bool)
                or not isinstance(max_iterations, int)
                or max_iterations < 0
            ):
                row_errors.append(f"{label}: provenance max_iterations must be null or a non-negative integer")
            if profile_qayd is not True:
                row_errors.append(f"{label}: provenance profile_qayd must be true")
            valid_max_iterations = max_iterations is None or (
                isinstance(max_iterations, int)
                and not isinstance(max_iterations, bool)
                and max_iterations >= 0
            )
            if (
                normalized_budgets
                and isinstance(threads, int)
                and not isinstance(threads, bool)
                and threads > 0
                and isinstance(memory_limit, int)
                and not isinstance(memory_limit, bool)
                and memory_limit >= 0
                and "max_iterations" in provenance
                and valid_max_iterations
                and profile_qayd is True
            ):
                configurations[label] = (
                    normalized_budgets,
                    threads,
                    memory_limit,
                    max_iterations,
                    profile_qayd,
                )
            solvers = provenance.get("solvers")
            if not isinstance(solvers, dict) or not isinstance(solvers.get(solver), str) or not solvers[solver]:
                row_errors.append(f"{label}: provenance does not identify solver {solver!r}")
            expected_engine = expected_engines.get(label)
            if expected_engine is not None and provenance.get("qayd_engine") != expected_engine:
                row_errors.append(
                    f"{label}: provenance qayd_engine is {provenance.get('qayd_engine')!r}, expected {expected_engine!r}"
                )
            if fingerprint is None:
                row_errors.append(f"{label}: provenance does not identify commit, source tree and Qayd artifact hashes")
            else:
                fingerprints[label] = fingerprint
        rows.append({
            key: source[key]
            for key in ("label", "path", "sha256", "records", "provenance_path", "provenance_sha256")
        } | {
            "qayd_engine": provenance.get("qayd_engine") if provenance else None,
            "budgets": provenance.get("budgets") if provenance else None,
            "seeds": provenance.get("seeds") if provenance else None,
            "threads": provenance.get("threads") if provenance else None,
            "memory_limit_mb": provenance.get("memory_limit_mb") if provenance else None,
            "max_iterations": provenance.get("max_iterations") if provenance else None,
            "profile_qayd": provenance.get("profile_qayd") if provenance else None,
            "solver_versions": provenance.get("solvers") if provenance else None,
            "host": provenance.get("host") if provenance else None,
            "qayd_artifact": provenance.get("qayd_artifact") if provenance else None,
            "errors": sorted(set(row_errors)),
            "pass": not row_errors,
        })
        errors.extend(row_errors)

    current_fingerprints = {fingerprints[label] for label in current_labels if label in fingerprints}
    if len(current_fingerprints) > 1:
        errors.append("current campaigns were produced by different Qayd source or extension artifacts")
    if len(current_fingerprints) != 1:
        errors.append("current campaigns do not provide one complete shared Qayd build fingerprint")
    current_configurations = {configurations[label] for label in current_labels if label in configurations}
    if len(current_configurations) != 1:
        errors.append(
            "current campaigns do not share one complete budget, thread, memory, "
            "max_iterations and profile_qayd configuration"
        )
    elif configurations and any(
        label.startswith("baseline-") and configuration != next(iter(current_configurations))
        for label, configuration in configurations.items()
    ):
        errors.append(
            "baseline and current campaigns do not share the same budget, thread, memory, "
            "max_iterations and profile_qayd configuration"
        )
    return {
        "sources": rows,
        "current_build_fingerprint": list(next(iter(current_fingerprints))) if len(current_fingerprints) == 1 else None,
        "current_configuration": {
            "budgets": list(next(iter(current_configurations))[0]),
            "threads": next(iter(current_configurations))[1],
            "memory_limit_mb": next(iter(current_configurations))[2],
            "max_iterations": next(iter(current_configurations))[3],
            "profile_qayd": next(iter(current_configurations))[4],
        } if len(current_configurations) == 1 else None,
        "errors": sorted(set(errors)),
        "pass": not errors,
    }


def audit_instance_sources(
    matrices: Sequence[tuple[str, dict[tuple[str, int], dict[str, Any]]]],
) -> dict[str, Any]:
    identities: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(lambda: {"paths": set(), "sha256": set()})
    missing = []
    for matrix_name, records in matrices:
        problem = matrix_name.split("-", 1)[0]
        for (instance, seed), record in records.items():
            path = record.get("instance_path")
            digest = record.get("instance_sha256")
            if not isinstance(path, str) or not path or not valid_sha256(digest):
                missing.append({"matrix": matrix_name, "instance": instance, "seed": seed})
                continue
            identities[(problem, instance)]["paths"].add(path)
            identities[(problem, instance)]["sha256"].add(digest)
    inconsistent = []
    rows = []
    for (problem, instance), values in sorted(identities.items()):
        row = {
            "problem": problem,
            "instance": instance,
            "paths": sorted(values["paths"]),
            "sha256": sorted(values["sha256"]),
        }
        rows.append(row)
        if len(values["sha256"]) != 1:
            inconsistent.append(row)
    errors = []
    if missing:
        errors.append(f"{len(missing)} runs do not reference a complete instance source artifact")
    if inconsistent:
        errors.append(f"{len(inconsistent)} paired instances have inconsistent source hashes")
    return {
        "instances": rows,
        "missing": missing,
        "inconsistent": inconsistent,
        "errors": errors,
        "pass": not errors,
    }


def audit_reference_provenance(
    sources: Sequence[dict[str, Any]], seeds: Sequence[int],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows = []
    errors = []
    for source in sources:
        label = source["label"]
        provenance = source["provenance"]
        row_errors = list(source["errors"])
        if provenance is not None:
            if provenance.get("schema_version") != 1:
                row_errors.append(f"{label}: provenance schema_version must be 1")
            if provenance.get("seeds") != list(seeds):
                row_errors.append(f"{label}: provenance seeds must be exactly {list(seeds)}")
            budgets = provenance.get("budgets")
            normalized_budgets = tuple(
                number
                for value in budgets
                if (number := finite(value)) is not None and number > 0
            ) if isinstance(budgets, list) else ()
            if not normalized_budgets or len(normalized_budgets) != len(budgets):
                row_errors.append(f"{label}: provenance must declare finite positive budgets")
            threads = non_negative_integer(provenance.get("threads"))
            if threads is None or threads <= 0:
                row_errors.append(f"{label}: provenance must declare a positive thread count")
            solvers = provenance.get("solvers")
            declared = {
                name
                for name, version in solvers.items()
                if name in REFERENCE_SOLVERS and isinstance(version, str) and version
            } if isinstance(solvers, dict) else set()
            if not declared:
                row_errors.append(
                    f"{label}: provenance must identify at least one CP-SAT or Hexaly reference solver"
                )
        rows.append({
            key: source[key]
            for key in (
                "label", "path", "sha256", "records", "provenance_path",
                "provenance_sha256",
            )
        } | {
            "budgets": provenance.get("budgets") if provenance else None,
            "seeds": provenance.get("seeds") if provenance else None,
            "threads": provenance.get("threads") if provenance else None,
            "solver_versions": provenance.get("solvers") if provenance else None,
            "errors": sorted(set(row_errors)),
            "pass": not row_errors,
        })
        errors.extend(row_errors)
    return rows, errors


def audit_jssp_references(
    sources: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
    jssp_ls: dict[tuple[str, int], dict[str, Any]],
    jssp_instances: Sequence[str],
    seeds: Sequence[int],
    max_rss_mb: float,
    limit_percent: float,
) -> dict[str, Any]:
    if not sources:
        return {
            "enabled": False,
            "limit_percent": limit_percent,
            "source_artifacts": [],
            "solvers": [],
            "instances": [],
            "missing": [],
            "duplicates": [],
            "invalid": [],
            "errors": [],
            "pass": True,
        }

    source_rows, errors = audit_reference_provenance(sources, seeds)
    configured = {instance.casefold() for instance in jssp_instances}
    targets = tuple(instance for instance in JSSP_REFERENCE_INSTANCES if instance in configured)
    omitted_targets = sorted(set(JSSP_REFERENCE_INSTANCES) - configured)
    if omitted_targets:
        errors.append(
            "JSSP reference comparison is missing configured Qayd instances: "
            + ", ".join(omitted_targets)
        )

    relevant = [
        record
        for record in records
        if record.get("problem") == "jssp"
        and canonical_instance(record) in set(JSSP_REFERENCE_INSTANCES)
    ]
    unsupported = [
        record_location(record)
        for record in relevant
        if record.get("solver") not in REFERENCE_SOLVERS
    ]
    if unsupported:
        errors.append(f"JSSP reference campaigns contain {len(unsupported)} unsupported solver records")
    relevant = [record for record in relevant if record.get("solver") in REFERENCE_SOLVERS]
    solvers = sorted({record["solver"] for record in relevant})
    if not solvers:
        errors.append("JSSP reference campaigns contain no CP-SAT or Hexaly records")

    groups: dict[tuple[str, str, int | None], list[dict[str, Any]]] = defaultdict(list)
    for record in relevant:
        groups[(record["solver"], canonical_instance(record), seed_value(record))].append(record)
    expected = {
        (solver, instance, seed)
        for solver in solvers
        for instance in targets
        for seed in seeds
    }
    selected: dict[tuple[str, str, int], dict[str, Any]] = {}
    duplicates = []
    invalid = []
    unexpected = []
    for key, runs in sorted(
        groups.items(), key=lambda item: (item[0][0], item[0][1], -1 if item[0][2] is None else item[0][2]),
    ):
        solver, instance, seed = key
        if len(runs) > 1:
            duplicates.append({
                "solver": solver,
                "instance": instance,
                "seed": seed,
                "locations": [record_location(record) for record in runs],
            })
        if seed is None or key not in expected:
            unexpected.append({
                "solver": solver,
                "instance": instance,
                "seed": seed,
                "locations": [record_location(record) for record in runs],
            })
            continue
        record = runs[0]
        selected[(solver, instance, seed)] = record
        issues = record_issues(record, "jssp", None, max_rss_mb)
        declared_solvers = record.get("_provenance_solvers")
        if (
            not isinstance(declared_solvers, dict)
            or not isinstance(declared_solvers.get(solver), str)
            or not declared_solvers[solver]
        ):
            issues.append("reference solver is not identified by its provenance sidecar")
        current = jssp_ls.get((instance, seed))
        if current is None or record.get("instance_sha256") != current.get("instance_sha256"):
            issues.append("reference instance_sha256 does not match the forced-LS Qayd source")
        if issues:
            invalid.append({
                "solver": solver,
                "instance": instance,
                "seed": seed,
                "location": record_location(record),
                "issues": issues,
            })
    missing = sorted(expected - set(selected))
    if missing:
        errors.append(f"JSSP reference campaigns are missing {len(missing)} matched solver/instance/seed runs")
    if unexpected:
        errors.append(f"JSSP reference campaigns contain {len(unexpected)} unexpected keys")
    if duplicates:
        errors.append(f"JSSP reference campaigns contain {len(duplicates)} duplicate keys")
    if invalid:
        errors.append(f"JSSP reference campaigns contain {len(invalid)} invalid runs")

    rows = []
    for instance in targets:
        qayd_values = [scalar_objective(jssp_ls.get((instance, seed), {})) for seed in seeds]
        qayd_complete = len(qayd_values) == len(seeds) and all(value is not None for value in qayd_values)
        qayd_median = (
            statistics.median(value for value in qayd_values if value is not None)
            if qayd_complete
            else None
        )
        reference_medians = {}
        for solver in solvers:
            values = [scalar_objective(selected.get((solver, instance, seed), {})) for seed in seeds]
            if len(values) == len(seeds) and all(value is not None for value in values):
                reference_medians[solver] = statistics.median(
                    value for value in values if value is not None
                )
        best_reference = min(reference_medians.values()) if reference_medians else None
        regression = (
            100.0 * (qayd_median - best_reference) / max(1.0, abs(best_reference))
            if qayd_median is not None and best_reference is not None
            else None
        )
        passed = regression is not None and regression <= limit_percent
        rows.append({
            "instance": instance,
            "qayd_median": qayd_median,
            "reference_medians": reference_medians,
            "best_reference_median": best_reference,
            "regression_percent": regression,
            "pass": passed,
        })
        if not passed:
            errors.append(
                f"forced-LS {instance} median must be within {limit_percent:g}% of the best matched reference median"
            )
    return {
        "enabled": True,
        "limit_percent": limit_percent,
        "source_artifacts": source_rows,
        "solvers": solvers,
        "instances": rows,
        "missing": [
            {"solver": solver, "instance": instance, "seed": seed}
            for solver, instance, seed in missing
        ],
        "unexpected": unexpected,
        "duplicates": duplicates,
        "invalid": invalid,
        "errors": sorted(set(errors)),
        "pass": not errors,
    }


def audit_rcpsp_scale_matrix(
    name: str,
    scale_group: str,
    records: Sequence[dict[str, Any]],
    expected_engine: str | None,
    seeds: Sequence[int],
    solver: str,
    max_rss_mb: float,
) -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    def belongs_to_scale(record: dict[str, Any]) -> bool:
        family = record.get("family")
        path = record.get("instance_path")
        identity = "/".join(
            value.casefold() for value in (family, path) if isinstance(value, str)
        )
        return scale_group in identity

    relevant = [
        record
        for record in records
        if record.get("problem") in {"rcpsp", "mrcpsp"}
        and record.get("solver") == solver
        and belongs_to_scale(record)
    ]
    instances = sorted({canonical_instance(record) for record in relevant if canonical_instance(record)})
    expected = {(instance, seed) for instance in instances for seed in seeds}
    groups: dict[tuple[str, int | None], list[dict[str, Any]]] = defaultdict(list)
    for record in relevant:
        groups[(canonical_instance(record), seed_value(record))].append(record)

    selected = {}
    duplicates = []
    unexpected = []
    invalid = []
    for (instance, seed), runs in sorted(
        groups.items(), key=lambda item: (item[0][0], -1 if item[0][1] is None else item[0][1]),
    ):
        if len(runs) > 1:
            duplicates.append({
                "instance": instance or None,
                "seed": seed,
                "locations": [record_location(record) for record in runs],
            })
        if seed is None or (instance, seed) not in expected:
            unexpected.append({
                "instance": instance or None,
                "seed": seed,
                "locations": [record_location(record) for record in runs],
            })
            continue
        record = runs[0]
        selected[(instance, seed)] = record
        issues = record_issues(record, "rcpsp", expected_engine, max_rss_mb)
        if issues:
            invalid.append({
                "instance": instance,
                "seed": seed,
                "location": record_location(record),
                "issues": issues,
            })
    missing = sorted(expected - set(selected))
    errors = []
    if not relevant:
        errors.append(f"{name}: campaign contains no Qayd RCPSP records")
    if missing:
        errors.append(f"{name}: missing {len(missing)} expected instance/seed runs")
    if unexpected:
        errors.append(f"{name}: found {len(unexpected)} unexpected instance/seed keys")
    if duplicates:
        errors.append(f"{name}: found {len(duplicates)} duplicate instance/seed keys")
    if invalid:
        errors.append(f"{name}: found {len(invalid)} runs without a valid incumbent")
    valid_incumbents = len(selected) - len(invalid)
    return ({
        "name": name,
        "scale_group": scale_group,
        "expected_engine": expected_engine,
        "instances": instances,
        "expected_runs": len(expected),
        "selected_runs": len(selected),
        "valid_incumbents": valid_incumbents,
        "incumbent_coverage_percent": (
            100.0 * valid_incumbents / len(expected) if expected else 0.0
        ),
        "missing": [{"instance": instance, "seed": seed} for instance, seed in missing],
        "unexpected": unexpected,
        "duplicates": duplicates,
        "invalid": invalid,
        "errors": errors,
        "pass": not errors and bool(expected) and valid_incumbents == len(expected),
    }, selected)


def quality_checks(
    jssp_ls: dict[tuple[str, int], dict[str, Any]],
    rcpsp_auto: dict[tuple[str, int], dict[str, Any]],
    rcpsp_ls: dict[tuple[str, int], dict[str, Any]],
    jssp_instances: Sequence[str],
    seeds: Sequence[int],
    rcpsp_best_known: dict[str, float],
) -> dict[str, Any]:
    configured = {value.casefold() for value in jssp_instances}

    ft06_values = [scalar_objective(jssp_ls.get(("ft06", seed), {})) for seed in seeds] if "ft06" in configured else []
    ft06 = {
        "enabled": "ft06" in configured,
        "target": FT06_TARGET,
        "objectives": ft06_values,
        "pass": "ft06" not in configured or (len(ft06_values) == len(seeds) and all(value == FT06_TARGET for value in ft06_values)),
    }

    orb10_values = [scalar_objective(jssp_ls.get(("orb10", seed), {})) for seed in seeds] if "orb10" in configured else []
    orb10_complete = len(orb10_values) == len(seeds) and all(value is not None for value in orb10_values)
    orb10_median = statistics.median(value for value in orb10_values if value is not None) if orb10_complete else None
    orb10 = {
        "enabled": "orb10" in configured,
        "limit": ORB10_MEDIAN_LIMIT,
        "objectives": orb10_values,
        "median_objective": orb10_median,
        "pass": "orb10" not in configured or (orb10_median is not None and orb10_median <= ORB10_MEDIAN_LIMIT),
    }

    def bks_check(records: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
        rows = []
        passed_all = True
        for (instance, seed), record in sorted(records.items()):
            objective = scalar_objective(record)
            manifest_best_known = rcpsp_best_known.get(instance)
            record_best_known = finite(record.get("best_known"))
            objective_matches = (
                objective is not None
                and manifest_best_known is not None
                and objective == manifest_best_known
            )
            record_matches = (
                record_best_known is not None
                and manifest_best_known is not None
                and record_best_known == manifest_best_known
            )
            passed = objective_matches and record_matches
            passed_all &= passed
            rows.append({
                "instance": instance,
                "seed": seed,
                "objective": objective,
                "manifest_best_known": manifest_best_known,
                "record_best_known": record_best_known,
                "objective_matches": objective_matches,
                "record_matches": record_matches,
                "pass": passed,
            })
        return {
            "runs": len(records),
            "matches": sum(row["pass"] for row in rows),
            "objective_matches": sum(row["objective_matches"] for row in rows),
            "record_matches": sum(row["record_matches"] for row in rows),
            "rows": rows,
            "pass": bool(rows) and passed_all,
        }

    rcpsp_auto_bks = bks_check(rcpsp_auto)
    rcpsp_ls_bks = bks_check(rcpsp_ls)
    errors = []
    if not ft06["pass"]:
        errors.append(f"forced-LS ft06 must equal {FT06_TARGET:g} for every seed")
    if not orb10["pass"]:
        errors.append(f"forced-LS orb10 median must be at most {ORB10_MEDIAN_LIMIT:g}")
    if not rcpsp_auto_bks["pass"]:
        errors.append(
            "automatic RCPSP objectives and record best_known fields must equal the external manifest"
        )
    if not rcpsp_ls_bks["pass"]:
        errors.append(
            "forced-LS RCPSP objectives and record best_known fields must equal the external manifest"
        )
    return {
        "ft06": ft06,
        "orb10": orb10,
        "rcpsp_auto_bks": rcpsp_auto_bks,
        "rcpsp_ls_bks": rcpsp_ls_bks,
        "errors": errors,
        "pass": not errors,
    }


def paired_regression(
    current: Sequence[tuple[str, dict[tuple[str, int], dict[str, Any]]]],
    baseline: Sequence[tuple[str, dict[tuple[str, int], dict[str, Any]]]],
    limit_percent: float,
) -> dict[str, Any]:
    if not baseline:
        return {
            "enabled": False,
            "limit_percent": limit_percent,
            "expected_pairs": 0,
            "pairs": 0,
            "regressions": [],
            "instances": [],
            "errors": [],
            "pass": True,
        }

    baseline_by_problem = dict(baseline)
    rows = []
    regressions = []
    errors = []
    expected_pairs = 0
    by_instance: dict[tuple[str, str], list[float]] = defaultdict(list)
    for problem, current_records in current:
        reference_records = baseline_by_problem.get(problem, {})
        expected_pairs += len(current_records)
        for (instance, seed), record in sorted(current_records.items()):
            reference = reference_records.get((instance, seed))
            current_value = scalar_objective(record)
            reference_value = scalar_objective(reference) if reference is not None else None
            if current_value is None or reference_value is None:
                errors.append(f"missing paired objective for {problem}/{instance}/seed={seed}")
                continue
            regression = 100.0 * (current_value - reference_value) / max(1.0, abs(reference_value))
            passed = regression <= limit_percent
            row = {
                "problem": problem,
                "instance": instance,
                "seed": seed,
                "current_objective": current_value,
                "baseline_objective": reference_value,
                "regression_percent": regression,
                "pass": passed,
            }
            rows.append(row)
            by_instance[(problem, instance)].append(regression)
            if not passed:
                regressions.append(row)
    instance_rows = [
        {
            "problem": problem,
            "instance": instance,
            "pairs": len(values),
            "median_regression_percent": statistics.median(values),
            "max_regression_percent": max(values),
        }
        for (problem, instance), values in sorted(by_instance.items())
    ]
    if len(rows) != expected_pairs:
        errors.append(f"paired regression has {len(rows)}/{expected_pairs} objective pairs")
    if regressions:
        errors.append(f"{len(regressions)} paired runs regress by more than {limit_percent:g}%")
    return {
        "enabled": True,
        "limit_percent": limit_percent,
        "expected_pairs": expected_pairs,
        "pairs": len(rows),
        "regressions": regressions,
        "instances": instance_rows,
        "rows": rows,
        "errors": errors,
        "pass": not errors,
    }


def build_summary(
    jssp_auto_path: Path,
    jssp_ls_path: Path,
    rcpsp_auto_path: Path,
    rcpsp_ls_path: Path,
    *,
    rcpsp_best_known_path: Path,
    baseline_paths: Sequence[Path] = (),
    jssp_reference_paths: Sequence[Path] = (),
    rcpsp_scale_paths: dict[str, Path] | None = None,
    jssp_instances: Sequence[str] = DEFAULT_JSSP_INSTANCES,
    rcpsp_instances: Sequence[str] = DEFAULT_RCPSP_INSTANCES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    solver: str = "qayd-api",
    max_rss_mb: float = MAX_RSS_MB,
    regression_limit_percent: float = 20.0,
    jssp_reference_limit_percent: float = JSSP_REFERENCE_REGRESSION_LIMIT_PERCENT,
    require_complete_acceptance: bool = False,
) -> dict[str, Any]:
    if not jssp_instances or not rcpsp_instances or not seeds:
        raise ValueError("JSSP instances, RCPSP instances and seeds must all be non-empty")
    if len({value.casefold() for value in jssp_instances}) != len(jssp_instances):
        raise ValueError("JSSP instance names must be unique")
    if len({value.casefold() for value in rcpsp_instances}) != len(rcpsp_instances):
        raise ValueError("RCPSP instance names must be unique")
    if len(set(seeds)) != len(seeds) or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("seeds must be unique non-negative integers")
    if not math.isfinite(max_rss_mb) or max_rss_mb <= 0:
        raise ValueError("max_rss_mb must be positive and finite")
    if not math.isfinite(regression_limit_percent) or regression_limit_percent < 0:
        raise ValueError("regression_limit_percent must be non-negative and finite")
    if not math.isfinite(jssp_reference_limit_percent) or jssp_reference_limit_percent < 0:
        raise ValueError("jssp_reference_limit_percent must be non-negative and finite")
    scale_paths = dict(rcpsp_scale_paths or {})
    unknown_scale_groups = sorted(set(scale_paths) - set(RCPSP_SCALE_GROUPS))
    if unknown_scale_groups:
        raise ValueError(f"unknown RCPSP scale groups: {', '.join(unknown_scale_groups)}")

    rcpsp_best_known_artifact, rcpsp_best_known = load_rcpsp_best_known(
        rcpsp_best_known_path, rcpsp_instances,
    )

    source_specs = [
        ("jssp-auto", jssp_auto_path),
        ("jssp-ls", jssp_ls_path),
        ("rcpsp-auto", rcpsp_auto_path),
        ("rcpsp-ls", rcpsp_ls_path),
        *((f"rcpsp-scale-{group}", scale_paths[group]) for group in RCPSP_SCALE_GROUPS if group in scale_paths),
        *((f"baseline-{index + 1}", path) for index, path in enumerate(baseline_paths)),
    ]
    loaded = [load_source(path, label) for label, path in source_specs]
    sources = [metadata for metadata, _records in loaded]
    records_by_label = {metadata["label"]: records for metadata, records in loaded}

    matrix_specs = (
        ("jssp-auto", "jssp", "auto", jssp_instances),
        ("jssp-ls", "jssp", "ls", jssp_instances),
        ("rcpsp-auto", "rcpsp", "auto", rcpsp_instances),
        ("rcpsp-ls", "rcpsp", "ls", rcpsp_instances),
    )
    matrices = {}
    selected = {}
    for name, problem, engine, instances in matrix_specs:
        matrices[name], selected[name] = audit_matrix(
            name,
            records_by_label[name],
            problem,
            engine,
            instances,
            seeds,
            solver,
            max_rss_mb,
        )

    baseline_records = [
        record
        for label, records in records_by_label.items()
        if label.startswith("baseline-")
        for record in records
    ]
    baseline_selected = {}
    if baseline_paths:
        matrices["baseline-jssp"], baseline_selected["jssp"] = audit_matrix(
            "baseline-jssp", baseline_records, "jssp", "ls", jssp_instances, seeds, solver, max_rss_mb,
        )
        matrices["baseline-rcpsp"], baseline_selected["rcpsp"] = audit_matrix(
            "baseline-rcpsp", baseline_records, "rcpsp", "ls", rcpsp_instances, seeds, solver, max_rss_mb,
        )

    scale_selected = {}
    scale_labels = []
    for group in RCPSP_SCALE_GROUPS:
        if group not in scale_paths:
            continue
        label = f"rcpsp-scale-{group}"
        scale_labels.append(label)
        source = next(source for source in sources if source["label"] == label)
        source_provenance = source["provenance"]
        expected_engine = (
            source_provenance.get("qayd_engine")
            if source_provenance is not None
            and source_provenance.get("qayd_engine") in {"auto", "ls"}
            else None
        )
        matrices[label], scale_selected[group] = audit_rcpsp_scale_matrix(
            label,
            group,
            records_by_label[label],
            expected_engine,
            seeds,
            solver,
            max_rss_mb,
        )
        if expected_engine is None:
            matrices[label]["errors"].append(
                f"{label}: provenance qayd_engine must be 'auto' or 'ls'"
            )
            matrices[label]["pass"] = False

    expected_engines = {name: engine for name, _problem, engine, _instances in matrix_specs}
    expected_engines.update((f"baseline-{index + 1}", "ls") for index in range(len(baseline_paths)))
    for label in scale_labels:
        source = next(source for source in sources if source["label"] == label)
        source_provenance = source["provenance"]
        expected_engines[label] = (
            source_provenance.get("qayd_engine")
            if source_provenance is not None
            and source_provenance.get("qayd_engine") in {"auto", "ls"}
            else None
        )
    provenance = audit_provenance(
        sources,
        expected_engines,
        seeds,
        tuple(name for name, _problem, _engine, _instances in matrix_specs) + tuple(scale_labels),
        solver,
    )
    instance_sources = audit_instance_sources([
        ("jssp-auto", selected["jssp-auto"]),
        ("jssp-ls", selected["jssp-ls"]),
        ("rcpsp-auto", selected["rcpsp-auto"]),
        ("rcpsp-ls", selected["rcpsp-ls"]),
        *((f"{problem}-baseline", records) for problem, records in baseline_selected.items()),
        *((f"rcpsp-scale-{group}", records) for group, records in scale_selected.items()),
    ])

    loaded_references = [
        load_source(path, f"jssp-reference-{index + 1}")
        for index, path in enumerate(jssp_reference_paths)
    ]
    reference_sources = [metadata for metadata, _records in loaded_references]
    reference_records = [record for _metadata, records in loaded_references for record in records]
    jssp_reference = audit_jssp_references(
        reference_sources,
        reference_records,
        selected["jssp-ls"],
        jssp_instances,
        seeds,
        max_rss_mb,
        jssp_reference_limit_percent,
    )
    quality = quality_checks(
        selected["jssp-ls"],
        selected["rcpsp-auto"],
        selected["rcpsp-ls"],
        jssp_instances,
        seeds,
        rcpsp_best_known,
    )
    regression = paired_regression(
        [("jssp", selected["jssp-ls"]), ("rcpsp", selected["rcpsp-ls"])],
        list(baseline_selected.items()),
        regression_limit_percent,
    )

    complete_errors = []
    missing_scale_groups = [group for group in RCPSP_SCALE_GROUPS if group not in scale_paths]
    if require_complete_acceptance and not jssp_reference_paths:
        complete_errors.append("complete acceptance requires at least one --jssp-reference campaign")
    if require_complete_acceptance and missing_scale_groups:
        complete_errors.append(
            "complete acceptance requires RCPSP scale campaigns for: "
            + ", ".join(missing_scale_groups)
        )
    complete_inputs_present = bool(jssp_reference_paths) and not missing_scale_groups
    complete_criteria_pass = (
        complete_inputs_present
        and jssp_reference["pass"]
        and all(matrices[f"rcpsp-scale-{group}"]["pass"] for group in RCPSP_SCALE_GROUPS)
    )

    errors = [
        error
        for matrix in matrices.values()
        for error in matrix["errors"]
    ]
    errors.extend(provenance["errors"])
    errors.extend(instance_sources["errors"])
    errors.extend(rcpsp_best_known_artifact["errors"])
    errors.extend(quality["errors"])
    errors.extend(regression["errors"])
    errors.extend(jssp_reference["errors"])
    errors.extend(complete_errors)
    accepted = not errors
    return {
        "schema_version": 1,
        "gate": "phase-13-scheduling-search",
        "solver": solver,
        "contract": {
            "jssp_instances": [value.casefold() for value in jssp_instances],
            "rcpsp_instances": [value.casefold() for value in rcpsp_instances],
            "seeds": list(seeds),
            "max_rss_mb_exclusive": max_rss_mb,
            "ft06_objective": FT06_TARGET,
            "orb10_median_limit": ORB10_MEDIAN_LIMIT,
            "regression_limit_percent": regression_limit_percent,
            "jssp_reference_instances": list(JSSP_REFERENCE_INSTANCES),
            "jssp_reference_limit_percent": jssp_reference_limit_percent,
            "rcpsp_scale_groups": list(RCPSP_SCALE_GROUPS),
            "rcpsp_best_known_sha256": rcpsp_best_known_artifact["sha256"],
        },
        "rcpsp_best_known": rcpsp_best_known_artifact,
        "source_artifacts": provenance["sources"],
        "provenance": {
            key: value for key, value in provenance.items() if key != "sources"
        },
        "instance_sources": instance_sources,
        "matrices": matrices,
        "quality": quality,
        "jssp_reference": jssp_reference,
        "complete_qualification": {
            "required": require_complete_acceptance,
            "inputs_present": complete_inputs_present,
            "missing_rcpsp_scale_groups": missing_scale_groups,
            "errors": complete_errors,
            "pass": accepted and complete_criteria_pass,
        },
        "regression": regression,
        "errors": sorted(set(errors)),
        "accepted": accepted,
        "pass": accepted,
    }


def shown(value: object, digits: int = 2) -> str:
    number = finite(value)
    return "n/a" if number is None else f"{number:.{digits}f}"


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 13 scheduling-search qualification",
        "",
        f"Gate: **{'PASS' if summary['accepted'] else 'FAIL'}**.",
        "",
        "## Source artifacts",
        "",
        "| Arm | JSONL | SHA-256 | Provenance | Engine | Gate |",
        "|---|---|---|---|---|---|",
    ]
    for source in summary["source_artifacts"]:
        lines.append(
            f"| {source['label']} | `{source['path']}` | `{source['sha256'] or 'missing'}` | "
            f"`{source['provenance_path']}` | {source['qayd_engine']} | "
            f"{'PASS' if source['pass'] else 'FAIL'} |"
        )
    best_known = summary["rcpsp_best_known"]
    lines.extend((
        "",
        "## External RCPSP best-known artifact",
        "",
        "| JSON | SHA-256 | Entries | Gate |",
        "|---|---|---:|---|",
        f"| `{best_known['path']}` | `{best_known['sha256'] or 'missing'}` | "
        f"{len(best_known['entries'])} | {'PASS' if best_known['pass'] else 'FAIL'} |",
    ))
    lines.extend((
        "",
        "## Exact matrices",
        "",
        "| Matrix | Selected | Missing | Unexpected | Duplicates | Invalid | Gate |",
        "|---|---:|---:|---:|---:|---:|---|",
    ))
    for matrix in summary["matrices"].values():
        lines.append(
            f"| {matrix['name']} | {matrix['selected_runs']}/{matrix['expected_runs']} | "
            f"{len(matrix['missing'])} | {len(matrix['unexpected'])} | "
            f"{len(matrix['duplicates'])} | {len(matrix['invalid'])} | "
            f"{'PASS' if matrix['pass'] else 'FAIL'} |"
        )

    quality = summary["quality"]
    lines.extend((
        "",
        "## Quality sentinels",
        "",
        "| Check | Result | Requirement | Gate |",
        "|---|---:|---:|---|",
        f"| forced-LS ft06 | {quality['ft06']['objectives']} | all = {quality['ft06']['target']:g} | "
        f"{'PASS' if quality['ft06']['pass'] else 'FAIL'} |",
        f"| forced-LS orb10 median | {shown(quality['orb10']['median_objective'])} | <= {quality['orb10']['limit']:g} | "
        f"{'PASS' if quality['orb10']['pass'] else 'FAIL'} |",
        f"| automatic RCPSP at external BKS | {quality['rcpsp_auto_bks']['matches']}/{quality['rcpsp_auto_bks']['runs']} | all | "
        f"{'PASS' if quality['rcpsp_auto_bks']['pass'] else 'FAIL'} |",
        f"| forced-LS RCPSP at external BKS | {quality['rcpsp_ls_bks']['matches']}/{quality['rcpsp_ls_bks']['runs']} | all | "
        f"{'PASS' if quality['rcpsp_ls_bks']['pass'] else 'FAIL'} |",
    ))

    reference = summary["jssp_reference"]
    lines.extend(("", "## Matched JSSP references", ""))
    if not reference["enabled"]:
        lines.append("No CP-SAT or Hexaly reference campaign was supplied.")
    else:
        lines.extend((
            "| Instance | Qayd median | Best reference median | Regression | Gate |",
            "|---|---:|---:|---:|---|",
        ))
        for row in reference["instances"]:
            lines.append(
                f"| {row['instance']} | {shown(row['qayd_median'])} | "
                f"{shown(row['best_reference_median'])} | {shown(row['regression_percent'])}% | "
                f"{'PASS' if row['pass'] else 'FAIL'} |"
            )

    scale_matrices = [
        matrix
        for name, matrix in summary["matrices"].items()
        if name.startswith("rcpsp-scale-")
    ]
    lines.extend(("", "## RCPSP scale incumbent coverage", ""))
    if not scale_matrices:
        lines.append("No j60, j90, j120 or MMLIB scale campaign was supplied.")
    else:
        lines.extend((
            "| Matrix | Instances | Valid incumbents | Coverage | Gate |",
            "|---|---:|---:|---:|---|",
        ))
        for matrix in scale_matrices:
            lines.append(
                f"| {matrix['name']} | {len(matrix['instances'])} | "
                f"{matrix['valid_incumbents']}/{matrix['expected_runs']} | "
                f"{shown(matrix['incumbent_coverage_percent'])}% | "
                f"{'PASS' if matrix['pass'] else 'FAIL'} |"
            )

    complete = summary["complete_qualification"]
    lines.extend((
        "",
        "## Complete qualification",
        "",
        f"Complete gate: **{'PASS' if complete['pass'] else 'FAIL'}**. "
        f"Required by this invocation: `{complete['required']}`.",
    ))

    regression = summary["regression"]
    lines.extend(("", "## Paired forced-LS regression", ""))
    if not regression["enabled"]:
        lines.append("No baseline was requested. The paired regression gate is disabled.")
    else:
        lines.extend((
            f"Pairs: {regression['pairs']}/{regression['expected_pairs']}; per-run limit: "
            f"{regression['limit_percent']:g}%.",
            "",
            "| Problem | Instance | Pairs | Median regression | Maximum regression |",
            "|---|---|---:|---:|---:|",
        ))
        for row in regression["instances"]:
            lines.append(
                f"| {row['problem']} | {row['instance']} | {row['pairs']} | "
                f"{shown(row['median_regression_percent'])}% | {shown(row['max_regression_percent'])}% |"
            )

    lines.extend(("", "## Diagnostics", ""))
    if summary["errors"]:
        lines.extend(f"- {error}" for error in summary["errors"])
        for matrix in summary["matrices"].values():
            for invalid in matrix["invalid"]:
                lines.append(
                    f"- {matrix['name']} {invalid['instance']}/seed={invalid['seed']} at "
                    f"`{invalid['location']}`: {'; '.join(invalid['issues'])}"
                )
            for duplicate in matrix["duplicates"]:
                lines.append(
                    f"- {matrix['name']} duplicate {duplicate['instance']}/seed={duplicate['seed']}: "
                    f"{', '.join(duplicate['locations'])}"
                )
        for invalid in reference["invalid"]:
            lines.append(
                f"- JSSP reference {invalid['solver']} {invalid['instance']}/seed={invalid['seed']} at "
                f"`{invalid['location']}`: {'; '.join(invalid['issues'])}"
            )
    else:
        lines.append("- All acceptance checks passed.")
    lines.append("")
    return "\n".join(lines)


def csv_values(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def csv_seeds(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error
    if not values or any(seed < 0 for seed in values):
        raise argparse.ArgumentTypeError("seeds must be non-negative and non-empty")
    return values


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jssp-auto", type=Path, required=True)
    parser.add_argument("--jssp-ls", type=Path, required=True)
    parser.add_argument("--rcpsp-auto", type=Path, required=True)
    parser.add_argument("--rcpsp-ls", type=Path, required=True)
    parser.add_argument("--rcpsp-best-known", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, action="append", default=[])
    parser.add_argument("--jssp-reference", type=Path, action="append", default=[])
    parser.add_argument("--rcpsp-j60", type=Path)
    parser.add_argument("--rcpsp-j90", type=Path)
    parser.add_argument("--rcpsp-j120", type=Path)
    parser.add_argument("--rcpsp-mmlib", type=Path)
    parser.add_argument("--jssp-instances", type=csv_values, default=DEFAULT_JSSP_INSTANCES)
    parser.add_argument("--rcpsp-instances", type=csv_values, default=DEFAULT_RCPSP_INSTANCES)
    parser.add_argument("--seeds", type=csv_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--solver", default="qayd-api")
    parser.add_argument("--max-rss-mb", type=float, default=MAX_RSS_MB)
    parser.add_argument("--regression-limit-percent", type=float, default=20.0)
    parser.add_argument(
        "--jssp-reference-limit-percent",
        type=float,
        default=JSSP_REFERENCE_REGRESSION_LIMIT_PERCENT,
    )
    parser.add_argument("--json", type=Path, required=True, dest="json_output")
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--require-acceptance", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = arguments(argv)
    rcpsp_scale_paths = {
        group: path
        for group, path in (
            ("j60", args.rcpsp_j60),
            ("j90", args.rcpsp_j90),
            ("j120", args.rcpsp_j120),
            ("mmlib", args.rcpsp_mmlib),
        )
        if path is not None
    }
    summary = build_summary(
        args.jssp_auto,
        args.jssp_ls,
        args.rcpsp_auto,
        args.rcpsp_ls,
        rcpsp_best_known_path=args.rcpsp_best_known,
        baseline_paths=args.baseline,
        jssp_reference_paths=args.jssp_reference,
        rcpsp_scale_paths=rcpsp_scale_paths,
        jssp_instances=args.jssp_instances,
        rcpsp_instances=args.rcpsp_instances,
        seeds=args.seeds,
        solver=args.solver,
        max_rss_mb=args.max_rss_mb,
        regression_limit_percent=args.regression_limit_percent,
        jssp_reference_limit_percent=args.jssp_reference_limit_percent,
        require_complete_acceptance=args.require_acceptance,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.write_text(markdown(summary), encoding="utf-8")
    print(f"Phase 13 scheduling gate: {'PASS' if summary['accepted'] else 'FAIL'}")
    print(args.markdown)
    return 2 if args.require_acceptance and not summary["accepted"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
