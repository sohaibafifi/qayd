#!/usr/bin/env python3
"""Fail-closed A/B gate for the bounded TSAB JSSP candidate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


FEASIBLE = {"SAT", "SATISFIABLE", "FEASIBLE", "OPTIMAL", "OPTIMUM"}
EXPECTED_OWNER_MASK = 0x40
MAX_PEAK_MEMORY_MB = 11_264.0
CONTROL_RETENTION = 0.90
QUALITY_OBJECTIVE = 936_343
ZERO_FIELDS = (
    "schedule_work_budget_overruns",
    "schedule_oracle_mismatches",
    "schedule_incumbent_incomplete_rejections",
    "schedule_incumbent_verification_rejections",
)
TSAB_COUNTERS = (
    "schedule_tsab_owner_worker_mask",
    "schedule_tsab_shortlists",
    "schedule_tsab_n5_generated",
    "schedule_tsab_ranked",
    "schedule_tsab_delta_probes",
    "schedule_tsab_additional_delta_probes",
    "schedule_tsab_full_oracle_commits",
    "schedule_tsab_selections",
    "schedule_tsab_selected_shortlist_rank_sum",
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
TSAB_NUMBERS = ("schedule_tsab_fast_elapsed_seconds",)
TSAB_OPTIONAL_COUNTERS = (
    "schedule_tsab_activation_boundary",
    "schedule_tsab_activation_objective",
    "schedule_tsab_best_committed_objective",
    "schedule_tsab_restart_best_base_objective",
    "schedule_tsab_restart_best_kicked_objective",
)


class GateError(ValueError):
    """Raised when inputs cannot support a qualified comparison."""


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


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GateError(f"{label} must be an integer >= {minimum}")
    return value


def _optional_integer(value: object, label: str, *, minimum: int = 0) -> int | None:
    if value is None:
        return None
    return _integer(value, label, minimum=minimum)


def _number(value: object, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise GateError(f"{label} must be finite and >= {minimum:g}")
    return result


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdefABCDEF" for character in value):
        raise GateError(f"{label} must be a SHA-256 hex digest")
    return value.lower()


def _profile_map(value: object, label: str) -> dict[int, int]:
    if not isinstance(value, str) or not value:
        raise GateError(f"{label} must be a non-empty profile map")
    result: dict[int, int] = {}
    for item in value.split(","):
        try:
            raw_key, raw_work = item.split("=", 1)
            key, work = int(raw_key), int(raw_work)
        except ValueError as error:
            raise GateError(f"{label} contains an invalid entry: {item!r}") from error
        if key in result or key < 0 or work < 0:
            raise GateError(f"{label} contains an invalid or duplicate profile: {item!r}")
        result[key] = work
    return result


def _provenance(campaign: Path, *, strategy: str) -> dict[str, Any]:
    value = _read_json(provenance_path(campaign))
    expected = {
        "threads": 7,
        "qayd_engine": "ls",
        "profile_qayd": True,
        "qayd_schedule_jssp_search": strategy,
        "qayd_schedule_constructor_workers": 0,
        "qayd_schedule_path_relink": False,
        "qayd_schedule_lns_shadow": False,
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise GateError(f"{campaign}: provenance {field} must be {expected_value!r}")
    host = value.get("host")
    artifact = value.get("qayd_artifact")
    if not isinstance(host, dict) or not isinstance(artifact, dict):
        raise GateError(f"{campaign}: provenance must contain host and qayd_artifact")
    _sha256(host.get("source_tree_sha256"), f"{campaign}: source_tree_sha256")
    _sha256(artifact.get("sha256"), f"{campaign}: artifact sha256")
    return value


def _pair_key(record: dict[str, Any], label: str) -> tuple[str, int, int]:
    digest = _sha256(record.get("instance_sha256"), f"{label}.instance_sha256")
    checkpoint = _integer(record.get("checkpoint_seconds"), f"{label}.checkpoint_seconds", minimum=1)
    seed = _integer(record.get("seed"), f"{label}.seed")
    return digest, checkpoint, seed


def _record_map(records: list[dict[str, Any]], label: str) -> dict[tuple[str, int, int], dict[str, Any]]:
    result: dict[tuple[str, int, int], dict[str, Any]] = {}
    for index, record in enumerate(records):
        key = _pair_key(record, f"{label}[{index}]")
        if key in result:
            raise GateError(f"{label}: duplicate comparison key {key}")
        result[key] = record
    return result


def _objective(record: dict[str, Any], label: str) -> int:
    objectives = record.get("objectives")
    if not isinstance(objectives, list) or len(objectives) != 1:
        raise GateError(f"{label}.objectives must contain exactly one value")
    return _integer(objectives[0], f"{label}.objectives[0]", minimum=1)


def _replay_valid(record: dict[str, Any], label: str) -> bool:
    if (
        record.get("solver") not in {"qayd-api", "qayd-native"}
        or record.get("problem") != "jssp"
        or record.get("engine") != "ls"
        or record.get("threads") != 7
        or record.get("requested_threads") != 7
        or record.get("return_code") != 0
        or record.get("verified") is not True
        or record.get("status") not in FEASIBLE
    ):
        return False
    length = _integer(record.get("start_vector_length"), f"{label}.start_vector_length", minimum=1)
    _sha256(record.get("start_vector_sha256"), f"{label}.start_vector_sha256")
    counts = record.get("verification_counts")
    if not isinstance(counts, dict):
        raise GateError(f"{label}.verification_counts must be an object")
    return (
        counts.get("starts") == length
        and _integer(counts.get("job_precedence_pairs"), f"{label}.job_precedence_pairs") > 0
        and _integer(counts.get("machine_non_overlap_pairs"), f"{label}.machine_non_overlap_pairs") > 0
        and counts.get("objective_checks") == 1
        and _integer(record.get("schedule_incumbent_verifications"), f"{label}.schedule_incumbent_verifications", minimum=1) > 0
    )


def _verification_timing_valid(record: dict[str, Any], label: str) -> bool:
    total = _number(record.get("schedule_incumbent_verification_seconds"), f"{label}.schedule_incumbent_verification_seconds")
    maximum = _number(
        record.get("schedule_incumbent_verification_max_seconds"),
        f"{label}.schedule_incumbent_verification_max_seconds",
    )
    return maximum <= total


def _legacy_clean(record: dict[str, Any], label: str) -> bool:
    missing = [field for field in (*TSAB_COUNTERS, *TSAB_NUMBERS, *TSAB_OPTIONAL_COUNTERS) if field not in record]
    if missing:
        raise GateError(f"{label} is missing TSAB metrics: {', '.join(missing)}")
    return (
        all(_integer(record.get(field), f"{label}.{field}") == 0 for field in TSAB_COUNTERS)
        and all(_number(record.get(field), f"{label}.{field}") == 0.0 for field in TSAB_NUMBERS)
        and all(_optional_integer(record.get(field), f"{label}.{field}") is None for field in TSAB_OPTIONAL_COUNTERS)
    )


def _candidate_checks(
    record: dict[str, Any], baseline: dict[str, Any], label: str
) -> tuple[list[str], bool, dict[int, float]]:
    failures: list[str] = []
    missing = [field for field in (*TSAB_COUNTERS, *TSAB_NUMBERS, *TSAB_OPTIONAL_COUNTERS) if field not in record]
    if missing:
        raise GateError(f"{label} is missing TSAB metrics: {', '.join(missing)}")
    counters = {field: _integer(record.get(field), f"{label}.{field}") for field in TSAB_COUNTERS}
    fast_elapsed = _number(record.get("schedule_tsab_fast_elapsed_seconds"), f"{label}.schedule_tsab_fast_elapsed_seconds")
    activation_boundary = _optional_integer(
        record.get("schedule_tsab_activation_boundary"),
        f"{label}.schedule_tsab_activation_boundary",
        minimum=1,
    )
    activation_objective = _optional_integer(
        record.get("schedule_tsab_activation_objective"),
        f"{label}.schedule_tsab_activation_objective",
        minimum=1,
    )
    best_committed_objective = _optional_integer(
        record.get("schedule_tsab_best_committed_objective"),
        f"{label}.schedule_tsab_best_committed_objective",
        minimum=1,
    )
    restart_best_base_objective = _optional_integer(
        record.get("schedule_tsab_restart_best_base_objective"),
        f"{label}.schedule_tsab_restart_best_base_objective",
        minimum=1,
    )
    restart_best_kicked_objective = _optional_integer(
        record.get("schedule_tsab_restart_best_kicked_objective"),
        f"{label}.schedule_tsab_restart_best_kicked_objective",
        minimum=1,
    )
    if counters["schedule_tsab_owner_worker_mask"] != EXPECTED_OWNER_MASK:
        failures.append("owner_mask")
    if (
        counters["schedule_tsab_activations"] != 1
        or activation_boundary != 2
        or counters["schedule_tsab_legacy_warmup_work_steps"] != 512
        or counters["schedule_tsab_activation_rebases"] != 1
        or activation_objective is None
    ):
        failures.append("phased_activation")
    active_boundaries = counters["schedule_tsab_active_boundaries"]
    burst_limit = counters["schedule_tsab_burst_work_limit"]
    burst_work = counters["schedule_tsab_burst_work_units"]
    if active_boundaries == 0 or burst_limit != 128 or burst_work > burst_limit * active_boundaries:
        failures.append("burst_accounting")
    shortlists = counters["schedule_tsab_shortlists"]
    generated = counters["schedule_tsab_n5_generated"]
    ranked = counters["schedule_tsab_ranked"]
    probes = counters["schedule_tsab_delta_probes"]
    selections = counters["schedule_tsab_selections"]
    fast_enabled = counters["schedule_tsab_fast_enabled"]
    fast_eligible = counters["schedule_tsab_fast_eligible"]
    fast_disabled = counters["schedule_tsab_fast_disabled"]
    fast_attempts = counters["schedule_tsab_fast_attempts"]
    fast_commits = counters["schedule_tsab_fast_commits"]
    fast_full_validations = counters["schedule_tsab_fast_full_validations"]
    fast_transitions = counters["schedule_tsab_fast_transitions"]
    fast_work = counters["schedule_tsab_fast_work_units"]
    fast_active = fast_enabled > 0 and fast_eligible > 0
    if not fast_active or fast_attempts == 0 or fast_transitions == 0 or fast_work == 0 or fast_full_validations == 0:
        failures.append("fast_activity")
    if fast_disabled != 0:
        failures.append("fast_disabled")
    if counters["schedule_tsab_fast_oracle_mismatches"] != 0:
        failures.append("fast_oracle_mismatch")
    if counters["schedule_tsab_fast_pending_discards"] != 0:
        failures.append("fast_pending_discard")
    if fast_commits != fast_transitions or fast_transitions > fast_attempts or fast_transitions > 4 * fast_work:
        failures.append("fast_transition_accounting")
    if counters["schedule_tsab_fast_fallbacks"] > fast_attempts:
        failures.append("fast_fallback_accounting")
    if counters["schedule_tsab_fast_queue_pops"] < counters["schedule_tsab_fast_date_changes"]:
        failures.append("fast_queue_accounting")
    if (
        counters["schedule_tsab_fast_pending_promotions"] + counters["schedule_tsab_fast_pending_discards"]
        > fast_full_validations
    ):
        failures.append("fast_validation_accounting")
    if fast_active and counters["schedule_tsab_fast_workspace_peak_bytes"] == 0:
        failures.append("fast_workspace")
    if fast_elapsed < 0.0:
        failures.append("fast_timing")
    if shortlists == 0 or generated == 0 or ranked != generated:
        failures.append("shortlist_activity")
    if not fast_active and (probes == 0 or probes > 4 * shortlists or probes > ranked):
        failures.append("probe_accounting")
    if counters["schedule_tsab_additional_delta_probes"] > probes:
        failures.append("additional_probe_accounting")
    commits = counters["schedule_tsab_full_oracle_commits"]
    if (not fast_active and commits == 0) or commits > selections:
        failures.append("oracle_commit_accounting")
    if selections == 0 or selections > shortlists:
        failures.append("selection_accounting")
    if counters["schedule_tsab_aspirations"] > commits:
        failures.append("aspiration_accounting")
    if counters["schedule_tsab_tabu_rejections"] > probes:
        failures.append("tabu_accounting")
    if counters["schedule_tsab_tabu_resets"] > shortlists:
        failures.append("tabu_reset_accounting")
    if counters["schedule_tsab_selected_shortlist_rank_sum"] > 3 * commits:
        failures.append("shortlist_rank_accounting")
    restart_attempts = counters["schedule_tsab_restart_attempts"]
    restart_rebases = counters["schedule_tsab_restart_global_rebases"]
    restart_generated = counters["schedule_tsab_restart_n6_generated"]
    restart_probes = counters["schedule_tsab_restart_delta_probes"]
    restart_commits = counters["schedule_tsab_restart_oracle_commits"]
    restart_rejections = counters["schedule_tsab_restart_rejections"]
    restart_interruptions = counters["schedule_tsab_restart_interruptions"]
    restart_work = counters["schedule_tsab_restart_work_units"]
    n6_kicks = counters["schedule_tsab_n6_kicks"]
    elite_restarts = counters["schedule_tsab_elite_restarts"]
    if restart_attempts == 0:
        failures.append("restart_activity")
    if restart_rebases == 0 or restart_rebases > elite_restarts:
        failures.append("restart_global_rebase")
    if restart_generated == 0 or restart_probes == 0 or restart_probes > restart_generated or restart_probes > 4 * restart_attempts:
        failures.append("restart_probe_accounting")
    if (
        restart_commits != n6_kicks
        or n6_kicks != counters["schedule_tsab_kick_moves"]
        or restart_commits > elite_restarts
        or elite_restarts > restart_attempts
    ):
        failures.append("restart_commit_accounting")
    if restart_commits + restart_rejections != restart_attempts:
        failures.append("restart_outcome_accounting")
    if restart_interruptions != 0:
        failures.append("restart_interruption")
    if restart_work == 0 or restart_work > burst_work:
        failures.append("restart_work_accounting")
    if counters["schedule_tsab_escape_signals"] != restart_attempts:
        failures.append("restart_escape_accounting")
    if restart_best_base_objective is None or (restart_best_kicked_objective is None) != (restart_commits == 0):
        failures.append("restart_objective_accounting")
    if counters["schedule_tsab_restart_shortlist_peak_bytes"] == 0:
        failures.append("restart_workspace")
    deferred = (
        "schedule_tsab_fingerprint_repeats",
        "schedule_tsab_n1_kicks",
        "schedule_tsab_ranking_audits",
        "schedule_tsab_exact_best_matches",
        "schedule_tsab_regret_sum",
        "schedule_tsab_regret_max",
    )
    if any(counters[field] != 0 for field in deferred):
        failures.append("deferred_tsab_activity")
    expected_masks = {
        "schedule_island_profile_mask": 0x7F,
        "schedule_baseline_island_profile_mask": 0x3F,
        "schedule_scored_island_profile_mask": 0x40,
    }
    if any(_integer(record.get(field), f"{label}.{field}") != expected for field, expected in expected_masks.items()):
        failures.append("portfolio_masks")
    if any(_integer(record.get(field), f"{label}.{field}") != 0 for field in ZERO_FIELDS):
        failures.append("integrity_counter")
    if not _replay_valid(record, label):
        failures.append("replay")
    if not _verification_timing_valid(record, label):
        failures.append("verification_timing")
    peak_memory = _number(record.get("peak_memory_mb"), f"{label}.peak_memory_mb")
    if peak_memory >= MAX_PEAK_MEMORY_MB:
        failures.append("memory")

    baseline_work = _profile_map(baseline.get("schedule_profile_work_steps"), f"{label}.baseline_profile_work")
    candidate_work = _profile_map(record.get("schedule_profile_work_steps"), f"{label}.candidate_profile_work")
    if set(baseline_work) != set(range(7)) or set(candidate_work) != set(range(7)):
        failures.append("profile_coverage")
        retention: dict[int, float] = {}
    else:
        retention = {
            profile: candidate_work[profile] / baseline_work[profile]
            for profile in range(6)
            if baseline_work[profile] > 0
        }
        if len(retention) != 6:
            failures.append("control_baseline_work")
        elif any(value < CONTROL_RETENTION for value in retention.values()):
            failures.append("control_work_retention")

    improving_commits = counters["schedule_tsab_improving_commits"]
    if improving_commits > commits + fast_commits or (improving_commits == 0) != (best_committed_objective is None):
        failures.append("improving_commit_accounting")
    post_restart_improvements = counters["schedule_tsab_post_restart_improvements"]
    if post_restart_improvements > improving_commits:
        failures.append("post_restart_improvement_accounting")
    search_signal = post_restart_improvements > 0 or counters["schedule_tsab_fast_pending_promotions"] > 0
    return failures, search_signal, retention


def evaluate(baseline_path: Path, candidate_path: Path) -> dict[str, Any]:
    baseline_provenance = _provenance(baseline_path, strategy="legacy")
    candidate_provenance = _provenance(candidate_path, strategy="tsab-candidate")
    for nested, field in (("host", "source_tree_sha256"), ("qayd_artifact", "sha256")):
        if baseline_provenance[nested][field] != candidate_provenance[nested][field]:
            raise GateError(f"baseline/candidate provenance mismatch: {nested}.{field}")

    baseline = _record_map(load_jsonl(baseline_path), "baseline")
    candidate = _record_map(load_jsonl(candidate_path), "candidate")
    if set(baseline) != set(candidate):
        raise GateError("baseline and candidate comparison keys differ")

    rows: list[dict[str, Any]] = []
    all_failures: list[str] = []
    for key in sorted(baseline):
        baseline_record, candidate_record = baseline[key], candidate[key]
        label = f"candidate[{key[1]}s,seed={key[2]}]"
        failures: list[str] = []
        if key[1] != 60:
            failures.append("checkpoint_not_60s")
        if not _legacy_clean(baseline_record, "baseline"):
            failures.append("baseline_tsab_activity")
        if not _replay_valid(baseline_record, "baseline"):
            failures.append("baseline_replay")
        if not _verification_timing_valid(baseline_record, "baseline"):
            failures.append("baseline_verification_timing")
        baseline_masks = {
            "schedule_island_profile_mask": 0x7F,
            "schedule_baseline_island_profile_mask": 0x3F,
            "schedule_scored_island_profile_mask": 0x40,
        }
        if any(
            _integer(baseline_record.get(field), f"baseline.{field}") != expected
            for field, expected in baseline_masks.items()
        ):
            failures.append("baseline_portfolio_masks")
        if any(_integer(baseline_record.get(field), f"baseline.{field}") != 0 for field in ZERO_FIELDS):
            failures.append("baseline_integrity_counter")
        if _number(baseline_record.get("peak_memory_mb"), "baseline.peak_memory_mb") >= MAX_PEAK_MEMORY_MB:
            failures.append("baseline_memory")
        candidate_failures, search_signal, retention = _candidate_checks(candidate_record, baseline_record, label)
        failures.extend(candidate_failures)
        baseline_objective = _objective(baseline_record, "baseline")
        candidate_objective = _objective(candidate_record, label)
        search_signal = search_signal and candidate_objective <= baseline_objective
        rows.append(
            {
                "instance_sha256": key[0],
                "checkpoint_seconds": key[1],
                "seed": key[2],
                "baseline_objective": baseline_objective,
                "candidate_objective": candidate_objective,
                "objective_delta": candidate_objective - baseline_objective,
                "control_work_retention": {str(profile): value for profile, value in sorted(retention.items())},
                "minimum_control_work_retention": min(retention.values(), default=0.0),
                "fast_attempts": candidate_record["schedule_tsab_fast_attempts"],
                "fast_transitions": candidate_record["schedule_tsab_fast_transitions"],
                "fast_work_units": candidate_record["schedule_tsab_fast_work_units"],
                "fast_fallbacks": candidate_record["schedule_tsab_fast_fallbacks"],
                "fast_full_validations": candidate_record["schedule_tsab_fast_full_validations"],
                "fast_elapsed_seconds": candidate_record["schedule_tsab_fast_elapsed_seconds"],
                "tsab_search_signal": search_signal,
                "failures": failures,
            }
        )
        all_failures.extend(failures)

    ready = not all_failures
    search_signal = ready and all(row["tsab_search_signal"] for row in rows)
    quality = search_signal and all(
        row["candidate_objective"] <= QUALITY_OBJECTIVE
        and row["candidate_objective"] <= row["baseline_objective"]
        for row in rows
    )
    return {
        "schema_version": 4,
        "gate": "jssp-tsab-phased-candidate",
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "tsab_ready": ready,
        "tsab_search_signal": search_signal,
        "tsab_quality_signal": quality,
        "quality_objective_limit": QUALITY_OBJECTIVE,
        "failure_count": len(all_failures),
        "failures": sorted(set(all_failures)),
        "rows": rows,
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# JSSP TSAB candidate gate",
        "",
        f"- Readiness: **{report['tsab_ready']}**",
        f"- Search signal: **{report['tsab_search_signal']}**",
        f"- Quality signal: **{report['tsab_quality_signal']}**",
        f"- Quality limit: **{report['quality_objective_limit']}**",
        f"- Failures: **{report['failure_count']}**",
        "",
        "| Checkpoint | Seed | Legacy | TSAB | Delta | Minimum control retention | Search signal | Failures |",
        "|---:|---:|---:|---:|---:|---:|:---:|:---|",
    ]
    for row in report["rows"]:
        failures = ", ".join(row["failures"]) or "none"
        lines.append(
            f"| {row['checkpoint_seconds']} | {row['seed']} | {row['baseline_objective']} | "
            f"{row['candidate_objective']} | {row['objective_delta']} | "
            f"{row['minimum_control_work_retention']:.3f} | {row['tsab_search_signal']} | {failures} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--require-readiness", action="store_true")
    parser.add_argument("--require-search-signal", action="store_true")
    parser.add_argument("--require-quality-signal", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = evaluate(args.baseline, args.candidate)
    except GateError as error:
        print(json.dumps({"error": str(error)}, sort_keys=True))
        return 1
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json:
        args.json.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    if args.markdown:
        args.markdown.write_text(markdown(report), encoding="utf-8")
    if args.require_readiness and not report["tsab_ready"]:
        return 2
    if args.require_search_signal and not report["tsab_search_signal"]:
        return 2
    if args.require_quality_signal and not report["tsab_quality_signal"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
