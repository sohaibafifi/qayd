#!/usr/bin/env python3
"""Report Qayd against the published Hexaly Large-TA 10-minute incumbents."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "bench" / "suites" / "hexaly-jssp-1000x1000.json"
DEFAULT_CHECKPOINT_SECONDS = 600.0
DEFAULT_SEED = 0
REQUIRED_MINIMUM_EXTERNAL_GRACE_SECONDS = 600.0
EXPECTED_INSTANCE_COUNT = 10
EXPECTED_JOBS = 1000
EXPECTED_MACHINES = 1000
EXPECTED_OPERATIONS = 1_000_000
EXPECTED_SCHEDULE_RESTART_WORK = 256
EXPECTED_VERIFICATION_COUNTS = {
    "starts": 1_000_000,
    "job_precedence_pairs": 999_000,
    "machine_non_overlap_pairs": 999_000,
    "objective_checks": 1,
}
EMPTY_VERIFICATION_COUNTS = {
    "starts": 0,
    "job_precedence_pairs": 0,
    "machine_non_overlap_pairs": 0,
    "objective_checks": 0,
}
EMPTY_START_VECTOR_SHA256 = (
    "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
)
FEASIBLE_STATUSES = {"SAT", "SATISFIABLE", "FEASIBLE", "OPTIMAL", "OPTIMUM"}


class ReportError(ValueError):
    """Raised when an input cannot support the strict comparison report."""


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ReportError(f"cannot read {label} {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ReportError(f"invalid JSON in {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ReportError(f"{label} {path} must contain a JSON object")
    return value


def load_campaign(path: Path) -> list[dict[str, Any]]:
    """Load a campaign JSONL and reject malformed or non-object records."""
    records: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ReportError(
                        f"{path}:{line_number}: invalid JSON: {error}"
                    ) from error
                if not isinstance(value, dict):
                    raise ReportError(
                        f"{path}:{line_number}: campaign record must be a JSON object"
                    )
                records.append(value)
    except OSError as error:
        raise ReportError(f"cannot read campaign {path}: {error}") from error
    return records


def _plain_filename(value: str) -> str:
    return value.replace("\\", "/").rsplit("/", 1)[-1]


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _valid_git_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) in {40, 64}
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _positive_number(value: object, label: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ReportError(f"{label} must be positive and finite")
    return value


def _nonnegative_number(value: object, label: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ReportError(f"{label} must be non-negative and finite")
    return value


def _manifest_instances(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    minimum_grace = manifest.get("minimum_external_grace_seconds")
    if (
        isinstance(minimum_grace, bool)
        or not isinstance(minimum_grace, (int, float))
        or not math.isfinite(float(minimum_grace))
        or float(minimum_grace) != REQUIRED_MINIMUM_EXTERNAL_GRACE_SECONDS
    ):
        raise ReportError(
            "manifest minimum_external_grace_seconds must be exactly "
            f"{REQUIRED_MINIMUM_EXTERNAL_GRACE_SECONDS:g}"
        )
    expected = manifest.get("expected_instances")
    if not isinstance(expected, dict) or len(expected) != EXPECTED_INSTANCE_COUNT:
        raise ReportError(
            "manifest expected_instances must contain exactly "
            f"{EXPECTED_INSTANCE_COUNT} entries"
        )

    try:
        references = manifest["reference_protocol"]["hexaly"][
            "reference_incumbent_10m"
        ]
    except (KeyError, TypeError) as error:
        raise ReportError(
            "manifest is missing reference_protocol.hexaly.reference_incumbent_10m"
        ) from error
    if not isinstance(references, dict):
        raise ReportError("reference_incumbent_10m must be a JSON object")

    reference_by_stem: dict[str, int | float] = {}
    for raw_name, raw_value in references.items():
        if not isinstance(raw_name, str) or not raw_name:
            raise ReportError("reference incumbent names must be non-empty strings")
        stem = Path(_plain_filename(raw_name)).stem
        if stem in reference_by_stem:
            raise ReportError(f"duplicate Hexaly reference incumbent for {stem}")
        reference_by_stem[stem] = _positive_number(
            raw_value, f"Hexaly reference incumbent for {stem}"
        )

    instances = []
    seen_filenames = set()
    for raw_filename, raw_digest in expected.items():
        if not isinstance(raw_filename, str) or not raw_filename:
            raise ReportError("expected instance names must be non-empty strings")
        filename = _plain_filename(raw_filename)
        if filename in seen_filenames:
            raise ReportError(f"duplicate expected instance filename: {filename}")
        seen_filenames.add(filename)
        if not _valid_sha256(raw_digest):
            raise ReportError(f"invalid SHA-256 in manifest for {filename}")
        stem = Path(filename).stem
        if stem not in reference_by_stem:
            raise ReportError(f"missing Hexaly reference incumbent for {stem}")
        instances.append(
            {
                "instance": stem,
                "instance_file": filename,
                "instance_sha256": raw_digest.lower(),
                "hexaly_reference_incumbent_10m": reference_by_stem[stem],
            }
        )

    expected_stems = {instance["instance"] for instance in instances}
    extra_references = sorted(set(reference_by_stem) - expected_stems)
    if extra_references:
        raise ReportError(
            "Hexaly reference incumbents contain unexpected instances: "
            + ", ".join(extra_references)
        )
    return instances


def campaign_provenance_path(campaign_path: Path) -> Path:
    return campaign_path.with_suffix(campaign_path.suffix + ".provenance.json")


def _single_number_list(
    value: object, expected: int | float, label: str,
) -> list[int | float]:
    if not isinstance(value, list) or len(value) != 1:
        raise ReportError(f"campaign provenance {label} must contain exactly {expected:g}")
    actual = value[0]
    if (
        isinstance(actual, bool)
        or not isinstance(actual, (int, float))
        or not math.isfinite(float(actual))
        or float(actual) != float(expected)
    ):
        raise ReportError(f"campaign provenance {label} must contain exactly {expected:g}")
    return list(value)


def _validate_campaign_provenance(
    provenance_path: Path,
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    checkpoint_seconds: float,
    seed: int,
) -> dict[str, Any]:
    if not provenance_path.is_file():
        raise ReportError(f"campaign provenance sidecar is required: {provenance_path}")
    provenance = _read_json(provenance_path, "campaign provenance sidecar")
    if provenance.get("schema_version") != 1:
        raise ReportError("campaign provenance schema_version must be exactly 1")
    if provenance.get("suite") != manifest:
        raise ReportError("campaign provenance suite content does not match the manifest")
    raw_suite_file = provenance.get("suite_file")
    if not isinstance(raw_suite_file, str) or not raw_suite_file:
        raise ReportError("campaign provenance suite_file must identify the manifest")
    if Path(raw_suite_file).expanduser().resolve() != manifest_path.resolve():
        raise ReportError("campaign provenance suite_file does not identify the manifest")

    budgets = _single_number_list(
        provenance.get("budgets"), checkpoint_seconds, "budgets"
    )
    seeds = _single_number_list(provenance.get("seeds"), seed, "seeds")
    threads = provenance.get("threads")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0:
        raise ReportError("campaign provenance threads must be a positive integer")
    grace_seconds = _nonnegative_number(
        provenance.get("grace_seconds"), "campaign provenance grace_seconds"
    )
    minimum_grace = float(manifest["minimum_external_grace_seconds"])
    if float(grace_seconds) < minimum_grace:
        raise ReportError(
            "campaign provenance grace_seconds is below the manifest minimum "
            f"of {minimum_grace:g}"
        )
    engine = provenance.get("qayd_engine")
    if engine not in {"auto", "exact", "ls"}:
        raise ReportError("campaign provenance qayd_engine must identify a Qayd engine")
    if provenance.get("qayd_prepared") is not True:
        raise ReportError("campaign provenance qayd_prepared must be true")

    artifact = provenance.get("qayd_artifact")
    if not isinstance(artifact, dict):
        raise ReportError("campaign provenance must identify the Qayd extension artifact")
    artifact_path = artifact.get("path")
    if (
        not isinstance(artifact_path, str)
        or not artifact_path
        or not Path(artifact_path).is_absolute()
        or not _valid_sha256(artifact.get("sha256"))
    ):
        raise ReportError(
            "campaign provenance Qayd artifact must have an absolute path and valid SHA-256"
        )

    host = provenance.get("host")
    if (
        not isinstance(host, dict)
        or not _valid_git_commit(host.get("commit"))
        or not _valid_sha256(host.get("source_tree_sha256"))
    ):
        raise ReportError(
            "campaign provenance host must identify the commit and source tree SHA-256"
        )
    solvers = provenance.get("solvers")
    if not isinstance(solvers, dict) or not solvers:
        raise ReportError("campaign provenance must identify solver versions")
    if any(not isinstance(value, str) or not value.strip() for value in solvers.values()):
        raise ReportError("campaign provenance solver versions must be non-empty strings")

    return {
        "path": str(provenance_path.resolve()),
        "suite_file": str(manifest_path.resolve()),
        "budgets": budgets,
        "seeds": seeds,
        "threads": threads,
        "grace_seconds": grace_seconds,
        "qayd_engine": engine,
        "qayd_prepared": True,
        "qayd_artifact": {
            "path": artifact_path,
            "sha256": str(artifact["sha256"]).lower(),
        },
        "host": {
            "commit": host["commit"],
            "source_tree_sha256": str(host["source_tree_sha256"]).lower(),
        },
        "solvers": dict(solvers),
    }


def _is_qayd(record: dict[str, Any]) -> bool:
    solver = record.get("solver")
    return isinstance(solver, str) and (solver == "qayd" or solver.startswith("qayd-"))


def _matches_checkpoint(record: dict[str, Any], checkpoint_seconds: float) -> bool:
    value = record.get("checkpoint_seconds")
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) == checkpoint_seconds
    )


def _matches_seed(record: dict[str, Any], selected_seed: int) -> bool:
    value = record.get("seed")
    return isinstance(value, int) and not isinstance(value, bool) and value == selected_seed


def _record_instance_file(
    record: dict[str, Any], expected_by_name: dict[str, dict[str, Any]],
    expected_by_stem: dict[str, dict[str, Any]],
) -> str | None:
    for field in ("instance_path", "instance"):
        raw_value = record.get(field)
        if not isinstance(raw_value, str) or not raw_value:
            continue
        name = _plain_filename(raw_value)
        if name in expected_by_name:
            return name
        stem = Path(name).stem
        if stem in expected_by_stem:
            return expected_by_stem[stem]["instance_file"]
    return None


def _scalar_makespan(record: dict[str, Any], label: str) -> int | float:
    objectives = record.get("objectives")
    if not isinstance(objectives, list) or len(objectives) != 1:
        raise ReportError(f"{label}: objectives must contain one scalar makespan")
    return _positive_number(objectives[0], f"{label}: makespan")


def _exact_integer(record: dict[str, Any], field: str, expected: int, label: str) -> int:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ReportError(f"{label}: {field} must be exactly {expected}")
    return value


def _verification_counts(record: dict[str, Any], label: str) -> dict[str, int]:
    counts = record.get("verification_counts")
    if not isinstance(counts, dict) or set(counts) != set(EXPECTED_VERIFICATION_COUNTS):
        raise ReportError(f"{label}: verification_counts must contain exactly the replay counters")
    for field, expected in EXPECTED_VERIFICATION_COUNTS.items():
        value = counts.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            raise ReportError(
                f"{label}: verification_counts.{field} must be exactly {expected}"
            )
    return dict(EXPECTED_VERIFICATION_COUNTS)


def _command_option(command: list[str], flag: str, label: str) -> str:
    positions = [index for index, value in enumerate(command) if value == flag]
    if len(positions) != 1:
        raise ReportError(f"{label}: command must contain {flag} exactly once")
    position = positions[0]
    if position + 1 >= len(command) or command[position + 1].startswith("--"):
        raise ReportError(f"{label}: command {flag} has no value")
    return command[position + 1]


def _command_number(value: str, flag: str, label: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise ReportError(f"{label}: command {flag} must be numeric") from error
    if not math.isfinite(number):
        raise ReportError(f"{label}: command {flag} must be finite")
    return number


def _validate_command(
    record: dict[str, Any],
    instance: dict[str, Any],
    *,
    checkpoint_seconds: float,
    seed: int,
    threads: int,
    engine: str,
) -> list[str]:
    label = instance["instance_file"]
    raw_command = record.get("command")
    if (
        not isinstance(raw_command, list)
        or not raw_command
        or any(not isinstance(value, str) or not value for value in raw_command)
    ):
        raise ReportError(f"{label}: command must be a non-empty string array")
    command = list(raw_command)
    if command.count("--json") != 1:
        raise ReportError(f"{label}: command must request JSON output exactly once")
    if command.count("--compact-json") != 1:
        raise ReportError(f"{label}: command must request compact mode exactly once")
    if not any(_plain_filename(value) == label for value in command):
        raise ReportError(f"{label}: command does not identify the selected instance")

    command_checkpoint = _command_number(
        _command_option(command, "--time-limit", label), "--time-limit", label
    )
    if command_checkpoint != checkpoint_seconds:
        raise ReportError(
            f"{label}: command --time-limit does not match checkpoint_seconds"
        )
    command_seed = _command_number(
        _command_option(command, "--seed", label), "--seed", label
    )
    if command_seed != seed:
        raise ReportError(f"{label}: command --seed does not match seed")
    command_threads = _command_number(
        _command_option(command, "--threads", label), "--threads", label
    )
    if command_threads != threads:
        raise ReportError(f"{label}: command --threads does not match threads")
    if _command_option(command, "--engine", label) != engine:
        raise ReportError(f"{label}: command --engine does not match engine")
    return command


def _uniform_protocol(
    records: Sequence[dict[str, Any]], provenance: dict[str, Any],
) -> dict[str, Any]:
    string_fields = {
        "solver": "solver variant",
        "solver_version": "solver version",
    }
    protocol: dict[str, Any] = {}
    for field, description in string_fields.items():
        values = []
        for record in records:
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ReportError(f"selected Qayd record has no valid {description}")
            values.append(value)
        unique = set(values)
        if len(unique) != 1:
            raise ReportError(
                f"selected Qayd records do not use one uniform {description}: "
                + ", ".join(sorted(unique))
            )
        protocol[field] = values[0]

    engines = []
    for record in records:
        value = record.get("engine")
        if not isinstance(value, str) or not value.strip():
            raw_status = record.get("status")
            if isinstance(raw_status, str) and raw_status.upper() in FEASIBLE_STATUSES:
                raise ReportError("selected feasible Qayd record has no valid engine")
            value = provenance["qayd_engine"]
        engines.append(value)
    unique_engines = set(engines)
    if len(unique_engines) != 1:
        raise ReportError(
            "selected Qayd records do not use one uniform engine: "
            + ", ".join(sorted(unique_engines))
        )
    protocol["engine"] = engines[0]

    for field in ("threads", "requested_threads"):
        values = []
        for record in records:
            value = record.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ReportError(f"selected Qayd record has no valid {field}")
            values.append(value)
        unique = set(values)
        if len(unique) != 1:
            raise ReportError(
                f"selected Qayd records do not use one uniform {field}: "
                + ", ".join(map(str, sorted(unique)))
            )
        protocol[field] = values[0]

    restart_work_values = []
    for record in records:
        value = record.get("schedule_restart_work")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ReportError(
                "selected Qayd record has no valid schedule_restart_work"
            )
        restart_work_values.append(value)
    unique_restart_work = set(restart_work_values)
    if len(unique_restart_work) != 1:
        raise ReportError(
            "selected Qayd records do not use one uniform schedule_restart_work: "
            + ", ".join(map(str, sorted(unique_restart_work)))
        )
    restart_work = restart_work_values[0]
    if restart_work != EXPECTED_SCHEDULE_RESTART_WORK:
        raise ReportError(
            "selected Qayd schedule_restart_work must be exactly "
            f"{EXPECTED_SCHEDULE_RESTART_WORK}, got {restart_work}"
        )
    protocol["schedule_restart_work"] = restart_work
    return protocol


def _validate_protocol_provenance(
    protocol: dict[str, Any], provenance: dict[str, Any],
) -> None:
    solver = protocol["solver"]
    solver_version = provenance["solvers"].get(solver)
    if not isinstance(solver_version, str) or not solver_version:
        raise ReportError(
            f"campaign provenance does not identify solver version for {solver}"
        )
    if protocol["solver_version"] != solver_version:
        raise ReportError(
            "selected Qayd solver_version does not match campaign provenance"
        )
    if protocol["engine"] != provenance["qayd_engine"]:
        raise ReportError("selected Qayd engine does not match campaign provenance")
    if (
        protocol["threads"] != provenance["threads"]
        or protocol["requested_threads"] != provenance["threads"]
    ):
        raise ReportError("selected Qayd threads do not match campaign provenance")


def _validated_row(
    record: dict[str, Any],
    instance: dict[str, Any],
    *,
    checkpoint_seconds: float,
    seed: int,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    label = instance["instance_file"]
    if record.get("problem") != "jssp":
        raise ReportError(f"{label}: problem must be jssp")
    digest = record.get("instance_sha256")
    if not isinstance(digest, str) or digest.lower() != instance["instance_sha256"]:
        raise ReportError(f"{label}: instance SHA-256 does not match the manifest")
    return_code = record.get("return_code")
    if isinstance(return_code, bool) or not isinstance(return_code, int):
        raise ReportError(f"{label}: return_code must be an integer")
    timed_out = record.get("timed_out")
    if not isinstance(timed_out, bool):
        raise ReportError(f"{label}: timed_out must be a boolean")
    raw_status = record.get("status")
    if not isinstance(raw_status, str) or not raw_status.strip():
        raise ReportError(f"{label}: status must be a non-empty string")
    status = raw_status.upper()
    feasible = status in FEASIBLE_STATUSES
    solve_elapsed = _nonnegative_number(
        record.get("elapsed_seconds"), f"{label}: elapsed_seconds"
    )
    if record.get("elapsed_seconds_scope") != "model.solve":
        raise ReportError(f"{label}: elapsed_seconds_scope must be model.solve")
    solve_seconds = _nonnegative_number(
        record.get("solve_seconds"), f"{label}: solve_seconds"
    )
    if not math.isclose(
        float(solve_elapsed), float(solve_seconds), rel_tol=0.0, abs_tol=1e-9
    ):
        raise ReportError(f"{label}: solve_seconds must match elapsed_seconds")
    wall_elapsed = _nonnegative_number(
        record.get("wall_seconds"), f"{label}: wall_seconds"
    )
    peak_rss = _nonnegative_number(
        record.get("peak_memory_mb"), f"{label}: peak_memory_mb"
    )
    grace_seconds = _nonnegative_number(
        record.get("grace_seconds"), f"{label}: grace_seconds"
    )
    if float(grace_seconds) != float(provenance["grace_seconds"]):
        raise ReportError(f"{label}: grace_seconds does not match campaign provenance")
    external_timeout_seconds = _positive_number(
        record.get("external_timeout_seconds"),
        f"{label}: external_timeout_seconds",
    )
    if float(external_timeout_seconds) != checkpoint_seconds + float(grace_seconds):
        raise ReportError(
            f"{label}: external_timeout_seconds must equal checkpoint plus grace"
        )
    command = _validate_command(
        record,
        instance,
        checkpoint_seconds=checkpoint_seconds,
        seed=seed,
        threads=record["threads"],
        engine=provenance["qayd_engine"],
    )
    reference = instance["hexaly_reference_incumbent_10m"]
    if feasible:
        if record.get("verified") is not True:
            raise ReportError(f"{label}: feasible record verified must be true")
        if return_code != 0:
            raise ReportError(f"{label}: feasible record return_code must be exactly 0")
        if timed_out:
            raise ReportError(f"{label}: feasible record timed_out must be false")
        makespan = _scalar_makespan(record, label)
        jobs = _exact_integer(record, "jobs", EXPECTED_JOBS, label)
        machines = _exact_integer(record, "machines", EXPECTED_MACHINES, label)
        operations = _exact_integer(record, "operations", EXPECTED_OPERATIONS, label)
        commitment_length = _exact_integer(
            record, "start_vector_length", EXPECTED_OPERATIONS, label
        )
        commitment_sha256 = record.get("start_vector_sha256")
        if not _valid_sha256(commitment_sha256):
            raise ReportError(f"{label}: start_vector_sha256 must be a valid SHA-256")
        verification_counts = _verification_counts(record, label)
        gap_percent: float | None = (
            100.0 * (float(makespan) - float(reference)) / float(reference)
        )
    else:
        if record.get("verified") is not False and record.get("verified") is not None:
            raise ReportError(f"{label}: non-feasible record verified must be false")
        if record.get("objectives") != []:
            raise ReportError(f"{label}: non-feasible record objectives must be empty")
        commitment_length = _exact_integer(record, "start_vector_length", 0, label)
        commitment_sha256 = record.get("start_vector_sha256")
        if (
            not isinstance(commitment_sha256, str)
            or commitment_sha256.lower() != EMPTY_START_VECTOR_SHA256
        ):
            raise ReportError(
                f"{label}: empty start-vector SHA-256 commitment is inconsistent"
            )
        counts = record.get("verification_counts")
        if counts != EMPTY_VERIFICATION_COUNTS:
            raise ReportError(
                f"{label}: non-feasible verification_counts must all be zero"
            )
        verification_counts = dict(EMPTY_VERIFICATION_COUNTS)
        makespan = None
        jobs = None
        machines = None
        operations = None
        gap_percent = None
    return {
        "instance": instance["instance"],
        "instance_file": instance["instance_file"],
        "instance_path": record.get("instance_path"),
        "instance_sha256": instance["instance_sha256"],
        "solver": record.get("solver"),
        "solver_version": record.get("solver_version"),
        "engine": record.get("engine") or provenance["qayd_engine"],
        "threads": record.get("threads"),
        "requested_threads": record.get("requested_threads"),
        "status": status,
        "return_code": return_code,
        "timed_out": timed_out,
        "feasible": feasible,
        "qayd_makespan": makespan,
        "hexaly_reference_incumbent_10m": reference,
        "qayd_vs_hexaly_percent": gap_percent,
        "solve_elapsed_seconds": solve_elapsed,
        "elapsed_seconds_scope": "model.solve",
        "solve_seconds": solve_seconds,
        "end_to_end_wall_seconds": wall_elapsed,
        "grace_seconds": grace_seconds,
        "external_timeout_seconds": external_timeout_seconds,
        "peak_rss_mb": peak_rss,
        "jobs": jobs,
        "machines": machines,
        "operations": operations,
        "start_vector_length": commitment_length,
        "start_vector_sha256": str(commitment_sha256).lower(),
        "verification_counts": verification_counts,
        "command": command,
    }


def build_summary(
    campaign_path: Path,
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    provenance_path: Path | None = None,
    checkpoint_seconds: float = DEFAULT_CHECKPOINT_SECONDS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Validate one complete checkpoint and build the JSON report payload."""
    if not math.isfinite(checkpoint_seconds) or checkpoint_seconds <= 0:
        raise ReportError("checkpoint_seconds must be positive and finite")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ReportError("seed must be a non-negative integer")

    manifest = _read_json(manifest_path, "manifest")
    expected = _manifest_instances(manifest)
    resolved_provenance_path = (
        provenance_path
        if provenance_path is not None
        else campaign_provenance_path(campaign_path)
    )
    provenance = _validate_campaign_provenance(
        resolved_provenance_path,
        manifest=manifest,
        manifest_path=manifest_path,
        checkpoint_seconds=checkpoint_seconds,
        seed=seed,
    )
    records = load_campaign(campaign_path)
    expected_by_name = {item["instance_file"]: item for item in expected}
    expected_by_stem = {item["instance"]: item for item in expected}
    selected: dict[str, list[dict[str, Any]]] = {
        name: [] for name in expected_by_name
    }
    for record in records:
        if (
            not _is_qayd(record)
            or not _matches_checkpoint(record, checkpoint_seconds)
            or not _matches_seed(record, seed)
        ):
            continue
        filename = _record_instance_file(record, expected_by_name, expected_by_stem)
        if filename is not None:
            selected[filename].append(record)

    coverage_errors = []
    for filename, matches in selected.items():
        if not matches:
            coverage_errors.append(
                f"missing Qayd record for {filename} at checkpoint "
                f"{checkpoint_seconds:g}, seed {seed}"
            )
        elif len(matches) > 1:
            coverage_errors.append(
                f"duplicate Qayd records for {filename} at checkpoint "
                f"{checkpoint_seconds:g}, seed {seed}: {len(matches)} records"
            )
    if coverage_errors:
        raise ReportError("; ".join(coverage_errors))

    selected_records = [selected[item["instance_file"]][0] for item in expected]
    qayd_protocol = _uniform_protocol(selected_records, provenance)
    _validate_protocol_provenance(qayd_protocol, provenance)
    rows = [
        _validated_row(
            record,
            item,
            checkpoint_seconds=checkpoint_seconds,
            seed=seed,
            provenance=provenance,
        )
        for record, item in zip(selected_records, expected)
    ]
    feasible_rows = [row for row in rows if row["feasible"]]
    gaps = [float(row["qayd_vs_hexaly_percent"]) for row in feasible_rows]
    status_counts = Counter(str(row["status"]) for row in rows)
    claim_ready = len(feasible_rows) == len(rows)
    aggregate = {
        "instance_count": len(rows),
        "selected_record_count": len(rows),
        "record_coverage_percent": 100.0 * len(rows) / EXPECTED_INSTANCE_COUNT,
        "feasible_count": len(feasible_rows),
        "verified_feasible_count": len(feasible_rows),
        "incumbent_coverage_percent": 100.0 * len(feasible_rows) / len(rows),
        "claim_ready": claim_ready,
        "qayd_better_count": sum(
            row["qayd_makespan"] < row["hexaly_reference_incumbent_10m"]
            for row in feasible_rows
        ),
        "equal_count": sum(
            row["qayd_makespan"] == row["hexaly_reference_incumbent_10m"]
            for row in feasible_rows
        ),
        "qayd_worse_count": sum(
            row["qayd_makespan"] > row["hexaly_reference_incumbent_10m"]
            for row in feasible_rows
        ),
        "status_counts": dict(sorted(status_counts.items())),
        "median_qayd_vs_hexaly_percent": statistics.median(gaps) if gaps else None,
        "mean_qayd_vs_hexaly_percent": statistics.fmean(gaps) if gaps else None,
    }
    hexaly = manifest.get("reference_protocol", {}).get("hexaly", {})
    return {
        "schema_version": 1,
        "suite": manifest.get("name", "hexaly-jssp-1000x1000"),
        "manifest_path": str(manifest_path.resolve()),
        "campaign_path": str(campaign_path.resolve()),
        "campaign_provenance": provenance,
        "selection": {
            "checkpoint_seconds": checkpoint_seconds,
            "seed": seed,
        },
        "qayd_protocol": {
            **qayd_protocol,
            "instance_shape": {
                "jobs": EXPECTED_JOBS,
                "machines": EXPECTED_MACHINES,
                "operations": EXPECTED_OPERATIONS,
            },
            "compact_schedule_commitment": {
                "start_vector_length": EXPECTED_OPERATIONS,
                "start_vector_sha256": "required",
                "role": "commitment only, not a feasibility certificate",
                "post_hoc_replayable": False,
            },
            "schedule_validation": {
                "online_replay_by_launcher": True,
                "post_hoc_replay_from_digest": False,
            },
            "verification_counts": dict(EXPECTED_VERIFICATION_COUNTS),
        },
        "reference": {
            "solver": "Hexaly",
            "version": hexaly.get("version"),
            "time_limit_seconds": hexaly.get("time_limit_seconds", 600),
            "kind": "fixed published reference incumbent",
        },
        "gap_definition": (
            "100 * (Qayd makespan - Hexaly reference incumbent) / "
            "Hexaly reference incumbent; positive means Qayd is worse"
        ),
        "aggregate": aggregate,
        "claim_ready": claim_ready,
        "instances": rows,
    }


def _display_number(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def markdown_report(summary: dict[str, Any]) -> str:
    """Render a compact per-instance comparison table and aggregates."""
    selection = summary["selection"]
    aggregate = summary["aggregate"]
    protocol = summary["qayd_protocol"]
    reference = summary["reference"]
    checkpoint = _display_number(selection["checkpoint_seconds"])
    version = reference.get("version") or "version recorded in the manifest"
    reference_seconds = _display_number(reference["time_limit_seconds"])
    lines = [
        f"# Qayd Large-TA comparison at {checkpoint} seconds",
        "",
        (
            f"The Hexaly {version} values are fixed published reference incumbents "
            f"from {reference_seconds}-second runs. This report treats them only as "
            "external reference incumbents."
        ),
        "",
        (
            "Signed gap is `100 * (Qayd - Hexaly) / Hexaly`. "
            "Positive values mean Qayd is worse."
        ),
        "",
        "## Qualified Qayd protocol",
        "",
        f"- Solver variant: `{protocol['solver']}`",
        f"- Solver version: `{protocol['solver_version']}`",
        f"- Engine: `{protocol['engine']}`",
        (
            f"- Threads: {protocol['threads']} effective, "
            f"{protocol['requested_threads']} requested"
        ),
        (
            "- Schedule restart work: "
            f"{protocol['schedule_restart_work']} local-search moves per worker boundary"
        ),
        (
            f"- Selection: checkpoint {checkpoint} seconds, seed "
            f"{selection['seed']}"
        ),
        (
            "- Instance shape: "
            f"{protocol['instance_shape']['jobs']} jobs, "
            f"{protocol['instance_shape']['machines']} machines, "
            f"{protocol['instance_shape']['operations']} operations"
        ),
        (
            "- Per-instance schedule commitment: "
            f"{protocol['compact_schedule_commitment']['start_vector_length']} starts "
            "with a valid SHA-256. The digest is a commitment only, not a "
            "feasibility certificate"
        ),
        (
            "- Online launcher replay for feasible rows: "
            f"{protocol['verification_counts']['starts']} starts, "
            f"{protocol['verification_counts']['job_precedence_pairs']} job pairs, "
            f"{protocol['verification_counts']['machine_non_overlap_pairs']} machine pairs, "
            f"{protocol['verification_counts']['objective_checks']} objective check"
        ),
        (
            "- Replay limitation: the launcher replayed each feasible schedule online, "
            "but the stored digest is not post-hoc replayable"
        ),
        (
            f"- External guard: {summary['campaign_provenance']['grace_seconds']:g} "
            "seconds beyond the internal solve checkpoint"
        ),
        "",
        "## Per-instance results",
        "",
        (
            "| Instance | Qayd makespan | Hexaly reference incumbent | "
            "Qayd vs Hexaly | Solve elapsed | End-to-end wall | Peak RSS | Status |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["instances"]:
        gap = (
            f"{row['qayd_vs_hexaly_percent']:+.3f}%"
            if row["qayd_vs_hexaly_percent"] is not None
            else "n/a"
        )
        lines.append(
            f"| `{row['instance']}` | {_display_number(row['qayd_makespan'])} | "
            f"{_display_number(row['hexaly_reference_incumbent_10m'])} | "
            f"{gap} | "
            f"{float(row['solve_elapsed_seconds']):.3f}s | "
            f"{float(row['end_to_end_wall_seconds']):.3f}s | "
            f"{float(row['peak_rss_mb']):.1f} MB | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            (
                f"- Selected record coverage: {aggregate['selected_record_count']}/"
                f"{aggregate['instance_count']} ({aggregate['record_coverage_percent']:.1f}%)"
            ),
            (
                f"- Verified feasible instances: {aggregate['verified_feasible_count']}/"
                f"{aggregate['instance_count']} "
                f"({aggregate['incumbent_coverage_percent']:.1f}%)"
            ),
            f"- Claim ready: {'yes' if aggregate['claim_ready'] else 'no'}",
            (
                "- Median signed Qayd-vs-Hexaly gap: "
                + (
                    f"{aggregate['median_qayd_vs_hexaly_percent']:+.3f}%"
                    if aggregate["median_qayd_vs_hexaly_percent"] is not None
                    else "n/a"
                )
            ),
            (
                "- Mean signed Qayd-vs-Hexaly gap: "
                + (
                    f"{aggregate['mean_qayd_vs_hexaly_percent']:+.3f}%"
                    if aggregate["mean_qayd_vs_hexaly_percent"] is not None
                    else "n/a"
                )
            ),
            (
                "- Counts: Qayd better "
                f"{aggregate['qayd_better_count']}, equal {aggregate['equal_count']}, "
                f"Qayd worse {aggregate['qayd_worse_count']}"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    summary: dict[str, Any], *, markdown_path: Path | None, json_path: Path | None,
) -> None:
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(markdown_report(summary), encoding="utf-8")
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path, help="Qayd campaign JSONL")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--provenance",
        type=Path,
        help="campaign provenance sidecar, default: CAMPAIGN.provenance.json",
    )
    parser.add_argument(
        "--checkpoint-seconds", "--checkpoint", dest="checkpoint_seconds",
        type=float, default=DEFAULT_CHECKPOINT_SECONDS,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--json", dest="json_output", type=Path)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="fail unless all ten selected rows have verified feasible incumbents",
    )
    args = parser.parse_args(argv)
    if args.markdown is None and args.json_output is None:
        parser.error("at least one of --markdown or --json is required")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = build_summary(
            args.campaign,
            manifest_path=args.manifest,
            provenance_path=args.provenance,
            checkpoint_seconds=args.checkpoint_seconds,
            seed=args.seed,
        )
        if args.require_complete and not summary["claim_ready"]:
            raise ReportError(
                "complete incumbent coverage is required but one or more rows are non-feasible"
            )
        write_outputs(
            summary, markdown_path=args.markdown, json_path=args.json_output
        )
    except ReportError as error:
        print(f"large-ta-report: {error}", file=__import__("sys").stderr)
        return 2
    outputs = [str(path) for path in (args.markdown, args.json_output) if path]
    print(f"large-ta-report: wrote {', '.join(outputs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
