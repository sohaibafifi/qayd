#!/usr/bin/env python3
"""Deterministic correctness and repeated timing harness for parallel CP search."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


MANIFEST_SCHEMA = "qayd.parallel.manifest/v1"
CORRECTNESS_SCHEMA = "qayd.parallel.correctness-result/v1"
TIMING_SCHEMA = "qayd.parallel.timing-report/v1"
PAIRING_SCHEMA = "qayd.parallel.paired-capture/v1"
COMPARISON_SCHEMA = "qayd.parallel.timing-comparison/v1"
DEFAULT_TOLERANCE = 0.10
PINNED_PRE_REFACTOR_REVISION = "6c09b9fddc2738f584e53d464d7a76b98cab4d6c"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path(__file__).with_name("manifest.v1.json")
KNOWN_STATUSES = {"SATISFIABLE", "UNSATISFIABLE", "OPTIMAL", "UNKNOWN"}
PAIRING_KEY_FIELDS = (
    "schema",
    "capture_nonce_ns",
    "manifest_sha256",
    "harness_sha256",
    "baseline_binary_sha256",
    "candidate_binary_sha256",
    "repetitions",
    "warmups",
    "schedule",
)


class HarnessError(Exception):
    """A manifest, execution, parsing, validation, or comparison error."""


@dataclass(frozen=True)
class CommandOutcome:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HarnessError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def repo_path(relative: str) -> Path:
    path = (REPO_ROOT / relative).resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise HarnessError("path escapes repository: {}".format(relative)) from error
    return path


def utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def effective_timeout(request: dict[str, Any], timeout_override: int | None) -> int:
    """Resolve and validate the timeout used for one solver invocation."""
    if timeout_override is not None:
        require(
            isinstance(timeout_override, int) and not isinstance(timeout_override, bool) and timeout_override > 0,
            "timeout override must be a positive integer",
        )
        return timeout_override
    timeout = request.get("timeout_seconds")
    require(
        isinstance(timeout, int) and not isinstance(timeout, bool) and timeout > 0,
        "scenario timeout must be a positive integer",
    )
    return timeout


def detect_revision() -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPO_ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise HarnessError("cannot determine source revision, pass --revision explicitly") from error
    return commit + ("+dirty" if dirty else "")


def _number(value: Any, context: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), "{} must be numeric".format(context))
    return float(value)


def _option_value(args: Sequence[str], option: str, context: str) -> str:
    indexes = [index for index, token in enumerate(args) if token == option]
    require(len(indexes) == 1, "{} must contain {} exactly once".format(context, option))
    index = indexes[0]
    require(index + 1 < len(args), "{} has no value for {}".format(context, option))
    return args[index + 1]


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HarnessError("cannot read manifest {}: {}".format(path, error)) from error
    require(isinstance(manifest, dict), "manifest must be a JSON object")
    require(manifest.get("schema") == MANIFEST_SCHEMA, "unsupported manifest schema")
    require(isinstance(manifest.get("corpus_version"), str), "manifest needs corpus_version")
    require(
        isinstance(manifest.get("pre_refactor_revision"), str)
        and re.fullmatch(r"[0-9a-f]{40}", manifest["pre_refactor_revision"]) is not None,
        "manifest needs a full pre_refactor_revision",
    )
    require(
        manifest["pre_refactor_revision"] == PINNED_PRE_REFACTOR_REVISION,
        "manifest pre_refactor_revision differs from the reviewed baseline",
    )
    tolerance = _number(manifest.get("default_tolerance"), "default_tolerance")
    require(0.0 <= tolerance <= 1.0, "default_tolerance must be between zero and one")

    fixtures = manifest.get("fixtures")
    require(isinstance(fixtures, list) and fixtures, "manifest fixtures must be a non-empty list")
    fixture_ids: set[str] = set()
    for fixture in fixtures:
        require(isinstance(fixture, dict), "fixture entries must be objects")
        fixture_id = fixture.get("id")
        require(isinstance(fixture_id, str) and fixture_id, "fixture needs an id")
        require(fixture_id not in fixture_ids, "duplicate fixture id {}".format(fixture_id))
        fixture_ids.add(fixture_id)
        relative = fixture.get("path")
        expected_hash = fixture.get("sha256")
        require(isinstance(relative, str), "fixture {} needs a path".format(fixture_id))
        require(isinstance(expected_hash, str) and re.fullmatch(r"[0-9a-f]{64}", expected_hash) is not None,
                "fixture {} needs a lowercase SHA-256".format(fixture_id))
        instance = repo_path(relative)
        require(instance.is_file(), "fixture {} is missing: {}".format(fixture_id, relative))
        observed_hash = sha256_file(instance)
        require(observed_hash == expected_hash,
                "fixture {} hash mismatch: expected {}, got {}".format(fixture_id, expected_hash, observed_hash))
        require(isinstance(fixture.get("oracle"), dict), "fixture {} needs an oracle".format(fixture_id))

    scenarios = manifest.get("scenarios")
    require(isinstance(scenarios, list) and scenarios, "manifest scenarios must be a non-empty list")
    scenario_ids: set[str] = set()
    for scenario in scenarios:
        require(isinstance(scenario, dict), "scenario entries must be objects")
        scenario_id = scenario.get("id")
        require(isinstance(scenario_id, str) and scenario_id, "scenario needs an id")
        require(scenario_id not in scenario_ids, "duplicate scenario id {}".format(scenario_id))
        scenario_ids.add(scenario_id)
        require(scenario.get("feature") in {"clause-sharing", "split", "probes", "lns"},
                "scenario {} has an unknown feature".format(scenario_id))
        require(scenario.get("fixture") in fixture_ids, "scenario {} names an unknown fixture".format(scenario_id))
        request = scenario.get("request")
        require(isinstance(request, dict), "scenario {} needs a request".format(scenario_id))
        args = request.get("args")
        require(isinstance(args, list) and args and all(isinstance(token, str) for token in args),
                "scenario {} request needs string args".format(scenario_id))
        require("{instance}" in args, "scenario {} request must contain {{instance}}".format(scenario_id))
        require(isinstance(request.get("seed"), int), "scenario {} request needs an integer seed".format(scenario_id))
        require(isinstance(request.get("threads"), int) and request["threads"] > 1,
                "scenario {} must exercise multiple threads".format(scenario_id))
        context = "scenario {} request".format(scenario_id)
        require(_option_value(args, "--seed", context) == str(request["seed"]), "{} seed metadata differs from args".format(scenario_id))
        require(_option_value(args, "--threads", context) == str(request["threads"]),
                "{} thread metadata differs from args".format(scenario_id))
        require("--verbose" in args, "scenario {} must expose mechanism metrics".format(scenario_id))
        require("--time" not in args and "-t" not in args, "scenario {} must not use a timed search budget".format(scenario_id))
        feature_option = {"clause-sharing": "--shared-pool-cap", "split": "--split", "probes": "--probe", "lns": "--lns"}[
            scenario["feature"]
        ]
        require(feature_option in args, "scenario {} does not request {}".format(scenario_id, feature_option))
        budget = request.get("budget")
        require(isinstance(budget, dict) and budget.get("kind") == "complete",
                "scenario {} correctness and timing must use complete search".format(scenario_id))
        require(isinstance(request.get("timeout_seconds"), int) and request["timeout_seconds"] > 0,
                "scenario {} needs a positive timeout".format(scenario_id))
        expected = scenario.get("expected")
        require(isinstance(expected, dict), "scenario {} needs expected semantics".format(scenario_id))
        statuses = expected.get("statuses")
        require(isinstance(statuses, list) and statuses and set(statuses) <= KNOWN_STATUSES,
                "scenario {} has invalid expected statuses".format(scenario_id))
        evidence = scenario.get("evidence")
        require(isinstance(evidence, list) and evidence, "scenario {} needs feature evidence".format(scenario_id))
        for rule in evidence:
            require(isinstance(rule, dict) and isinstance(rule.get("metric"), str),
                    "scenario {} has invalid evidence".format(scenario_id))
            require("minimum" in rule, "scenario {} evidence needs minimum".format(scenario_id))
            _number(rule["minimum"], "scenario {} evidence minimum".format(scenario_id))
    return manifest


def manifest_maps(manifest: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    fixtures = {fixture["id"]: fixture for fixture in manifest["fixtures"]}
    scenarios = {scenario["id"]: scenario for scenario in manifest["scenarios"]}
    return fixtures, scenarios


def selected_scenarios(manifest: dict[str, Any], requested: Sequence[str]) -> list[dict[str, Any]]:
    _, scenarios = manifest_maps(manifest)
    if not requested:
        return list(manifest["scenarios"])
    unknown = sorted(set(requested) - set(scenarios))
    require(not unknown, "unknown scenarios: {}".format(", ".join(unknown)))
    requested_set = set(requested)
    return [scenario for scenario in manifest["scenarios"] if scenario["id"] in requested_set]


def expand_argv(binary: Path, fixture: dict[str, Any], request: dict[str, Any]) -> list[str]:
    instance = repo_path(fixture["path"])
    argv = [str(binary)]
    for token in request["args"]:
        expanded = token.replace("{instance}", str(instance)).replace("{repo}", str(REPO_ROOT))
        require("{" not in expanded and "}" not in expanded, "unknown command placeholder in {!r}".format(token))
        argv.append(expanded)
    return argv


def execute_solver(argv: Sequence[str], timeout_seconds: int) -> CommandOutcome:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            list(argv),
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError as error:
        raise HarnessError("solver binary not found: {}".format(argv[0])) from error
    except subprocess.TimeoutExpired as error:
        raise HarnessError("solver timed out after {} seconds: {}".format(timeout_seconds, " ".join(argv))) from error
    elapsed = time.perf_counter() - started
    outcome = CommandOutcome(list(argv), completed.returncode, completed.stdout, completed.stderr, elapsed)
    if completed.returncode != 0:
        raise HarnessError(
            "solver exited {}: {}\nstdout:\n{}\nstderr:\n{}".format(
                completed.returncode, " ".join(argv), completed.stdout, completed.stderr
            )
        )
    return outcome


def _solution_block(lines: Sequence[str], name: str) -> list[str]:
    start_marker = "v <{}>".format(name)
    end_marker = "v </{}>".format(name)
    if start_marker not in lines:
        return []
    start = lines.index(start_marker) + 1
    try:
        end = lines.index(end_marker, start)
    except ValueError as error:
        raise HarnessError("unterminated XCSP {} block".format(name)) from error
    tokens: list[str] = []
    for line in lines[start:end]:
        require(line.startswith("v "), "malformed XCSP {} block".format(name))
        tokens.extend(line[2:].split())
    return tokens


METRIC_PATTERNS = {
    "nodes": re.compile(r"^c nodes (\d+) failures (\d+)$"),
    "shared_clauses": re.compile(r"^c shared clauses (\d+) imported (\d+)$"),
    "split_jobs": re.compile(r"^c split jobs (\d+) completed (\d+)$"),
    "probe_attempts": re.compile(r"^c probes attempts (\d+) unsat (\d+)$"),
    "lns_attempts": re.compile(r"^c lns attempts (\d+) improved (\d+)$"),
}


def parse_xcsp_output(stdout: str) -> dict[str, Any]:
    lines = [re.sub(r"^v\s+", "v ", line.strip()) for line in stdout.splitlines()]
    statuses: list[str] = []
    status_map = {
        "s SATISFIABLE": "SATISFIABLE",
        "s UNSATISFIABLE": "UNSATISFIABLE",
        "s OPTIMUM FOUND": "OPTIMAL",
        "s UNKNOWN": "UNKNOWN",
    }
    for line in lines:
        if line in status_map:
            statuses.append(status_map[line])
    require(statuses, "competition status line is missing")
    require(len(set(statuses)) == 1, "conflicting competition statuses: {}".format(statuses))

    objectives: list[int] = []
    for line in lines:
        match = re.fullmatch(r"o\s+(-?\d+)", line)
        if match:
            objectives.append(int(match.group(1)))

    names = _solution_block(lines, "list")
    values = _solution_block(lines, "values")
    solution = None
    if names or values:
        require(len(names) == len(values), "XCSP name/value length mismatch")
        require(len(set(names)) == len(names), "XCSP solution contains duplicate variable names")
        solution = {}
        for name, value in zip(names, values):
            solution[name] = value if value == "*" else int(value)

    metrics: dict[str, int] = {}
    for line in lines:
        match = METRIC_PATTERNS["nodes"].fullmatch(line)
        if match:
            metrics["nodes"] = int(match.group(1))
            metrics["failures"] = int(match.group(2))
        match = METRIC_PATTERNS["shared_clauses"].fullmatch(line)
        if match:
            metrics["shared_clauses"] = int(match.group(1))
            metrics["imported_clauses"] = int(match.group(2))
        match = METRIC_PATTERNS["split_jobs"].fullmatch(line)
        if match:
            metrics["split_jobs"] = int(match.group(1))
            metrics["completed_jobs"] = int(match.group(2))
        match = METRIC_PATTERNS["probe_attempts"].fullmatch(line)
        if match:
            metrics["probe_attempts"] = int(match.group(1))
            metrics["probe_unsat"] = int(match.group(2))
        match = METRIC_PATTERNS["lns_attempts"].fullmatch(line)
        if match:
            metrics["lns_attempts"] = int(match.group(1))
            metrics["lns_improved"] = int(match.group(2))
    return {
        "status": statuses[-1],
        "objectives": objectives,
        "objective": objectives[-1] if objectives else None,
        "solution": solution,
        "metrics": metrics,
        "stdout_sha256": sha256_bytes(stdout.encode("utf-8")),
    }


def validate_semantics(fixture: dict[str, Any], scenario: dict[str, Any], result: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected = scenario["expected"]
    if result.get("status") not in expected["statuses"]:
        errors.append("status {} is not one of {}".format(result.get("status"), expected["statuses"]))
    if "objective" in expected and result.get("objective") != expected["objective"]:
        errors.append("objective {} differs from expected {}".format(result.get("objective"), expected["objective"]))

    oracle = fixture["oracle"]
    if oracle.get("kind") == "pigeonhole-unsat":
        pigeons = int(oracle["pigeons"])
        holes = int(oracle["holes"])
        if pigeons <= holes:
            errors.append("pigeonhole oracle is not structurally unsatisfiable")
        if result.get("status") != "UNSATISFIABLE":
            errors.append("pigeonhole fixture was not proved unsatisfiable")
    elif oracle.get("kind") == "golomb-ruler":
        order = int(oracle["order"])
        optimum = int(oracle["optimum"])
        solution = result.get("solution")
        if result.get("status") != "OPTIMAL":
            errors.append("Golomb fixture was not proved optimal")
        if result.get("objective") != optimum:
            errors.append("Golomb objective is {}, expected {}".format(result.get("objective"), optimum))
        if not isinstance(solution, dict):
            errors.append("Golomb solution is missing")
        else:
            names = ["x[{}]".format(index) for index in range(order)]
            missing = [name for name in names if name not in solution]
            if missing:
                errors.append("Golomb solution is missing marks {}".format(", ".join(missing)))
            elif any(isinstance(solution[name], str) for name in names):
                errors.append("Golomb solution contains unassigned marks")
            else:
                marks = [int(solution[name]) for name in names]
                if marks[0] != 0:
                    errors.append("Golomb first mark is not zero")
                if any(left >= right for left, right in zip(marks, marks[1:])):
                    errors.append("Golomb marks are not strictly increasing")
                differences = [marks[j] - marks[i] for i in range(order) for j in range(i + 1, order)]
                if len(set(differences)) != len(differences):
                    errors.append("Golomb pairwise distances are not unique")
                if marks[-1] != optimum:
                    errors.append("Golomb final mark does not match the optimum")
                distance_names = ["d[{}]".format(index) for index in range(len(differences))]
                missing_distances = [name for name in distance_names if name not in solution]
                if missing_distances:
                    errors.append("Golomb solution is missing distances {}".format(", ".join(missing_distances)))
                elif any(isinstance(solution[name], str) for name in distance_names):
                    errors.append("Golomb solution contains unassigned distances")
                else:
                    published_distances = [int(solution[name]) for name in distance_names]
                    if published_distances != differences:
                        errors.append("Golomb published distances do not match the marks")
    else:
        errors.append("unknown oracle {}".format(oracle.get("kind")))
    return errors


def validate_evidence(scenario: dict[str, Any], result: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        return ["parallel metrics are missing"]
    for rule in scenario["evidence"]:
        name = rule["metric"]
        minimum = float(rule["minimum"])
        value = metrics.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            errors.append("feature metric {} is missing".format(name))
        elif float(value) < minimum:
            errors.append("feature metric {} is {}, below {}".format(name, value, rule["minimum"]))
    return errors


def validate_result(fixture: dict[str, Any], scenario: dict[str, Any], result: dict[str, Any]) -> list[str]:
    return validate_semantics(fixture, scenario, result) + validate_evidence(scenario, result)


def campaign_metadata(
    manifest_path: Path,
    manifest: dict[str, Any],
    binary: Path,
    label: str,
    revision: str | None,
) -> dict[str, Any]:
    resolved_binary = binary.resolve()
    require(resolved_binary.is_file(), "solver binary does not exist: {}".format(resolved_binary))
    return {
        "label": label,
        "source_revision": revision or detect_revision(),
        "binary_path": str(resolved_binary),
        "binary_sha256": sha256_file(resolved_binary),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path.resolve()),
        "harness_sha256": sha256_file(Path(__file__).resolve()),
        "corpus_version": manifest["corpus_version"],
        "pre_refactor_revision": manifest["pre_refactor_revision"],
        "captured_at": utc_timestamp(),
    }


def result_record(
    campaign: dict[str, Any],
    fixture: dict[str, Any],
    scenario: dict[str, Any],
    outcome: CommandOutcome,
    result: dict[str, Any],
    errors: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema": CORRECTNESS_SCHEMA,
        "campaign": campaign,
        "scenario_id": scenario["id"],
        "feature": scenario["feature"],
        "instance": {"path": fixture["path"], "sha256": fixture["sha256"]},
        "request": copy.deepcopy(scenario["request"]),
        "execution": {"argv": outcome.argv, "returncode": outcome.returncode},
        "result": result,
        "validation": {"valid": not errors, "errors": list(errors)},
        "stderr_sha256": sha256_bytes(outcome.stderr.encode("utf-8")),
    }


def paths_alias(left: Path, right: Path) -> bool:
    """Return whether two paths resolve to the same file or destination."""
    try:
        resolved_left = left.resolve()
        resolved_right = right.resolve()
    except (OSError, RuntimeError) as error:
        raise HarnessError("cannot resolve output path: {}".format(error)) from error
    if resolved_left == resolved_right:
        return True
    try:
        return resolved_left.exists() and resolved_right.exists() and os.path.samefile(resolved_left, resolved_right)
    except OSError:
        return False


def validate_output_paths(outputs: Sequence[Path], protected: Sequence[Path]) -> None:
    """Reject outputs that alias each other or an input needed by the capture."""
    for index, output in enumerate(outputs):
        for other in outputs[index + 1:]:
            require(not paths_alias(output, other), "output paths must be distinct")
        for source in protected:
            require(
                not paths_alias(output, source),
                "output {} aliases protected input {}".format(output, source),
            )


def write_text_atomic(path: Path, value: str) -> None:
    """Replace a text file atomically after its complete contents are durable."""
    descriptor: int | None = None
    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".{}.".format(path.name), suffix=".tmp", dir=str(path.parent)
        )
        temporary_path = Path(temporary_name)
        destination = os.fdopen(descriptor, "w", encoding="utf-8")
        descriptor = None
        with destination:
            destination.write(value)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_path, path)
    except OSError as error:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass
        raise HarnessError("cannot write {}: {}".format(path, error)) from error


def write_json(path: Path, value: Any) -> None:
    write_text_atomic(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, values: Iterable[Any]) -> None:
    write_text_atomic(path, "".join(canonical_json(value) + "\n" for value in values))


def run_correctness(
    manifest_path: Path,
    binary: Path,
    output: Path,
    label: str,
    revision: str | None,
    requested: Sequence[str],
    timeout_override: int | None = None,
) -> list[dict[str, Any]]:
    manifest = load_manifest(manifest_path)
    fixtures, _ = manifest_maps(manifest)
    validate_output_paths(
        [output],
        [manifest_path, binary, Path(__file__)] + [repo_path(fixture["path"]) for fixture in fixtures.values()],
    )
    scenarios = selected_scenarios(manifest, requested)
    campaign = campaign_metadata(manifest_path, manifest, binary, label, revision)
    records: list[dict[str, Any]] = []
    for scenario in scenarios:
        fixture = fixtures[scenario["fixture"]]
        argv = expand_argv(binary.resolve(), fixture, scenario["request"])
        timeout = effective_timeout(scenario["request"], timeout_override)
        print("correctness {}".format(scenario["id"]), file=sys.stderr, flush=True)
        outcome = execute_solver(argv, timeout)
        result = parse_xcsp_output(outcome.stdout)
        errors = validate_result(fixture, scenario, result)
        records.append(result_record(campaign, fixture, scenario, outcome, result, errors))
    write_jsonl(output, records)
    return records


def median_seconds(samples: Sequence[float]) -> float:
    require(bool(samples), "cannot compute a median without samples")
    for sample in samples:
        require(isinstance(sample, (int, float)) and not isinstance(sample, bool) and sample > 0,
                "timing samples must be positive numbers")
    return float(statistics.median(float(sample) for sample in samples))


def _run_timing_sample(
    binary: Path,
    fixture: dict[str, Any],
    scenario: dict[str, Any],
    timeout_override: int | None,
) -> tuple[CommandOutcome, dict[str, Any]]:
    argv = expand_argv(binary, fixture, scenario["request"])
    timeout = effective_timeout(scenario["request"], timeout_override)
    outcome = execute_solver(argv, timeout)
    result = parse_xcsp_output(outcome.stdout)
    errors = validate_result(fixture, scenario, result)
    require(not errors, "{} produced an invalid timing sample: {}".format(scenario["id"], "; ".join(errors)))
    return outcome, result


def paired_order(repetition: int, scenario_index: int) -> tuple[str, str]:
    """Return a deterministic alternating baseline/candidate execution order."""
    if (repetition + scenario_index) % 2 == 0:
        return "baseline", "candidate"
    return "candidate", "baseline"


def expected_pair_schedule(scenario_ids: Sequence[str], repetitions: int) -> list[dict[str, Any]]:
    """Build the canonical rotated schedule for a paired capture."""
    identifiers = list(scenario_ids)
    require(bool(identifiers), "paired schedule requires at least one scenario")
    indexes = {scenario_id: index for index, scenario_id in enumerate(identifiers)}
    schedule: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        offset = repetition % len(identifiers)
        ordered_ids = identifiers[offset:] + identifiers[:offset]
        for scenario_id in ordered_ids:
            schedule.append({
                "repetition": repetition + 1,
                "scenario_id": scenario_id,
                "first": paired_order(repetition, indexes[scenario_id])[0],
            })
    return schedule


def _timing_report(
    campaign: dict[str, Any],
    manifest: dict[str, Any],
    fixtures: dict[str, dict[str, Any]],
    scenarios: Sequence[dict[str, Any]],
    samples: dict[str, list[dict[str, Any]]],
    repetitions: int,
    warmups: int,
    pairing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cases = []
    for scenario in scenarios:
        fixture = fixtures[scenario["fixture"]]
        scenario_samples = samples[scenario["id"]]
        elapsed = [sample["elapsed_seconds"] for sample in scenario_samples]
        cases.append({
            "scenario_id": scenario["id"],
            "feature": scenario["feature"],
            "instance": {"path": fixture["path"], "sha256": fixture["sha256"]},
            "request": copy.deepcopy(scenario["request"]),
            "repetitions": repetitions,
            "warmups": warmups,
            "samples": scenario_samples,
            "median_seconds": median_seconds(elapsed),
        })
    report = {
        "schema": TIMING_SCHEMA,
        "campaign": campaign,
        "default_tolerance": float(manifest["default_tolerance"]),
        "cases": cases,
    }
    if pairing is not None:
        report["pairing"] = pairing
    return report


def run_timing(
    manifest_path: Path,
    binary: Path,
    output: Path,
    label: str,
    revision: str | None,
    requested: Sequence[str],
    repetitions: int,
    warmups: int,
    timeout_override: int | None = None,
) -> dict[str, Any]:
    require(repetitions >= 3, "timing campaigns require at least three repetitions")
    require(warmups >= 0, "warmups cannot be negative")
    manifest = load_manifest(manifest_path)
    fixtures, _ = manifest_maps(manifest)
    validate_output_paths(
        [output],
        [manifest_path, binary, Path(__file__)] + [repo_path(fixture["path"]) for fixture in fixtures.values()],
    )
    scenarios = selected_scenarios(manifest, requested)
    binary = binary.resolve()
    campaign = campaign_metadata(manifest_path, manifest, binary, label, revision)
    samples: dict[str, list[dict[str, Any]]] = {scenario["id"]: [] for scenario in scenarios}

    for warmup in range(warmups):
        for scenario in scenarios:
            print("warmup {}/{} {}".format(warmup + 1, warmups, scenario["id"]), file=sys.stderr, flush=True)
            _run_timing_sample(binary, fixtures[scenario["fixture"]], scenario, timeout_override)

    for repetition in range(repetitions):
        offset = repetition % len(scenarios)
        ordered = scenarios[offset:] + scenarios[:offset]
        for scenario in ordered:
            print("timing {}/{} {}".format(repetition + 1, repetitions, scenario["id"]), file=sys.stderr, flush=True)
            outcome, result = _run_timing_sample(binary, fixtures[scenario["fixture"]], scenario, timeout_override)
            samples[scenario["id"]].append({
                "elapsed_seconds": outcome.elapsed_seconds,
                "result": result,
                "stderr_sha256": sha256_bytes(outcome.stderr.encode("utf-8")),
            })

    report = _timing_report(campaign, manifest, fixtures, scenarios, samples, repetitions, warmups)
    write_json(output, report)
    return report


def run_timing_pair(
    manifest_path: Path,
    baseline_binary: Path,
    candidate_binary: Path,
    baseline_output: Path,
    candidate_output: Path,
    baseline_label: str,
    candidate_label: str,
    baseline_revision: str | None,
    candidate_revision: str | None,
    requested: Sequence[str],
    repetitions: int,
    warmups: int,
    timeout_override: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Capture baseline and candidate samples as adjacent alternating pairs."""
    require(repetitions >= 4, "paired timing campaigns require at least four repetitions")
    require(repetitions % 2 == 0, "paired timing campaigns require an even repetition count")
    require(warmups >= 0, "warmups cannot be negative")
    require(isinstance(baseline_revision, str) and bool(baseline_revision.strip()), "baseline revision is required")
    require(isinstance(candidate_revision, str) and bool(candidate_revision.strip()), "candidate revision is required")
    manifest = load_manifest(manifest_path)
    fixtures, _ = manifest_maps(manifest)
    validate_output_paths(
        [baseline_output, candidate_output],
        [manifest_path, baseline_binary, candidate_binary, Path(__file__)]
        + [repo_path(fixture["path"]) for fixture in fixtures.values()],
    )
    require(
        baseline_revision == manifest["pre_refactor_revision"],
        "baseline revision must equal the pinned pre-refactor revision",
    )
    scenarios = selected_scenarios(manifest, requested)
    baseline_binary = baseline_binary.resolve()
    candidate_binary = candidate_binary.resolve()
    campaigns = {
        "baseline": campaign_metadata(
            manifest_path,
            manifest,
            baseline_binary,
            baseline_label,
            baseline_revision,
        ),
        "candidate": campaign_metadata(
            manifest_path,
            manifest,
            candidate_binary,
            candidate_label,
            candidate_revision,
        ),
    }
    binaries = {"baseline": baseline_binary, "candidate": candidate_binary}
    samples: dict[str, dict[str, list[dict[str, Any]]]] = {
        side: {scenario["id"]: [] for scenario in scenarios} for side in binaries
    }
    scenario_indexes = {scenario["id"]: index for index, scenario in enumerate(scenarios)}

    for warmup in range(warmups):
        for scenario in scenarios:
            order = paired_order(warmup, scenario_indexes[scenario["id"]])
            for side in order:
                print(
                    "paired warmup {}/{} {} {}".format(warmup + 1, warmups, scenario["id"], side),
                    file=sys.stderr,
                    flush=True,
                )
                _run_timing_sample(binaries[side], fixtures[scenario["fixture"]], scenario, timeout_override)

    measured_schedule = expected_pair_schedule([scenario["id"] for scenario in scenarios], repetitions)
    schedule_index = 0
    for repetition in range(repetitions):
        offset = repetition % len(scenarios)
        ordered_scenarios = scenarios[offset:] + scenarios[:offset]
        for scenario in ordered_scenarios:
            order = paired_order(repetition, scenario_indexes[scenario["id"]])
            require(
                measured_schedule[schedule_index]
                == {"repetition": repetition + 1, "scenario_id": scenario["id"], "first": order[0]},
                "internal paired schedule mismatch",
            )
            schedule_index += 1
            for position, side in enumerate(order, start=1):
                print(
                    "paired timing {}/{} {} {} ({}/2)".format(
                        repetition + 1, repetitions, scenario["id"], side, position
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                outcome, result = _run_timing_sample(
                    binaries[side], fixtures[scenario["fixture"]], scenario, timeout_override
                )
                samples[side][scenario["id"]].append({
                    "elapsed_seconds": outcome.elapsed_seconds,
                    "result": result,
                    "stderr_sha256": sha256_bytes(outcome.stderr.encode("utf-8")),
                    "pair": {"repetition": repetition + 1, "position": position},
                })

    pairing_key = {
        "schema": PAIRING_SCHEMA,
        "capture_nonce_ns": time.time_ns(),
        "manifest_sha256": campaigns["baseline"]["manifest_sha256"],
        "harness_sha256": campaigns["baseline"]["harness_sha256"],
        "baseline_binary_sha256": campaigns["baseline"]["binary_sha256"],
        "candidate_binary_sha256": campaigns["candidate"]["binary_sha256"],
        "repetitions": repetitions,
        "warmups": warmups,
        "schedule": measured_schedule,
    }
    pairing = copy.deepcopy(pairing_key)
    pairing["capture_id"] = sha256_bytes(canonical_json(pairing_key).encode("utf-8"))
    baseline_report = _timing_report(
        campaigns["baseline"], manifest, fixtures, scenarios, samples["baseline"], repetitions, warmups, pairing
    )
    candidate_report = _timing_report(
        campaigns["candidate"], manifest, fixtures, scenarios, samples["candidate"], repetitions, warmups, pairing
    )
    write_json(baseline_output, baseline_report)
    write_json(candidate_output, candidate_report)
    return baseline_report, candidate_report


def load_timing_report(path: Path) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HarnessError("cannot read timing report {}: {}".format(path, error)) from error
    errors = timing_report_errors(report, str(path))
    require(not errors, "invalid timing report {}: {}".format(path, "; ".join(errors)))
    return report


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def timing_report_errors(report: Any, side: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(report, dict):
        return ["{} report is not an object".format(side)]
    if report.get("schema") != TIMING_SCHEMA:
        errors.append("{} schema is invalid".format(side))
    campaign = report.get("campaign")
    if not isinstance(campaign, dict):
        errors.append("{} campaign is missing".format(side))
    else:
        for field in ("binary_sha256", "manifest_sha256", "harness_sha256"):
            if not _sha256(campaign.get(field)):
                errors.append("{} campaign {} is missing or invalid".format(side, field))
        for field in ("label", "source_revision", "binary_path", "manifest_path", "corpus_version", "captured_at"):
            if not isinstance(campaign.get(field), str) or not campaign[field]:
                errors.append("{} campaign {} is missing".format(side, field))
        if not isinstance(campaign.get("pre_refactor_revision"), str) or re.fullmatch(
            r"[0-9a-f]{40}", campaign["pre_refactor_revision"]
        ) is None:
            errors.append("{} campaign pre_refactor_revision is missing or invalid".format(side))

    pairing = report.get("pairing")
    paired = pairing is not None
    if paired:
        if not isinstance(pairing, dict):
            errors.append("{} pairing metadata is invalid".format(side))
        else:
            expected_fields = set(PAIRING_KEY_FIELDS) | {"capture_id"}
            if set(pairing) != expected_fields:
                errors.append("{} pairing fields are incomplete or unexpected".format(side))
            if pairing.get("schema") != PAIRING_SCHEMA:
                errors.append("{} pairing schema is invalid".format(side))
            nonce = pairing.get("capture_nonce_ns")
            if not isinstance(nonce, int) or isinstance(nonce, bool) or nonce <= 0:
                errors.append("{} pairing capture_nonce_ns is missing or invalid".format(side))
            for field in (
                "manifest_sha256",
                "harness_sha256",
                "baseline_binary_sha256",
                "candidate_binary_sha256",
            ):
                if not _sha256(pairing.get(field)):
                    errors.append("{} pairing {} is missing or invalid".format(side, field))
            pairing_repetitions = pairing.get("repetitions")
            if (
                not isinstance(pairing_repetitions, int)
                or isinstance(pairing_repetitions, bool)
                or pairing_repetitions < 4
                or pairing_repetitions % 2 != 0
            ):
                errors.append("{} pairing repetitions must be an even integer of at least four".format(side))
            pairing_warmups = pairing.get("warmups")
            if not isinstance(pairing_warmups, int) or isinstance(pairing_warmups, bool) or pairing_warmups < 0:
                errors.append("{} pairing warmups are missing or invalid".format(side))
            if not isinstance(pairing.get("schedule"), list) or not pairing["schedule"]:
                errors.append("{} pairing schedule is missing".format(side))
            if not _sha256(pairing.get("capture_id")):
                errors.append("{} pairing capture_id is missing or invalid".format(side))
            if all(field in pairing for field in PAIRING_KEY_FIELDS):
                pairing_key = {field: pairing[field] for field in PAIRING_KEY_FIELDS}
                try:
                    expected_capture_id = sha256_bytes(canonical_json(pairing_key).encode("utf-8"))
                except (TypeError, ValueError):
                    errors.append("{} pairing key is not canonical JSON".format(side))
                else:
                    if pairing.get("capture_id") != expected_capture_id:
                        errors.append("{} pairing capture_id does not match its contents".format(side))
            if isinstance(campaign, dict):
                for field in ("manifest_sha256", "harness_sha256"):
                    if pairing.get(field) != campaign.get(field):
                        errors.append("{} pairing {} differs from campaign".format(side, field))
                campaign_binary = campaign.get("binary_sha256")
                if campaign_binary not in {
                    pairing.get("baseline_binary_sha256"),
                    pairing.get("candidate_binary_sha256"),
                }:
                    errors.append("{} pairing binary hashes do not contain campaign binary".format(side))

    cases = report.get("cases")
    if not isinstance(cases, list) or not cases:
        errors.append("{} cases are missing".format(side))
        return errors
    scenario_ids: list[str] = []
    for index, case in enumerate(cases):
        prefix = "{} case {}".format(side, index)
        if not isinstance(case, dict):
            errors.append("{} is not an object".format(prefix))
            continue
        if not isinstance(case.get("scenario_id"), str) or not case["scenario_id"]:
            errors.append("{} scenario_id is missing".format(prefix))
        else:
            scenario_ids.append(case["scenario_id"])
        if case.get("feature") not in {"clause-sharing", "split", "probes", "lns"}:
            errors.append("{} feature is missing or invalid".format(prefix))
        instance = case.get("instance")
        if not isinstance(instance, dict):
            errors.append("{} instance is missing".format(prefix))
        else:
            if not isinstance(instance.get("path"), str) or not instance["path"]:
                errors.append("{} instance path is missing".format(prefix))
            if not _sha256(instance.get("sha256")):
                errors.append("{} instance sha256 is missing or invalid".format(prefix))
        request = case.get("request")
        if not isinstance(request, dict):
            errors.append("{} request is missing".format(prefix))
        else:
            args = request.get("args")
            if not isinstance(args, list) or not args or not all(isinstance(token, str) for token in args):
                errors.append("{} request args are missing or invalid".format(prefix))
            if not isinstance(request.get("seed"), int) or isinstance(request.get("seed"), bool):
                errors.append("{} request seed is missing".format(prefix))
            if not isinstance(request.get("threads"), int) or isinstance(request.get("threads"), bool):
                errors.append("{} request threads are missing".format(prefix))
            if not isinstance(request.get("budget"), dict):
                errors.append("{} request budget is missing".format(prefix))
        repetitions = case.get("repetitions")
        warmups = case.get("warmups")
        minimum_repetitions = 4 if paired else 3
        if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions < minimum_repetitions:
            errors.append("{} repetitions are missing or invalid".format(prefix))
        elif paired and repetitions % 2 != 0:
            errors.append("{} repetitions must be even for a paired capture".format(prefix))
        if not isinstance(warmups, int) or isinstance(warmups, bool) or warmups < 0:
            errors.append("{} warmups are missing or invalid".format(prefix))
        samples = case.get("samples")
        if not isinstance(samples, list):
            errors.append("{} samples are missing".format(prefix))
            continue
        if isinstance(repetitions, int) and not isinstance(repetitions, bool) and len(samples) != repetitions:
            errors.append("{} repetition count differs from samples".format(prefix))
        for sample_index, sample in enumerate(samples):
            sample_prefix = "{} sample {}".format(prefix, sample_index)
            if not isinstance(sample, dict):
                errors.append("{} is not an object".format(sample_prefix))
                continue
            elapsed = sample.get("elapsed_seconds")
            if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool) or elapsed <= 0:
                errors.append("{} elapsed_seconds is missing or invalid".format(sample_prefix))
            if not _sha256(sample.get("stderr_sha256")):
                errors.append("{} stderr_sha256 is missing or invalid".format(sample_prefix))
            result = sample.get("result")
            if not isinstance(result, dict):
                errors.append("{} result is missing".format(sample_prefix))
            else:
                if result.get("status") not in KNOWN_STATUSES:
                    errors.append("{} result status is missing or invalid".format(sample_prefix))
                for field, kind in (("objectives", list), ("metrics", dict)):
                    if not isinstance(result.get(field), kind):
                        errors.append("{} result {} is missing or invalid".format(sample_prefix, field))
                if "objective" not in result or "solution" not in result:
                    errors.append("{} result payload is incomplete".format(sample_prefix))
                if not _sha256(result.get("stdout_sha256")):
                    errors.append("{} result stdout_sha256 is missing or invalid".format(sample_prefix))
            pair = sample.get("pair")
            if paired:
                if not isinstance(pair, dict):
                    errors.append("{} pair position is missing".format(sample_prefix))
                else:
                    if pair.get("repetition") != sample_index + 1:
                        errors.append("{} pair repetition does not match sample position".format(sample_prefix))
                    if pair.get("position") not in {1, 2}:
                        errors.append("{} pair position is missing or invalid".format(sample_prefix))
            elif pair is not None:
                errors.append("{} has pair data without pairing metadata".format(sample_prefix))

    if len(scenario_ids) != len(set(scenario_ids)):
        errors.append("{} report repeats a paired scenario".format(side))
    if paired and isinstance(pairing, dict):
        pairing_repetitions = pairing.get("repetitions")
        pairing_warmups = pairing.get("warmups")
        for index, case in enumerate(cases):
            if not isinstance(case, dict):
                continue
            if case.get("repetitions") != pairing_repetitions:
                errors.append("{} case {} repetitions differ from pairing".format(side, index))
            if case.get("warmups") != pairing_warmups:
                errors.append("{} case {} warmups differ from pairing".format(side, index))
        if (
            len(scenario_ids) == len(cases)
            and len(scenario_ids) == len(set(scenario_ids))
            and isinstance(pairing_repetitions, int)
            and not isinstance(pairing_repetitions, bool)
            and pairing_repetitions >= 0
            and isinstance(pairing.get("schedule"), list)
        ):
            expected_schedule = expected_pair_schedule(scenario_ids, pairing_repetitions)
            if pairing["schedule"] != expected_schedule:
                errors.append("{} pairing schedule does not match cases and repetitions".format(side))
    return errors


def paired_comparison_errors(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    """Cross-check paired provenance and complementary sample positions."""
    errors: list[str] = []
    reports = {"baseline": baseline, "candidate": candidate}
    pairings = {side: report.get("pairing") for side, report in reports.items()}
    if not all(isinstance(pairing, dict) for pairing in pairings.values()):
        return errors

    for side, report in reports.items():
        pairing = pairings[side]
        assert isinstance(pairing, dict)
        campaign = report.get("campaign")
        if not isinstance(campaign, dict):
            continue
        for field in ("manifest_sha256", "harness_sha256"):
            if pairing.get(field) != campaign.get(field):
                errors.append("{} pairing {} differs from campaign".format(side, field))
        binary_field = "{}_binary_sha256".format(side)
        if pairing.get(binary_field) != campaign.get("binary_sha256"):
            errors.append("{} pairing binary hash differs from campaign".format(side))

    baseline_pairing = pairings["baseline"]
    candidate_pairing = pairings["candidate"]
    assert isinstance(baseline_pairing, dict) and isinstance(candidate_pairing, dict)
    if baseline_pairing != candidate_pairing:
        return errors
    schedule = baseline_pairing.get("schedule")
    if not isinstance(schedule, list):
        return errors
    baseline_cases, _ = _case_map(baseline, "baseline")
    candidate_cases, _ = _case_map(candidate, "candidate")
    for entry in schedule:
        if not isinstance(entry, dict):
            continue
        scenario_id = entry.get("scenario_id")
        repetition = entry.get("repetition")
        first = entry.get("first")
        if (
            not isinstance(scenario_id, str)
            or not isinstance(repetition, int)
            or isinstance(repetition, bool)
            or repetition < 1
            or first not in {"baseline", "candidate"}
            or scenario_id not in baseline_cases
            or scenario_id not in candidate_cases
        ):
            continue
        baseline_samples = baseline_cases[scenario_id].get("samples")
        candidate_samples = candidate_cases[scenario_id].get("samples")
        if not isinstance(baseline_samples, list) or not isinstance(candidate_samples, list):
            continue
        if repetition > len(baseline_samples) or repetition > len(candidate_samples):
            continue
        baseline_sample = baseline_samples[repetition - 1]
        candidate_sample = candidate_samples[repetition - 1]
        if not isinstance(baseline_sample, dict) or not isinstance(candidate_sample, dict):
            continue
        baseline_pair = baseline_sample.get("pair", {})
        candidate_pair = candidate_sample.get("pair", {})
        baseline_position = baseline_pair.get("position") if isinstance(baseline_pair, dict) else None
        candidate_position = candidate_pair.get("position") if isinstance(candidate_pair, dict) else None
        expected_baseline = 1 if first == "baseline" else 2
        expected_candidate = 1 if first == "candidate" else 2
        if baseline_position != expected_baseline or candidate_position != expected_candidate:
            errors.append(
                "paired sample positions do not match schedule for {} repetition {}".format(
                    scenario_id, repetition
                )
            )
        elif {baseline_position, candidate_position} != {1, 2}:
            errors.append(
                "paired sample positions are not complementary for {} repetition {}".format(
                    scenario_id, repetition
                )
            )
    return errors


def _case_map(report: dict[str, Any], side: str) -> tuple[dict[str, dict[str, Any]], list[str]]:
    errors: list[str] = []
    result: dict[str, dict[str, Any]] = {}
    cases = report.get("cases")
    if not isinstance(cases, list):
        return result, ["{} report cases are invalid".format(side)]
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("scenario_id"), str):
            errors.append("{} report has an invalid case".format(side))
            continue
        scenario_id = case["scenario_id"]
        if scenario_id in result:
            errors.append("{} report repeats scenario {}".format(side, scenario_id))
        result[scenario_id] = case
    return result, errors


def _sample_signature(sample: dict[str, Any]) -> tuple[Any, Any]:
    result = sample.get("result", {})
    return result.get("status"), result.get("objective")


def compare_reports(baseline: dict[str, Any], candidate: dict[str, Any], tolerance: float = DEFAULT_TOLERANCE) -> dict[str, Any]:
    require(0.0 <= tolerance <= 1.0, "tolerance must be between zero and one")
    errors = timing_report_errors(baseline, "baseline") + timing_report_errors(candidate, "candidate")
    baseline_report = baseline if isinstance(baseline, dict) else {}
    candidate_report = candidate if isinstance(candidate, dict) else {}
    baseline_campaign = baseline_report.get("campaign", {})
    candidate_campaign = candidate_report.get("campaign", {})
    baseline_campaign = baseline_campaign if isinstance(baseline_campaign, dict) else {}
    candidate_campaign = candidate_campaign if isinstance(candidate_campaign, dict) else {}
    for field in ("manifest_sha256", "harness_sha256", "corpus_version", "pre_refactor_revision"):
        if baseline_campaign.get(field) != candidate_campaign.get(field):
            errors.append("campaign {} differs".format(field))
    if baseline_report.get("pairing") != candidate_report.get("pairing"):
        errors.append("campaign pairing metadata differs")
    errors.extend(paired_comparison_errors(baseline_report, candidate_report))
    if baseline_campaign.get("source_revision") != PINNED_PRE_REFACTOR_REVISION:
        errors.append("baseline source_revision is not the pinned pre-refactor revision")

    baseline_cases, baseline_errors = _case_map(baseline_report, "baseline")
    candidate_cases, candidate_errors = _case_map(candidate_report, "candidate")
    errors.extend(baseline_errors)
    errors.extend(candidate_errors)
    if set(baseline_cases) != set(candidate_cases):
        errors.append("scenario sets differ")

    comparisons: list[dict[str, Any]] = []
    for scenario_id in sorted(set(baseline_cases) & set(candidate_cases)):
        before = baseline_cases[scenario_id]
        after = candidate_cases[scenario_id]
        identity_errors: list[str] = []
        if before.get("feature") != after.get("feature"):
            identity_errors.append("feature differs")
        if before.get("instance") != after.get("instance"):
            identity_errors.append("instance hash or path differs")
        if canonical_json(before.get("request")) != canonical_json(after.get("request")):
            identity_errors.append("request differs")

        before_samples = before.get("samples")
        after_samples = after.get("samples")
        if not isinstance(before_samples, list) or not before_samples:
            identity_errors.append("baseline samples are missing")
        elif len(before_samples) < 3:
            identity_errors.append("baseline has fewer than three samples")
        if not isinstance(after_samples, list) or not after_samples:
            identity_errors.append("candidate samples are missing")
        elif len(after_samples) < 3:
            identity_errors.append("candidate has fewer than three samples")
        if isinstance(before_samples, list) and before.get("repetitions") != len(before_samples):
            identity_errors.append("baseline repetition count differs from samples")
        if isinstance(after_samples, list) and after.get("repetitions") != len(after_samples):
            identity_errors.append("candidate repetition count differs from samples")
        if before.get("repetitions") != after.get("repetitions"):
            identity_errors.append("repetition counts differ")
        if before.get("warmups") != after.get("warmups"):
            identity_errors.append("warmup counts differ")
        before_median = None
        after_median = None
        signatures_before: set[tuple[Any, Any]] = set()
        signatures_after: set[tuple[Any, Any]] = set()
        if isinstance(before_samples, list) and before_samples:
            try:
                before_median = median_seconds([sample["elapsed_seconds"] for sample in before_samples])
                signatures_before = {_sample_signature(sample) for sample in before_samples}
            except (HarnessError, KeyError, TypeError, AttributeError, ValueError):
                identity_errors.append("baseline samples are invalid")
        if isinstance(after_samples, list) and after_samples:
            try:
                after_median = median_seconds([sample["elapsed_seconds"] for sample in after_samples])
                signatures_after = {_sample_signature(sample) for sample in after_samples}
            except (HarnessError, KeyError, TypeError, AttributeError, ValueError):
                identity_errors.append("candidate samples are invalid")
        if before_median is not None:
            try:
                if abs(float(before["median_seconds"]) - before_median) > 1e-12:
                    identity_errors.append("baseline median does not match samples")
            except (KeyError, TypeError, ValueError):
                identity_errors.append("baseline median is invalid")
        if after_median is not None:
            try:
                if abs(float(after["median_seconds"]) - after_median) > 1e-12:
                    identity_errors.append("candidate median does not match samples")
            except (KeyError, TypeError, ValueError):
                identity_errors.append("candidate median is invalid")
        if signatures_before != signatures_after or len(signatures_before) != 1:
            identity_errors.append("result status or objective differs")

        ratio = None
        limit = None
        regression = False
        if not identity_errors and before_median is not None and after_median is not None:
            limit = before_median * (1.0 + tolerance)
            ratio = after_median / before_median
            regression = after_median > limit
        comparisons.append({
            "scenario_id": scenario_id,
            "baseline_median_seconds": before_median,
            "candidate_median_seconds": after_median,
            "allowed_seconds": limit,
            "candidate_over_baseline": ratio,
            "regression": regression,
            "comparable": not identity_errors,
            "errors": identity_errors,
        })
        errors.extend("{}: {}".format(scenario_id, error) for error in identity_errors)

    regressions = [entry["scenario_id"] for entry in comparisons if entry["regression"]]
    return {
        "schema": COMPARISON_SCHEMA,
        "baseline": {
            "label": baseline_campaign.get("label"),
            "source_revision": baseline_campaign.get("source_revision"),
            "binary_sha256": baseline_campaign.get("binary_sha256"),
        },
        "candidate": {
            "label": candidate_campaign.get("label"),
            "source_revision": candidate_campaign.get("source_revision"),
            "binary_sha256": candidate_campaign.get("binary_sha256"),
        },
        "tolerance": tolerance,
        "comparable": not errors,
        "regressions": regressions,
        "passed": not errors and not regressions,
        "errors": errors,
        "cases": comparisons,
    }


def comparison_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "# Parallel timing comparison",
        "",
        "Tolerance: {:.1%}".format(comparison["tolerance"]),
        "",
        "| Scenario | Baseline median | Candidate median | Ratio | Verdict |",
        "|---|---:|---:|---:|---|",
    ]
    for case in comparison["cases"]:
        before = case["baseline_median_seconds"]
        after = case["candidate_median_seconds"]
        ratio = case["candidate_over_baseline"]
        if not case["comparable"]:
            verdict = "invalid"
        elif case["regression"]:
            verdict = "regression"
        else:
            verdict = "pass"
        lines.append("| {} | {} | {} | {} | {} |".format(
            case["scenario_id"],
            "{:.6f}s".format(before) if before is not None else "n/a",
            "{:.6f}s".format(after) if after is not None else "n/a",
            "{:.3f}x".format(ratio) if ratio is not None else "n/a",
            verdict,
        ))
    if comparison["errors"]:
        lines.extend(["", "Comparison errors:"])
        lines.extend("- {}".format(error) for error in comparison["errors"])
    return "\n".join(lines) + "\n"


def _add_capture_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--binary", type=Path, default=REPO_ROOT / "target" / "release" / "qayd")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--revision", help="source revision used to build --binary; detected from this worktree if omitted")
    parser.add_argument("--case", action="append", default=[], dest="cases")
    parser.add_argument("--timeout", type=int, help="override each scenario timeout")


def _add_pair_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--baseline-binary", type=Path, required=True)
    parser.add_argument("--candidate-binary", type=Path, required=True)
    parser.add_argument("--baseline-out", type=Path, required=True)
    parser.add_argument("--candidate-out", type=Path, required=True)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--baseline-revision", required=True)
    parser.add_argument("--candidate-revision", required=True)
    parser.add_argument("--case", action="append", default=[], dest="cases")
    parser.add_argument("--timeout", type=int, help="override each scenario timeout")
    parser.add_argument("--repetitions", type=int, default=6)
    parser.add_argument("--warmups", type=int, default=1)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check", help="run semantic and feature checks without timing thresholds")
    _add_capture_arguments(check)
    measure = subparsers.add_parser("measure", help="capture repeated wall-time samples and medians")
    _add_capture_arguments(measure)
    measure.add_argument("--repetitions", type=int, default=5)
    measure.add_argument("--warmups", type=int, default=1)
    measure_pair = subparsers.add_parser(
        "measure-pair", help="capture baseline and candidate as adjacent samples in alternating order"
    )
    _add_pair_arguments(measure_pair)
    compare = subparsers.add_parser("compare", help="compare two existing timing reports without running the solver")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    compare.add_argument("--json", type=Path, dest="json_out")
    compare.add_argument("--markdown", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "check":
            records = run_correctness(
                args.manifest, args.binary, args.out, args.label, args.revision, args.cases, args.timeout
            )
            invalid = [record["scenario_id"] for record in records if not record["validation"]["valid"]]
            if invalid:
                print("invalid correctness scenarios: {}".format(", ".join(invalid)), file=sys.stderr)
                return 1
            print("correctness: {} scenarios passed".format(len(records)), file=sys.stderr)
            return 0
        if args.command == "measure":
            report = run_timing(
                args.manifest,
                args.binary,
                args.out,
                args.label,
                args.revision,
                args.cases,
                args.repetitions,
                args.warmups,
                args.timeout,
            )
            print("timing: {} scenario medians captured".format(len(report["cases"])), file=sys.stderr)
            return 0
        if args.command == "measure-pair":
            baseline_report, candidate_report = run_timing_pair(
                args.manifest,
                args.baseline_binary,
                args.candidate_binary,
                args.baseline_out,
                args.candidate_out,
                args.baseline_label,
                args.candidate_label,
                args.baseline_revision,
                args.candidate_revision,
                args.cases,
                args.repetitions,
                args.warmups,
                args.timeout,
            )
            print(
                "paired timing: {} scenario medians captured for each binary".format(len(baseline_report["cases"])),
                file=sys.stderr,
            )
            return 0
        baseline = load_timing_report(args.baseline)
        candidate = load_timing_report(args.candidate)
        comparison = compare_reports(baseline, candidate, args.tolerance)
        rendered = comparison_markdown(comparison)
        comparison_outputs = [path for path in (args.json_out, args.markdown) if path is not None]
        validate_output_paths(comparison_outputs, [args.baseline, args.candidate, Path(__file__)])
        if args.json_out:
            write_json(args.json_out, comparison)
        if args.markdown:
            write_text_atomic(args.markdown, rendered)
        print(rendered, end="")
        return 0 if comparison["passed"] else 1
    except HarnessError as error:
        print("parallel harness error: {}".format(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
