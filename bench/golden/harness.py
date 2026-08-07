#!/usr/bin/env python3
"""Versioned, read-only architecture golden harness for qayd."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


MANIFEST_SCHEMA = "qayd.golden.manifest/v1"
RESULT_PREFIX = "QAYD_GOLDEN_RESULT="
KNOWN_STATUSES = {"SATISFIABLE", "UNSATISFIABLE", "OPTIMAL", "UNKNOWN", "UNSUPPORTED"}
FEASIBLE_STATUSES = {"SATISFIABLE", "OPTIMAL"}
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path(__file__).with_name("manifest.v1.json")


class GoldenError(Exception):
    """A deterministic harness, command, parsing, or comparison failure."""


@dataclass
class OracleOutcome:
    feasible_exists: bool
    valid_solution: bool
    actual_objectives: Optional[List[int]]
    optimum: Optional[List[int]]
    exhaustive: bool = True


@dataclass
class CommandResult:
    argv: List[str]
    returncode: int
    stdout: str
    stderr: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_path(relative: str) -> Path:
    candidate = (REPO_ROOT / relative).resolve()
    try:
        candidate.relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise GoldenError("path escapes repository: {}".format(relative)) from error
    return candidate


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GoldenError(message)


def canonical_gap(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the canonical primary-objective gap represented by a record."""
    objectives = record.get("objectives")
    senses = record.get("senses")
    bound = record.get("bound")
    if not isinstance(objectives, list) or not objectives:
        return None
    if not isinstance(senses, list) or not senses:
        return None
    if not isinstance(bound, dict):
        return None
    values = bound.get("values")
    if not isinstance(values, list) or not values:
        return None

    primal = objectives[0]
    dual = values[0]
    sense = senses[0]
    require(isinstance(primal, int) and not isinstance(primal, bool), "gap primal must be an integer")
    require(isinstance(dual, int) and not isinstance(dual, bool), "gap bound must be an integer")
    require(sense in ("minimize", "maximize"), "gap sense must be minimize or maximize")
    signed = primal - dual if sense == "minimize" else dual - primal
    if signed < 0:
        return None
    denominator = max(abs(primal), abs(dual), 1)
    return {"absolute": signed, "relative": signed / denominator}


def json_semantics(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def unique_json_object(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    value: Dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise GoldenError("duplicate JSON key {!r}".format(key))
        value[key] = item
    return value


def load_manifest(path: Path) -> Dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_json_object)
    except (OSError, json.JSONDecodeError) as error:
        raise GoldenError("cannot read manifest {}: {}".format(path, error)) from error

    require(isinstance(manifest, dict), "manifest must be a JSON object")
    require(manifest.get("schema") == MANIFEST_SCHEMA, "unsupported manifest schema")
    require(isinstance(manifest.get("corpus_version"), str), "manifest needs corpus_version")
    require(isinstance(manifest.get("prepare"), list), "manifest prepare must be a list")
    for step in manifest["prepare"]:
        require(isinstance(step, dict), "prepare step must be an object")
        require(isinstance(step.get("id"), str) and step["id"], "prepare step needs an id")
        require(isinstance(step.get("argv"), list) and step["argv"], "prepare step needs non-empty argv")
        require(all(isinstance(token, str) for token in step["argv"]), "prepare argv entries must be strings")
    cases = manifest.get("cases")
    require(isinstance(cases, list) and cases, "manifest cases must be a non-empty list")

    seen = set()
    equivalence_groups: Dict[str, List[Dict[str, Any]]] = {}
    for index, case in enumerate(cases):
        prefix = "case {}".format(index)
        require(isinstance(case, dict), "{} must be an object".format(prefix))
        case_id = case.get("id")
        require(isinstance(case_id, str) and case_id, "{} needs an id".format(prefix))
        require(case_id not in seen, "duplicate case id {}".format(case_id))
        seen.add(case_id)
        require(isinstance(case.get("surface"), str), "{} needs a surface".format(case_id))
        equivalence_group = case.get("equivalence_group")
        require(
            equivalence_group is None or isinstance(equivalence_group, str) and bool(equivalence_group),
            "{} equivalence_group must be a non-empty string".format(case_id),
        )
        if equivalence_group is not None:
            equivalence_groups.setdefault(equivalence_group, []).append(case)
        require(isinstance(case.get("parser"), str), "{} needs a parser".format(case_id))
        require(isinstance(case.get("oracle"), dict), "{} needs an oracle".format(case_id))
        require(isinstance(case.get("request"), dict), "{} needs a request".format(case_id))
        require(isinstance(case["request"].get("argv"), list) and case["request"]["argv"], "{} request needs non-empty argv".format(case_id))
        require(all(isinstance(token, str) for token in case["request"]["argv"]), "{} argv entries must be strings".format(case_id))
        require(case["request"].get("budget", {}).get("kind") == "complete", "{} is not a blocking complete-search golden".format(case_id))
        expected = case.get("expected")
        require(isinstance(expected, dict), "{} needs expected semantics".format(case_id))
        for field in ("status", "senses", "objectives", "solution", "validity", "bound", "gap", "proof"):
            require(field in expected, "{} expected record lacks {}".format(case_id, field))
        require(expected["status"] in KNOWN_STATUSES, "{} has unknown expected status".format(case_id))
        require(isinstance(expected["senses"], list), "{} expected senses must be a list".format(case_id))
        require(isinstance(expected["objectives"], list), "{} expected objectives must be a list".format(case_id))
        require(case["request"].get("objective_senses") == expected["senses"], "{} request and expected senses differ".format(case_id))
        if expected["status"] in FEASIBLE_STATUSES:
            require(expected["solution"] is not None and expected["validity"] is True,
                    "{} feasible golden needs a valid canonical solution".format(case_id))
        if expected["status"] in ("OPTIMAL", "UNSATISFIABLE"):
            require(isinstance(expected["proof"], dict) and expected["proof"].get("verified") is True,
                    "{} exact conclusion needs a verified proof record".format(case_id))
        if expected["status"] == "OPTIMAL":
            require(bool(expected["senses"]) and bool(expected["objectives"]),
                    "{} OPTIMAL needs an objective vector and senses".format(case_id))
            require(expected["bound"] is not None, "{} OPTIMAL needs a terminal bound".format(case_id))
        if expected["status"] == "UNSATISFIABLE":
            require(expected["solution"] is None, "{} UNSATISFIABLE must not record a solution".format(case_id))
        if expected["status"] == "SATISFIABLE":
            require(expected["proof"] is None, "{} SATISFIABLE must not carry an exact proof".format(case_id))
        try:
            derived_gap = canonical_gap(expected)
        except GoldenError as error:
            raise GoldenError("{} cannot derive expected gap: {}".format(case_id, error)) from error
        require(
            json_semantics(expected["gap"]) == json_semantics(derived_gap),
            "{} expected gap is not canonical: expected {}, got {}".format(
                case_id,
                json_semantics(derived_gap),
                json_semantics(expected["gap"]),
            ),
        )

        instance = case.get("instance")
        require(isinstance(instance, dict), "{} needs an instance record".format(case_id))
        relative = instance.get("path")
        expected_hash = instance.get("sha256")
        require(isinstance(relative, str), "{} instance needs path".format(case_id))
        require(isinstance(expected_hash, str) and re.fullmatch(r"[0-9a-f]{64}", expected_hash) is not None,
                "{} instance needs a lowercase SHA-256".format(case_id))
        instance_path = repo_path(relative)
        require(instance_path.is_file(), "{} instance does not exist: {}".format(case_id, relative))
        observed_hash = sha256_file(instance_path)
        require(observed_hash == expected_hash,
                "{} instance hash mismatch: expected {}, got {}".format(case_id, expected_hash, observed_hash))

    for group, members in equivalence_groups.items():
        surfaces = {member["surface"] for member in members}
        require(len(members) >= 2, "equivalence group {} needs at least two cases".format(group))
        require(len(surfaces) >= 2, "equivalence group {} needs at least two surfaces".format(group))
    expected_by_id = {case["id"]: case["expected"] for case in cases}
    equivalence_errors = compare_equivalence_groups(cases, expected_by_id)
    require(
        not equivalence_errors,
        "manifest expected equivalence mismatch:\n{}".format("\n".join(equivalence_errors)),
    )
    return manifest


def snapshot_goldens(manifest_path: Path, manifest: Dict[str, Any]) -> Dict[Path, str]:
    paths = {manifest_path.resolve(), Path(__file__).with_name("schema.v1.json").resolve()}
    paths.update(repo_path(case["instance"]["path"]) for case in manifest["cases"])
    return {path: sha256_file(path) for path in paths}


def assert_goldens_unchanged(snapshot: Dict[Path, str]) -> None:
    changed = []
    for path, expected_hash in snapshot.items():
        if not path.is_file() or sha256_file(path) != expected_hash:
            changed.append(str(path))
    require(not changed, "a test command modified protected golden files: {}".format(", ".join(sorted(changed))))


def expand_argv(argv: Sequence[str], instance: Optional[Path], proof: Optional[Path]) -> List[str]:
    replacements = {
        "{repo}": str(REPO_ROOT),
        "{instance}": str(instance) if instance is not None else "",
        "{proof}": str(proof) if proof is not None else "",
    }
    expanded = []
    for token in argv:
        require(isinstance(token, str), "command argv entries must be strings")
        value = token
        for marker, replacement in replacements.items():
            value = value.replace(marker, replacement)
        require("{" not in value and "}" not in value, "unknown command placeholder in {!r}".format(token))
        expanded.append(value)
    return expanded


def execute(argv: Sequence[str], allowed_exit_codes: Iterable[int], timeout_seconds: int, verbose: bool) -> CommandResult:
    command = list(argv)
    if verbose:
        print("  $ {}".format(" ".join(command)))
    try:
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError as error:
        raise GoldenError("command not found: {}".format(command[0])) from error
    except subprocess.TimeoutExpired as error:
        raise GoldenError("command timed out after {} seconds: {}".format(timeout_seconds, " ".join(command))) from error
    result = CommandResult(command, completed.returncode, completed.stdout, completed.stderr)
    if completed.returncode not in set(allowed_exit_codes):
        raise GoldenError(
            "command exited {} (allowed {}): {}\nstdout:\n{}\nstderr:\n{}".format(
                completed.returncode,
                sorted(set(allowed_exit_codes)),
                " ".join(command),
                completed.stdout,
                completed.stderr,
            )
        )
    if verbose and completed.stdout:
        print("  stdout:\n{}".format(completed.stdout.rstrip()))
    if verbose and completed.stderr:
        print("  stderr:\n{}".format(completed.stderr.rstrip()))
    return result


def decode_marker(text: str) -> Dict[str, Any]:
    index = text.find(RESULT_PREFIX)
    if index < 0:
        raise GoldenError("structured result marker is missing")
    payload = text[index + len(RESULT_PREFIX):]
    try:
        value, _ = json.JSONDecoder(object_pairs_hook=unique_json_object).raw_decode(payload)
    except (json.JSONDecodeError, GoldenError) as error:
        raise GoldenError("invalid structured result marker: {}".format(error)) from error
    require(isinstance(value, dict), "structured result must be an object")
    return value


def status_from_competition_output(text: str) -> str:
    statuses = []
    for raw in text.splitlines():
        line = raw.strip()
        if line == "s SATISFIABLE":
            statuses.append("SATISFIABLE")
        elif line == "s UNSATISFIABLE":
            statuses.append("UNSATISFIABLE")
        elif line == "s OPTIMUM FOUND":
            statuses.append("OPTIMAL")
        elif line == "s UNKNOWN":
            statuses.append("UNKNOWN")
        elif line == "s UNSUPPORTED":
            statuses.append("UNSUPPORTED")
    require(bool(statuses), "competition status line is missing")
    require(len(set(statuses)) == 1, "conflicting competition statuses: {}".format(statuses))
    return statuses[-1]


def objective_lines(text: str) -> List[int]:
    values = []
    for raw in text.splitlines():
        match = re.fullmatch(r"o\s+(-?\d+)", raw.strip())
        if match:
            values.append(int(match.group(1)))
    return values


def proof_claim(status: str, kind: str) -> Optional[Dict[str, Any]]:
    if status == "OPTIMAL":
        return {"claim": "optimality", "kind": kind, "verified": False}
    if status == "UNSATISFIABLE":
        return {"claim": "unsatisfiability", "kind": kind, "verified": False}
    return None


def parse_xcsp(case: Dict[str, Any], result: CommandResult, _: Optional[Path]) -> Dict[str, Any]:
    text = result.stdout
    status = status_from_competition_output(text)
    lines = [line.strip() for line in text.splitlines()]

    def block(name: str) -> List[str]:
        start_marker = "v <{}>".format(name)
        end_marker = "v </{}>".format(name)
        if start_marker not in lines:
            return []
        start = lines.index(start_marker) + 1
        try:
            end = lines.index(end_marker, start)
        except ValueError as error:
            raise GoldenError("unterminated XCSP {} block".format(name)) from error
        tokens = []
        for line in lines[start:end]:
            require(line.startswith("v "), "malformed XCSP value line")
            tokens.extend(line[2:].split())
        return tokens

    names = block("list")
    values = block("values")
    solution = None
    if status in FEASIBLE_STATUSES:
        require(len(names) == len(values), "XCSP name/value length mismatch")
        parsed_values = {name: (value if value == "*" else int(value)) for name, value in zip(names, values)}
        solution = {"assignments": parsed_values}
    objectives = objective_lines(text)
    final_objectives = objectives[-1:] if objectives else []
    bound = {"values": list(final_objectives), "source": "terminal-status"} if status == "OPTIMAL" else None
    return {
        "status": status,
        "senses": list(case["request"].get("objective_senses", [])),
        "objectives": final_objectives,
        "solution": solution,
        "bound": bound,
        "proof": proof_claim(status, "terminal-status"),
    }


def parse_flatzinc(case: Dict[str, Any], result: CommandResult, _: Optional[Path]) -> Dict[str, Any]:
    text = result.stdout
    if "=====UNSATISFIABLE=====" in text:
        status = "UNSATISFIABLE"
    elif "=====UNKNOWN=====" in text:
        status = "UNKNOWN"
    elif "==========" in text and case["request"].get("objective_senses"):
        status = "OPTIMAL"
    elif "----------" in text:
        status = "SATISFIABLE"
    else:
        raise GoldenError("FlatZinc terminal marker is missing")
    assignments: Dict[str, Any] = {}
    pattern = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(true|false|-?\d+)\s*;$")
    for raw in text.splitlines():
        match = pattern.fullmatch(raw.strip())
        if match:
            token = match.group(2)
            assignments[match.group(1)] = token == "true" if token in ("true", "false") else int(token)
    solution = {"assignments": assignments} if status in FEASIBLE_STATUSES else None
    bound = {"values": None, "source": "completion-marker"} if status == "OPTIMAL" else None
    return {
        "status": status,
        "senses": list(case["request"].get("objective_senses", [])),
        "objectives": [],
        "solution": solution,
        "bound": bound,
        "proof": proof_claim(status, "completion-marker"),
    }


def parse_boolean_model(line: str) -> Dict[str, bool]:
    assignment: Dict[str, bool] = {}
    for token in line.split()[1:]:
        if token == "0":
            continue
        if token.startswith("-x"):
            assignment[token[1:]] = False
        elif token.startswith("x"):
            assignment[token] = True
        else:
            literal = int(token)
            assignment["x{}".format(abs(literal))] = literal > 0
    return assignment


def parse_sat(case: Dict[str, Any], result: CommandResult, proof_path: Optional[Path]) -> Dict[str, Any]:
    status = status_from_competition_output(result.stdout)
    model_lines = [line.strip() for line in result.stdout.splitlines() if line.strip().startswith("v ")]
    solution = None
    if status in FEASIBLE_STATUSES:
        require(bool(model_lines), "SAT assignment line is missing")
        solution = {"assignments": parse_boolean_model(model_lines[-1])}
    proof = None
    if status == "UNSATISFIABLE" and proof_path is not None:
        require(proof_path.is_file(), "SAT proof artifact was not created")
        proof = {
            "claim": "unsatisfiability",
            "kind": "drat-rup",
            "verified": False,
            "artifact_sha256": sha256_file(proof_path),
            "_artifact_path": str(proof_path),
        }
    elif status == "UNSATISFIABLE":
        proof = proof_claim(status, "terminal-status")
    return {
        "status": status,
        "senses": list(case["request"].get("objective_senses", [])),
        "objectives": [],
        "solution": solution,
        "bound": None,
        "proof": proof,
    }


def parse_pb(case: Dict[str, Any], result: CommandResult, _: Optional[Path]) -> Dict[str, Any]:
    status = status_from_competition_output(result.stdout)
    model_lines = [line.strip() for line in result.stdout.splitlines() if line.strip().startswith("v ")]
    solution = None
    if status in FEASIBLE_STATUSES:
        require(bool(model_lines), "PB assignment line is missing")
        solution = {"assignments": parse_boolean_model(model_lines[-1])}
    objectives = objective_lines(result.stdout)
    final_objectives = objectives[-1:] if objectives else []
    bound = {"values": list(final_objectives), "source": "terminal-status"} if status == "OPTIMAL" else None
    return {
        "status": status,
        "senses": list(case["request"].get("objective_senses", [])),
        "objectives": final_objectives,
        "solution": solution,
        "bound": bound,
        "proof": proof_claim(status, "terminal-status"),
    }


PARSERS = {
    "structured-json": lambda case, result, proof: decode_marker(result.stdout + "\n" + result.stderr),
    "xcsp": parse_xcsp,
    "flatzinc": parse_flatzinc,
    "sat": parse_sat,
    "pb": parse_pb,
}


def relation_holds(value: int, relation: str, rhs: int) -> bool:
    if relation == "eq":
        return value == rhs
    if relation == "ne":
        return value != rhs
    if relation == "le":
        return value <= rhs
    if relation == "ge":
        return value >= rhs
    if relation == "lt":
        return value < rhs
    if relation == "gt":
        return value > rhs
    raise GoldenError("unsupported oracle relation {}".format(relation))


def linear_value(assignment: Dict[str, Any], expression: Dict[str, Any]) -> int:
    total = int(expression.get("constant", 0))
    for name, coefficient in expression.get("terms", {}).items():
        total += int(coefficient) * int(assignment[name])
    return total


def lex_key(values: Sequence[int], objectives: Sequence[Dict[str, Any]]) -> Tuple[int, ...]:
    return tuple(value if objective["sense"] == "minimize" else -value for value, objective in zip(values, objectives))


def finite_domain_oracle(model: Dict[str, Any], actual: Dict[str, Any]) -> OracleOutcome:
    variables = model["variables"]
    names = [entry["name"] for entry in variables]
    domains = []
    for entry in variables:
        domain = entry["domain"]
        if isinstance(domain, dict):
            domains.append(list(range(int(domain["min"]), int(domain["max"]) + 1)))
        else:
            domains.append([int(value) for value in domain])
    objectives = model.get("objectives", [])
    feasible: List[Tuple[Dict[str, int], List[int]]] = []
    for values in itertools.product(*domains):
        assignment = dict(zip(names, values))
        if all(
            relation_holds(linear_value(assignment, constraint), constraint["relation"], int(constraint["rhs"]))
            for constraint in model.get("constraints", [])
        ):
            vector = [linear_value(assignment, objective) for objective in objectives]
            feasible.append((assignment, vector))

    optimum = min((vector for _, vector in feasible), key=lambda vector: lex_key(vector, objectives)) if feasible and objectives else ([] if feasible else None)
    solution = actual.get("solution")
    valid = False
    actual_vector = None
    if actual.get("status") in FEASIBLE_STATUSES and isinstance(solution, dict):
        candidate = solution.get("assignments")
        if isinstance(candidate, dict) and set(candidate) == set(names):
            candidate_int = {name: int(candidate[name]) for name in names}
            in_domains = all(candidate_int[name] in domains[index] for index, name in enumerate(names))
            constraints_hold = in_domains and all(
                relation_holds(linear_value(candidate_int, constraint), constraint["relation"], int(constraint["rhs"]))
                for constraint in model.get("constraints", [])
            )
            valid = constraints_hold
            if valid:
                actual_vector = [linear_value(candidate_int, objective) for objective in objectives]
    return OracleOutcome(bool(feasible), valid, actual_vector, optimum)


def parse_cnf(path: Path) -> Tuple[int, List[List[int]]]:
    variables = None
    expected_clauses = None
    clauses: List[List[int]] = []
    current: List[int] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("c"):
            continue
        if line.startswith("p "):
            fields = line.split()
            require(len(fields) == 4 and fields[:2] == ["p", "cnf"], "invalid CNF header")
            variables, expected_clauses = int(fields[2]), int(fields[3])
            continue
        for token in line.split():
            literal = int(token)
            if literal == 0:
                clauses.append(current)
                current = []
            else:
                current.append(literal)
    require(variables is not None and expected_clauses is not None, "CNF header is missing")
    require(not current, "unterminated CNF clause")
    require(len(clauses) == expected_clauses, "CNF clause count mismatch")
    return variables, clauses


def cnf_satisfied(clauses: Sequence[Sequence[int]], assignment: Dict[str, Any]) -> bool:
    return all(
        any(bool(assignment["x{}".format(abs(literal))]) == (literal > 0) for literal in clause)
        for clause in clauses
    )


def cnf_oracle(path: Path, actual: Dict[str, Any]) -> OracleOutcome:
    variable_count, clauses = parse_cnf(path)
    names = ["x{}".format(index) for index in range(1, variable_count + 1)]
    feasible = []
    for values in itertools.product((False, True), repeat=variable_count):
        assignment = dict(zip(names, values))
        if cnf_satisfied(clauses, assignment):
            feasible.append(assignment)
    solution = actual.get("solution")
    valid = False
    if actual.get("status") in FEASIBLE_STATUSES and isinstance(solution, dict):
        candidate = solution.get("assignments")
        valid = isinstance(candidate, dict) and set(candidate) == set(names) and cnf_satisfied(clauses, candidate)
    return OracleOutcome(bool(feasible), valid, [] if valid else None, [] if feasible else None)


def route_cost(route: Sequence[int], travel: Sequence[Sequence[int]], depot: int) -> int:
    nodes = [depot] + list(route) + [depot]
    return sum(int(travel[left][right]) for left, right in zip(nodes, nodes[1:]))


def ordered_route_partitions(items: Sequence[int], count: int, optional: bool) -> Iterable[List[List[int]]]:
    slots: List[Optional[int]] = list(range(count))
    if optional:
        slots.append(None)
    for allocation in itertools.product(slots, repeat=len(items)):
        unordered = [[] for _ in range(count)]
        for item, slot in zip(items, allocation):
            if slot is not None:
                unordered[slot].append(item)
        orderings = [list(itertools.permutations(route)) if route else [tuple()] for route in unordered]
        for ordered in itertools.product(*orderings):
            yield [list(route) for route in ordered]


def route_oracle(spec: Dict[str, Any], actual: Dict[str, Any]) -> OracleOutcome:
    kind = spec["kind"]
    items = spec["items"] if kind == "lists" else spec["customers"]
    count = int(spec["list_count"] if kind == "lists" else spec["vehicles"])
    optional = bool(spec.get("optional", False))
    key = "lists" if kind == "lists" else "routes"
    travel = spec["travel"]
    depot = int(spec["depot"])
    sense = spec["objective"]["sense"]

    candidates = list(ordered_route_partitions(items, count, optional))
    values = [sum(route_cost(route, travel, depot) for route in routes) for routes in candidates]
    optimum_value = (min(values) if sense == "minimize" else max(values)) if values else None

    valid = False
    actual_vector = None
    solution = actual.get("solution")
    if actual.get("status") in FEASIBLE_STATUSES and isinstance(solution, dict):
        routes = solution.get(key)
        if isinstance(routes, list) and len(routes) == count and all(isinstance(route, list) for route in routes):
            flattened = [item for route in routes for item in route]
            no_duplicates = len(flattened) == len(set(flattened))
            allowed = set(flattened).issubset(set(items))
            coverage = optional or set(flattened) == set(items)
            valid = no_duplicates and allowed and coverage
            if valid:
                actual_vector = [sum(route_cost(route, travel, depot) for route in routes)]
    return OracleOutcome(bool(candidates), valid, actual_vector, None if optimum_value is None else [optimum_value])


def schedule_feasible(spec: Dict[str, Any], starts: Dict[str, int]) -> bool:
    tasks = {str(entry["id"]): entry for entry in spec["tasks"]}
    if set(starts) != set(tasks):
        return False
    for task_id, task in tasks.items():
        start = starts[task_id]
        if not isinstance(start, int) or start < 0 or start + int(task["duration"]) > int(spec["horizon"]):
            return False
    for before, after in spec.get("precedences", []):
        before_key, after_key = str(before), str(after)
        if starts[before_key] + int(tasks[before_key]["duration"]) > starts[after_key]:
            return False
    for resource in spec.get("resources", []):
        demand_index = int(resource["demand_index"])
        capacity = int(resource["capacity"])
        for instant in range(int(spec["horizon"])):
            load = 0
            for task_id, task in tasks.items():
                start = starts[task_id]
                if start <= instant < start + int(task["duration"]):
                    load += int(task.get("demands", [])[demand_index])
            if load > capacity:
                return False
    return True


def schedule_objective(spec: Dict[str, Any], starts: Dict[str, int]) -> int:
    tasks = {str(entry["id"]): entry for entry in spec["tasks"]}
    return max(starts[task_id] + int(task["duration"]) for task_id, task in tasks.items())


def schedule_oracle(spec: Dict[str, Any], actual: Dict[str, Any]) -> OracleOutcome:
    task_entries = spec["tasks"]
    names = [str(entry["id"]) for entry in task_entries]
    domains = [range(0, int(spec["horizon"]) - int(entry["duration"]) + 1) for entry in task_entries]
    feasible_values = []
    for values in itertools.product(*domains):
        starts = dict(zip(names, values))
        if schedule_feasible(spec, starts):
            feasible_values.append(schedule_objective(spec, starts))
    sense = spec["objective"]["sense"]
    optimum_value = (min(feasible_values) if sense == "minimize" else max(feasible_values)) if feasible_values else None

    valid = False
    actual_vector = None
    solution = actual.get("solution")
    if actual.get("status") in FEASIBLE_STATUSES and isinstance(solution, dict) and isinstance(solution.get("starts"), dict):
        starts = solution["starts"]
        valid = schedule_feasible(spec, starts)
        if valid:
            actual_vector = [schedule_objective(spec, starts)]
    return OracleOutcome(bool(feasible_values), valid, actual_vector, None if optimum_value is None else [optimum_value])


def rup_conflict(clauses: Sequence[Sequence[int]], added_clause: Sequence[int]) -> bool:
    assignment: Dict[int, bool] = {}

    def assign(literal: int) -> Optional[bool]:
        variable, value = abs(literal), literal > 0
        if variable in assignment:
            return False if assignment[variable] == value else None
        assignment[variable] = value
        return True

    for literal in added_clause:
        if assign(-literal) is None:
            return True
    changed = True
    while changed:
        changed = False
        for clause in clauses:
            unresolved = []
            satisfied = False
            for literal in clause:
                variable = abs(literal)
                if variable not in assignment:
                    unresolved.append(literal)
                elif assignment[variable] == (literal > 0):
                    satisfied = True
                    break
            if satisfied:
                continue
            if not unresolved:
                return True
            if len(unresolved) == 1:
                assigned = assign(unresolved[0])
                if assigned is None:
                    return True
                changed = changed or assigned
    return False


def verify_rup_drat(instance_path: Path, proof_path: Path) -> bool:
    _, clauses = parse_cnf(instance_path)
    database = [list(clause) for clause in clauses]
    derived_empty = False
    try:
        lines = proof_path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeDecodeError):
        return False
    if not lines:
        return False
    for raw in lines:
        fields = raw.split()
        deleting = bool(fields and fields[0] == "d")
        if deleting:
            fields = fields[1:]
        if not fields or fields[-1] != "0":
            return False
        try:
            clause = [int(token) for token in fields[:-1]]
        except ValueError:
            return False
        if any(literal == 0 for literal in clause):
            return False
        if deleting:
            try:
                database.remove(clause)
            except ValueError:
                return False
            continue
        if not rup_conflict(database, clause):
            return False
        database.append(clause)
        if not clause:
            derived_empty = True
    return derived_empty


def oracle_for(case: Dict[str, Any], actual: Dict[str, Any], instance_path: Path) -> OracleOutcome:
    oracle = case["oracle"]
    kind = oracle["kind"]
    if kind == "finite-domain":
        return finite_domain_oracle(oracle["model"], actual)
    if kind == "cnf":
        return cnf_oracle(instance_path, actual)
    if kind == "python-spec":
        spec = json.loads(instance_path.read_text(encoding="utf-8"))
        if spec["kind"] == "integer":
            return finite_domain_oracle(spec, actual)
        if spec["kind"] in ("lists", "routing"):
            return route_oracle(spec, actual)
        if spec["kind"] == "scheduling":
            return schedule_oracle(spec, actual)
        raise GoldenError("unsupported Python oracle kind {}".format(spec["kind"]))
    raise GoldenError("unsupported oracle kind {}".format(kind))


def apply_oracle(case: Dict[str, Any], actual: Dict[str, Any], instance_path: Path) -> OracleOutcome:
    status = actual.get("status")
    require(status in KNOWN_STATUSES, "parser produced unknown status {!r}".format(status))
    outcome = oracle_for(case, actual, instance_path)

    if status in FEASIBLE_STATUSES:
        actual["validity"] = outcome.valid_solution
        if outcome.valid_solution and not actual.get("objectives") and outcome.actual_objectives is not None:
            actual["objectives"] = list(outcome.actual_objectives)
    elif status == "UNSATISFIABLE":
        actual["validity"] = not outcome.feasible_exists
    else:
        actual["validity"] = actual.get("solution") is None

    bound = actual.get("bound")
    if isinstance(bound, dict) and bound.get("values") is None:
        bound["values"] = list(actual.get("objectives", []))
    actual["gap"] = canonical_gap(actual)

    proof = actual.get("proof")
    if isinstance(proof, dict):
        claim = proof.get("claim")
        if claim == "optimality":
            proof["verified"] = bool(
                outcome.exhaustive
                and status == "OPTIMAL"
                and outcome.valid_solution
                and outcome.optimum is not None
                and list(actual.get("objectives", [])) == list(outcome.optimum)
            )
        elif claim == "unsatisfiability":
            proof["verified"] = bool(outcome.exhaustive and status == "UNSATISFIABLE" and not outcome.feasible_exists)
        else:
            proof["verified"] = False
        artifact = proof.pop("_artifact_path", None)
        if proof.get("kind") == "drat-rup":
            proof["verified"] = bool(
                proof["verified"] and artifact is not None and verify_rup_drat(instance_path, Path(artifact))
            )
    return outcome


def compare_semantics(case: Dict[str, Any], actual: Dict[str, Any], outcome: OracleOutcome) -> List[str]:
    expected = case["expected"]
    errors = []
    for field in ("status", "senses", "objectives", "solution", "validity", "bound", "gap", "proof"):
        expected_json = json_semantics(expected.get(field))
        actual_json = json_semantics(actual.get(field))
        if actual_json != expected_json:
            errors.append(
                "{} mismatch: expected {}, got {}".format(
                    field,
                    expected_json,
                    actual_json,
                )
            )

    try:
        derived_gap = canonical_gap(actual)
    except GoldenError as error:
        errors.append("gap cannot be derived: {}".format(error))
    else:
        if json_semantics(actual.get("gap")) != json_semantics(derived_gap):
            errors.append(
                "gap is not canonical: expected {}, got {}".format(
                    json_semantics(derived_gap),
                    json_semantics(actual.get("gap")),
                )
            )

    status = actual.get("status")
    if status in FEASIBLE_STATUSES and not actual.get("validity"):
        errors.append("published solution is not valid under the independent oracle")
    if status == "UNSATISFIABLE" and outcome.feasible_exists:
        errors.append("solver claimed UNSATISFIABLE but the oracle found a solution")
    if status == "OPTIMAL":
        if not isinstance(actual.get("proof"), dict) or actual["proof"].get("verified") is not True:
            errors.append("OPTIMAL lacks a verified proof claim")
        if outcome.optimum is None or actual.get("objectives") != outcome.optimum:
            errors.append("OPTIMAL objective does not match the exhaustive oracle")
    if status == "UNSATISFIABLE":
        if not isinstance(actual.get("proof"), dict) or actual["proof"].get("verified") is not True:
            errors.append("UNSATISFIABLE lacks a verified proof claim")

    bound = actual.get("bound")
    objectives = actual.get("objectives") or []
    senses = actual.get("senses") or []
    if isinstance(bound, dict) and bound.get("values") and objectives:
        values = bound["values"]
        if len(values) > len(objectives):
            errors.append("bound vector is longer than objective vector")
        for index, value in enumerate(values):
            if index >= len(objectives) or index >= len(senses):
                break
            if senses[index] == "minimize" and value > objectives[index]:
                errors.append("minimization bound exceeds incumbent at objective {}".format(index))
            if senses[index] == "maximize" and value < objectives[index]:
                errors.append("maximization bound is below incumbent at objective {}".format(index))
        if status == "OPTIMAL" and values != objectives[:len(values)]:
            errors.append("terminal optimal bound does not close the represented objective prefix")
    return errors


def equivalence_projection(record: Dict[str, Any]) -> Dict[str, Any]:
    bound = record.get("bound")
    bound_values = bound.get("values") if isinstance(bound, dict) else None
    return {
        "status": record.get("status"),
        "senses": record.get("senses"),
        "objectives": record.get("objectives"),
        "solution": record.get("solution"),
        "validity": record.get("validity"),
        "bound_values": bound_values,
        "gap": record.get("gap"),
    }


def compare_equivalence_groups(
    cases: Sequence[Dict[str, Any]],
    records_by_case_id: Dict[str, Dict[str, Any]],
) -> List[str]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for case in cases:
        group = case.get("equivalence_group")
        if isinstance(group, str) and case["id"] in records_by_case_id:
            groups.setdefault(group, []).append(case)

    errors = []
    for group, members in groups.items():
        if len(members) < 2:
            continue
        reference_case = members[0]
        reference = equivalence_projection(records_by_case_id[reference_case["id"]])
        for case in members[1:]:
            candidate = equivalence_projection(records_by_case_id[case["id"]])
            for field, reference_value in reference.items():
                candidate_value = candidate[field]
                if json_semantics(candidate_value) != json_semantics(reference_value):
                    errors.append(
                        "equivalence group {} {} mismatch between {} [{}] and {} [{}]: {} != {}".format(
                            group,
                            field,
                            reference_case["id"],
                            reference_case["surface"],
                            case["id"],
                            case["surface"],
                            json_semantics(reference_value),
                            json_semantics(candidate_value),
                        )
                    )
    return errors


def run_case(case: Dict[str, Any], verbose: bool) -> Tuple[Dict[str, Any], OracleOutcome]:
    instance_path = repo_path(case["instance"]["path"])
    request = case["request"]
    with tempfile.TemporaryDirectory(prefix="qayd-golden-") as temporary:
        proof_path = Path(temporary) / "proof.drat" if "{proof}" in request["argv"] else None
        argv = expand_argv(request["argv"], instance_path, proof_path)
        result = execute(
            argv,
            request.get("allowed_exit_codes", [0]),
            int(request.get("execution_timeout_seconds", 60)),
            verbose,
        )
        try:
            parser = PARSERS[case["parser"]]
        except KeyError as error:
            raise GoldenError("unknown parser {}".format(case["parser"])) from error
        actual = parser(case, result, proof_path)
        outcome = apply_oracle(case, actual, instance_path)
        errors = compare_semantics(case, actual, outcome)
        if errors:
            raise GoldenError("\n".join(errors))
        return actual, outcome


def run_prepare(manifest: Dict[str, Any], verbose: bool) -> None:
    for step in manifest["prepare"]:
        require(isinstance(step, dict), "prepare step must be an object")
        name = step.get("id", "prepare")
        print("PREP {}".format(name))
        argv = expand_argv(step["argv"], None, None)
        execute(argv, step.get("allowed_exit_codes", [0]), int(step.get("timeout_seconds", 900)), verbose)


def select_cases(manifest: Dict[str, Any], case_ids: Sequence[str], surfaces: Sequence[str]) -> List[Dict[str, Any]]:
    cases = manifest["cases"]
    known_ids = {case["id"] for case in cases}
    missing = sorted(set(case_ids) - known_ids)
    require(not missing, "unknown case ids: {}".format(", ".join(missing)))
    selected = [
        case for case in cases
        if (not case_ids or case["id"] in case_ids) and (not surfaces or case["surface"] in surfaces)
    ]
    require(bool(selected), "case filters selected nothing")
    return selected


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--case", action="append", default=[], help="run one case id, repeatable")
    parser.add_argument("--surface", action="append", default=[], help="run one surface, repeatable")
    parser.add_argument("--list", action="store_true", help="list selected cases without running commands")
    parser.add_argument("--no-prepare", action="store_true", help="use existing Rust binaries and Python extension")
    parser.add_argument("--keep-going", action="store_true", help="continue after a failed case")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    try:
        manifest_path = args.manifest.resolve()
        manifest = load_manifest(manifest_path)
        protected_goldens = snapshot_goldens(manifest_path, manifest)
        cases = select_cases(manifest, args.case, args.surface)
        if args.list:
            for case in cases:
                print("{}\t{}".format(case["id"], case["surface"]))
            return 0
        if not args.no_prepare:
            run_prepare(manifest, args.verbose)
            assert_goldens_unchanged(protected_goldens)
    except GoldenError as error:
        print("ERROR {}".format(error), file=sys.stderr)
        return 2

    failures = []
    records_by_case_id: Dict[str, Dict[str, Any]] = {}
    for case in cases:
        try:
            actual, _ = run_case(case, args.verbose)
            assert_goldens_unchanged(protected_goldens)
            records_by_case_id[case["id"]] = actual
            print(
                "PASS {} [{}] status={} objectives={}".format(
                    case["id"], case["surface"], actual["status"], actual["objectives"]
                )
            )
        except GoldenError as error:
            try:
                assert_goldens_unchanged(protected_goldens)
            except GoldenError as mutation_error:
                error = GoldenError("{}\n{}".format(error, mutation_error))
            failures.append((case["id"], str(error)))
            print("FAIL {} [{}]\n{}".format(case["id"], case["surface"], error), file=sys.stderr)
            if not args.keep_going:
                break
    equivalence_errors = [] if failures else compare_equivalence_groups(cases, records_by_case_id)
    if equivalence_errors:
        message = "\n".join(equivalence_errors)
        failures.append(("equivalence", message))
        print("FAIL equivalence groups\n{}".format(message), file=sys.stderr)
    if failures:
        print("{} of {} selected golden cases failed".format(len(failures), len(cases)), file=sys.stderr)
        return 1
    print("PASS all {} selected golden cases, corpus {}".format(len(cases), manifest["corpus_version"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
