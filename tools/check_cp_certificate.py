#!/usr/bin/env python3
"""Independent bounded checker for the frozen qayd-cp-v1 certificate grammar.

This module uses only Python's standard library. It neither imports Qayd nor
calls a solver. Linear deductions use interval arithmetic; table deductions
scan the original supports. No search or input-code evaluation is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import sys
import time
from dataclasses import dataclass
from typing import Optional

I32_MIN, I32_MAX = -(1 << 31), (1 << 31) - 1
I64_MIN, I64_MAX = -(1 << 63), (1 << 63) - 1
I128_MIN, I128_MAX = -(1 << 127), (1 << 127) - 1
MAX_VARIABLES, MAX_CONSTRAINTS, MAX_INPUT_ENTRIES = 64, 512, 100_000
RELATIONS = frozenset(("eq", "ne", "le", "lt", "ge", "gt"))
NEGATED = {"eq": "ne", "ne": "eq", "le": "gt", "lt": "ge", "ge": "lt", "gt": "le"}
DECIMAL = re.compile(r"-?[0-9]+\Z", re.ASCII)
UNSIGNED = re.compile(r"[0-9]+\Z", re.ASCII)
SHA256 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
RECORD_ASCII = re.compile(r"[ -~]+\Z", re.ASCII)


class CertificateRejected(Exception):
    """Malformed input or an unjustified claimed conclusion."""


class CheckLimitExceeded(Exception):
    """Checking did not finish within its explicit resource bounds."""


@dataclass(frozen=True)
class VerificationResult:
    model_sha256: str
    claim: str
    steps: int
    assumption_count: int

    def acknowledgement(self) -> str:
        return f"VERIFIED qayd-cp-v1 {self.model_sha256} {self.claim} {self.steps} {self.assumption_count}"


class _Budget:
    def __init__(self, max_work: int, time_limit: Optional[float]):
        self.remaining = max_work
        self.deadline = None if time_limit is None else time.monotonic() + time_limit

    def spend(self, work: int = 1) -> None:
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise CheckLimitExceeded("time limit reached")
        if work > self.remaining:
            raise CheckLimitExceeded("work limit reached")
        self.remaining -= work


class _Records:
    def __init__(self, data: bytes, label: str, limit: int, budget: _Budget, max_records: int):
        budget.spend()
        if not isinstance(data, bytes):
            raise CertificateRejected(f"{label} must be bytes")
        if len(data) > limit:
            raise CheckLimitExceeded(f"{label} byte limit reached")
        if not data or not data.endswith(b"\n"):
            raise CertificateRejected(f"{label} is empty or not LF terminated")
        if data.count(b"\n") > max_records:
            raise CheckLimitExceeded(f"{label} record limit reached")
        try:
            text = data.decode("ascii")
        except UnicodeDecodeError as error:
            raise CertificateRejected(f"{label} is not ASCII") from error
        self.lines = text.split("\n")
        self.lines.pop()
        self.index = 0
        self.label = label
        self.budget = budget

    def next(self) -> list[str]:
        self.budget.spend()
        if self.index >= len(self.lines):
            raise CertificateRejected(f"truncated {self.label}")
        line = self.lines[self.index]
        self.index += 1
        # LF is the sole record delimiter. Tabs, CR and other controls are not
        # interchangeable whitespace in the certificate grammar.
        if not RECORD_ASCII.fullmatch(line):
            raise CertificateRejected(f"invalid {self.label} record {self.index}")
        self.budget.spend(line.count(" ") + 1)
        tokens = line.split(" ")
        if any(not token for token in tokens):
            raise CertificateRejected(f"invalid spacing in {self.label} record {self.index}")
        return tokens

    def end(self) -> None:
        if self.index != len(self.lines):
            raise CertificateRejected(f"trailing {self.label} records")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CertificateRejected(message)


def _integer(token: str, *, minimum: int = I128_MIN, maximum: int = I128_MAX) -> int:
    # All grammar integers fit i128. Reject huge digit strings before int(),
    # independently of the interpreter's configurable conversion limit.
    if len(token) > 40 or not DECIMAL.fullmatch(token):
        raise CertificateRejected("invalid exact integer")
    value = int(token)
    _require(minimum <= value <= maximum, "integer outside allowed range")
    return value


def _natural(token: str) -> int:
    _require(bool(UNSIGNED.fullmatch(token)), "invalid nonnegative integer")
    return _integer(token, minimum=0)


def _variable(token: str, count: int) -> int:
    value = _natural(token)
    _require(value < count, "unknown variable")
    return value


def _relation(token: str) -> str:
    _require(token in RELATIONS, "unknown relation")
    return token


def _predicate(tokens: list[str], count: int) -> tuple[int, str, int]:
    _require(len(tokens) == 3, "predicate arity mismatch")
    return (_variable(tokens[0], count), _relation(tokens[1]), _integer(tokens[2], minimum=I32_MIN, maximum=I32_MAX))


def _holds(left: int, relation: str, right: int) -> bool:
    if relation == "eq":
        return left == right
    if relation == "ne":
        return left != right
    if relation == "le":
        return left <= right
    if relation == "lt":
        return left < right
    if relation == "ge":
        return left >= right
    return left > right


# Domains are immutable tuples of disjoint inclusive integer ranges.
def _restrict(domain: tuple, relation: str, value: int, budget: _Budget) -> tuple:
    result = []
    for lo, hi in domain:
        budget.spend()
        if relation == "eq":
            if lo <= value <= hi:
                result.append((value, value))
        elif relation == "ne":
            if value < lo or value > hi:
                result.append((lo, hi))
            else:
                if lo < value:
                    result.append((lo, value - 1))
                if value < hi:
                    result.append((value + 1, hi))
        elif relation in ("le", "lt"):
            last = min(hi, value if relation == "le" else value - 1)
            if lo <= last:
                result.append((lo, last))
        else:
            first = max(lo, value if relation == "ge" else value + 1)
            if first <= hi:
                result.append((first, hi))
    return tuple(result)


def _apply(domains: list, predicate: tuple, budget: _Budget) -> None:
    variable, relation, value = predicate
    domains[variable] = _restrict(domains[variable], relation, value, budget)


def _contains(domain: tuple, value: int, budget: _Budget) -> bool:
    for lo, hi in domain:
        budget.spend()
        if lo <= value <= hi:
            return True
        if lo > value:
            return False
    return False


@dataclass(frozen=True)
class _Linear:
    relation: str
    rhs: int
    terms: tuple


@dataclass(frozen=True)
class _Table:
    variables: tuple
    rows: tuple


@dataclass(frozen=True)
class _Model:
    domains: tuple
    constraints: tuple
    assumptions: tuple
    objective: Optional[tuple]


def _model(data: bytes, limit: int, budget: _Budget) -> _Model:
    records = _Records(data, "model", limit, budget, MAX_INPUT_ENTRIES + MAX_VARIABLES + MAX_CONSTRAINTS + 10)
    _require(records.next() == ["qayd-cp-model", "1"], "unsupported model header")
    tokens = records.next()
    _require(len(tokens) == 2 and tokens[0] == "vars", "expected variable count")
    count = _natural(tokens[1])
    if count > MAX_VARIABLES:
        raise CheckLimitExceeded("variable limit reached")
    domains = []
    entries = 0

    def add_entries(amount: int) -> None:
        nonlocal entries
        entries += amount
        budget.spend(amount)
        if entries > MAX_INPUT_ENTRIES:
            raise CheckLimitExceeded("model entry limit reached")

    for index in range(count):
        tokens = records.next()
        _require(len(tokens) >= 4 and tokens[0] == "var" and tokens[2] == "ranges", "expected variable ranges")
        _require(_natural(tokens[1]) == index, "variable IDs must be consecutive")
        ranges = _natural(tokens[3])
        _require(ranges > 0 and len(tokens) == 4 + 2 * ranges, "domain range count mismatch")
        add_entries(ranges)
        domain = []
        for position in range(ranges):
            lo = _integer(tokens[4 + 2 * position], minimum=I32_MIN, maximum=I32_MAX)
            hi = _integer(tokens[5 + 2 * position], minimum=I32_MIN, maximum=I32_MAX)
            _require(lo <= hi, "empty root range")
            _require(not domain or domain[-1][1] + 1 < lo, "root ranges must be increasing, disjoint and nonadjacent")
            domain.append((lo, hi))
        domains.append(tuple(domain))
    tokens = records.next()
    _require(len(tokens) == 2 and tokens[0] == "constraints", "expected constraint count")
    constraint_count = _natural(tokens[1])
    if constraint_count > MAX_CONSTRAINTS:
        raise CheckLimitExceeded("constraint limit reached")
    constraints = []
    for index in range(constraint_count):
        tokens = records.next()
        _require(len(tokens) >= 2 and _natural(tokens[1]) == index, "constraint IDs must be consecutive")
        if tokens[0] == "linear":
            _require(len(tokens) >= 5, "truncated linear constraint")
            relation, rhs, term_count = _relation(tokens[2]), _integer(tokens[3]), _natural(tokens[4])
            _require(len(tokens) == 5 + 2 * term_count, "linear term count mismatch")
            add_entries(term_count)
            combined = {}
            for position in range(term_count):
                coefficient = _integer(tokens[5 + 2 * position])
                variable = _variable(tokens[6 + 2 * position], count)
                coefficient += combined.get(variable, 0)
                _require(I128_MIN <= coefficient <= I128_MAX, "combined coefficient overflows i128")
                combined[variable] = coefficient
            terms = tuple((variable, coefficient) for variable, coefficient in sorted(combined.items()) if coefficient)
            constraints.append(_Linear(relation, rhs, terms))
        elif tokens[0] == "table":
            _require(len(tokens) >= 4, "truncated table constraint")
            arity, row_count = _natural(tokens[2]), _natural(tokens[3])
            _require(len(tokens) == 4 + arity, "table arity mismatch")
            add_entries(arity + row_count + arity * row_count)
            variables = tuple(_variable(token, count) for token in tokens[4:])
            rows = []
            for _ in range(row_count):
                row = records.next()
                _require(len(row) == 1 + arity and row[0] == "row", "table row arity mismatch")
                rows.append(tuple(None if cell == "*" else _integer(cell, minimum=I32_MIN, maximum=I32_MAX) for cell in row[1:]))
            constraints.append(_Table(variables, tuple(rows)))
        else:
            raise CertificateRejected("unknown constraint kind")
    tokens = records.next()
    _require(len(tokens) == 2 and tokens[0] == "assumptions", "expected assumption count")
    assumption_count = _natural(tokens[1])
    add_entries(assumption_count)
    assumptions = []
    for _ in range(assumption_count):
        tokens = records.next()
        _require(tokens[0] == "assume", "expected assumption")
        assumptions.append(_predicate(tokens[1:], count))
    tokens = records.next()
    if tokens == ["objective", "none"]:
        objective = None
    else:
        _require(len(tokens) == 4 and tokens[0] == "objective" and tokens[1] in ("min", "max"), "invalid objective")
        _require(tokens[2] in ("var", "const"), "unsupported objective")
        value = _variable(tokens[3], count) if tokens[2] == "var" else _integer(tokens[3], minimum=I64_MIN, maximum=I64_MAX)
        objective = (tokens[1], tokens[2], value)
    _require(records.next() == ["end-model"], "expected end-model")
    records.end()
    return _Model(tuple(domains), tuple(constraints), tuple(assumptions), objective)


def _activity_impossible(minimum: int, maximum: int, relation: str, rhs: int) -> bool:
    if relation == "eq":
        return rhs < minimum or rhs > maximum
    if relation == "ne":
        return minimum == maximum == rhs
    if relation == "le":
        return minimum > rhs
    if relation == "lt":
        return minimum >= rhs
    if relation == "ge":
        return maximum < rhs
    return maximum <= rhs


def _constraint_impossible(constraint, domains: list, budget: _Budget) -> bool:
    if isinstance(constraint, _Linear):
        minimum = maximum = 0
        for variable, coefficient in constraint.terms:
            budget.spend()
            lo, hi = domains[variable][0][0], domains[variable][-1][1]
            minimum += coefficient * (lo if coefficient > 0 else hi)
            maximum += coefficient * (hi if coefficient > 0 else lo)
        return _activity_impossible(minimum, maximum, constraint.relation, constraint.rhs)
    for row in constraint.rows:
        budget.spend()
        fixed = {}
        compatible = True
        for variable, cell in zip(constraint.variables, row):
            budget.spend()
            if cell is None:
                continue
            if variable in fixed and fixed[variable] != cell:
                compatible = False
                break
            if not _contains(domains[variable], cell, budget):
                compatible = False
                break
            fixed[variable] = cell
        if compatible:
            return False
    return True


def _check_witness(model: _Model, witness: tuple, budget: _Budget) -> None:
    for variable, value in enumerate(witness):
        _require(_contains(model.domains[variable], value, budget), "witness outside original domain")
    for variable, relation, value in model.assumptions:
        budget.spend()
        _require(_holds(witness[variable], relation, value), "witness violates an assumption")
    for constraint in model.constraints:
        budget.spend()
        if isinstance(constraint, _Linear):
            total = 0
            for variable, coefficient in constraint.terms:
                budget.spend()
                total += coefficient * witness[variable]
            valid = _holds(total, constraint.relation, constraint.rhs)
        else:
            valid = False
            for row in constraint.rows:
                budget.spend()
                row_matches = True
                for variable, cell in zip(constraint.variables, row):
                    budget.spend()
                    if cell is not None and witness[variable] != cell:
                        row_matches = False
                        break
                if row_matches:
                    valid = True
                    break
        _require(valid, "witness violates a model constraint")


@dataclass
class _Frame:
    split_id: int
    variable: int
    pivot: int
    parent_domains: list
    parent_path: tuple
    history_length: int
    right_id: Optional[int] = None
    left_proof: Optional[int] = None


def _verify(model: _Model, proof: bytes, model_hash: str, claim: str, max_proof_bytes: int,
            max_steps: int, max_depth: int, budget: _Budget) -> VerificationResult:
    records = _Records(proof, "proof", max_proof_bytes, budget, max_steps + 5)
    _require(records.next() == ["qayd-cp-proof", "1", model_hash, claim], "proof header hash or claim mismatch")
    tokens = records.next()
    if claim == "unsat":
        _require(tokens == ["witness", "none"], "UNSAT requires witness none")
        witness = None
    else:
        _require(len(tokens) >= 2 and tokens[0] == "witness", "missing witness")
        _require(_natural(tokens[1]) == len(model.domains) and len(tokens) == 2 + len(model.domains), "witness size mismatch")
        witness = tuple(_integer(token, minimum=I32_MIN, maximum=I32_MAX) for token in tokens[2:])
        _check_witness(model, witness, budget)
    tokens = records.next()
    if claim == "optimal":
        _require(model.objective is not None, "optimal claim requires an objective")
        direction, kind, objective_value = model.objective
        evaluated = witness[objective_value] if kind == "var" else objective_value
        _require(len(tokens) == 3 and tokens[:2] == ["bound", direction], "objective bound direction mismatch")
        bound = _integer(tokens[2], minimum=I64_MIN, maximum=I64_MAX)
        _require(bound == evaluated, "objective bound does not match witness")
    else:
        _require(tokens == ["bound", "none"], "SAT and UNSAT require bound none")
        bound = None
    _require(records.next() == ["begin"], "expected begin")
    if claim == "sat":
        _require(records.next() == ["finish-sat"], "SAT allows only finish-sat after begin")
        records.end()
        return VerificationResult(model_hash, claim, 0, len(model.assumptions))

    base = list(model.domains)
    for predicate in model.assumptions:
        _apply(base, predicate, budget)
    current = list(base)
    facts = {}
    history = []
    path = []
    frames = []
    pending_conflict = None
    closed = None
    steps = 0

    def references(tokens: list[str]) -> list[int]:
        _require(bool(tokens), "missing premise count")
        count = _natural(tokens[0])
        _require(len(tokens) == count + 1, "premise count mismatch")
        result = []
        previous = 0
        for token in tokens[1:]:
            budget.spend()
            fact = _natural(token)
            _require(fact > previous and fact in facts, "premise is unordered, duplicated, future or outside the active scope")
            previous = fact
            result.append(fact)
        return result

    def impossible(cid: str, refs: list[int], negated=None) -> bool:
        if cid == "C":
            _require(claim == "optimal", "objective-cut rule outside an optimal proof")
            constraint = None
        elif cid == "D":
            constraint = None
        else:
            index = _natural(cid)
            _require(index < len(model.constraints), "unknown constraint ID")
            constraint = model.constraints[index]
        reconstructed = list(base)
        budget.spend(len(base))
        for fact in refs:
            _apply(reconstructed, facts[fact], budget)
        if negated is not None:
            variable, relation, value = negated
            _apply(reconstructed, (variable, NEGATED[relation], value), budget)
        if any(not domain for domain in reconstructed):
            return True
        if cid == "D":
            return False
        if cid == "C":
            direction, kind, value = model.objective
            relation = "lt" if direction == "min" else "gt"
            if kind == "const":
                return not _holds(value, relation, bound)
            domain = reconstructed[value]
            return _activity_impossible(domain[0][0], domain[-1][1], relation, bound)
        return _constraint_impossible(constraint, reconstructed, budget)

    def add_fact(identifier: int, predicate: tuple) -> None:
        facts[identifier] = predicate
        history.append(identifier)
        _apply(current, predicate, budget)

    def restore(frame: _Frame) -> None:
        nonlocal current, path
        while len(history) > frame.history_length:
            budget.spend()
            del facts[history.pop()]
        current = list(frame.parent_domains)
        path = list(frame.parent_path)
        budget.spend(len(current) + len(path))

    while True:
        tokens = records.next()
        tag = tokens[0]
        if tag == "finish":
            _require(len(tokens) == 2, "finish arity mismatch")
            _require(pending_conflict is None and not frames and not path, "finish before complete root discharge")
            _require(closed == (_natural(tokens[1]), ()), "finish does not reference the checked root proof")
            records.end()
            return VerificationResult(model_hash, claim, steps, len(model.assumptions))
        _require(tag in ("infer", "conflict", "learn", "split", "right", "resolve"), "unknown derivation record")
        _require(len(tokens) >= 2, "missing step ID")
        identifier = _natural(tokens[1])
        _require(identifier == steps + 1, "step IDs must increase by exactly one")
        if steps >= max_steps:
            raise CheckLimitExceeded("step limit reached")
        steps += 1
        _require(pending_conflict is None or tag == "learn", "conflict must be immediately followed by learn")
        _require(closed is None or tag in ("right", "resolve"), "inference after a closed branch")
        if tag == "infer":
            _require(len(tokens) >= 7, "truncated infer record")
            predicate = _predicate(tokens[3:6], len(model.domains))
            refs = references(tokens[6:])
            _require(impossible(tokens[2], refs, predicate), "inference is not justified by its cited facts")
            add_fact(identifier, predicate)
        elif tag == "conflict":
            _require(len(tokens) >= 4, "truncated conflict record")
            refs = references(tokens[3:])
            _require(impossible(tokens[2], refs), "conflict is not justified by its cited facts")
            pending_conflict = identifier
        elif tag == "learn":
            _require(len(tokens) >= 4, "truncated learn record")
            _require(pending_conflict is not None and _natural(tokens[2]) == pending_conflict, "learn must discharge the immediately checked conflict")
            count = _natural(tokens[3])
            _require(len(tokens) == 4 + count, "learn path length mismatch")
            learned_path = tuple(_natural(token) for token in tokens[4:])
            _require(learned_path == tuple(path), "learn must negate exactly the active decision path")
            closed = (identifier, learned_path)
            pending_conflict = None
        elif tag == "split":
            _require(len(tokens) == 4, "split arity mismatch")
            if len(frames) >= max_depth:
                raise CheckLimitExceeded("depth limit reached")
            variable = _variable(tokens[2], len(model.domains))
            pivot = _integer(tokens[3], minimum=I32_MIN, maximum=I32_MAX)
            _require(bool(_restrict(current[variable], "le", pivot, budget)), "split left half is empty")
            _require(bool(_restrict(current[variable], "gt", pivot, budget)), "split right half is empty")
            budget.spend(len(current) + len(path))
            frames.append(_Frame(identifier, variable, pivot, list(current), tuple(path), len(history)))
            path.append(identifier)
            add_fact(identifier, (variable, "le", pivot))
        elif tag == "right":
            _require(len(tokens) == 4 and bool(frames), "right without a split")
            frame = frames[-1]
            _require(_natural(tokens[2]) == frame.split_id and frame.right_id is None, "wrong or repeated split transition")
            left_proof = _natural(tokens[3])
            _require(closed == (left_proof, frame.parent_path + (frame.split_id,)), "left branch proof or path mismatch")
            frame.left_proof = left_proof
            frame.right_id = identifier
            restore(frame)
            closed = None
            path.append(identifier)
            add_fact(identifier, (frame.variable, "gt", frame.pivot))
        else:
            _require(len(tokens) == 4 and bool(frames), "resolve without a split")
            frame = frames[-1]
            _require(_natural(tokens[2]) == frame.split_id and frame.right_id is not None and frame.left_proof is not None, "resolve before both complementary branches")
            _require(closed == (_natural(tokens[3]), frame.parent_path + (frame.right_id,)), "right branch proof or path mismatch")
            restore(frame)
            frames.pop()
            closed = (identifier, frame.parent_path)


def _limit(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CheckLimitExceeded(f"{name} must be a nonnegative integer")
    return value


def verify_certificate(model_bytes: bytes, proof_bytes: bytes, *, expected_model_sha256: str,
                       expected_claim: str, expected_proof_sha256: Optional[str] = None,
                       max_work: int = 20_000_000, max_steps: int = 200_000,
                       max_depth: int = 128, max_model_bytes: int = 8 * 1024 * 1024,
                       max_proof_bytes: int = 32 * 1024 * 1024,
                       time_limit: Optional[float] = None) -> VerificationResult:
    """Check complete byte strings or raise a rejection/coverage exception.

    The expected model hash and claim are required out-of-band. An optional
    expected proof hash binds the exact already-read proof bytes independently
    of filesystem replacement. Resource exhaustion never produces a verified
    result. A zero time limit interrupts checking.
    """
    max_work = _limit(max_work, "max_work")
    max_steps = _limit(max_steps, "max_steps")
    max_depth = _limit(max_depth, "max_depth")
    max_model_bytes = _limit(max_model_bytes, "max_model_bytes")
    max_proof_bytes = _limit(max_proof_bytes, "max_proof_bytes")
    if time_limit is not None:
        try:
            valid_time = not isinstance(time_limit, bool) and isinstance(time_limit, (int, float)) and math.isfinite(time_limit) and time_limit >= 0
        except OverflowError:
            valid_time = False
        if not valid_time:
            raise CheckLimitExceeded("time_limit must be nonnegative and finite")
    _require(isinstance(expected_model_sha256, str) and bool(SHA256.fullmatch(expected_model_sha256)), "invalid expected model hash")
    _require(expected_claim in ("sat", "unsat", "optimal"), "invalid expected claim")
    if expected_proof_sha256 is not None:
        _require(isinstance(expected_proof_sha256, str) and bool(SHA256.fullmatch(expected_proof_sha256)), "invalid expected proof hash")
    budget = _Budget(max_work, time_limit)
    budget.spend()
    if not isinstance(model_bytes, bytes) or not isinstance(proof_bytes, bytes):
        raise CertificateRejected("model and proof must be bytes")
    if len(model_bytes) > max_model_bytes or len(proof_bytes) > max_proof_bytes:
        raise CheckLimitExceeded("input byte limit reached")
    if expected_proof_sha256 is not None:
        _require(hashlib.sha256(proof_bytes).hexdigest() == expected_proof_sha256, "proof SHA-256 mismatch")
        budget.spend()
    actual_hash = hashlib.sha256(model_bytes).hexdigest()
    _require(actual_hash == expected_model_sha256, "model SHA-256 mismatch")
    model = _model(model_bytes, max_model_bytes, budget)
    return _verify(model, proof_bytes, actual_hash, expected_claim, max_proof_bytes, max_steps, max_depth, budget)


def _cli_natural(token: str) -> int:
    if len(token) > 40 or not UNSIGNED.fullmatch(token):
        raise argparse.ArgumentTypeError("expected a nonnegative integer")
    return int(token)


def _cli_time(token: str) -> float:
    try:
        value = float(token)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a finite nonnegative time limit") from error
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("expected a finite nonnegative time limit")
    return value


def _read(path: str, limit: int, deadline: Optional[float]) -> bytes:
    chunks = []
    total = 0
    with open(path, "rb") as stream:
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                raise CheckLimitExceeded("time limit reached while reading inputs")
            chunk = stream.read(min(65536, limit - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise CheckLimitExceeded("input byte limit reached")
            chunks.append(chunk)
    return b"".join(chunks)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", required=True)
    parser.add_argument("--proof", required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--expected-proof-sha256")
    parser.add_argument("--expected-claim", required=True, choices=("sat", "unsat", "optimal"))
    parser.add_argument("--max-work", type=_cli_natural, default=20_000_000)
    parser.add_argument("--max-steps", type=_cli_natural, default=200_000)
    parser.add_argument("--max-depth", type=_cli_natural, default=128)
    parser.add_argument("--max-model-bytes", type=_cli_natural, default=8 * 1024 * 1024)
    parser.add_argument("--max-proof-bytes", type=_cli_natural, default=32 * 1024 * 1024)
    parser.add_argument("--time-limit", type=_cli_time)
    args = parser.parse_args(argv)
    started = time.monotonic()
    deadline = None if args.time_limit is None else started + args.time_limit
    try:
        model = _read(args.model, args.max_model_bytes, deadline)
        proof = _read(args.proof, args.max_proof_bytes, deadline)
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        result = verify_certificate(
            model, proof, expected_model_sha256=args.expected_model_sha256, expected_claim=args.expected_claim,
            expected_proof_sha256=args.expected_proof_sha256,
            max_work=args.max_work, max_steps=args.max_steps, max_depth=args.max_depth,
            max_model_bytes=args.max_model_bytes, max_proof_bytes=args.max_proof_bytes, time_limit=remaining,
        )
    except CertificateRejected as error:
        print(f"REJECTED: {error}", file=sys.stderr)
        return 1
    except (CheckLimitExceeded, OSError, MemoryError, KeyboardInterrupt) as error:
        print(f"UNCHECKED: {error}", file=sys.stderr)
        return 2
    print(result.acknowledgement())
    return 0


if __name__ == "__main__":
    sys.exit(main())
