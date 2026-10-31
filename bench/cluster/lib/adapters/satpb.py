"""Independent assignment replay for DIMACS SAT and linear OPB campaigns."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class VerificationError(ValueError):
    """The instance or solver assignment cannot be interpreted exactly."""


def _result(attempted: bool, valid: Optional[bool], reason: str, **extra):
    return {"attempted": attempted, "valid": valid, "reason": reason, **extra}


def verify_solution(kind: str, instance: Path, stdout: str, status: str, objective: Optional[int]) -> dict:
    """Replay one feasible solver output without trusting its reported objective."""
    if status not in ("SAT", "OPTIMUM"):
        return _result(False, None, "no-feasible-claim")
    try:
        source = instance.read_text(encoding="utf-8")
        if kind == "sat":
            return verify_sat(source, stdout)
        if kind == "pb":
            return verify_pb(source, stdout, objective)
        return _result(False, None, "verification-disabled")
    except (OSError, UnicodeError, VerificationError) as error:
        return _result(True, None, "verification-error", message=str(error))


def _assignment_tokens(stdout: str) -> list[str]:
    tokens = []
    for line in stdout.splitlines():
        if line.startswith("v ") or line == "v":
            tokens.extend(line[1:].split())
    if not tokens:
        raise VerificationError("feasible output has no assignment line")
    return tokens


def _store(assignment: dict[int, bool], variable: int, value: bool) -> None:
    if variable <= 0:
        raise VerificationError(f"invalid assignment variable x{variable}")
    previous = assignment.get(variable)
    if previous is not None and previous != value:
        raise VerificationError(f"conflicting values for variable x{variable}")
    assignment[variable] = value


def _complete_assignment(assignment: dict[int, bool], variables: int) -> list[bool]:
    missing = [variable for variable in range(1, variables + 1) if variable not in assignment]
    if missing:
        sample = ", ".join(f"x{variable}" for variable in missing[:3])
        raise VerificationError(f"assignment omits {len(missing)} variable(s): {sample}")
    extra = sorted(variable for variable in assignment if variable > variables)
    if extra:
        raise VerificationError(f"assignment references variable x{extra[0]} above x{variables}")
    return [assignment[variable] for variable in range(1, variables + 1)]


def _parse_dimacs(source: str) -> tuple[int, list[list[int]]]:
    variables = None
    expected_clauses = None
    tokens = []
    data_lines = []
    for line_number, raw in enumerate(source.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("c"):
            continue
        if line.startswith("%"):
            break
        if line.startswith("p"):
            fields = line.split()
            if variables is not None or len(fields) != 4 or fields[:2] != ["p", "cnf"]:
                raise VerificationError(f"invalid DIMACS problem line at line {line_number}")
            try:
                variables, expected_clauses = int(fields[2]), int(fields[3])
            except ValueError as error:
                raise VerificationError(f"invalid DIMACS counts at line {line_number}") from error
            continue
        fields = line.split()
        data_lines.append(fields)
        tokens.extend(fields)
    if variables is None or expected_clauses is None:
        raise VerificationError("missing DIMACS problem line")
    clauses = []
    current = []
    for token in tokens:
        try:
            literal = int(token)
        except ValueError as error:
            raise VerificationError(f"invalid DIMACS literal {token!r}") from error
        if literal == 0:
            clauses.append(current)
            current = []
        elif not 1 <= abs(literal) <= variables:
            raise VerificationError(f"DIMACS literal {literal} is outside the declared range")
        else:
            current.append(literal)
    if current:
        if len(data_lines) != expected_clauses:
            raise VerificationError("unterminated DIMACS clause")
        clauses = []
        for fields in data_lines:
            try:
                line_clause = [int(token) for token in fields if token != "0"]
            except ValueError as error:
                raise VerificationError("invalid DIMACS line-based clause") from error
            clauses.append(line_clause)
    if len(clauses) != expected_clauses:
        raise VerificationError(f"expected {expected_clauses} clauses, found {len(clauses)}")
    return variables, clauses


def verify_sat(source: str, stdout: str) -> dict:
    variables, clauses = _parse_dimacs(source)
    assignment = {}
    for token in _assignment_tokens(stdout):
        try:
            literal = int(token)
        except ValueError as error:
            raise VerificationError(f"invalid SAT assignment literal {token!r}") from error
        if literal == 0:
            continue
        _store(assignment, abs(literal), literal > 0)
    values = _complete_assignment(assignment, variables)
    for index, clause in enumerate(clauses, 1):
        if not any(values[abs(literal) - 1] == (literal > 0) for literal in clause):
            return _result(True, False, "unsatisfied-clause", constraint=index)
    return _result(True, True, "accepted", assigned_variables=variables)


@dataclass(frozen=True)
class PbTerm:
    coefficient: int
    variable: int
    negated: bool


@dataclass(frozen=True)
class PbConstraint:
    terms: tuple[PbTerm, ...]
    relation: str
    rhs: int


@dataclass(frozen=True)
class PbProblem:
    variables: int
    constraints: tuple[PbConstraint, ...]
    objective: Optional[tuple[PbTerm, ...]]


_VARIABLE_RE = re.compile(r"(?:~)?x([1-9][0-9]*)$")
_HEADER_RE = re.compile(r"#variable\s*=\s*([0-9]+)")
_RELATION_RE = re.compile(r"(>=|<=|=|>|<)")


def _parse_pb_terms(source: str) -> tuple[PbTerm, ...]:
    tokens = source.split()
    terms = []
    position = 0
    while position < len(tokens):
        try:
            coefficient = int(tokens[position])
        except ValueError as error:
            raise VerificationError(f"expected PB coefficient, found {tokens[position]!r}") from error
        position += 1
        if position >= len(tokens):
            raise VerificationError("PB coefficient has no literal")
        literal = tokens[position]
        match = _VARIABLE_RE.fullmatch(literal)
        if match is None:
            raise VerificationError(f"expected linear PB literal, found {literal!r}")
        terms.append(PbTerm(coefficient, int(match.group(1)), literal.startswith("~")))
        position += 1
        if position < len(tokens) and _VARIABLE_RE.fullmatch(tokens[position]):
            raise VerificationError("nonlinear PB product is unsupported by the verifier")
    return tuple(terms)


def _parse_opb(source: str) -> PbProblem:
    statements = []
    header_variables = 0
    buffered = []
    for raw in source.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("*"):
            match = _HEADER_RE.search(stripped)
            if match is not None:
                header_variables = max(header_variables, int(match.group(1)))
            continue
        buffered.append(stripped)
    for statement in " ".join(buffered).split(";"):
        statement = statement.strip()
        if statement:
            statements.append(statement)

    constraints = []
    objective = None
    max_variable = header_variables
    for statement in statements:
        lowered = statement.lower()
        if lowered.startswith("min:") or lowered.startswith("max:"):
            if objective is not None:
                raise VerificationError("multiple PB objectives")
            objective = _parse_pb_terms(statement.split(":", 1)[1])
            terms = objective
        else:
            match = _RELATION_RE.search(statement)
            if match is None:
                raise VerificationError("PB constraint has no relation")
            terms = _parse_pb_terms(statement[: match.start()])
            try:
                rhs = int(statement[match.end() :].strip())
            except ValueError as error:
                raise VerificationError("invalid PB right-hand side") from error
            constraints.append(PbConstraint(terms, match.group(1), rhs))
        for term in terms:
            max_variable = max(max_variable, term.variable)
    return PbProblem(max_variable, tuple(constraints), objective)


def _parse_pb_assignment(stdout: str, variables: int) -> list[bool]:
    assignment = {}
    for token in _assignment_tokens(stdout):
        if token == "0":
            continue
        match = re.fullmatch(r"([~-]?)(?:x)?([1-9][0-9]*)", token)
        if match is None:
            raise VerificationError(f"invalid PB assignment literal {token!r}")
        _store(assignment, int(match.group(2)), match.group(1) not in ("-", "~"))
    return _complete_assignment(assignment, variables)


def _pb_sum(terms: tuple[PbTerm, ...], values: list[bool]) -> int:
    return sum(
        term.coefficient * int(values[term.variable - 1] != term.negated)
        for term in terms
    )


def _holds(value: int, relation: str, rhs: int) -> bool:
    return {
        ">=": value >= rhs,
        "<=": value <= rhs,
        "=": value == rhs,
        ">": value > rhs,
        "<": value < rhs,
    }[relation]


def verify_pb(source: str, stdout: str, reported_objective: Optional[int]) -> dict:
    problem = _parse_opb(source)
    values = _parse_pb_assignment(stdout, problem.variables)
    for index, constraint in enumerate(problem.constraints, 1):
        value = _pb_sum(constraint.terms, values)
        if not _holds(value, constraint.relation, constraint.rhs):
            return _result(
                True,
                False,
                "unsatisfied-constraint",
                constraint=index,
                lhs=value,
                relation=constraint.relation,
                rhs=constraint.rhs,
            )
    computed = _pb_sum(problem.objective, values) if problem.objective is not None else None
    if problem.objective is not None and reported_objective is None:
        return _result(True, False, "missing-objective", computed_objective=computed)
    if computed != reported_objective and problem.objective is not None:
        return _result(
            True,
            False,
            "objective-mismatch",
            computed_objective=computed,
            reported_objective=reported_objective,
        )
    return _result(
        True,
        True,
        "accepted",
        assigned_variables=problem.variables,
        computed_objective=computed,
    )
