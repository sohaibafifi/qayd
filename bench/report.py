#!/usr/bin/env python3
"""Aggregate competitive JSONL campaigns and audit whether claims are ready."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable

from common.competitive import FEASIBLE_STATUSES, percentile


SOLVER_PROBLEMS = {
    "qayd-api": {"cvrp", "cvrptw", "jssp", "rcpsp"},
    "qayd-native": {"cvrp", "cvrptw", "jssp", "rcpsp"},
    "hexaly": {"cvrp", "cvrptw", "jssp", "rcpsp"},
    "hgs": {"cvrp"},
    "lkh": {"cvrp", "cvrptw"},
    "ortools-cp-sat": {"jssp", "rcpsp"},
}


def finite_numbers(values: Iterable[object]) -> list[float]:
    result = []
    for value in values:
        if isinstance(value, (int, float)) and math.isfinite(value):
            result.append(float(value))
    return result


def median(values: Iterable[object]) -> float | None:
    numbers = finite_numbers(values)
    return statistics.median(numbers) if numbers else None


def load_records(paths: list[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    records = []
    errors = []
    seen = set()
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    errors.append(f"{path}:{line_number}: invalid JSON: {error}")
                    continue
                run_id = record.get("run_id")
                if not run_id:
                    errors.append(f"{path}:{line_number}: missing run_id")
                elif run_id in seen:
                    errors.append(f"duplicate run_id: {run_id}")
                    continue
                seen.add(run_id)
                record["_campaign_file"] = str(path.resolve())
                records.append(record)
    return records, errors


def load_provenance(paths: list[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    documents = []
    errors = []
    for path in paths:
        sidecar = path.with_suffix(path.suffix + ".provenance.json")
        if not sidecar.is_file():
            errors.append(f"missing provenance sidecar: {sidecar}")
            continue
        try:
            document = json.loads(sidecar.read_text(encoding="utf-8"))
            document["_campaign_file"] = str(path.resolve())
            documents.append(document)
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"invalid provenance sidecar {sidecar}: {error}")
    return documents, errors


def feasible(record: dict[str, Any]) -> bool:
    return record.get("status") in FEASIBLE_STATUSES and bool(record.get("objectives"))


def lex_gap_percent(candidate: list[object], reference: list[object]) -> float | None:
    if not candidate or len(candidate) != len(reference):
        return None
    for raw_candidate, raw_reference in zip(candidate, reference):
        if not isinstance(raw_candidate, (int, float)) or not isinstance(raw_reference, (int, float)):
            return None
        if raw_candidate == raw_reference:
            continue
        return max(0.0, float(raw_candidate) - float(raw_reference)) * 100.0 / max(1.0, abs(float(raw_reference)))
    return 0.0


def best_observed(records: list[dict[str, Any]]) -> dict[tuple[object, ...], list[object]]:
    best = {}
    for record in records:
        if not feasible(record) or not record.get("verified"):
            continue
        key = (record.get("problem"), record.get("instance_path"), record.get("checkpoint_seconds"), record.get("seed"))
        objectives = list(record["objectives"])
        if key not in best or tuple(objectives) < tuple(best[key]):
            best[key] = objectives
    return best


def audit_claims(
    records: list[dict[str, Any]], provenance: list[dict[str, Any]], min_seeds: int,
) -> list[str]:
    errors = []
    if not records:
        return ["campaign contains no records"]
    for document in provenance:
        host = document.get("host", {})
        if host.get("dirty"):
            errors.append("campaign provenance reports a dirty qayd worktree")
        if not host.get("commit"):
            errors.append("campaign provenance has no qayd commit")
        for solver, version in document.get("solvers", {}).items():
            text = str(version).lower()
            if "missing" in text or text == "unknown":
                errors.append(f"missing reproducible version for {solver}")

    for record in records:
        label = f"{record.get('solver')}/{record.get('instance_path')}/t={record.get('checkpoint_seconds')}/seed={record.get('seed')}"
        missing_fields = [
            field for field in ("solver", "problem", "instance_path", "checkpoint_seconds", "seed", "status")
            if record.get(field) is None
        ]
        if missing_fields:
            errors.append(f"record missing required fields {missing_fields}: {label}")
        if record.get("return_code") != 0 or record.get("timed_out"):
            errors.append(f"failed run: {label}")
        if feasible(record) and not record.get("verified"):
            errors.append(f"feasible result was not independently verified: {label}")
        if record.get("status") == "ERROR":
            errors.append(f"solver error: {label}")
        if feasible(record) and not record.get("objective_convention"):
            errors.append(f"missing objective convention: {label}")

    logical_runs: dict[tuple[object, ...], int] = defaultdict(int)
    for record in records:
        key = (
            record.get("solver"), record.get("problem"), record.get("instance_path"),
            record.get("checkpoint_seconds"), record.get("seed"),
        )
        logical_runs[key] += 1
    for key, count in logical_runs.items():
        if count > 1:
            errors.append(f"duplicate logical run ({count} copies): {key}")

    conventions: dict[tuple[object, object], set[object]] = defaultdict(set)
    vrptw_conventions: dict[object, set[tuple[object, object]]] = defaultdict(set)
    instance_hashes: dict[object, set[object]] = defaultdict(set)
    for record in records:
        instance_hashes[record.get("instance_path")].add(record.get("instance_sha256"))
        if feasible(record):
            key = (record.get("problem"), record.get("instance_path"))
            conventions[key].add(record.get("objective_convention"))
            if record.get("problem") == "cvrptw":
                vrptw_conventions[record.get("instance_path")].add((record.get("distance_scale"), record.get("rounding")))
    for key, values in conventions.items():
        if len(values) > 1:
            errors.append(f"objective convention mismatch for {key[1]}: {sorted(map(str, values))}")
    for instance, values in vrptw_conventions.items():
        if len(values) > 1 or any(scale is None or rounding is None for scale, rounding in values):
            errors.append(f"VRPTW distance convention mismatch for {instance}: {sorted(map(str, values))}")
    for instance, values in instance_hashes.items():
        if None in values or len(values) > 1:
            errors.append(f"missing or inconsistent instance hash for {instance}")

    by_instance: dict[tuple[object, object], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_instance[(record.get("problem"), record.get("instance_path"))].append(record)
    for (_problem, instance), runs in by_instance.items():
        valid = [record for record in runs if feasible(record) and record.get("verified")]
        if valid and any(record.get("status") == "UNSAT" for record in runs):
            errors.append(f"soundness contradiction, feasible and UNSAT results for {instance}")
        optimums = {tuple(record["objectives"]) for record in valid if record.get("status") == "OPTIMAL"}
        if len(optimums) > 1:
            errors.append(f"soundness contradiction, different proven optima for {instance}: {sorted(optimums)}")
        if optimums:
            optimum = min(optimums)
            if any(tuple(record["objectives"]) < optimum for record in valid):
                errors.append(f"soundness contradiction, feasible result beats the proven optimum for {instance}")
        if valid:
            best_primary = min(float(record["objectives"][0]) for record in valid)
            for record in valid:
                dual = record.get("dual_bound")
                if isinstance(dual, (int, float)) and math.isfinite(dual) and float(dual) > best_primary + 1e-9:
                    errors.append(f"invalid minimization dual from {record.get('solver')} for {instance}: {dual} > {best_primary}")

    matrices = provenance or [{"_campaign_file": None}]
    for document in matrices:
        source = document.get("_campaign_file")
        source_records = [record for record in records if source is None or record.get("_campaign_file") == source]
        selected_solvers = set(document.get("solvers", {})) or {record.get("solver") for record in source_records}
        expected_budgets = set(document.get("budgets", [])) or {record.get("checkpoint_seconds") for record in source_records}
        expected_seeds = set(document.get("seeds", [])) or {record.get("seed") for record in source_records}
        if len(expected_seeds) < min_seeds:
            errors.append(f"only {len(expected_seeds)} campaign seeds in {source or 'input'}, at least {min_seeds} required")
        actual = {
            (
                record.get("solver"), record.get("problem"), record.get("instance_path"),
                record.get("checkpoint_seconds"), record.get("seed"),
            )
            for record in source_records
        }
        instances = {(record.get("problem"), record.get("instance_path")) for record in source_records}
        missing = []
        for problem, instance in sorted(instances, key=lambda values: tuple(map(str, values))):
            for solver in sorted(selected_solvers):
                if problem not in SOLVER_PROBLEMS.get(solver, set()):
                    continue
                for budget in expected_budgets:
                    for seed in expected_seeds:
                        key = (solver, problem, instance, budget, seed)
                        if key not in actual:
                            missing.append(f"{solver}/{instance}/t={budget}/seed={seed}")
        if missing:
            errors.append(f"{len(missing)} missing run(s) in {source or 'input'}")
            errors.extend(f"missing run: {label}" for label in missing[:20])
    return sorted(set(errors))


def aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    references = best_observed(records)
    groups: dict[tuple[object, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (record.get("problem"), record.get("family"), record.get("checkpoint_seconds"), record.get("solver"))
        groups[key].append(record)

    rows = []
    for (problem, family, budget, solver), runs in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        valid = [record for record in runs if feasible(record) and record.get("verified")]
        gaps = []
        best_known_gaps = []
        for record in valid:
            key = (problem, record.get("instance_path"), budget, record.get("seed"))
            reference = references.get(key)
            if reference is not None:
                gap = lex_gap_percent(record["objectives"], reference)
                if gap is not None:
                    gaps.append(gap)
            best_known_value = record.get("best_known")
            if len(record["objectives"]) == 1 and isinstance(best_known_value, (int, float)) and best_known_value:
                best_known_gaps.append(
                    max(0.0, float(record["objectives"][0]) - float(best_known_value)) * 100.0 / abs(float(best_known_value))
                )
        objective_width = max((len(record["objectives"]) for record in valid), default=0)
        median_objectives = [median(record["objectives"][tier] for record in valid if len(record["objectives"]) > tier) for tier in range(objective_width)]
        duals = finite_numbers(record.get("dual_bound") for record in valid)
        rows.append({
            "problem": problem,
            "family": family,
            "checkpoint_seconds": budget,
            "solver": solver,
            "threads": sorted({record.get("threads") for record in runs if isinstance(record.get("threads"), int)}),
            "instances": len({record.get("instance_path") for record in runs}),
            "runs": len(runs),
            "successful_runs": len(valid),
            "success_percent": 100.0 * len(valid) / len(runs),
            "median_objectives": median_objectives,
            "median_best_observed_gap_percent": median(gaps),
            "median_best_known_gap_percent": median(best_known_gaps),
            "dual_coverage_percent": 100.0 * len(duals) / len(valid) if valid else 0.0,
            "median_dual_bound": median(duals),
            "median_relative_gap_percent": median(
                100.0 * value for value in finite_numbers(record.get("relative_gap") for record in valid)
            ),
            "median_wall_seconds": median(record.get("wall_seconds") for record in runs),
            "p95_wall_seconds": percentile(finite_numbers(record.get("wall_seconds") for record in runs), 0.95),
            "median_peak_memory_mb": median(record.get("peak_memory_mb") for record in runs),
            "median_candidates_per_second": median(record.get("candidates_per_second") for record in valid),
        })
    return rows


def display(value: object, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_report(summary: dict[str, Any]) -> str:
    gate = summary["claim_gate"]
    lines = [
        "# Competitive campaign report",
        "",
        f"Claim gate: **{'READY' if gate['ready'] else 'NOT READY'}**",
        "",
        f"Records: {summary['record_count']}. Minimum seeds: {gate['minimum_seeds']}.",
        "",
    ]
    if gate["errors"]:
        lines.extend(("## Claim gate failures", ""))
        lines.extend(f"- {error}" for error in gate["errors"])
        lines.append("")
    lines.extend((
        "## Anytime quality and resource summary",
        "",
        "| Problem | Family | Budget | Solver | Threads | Success | Median objective | BKS gap | Best-observed gap | Dual coverage | Median certified gap | Wall p95 | RSS median | Candidates/s |",
        "|---|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ))
    for row in summary["rows"]:
        objectives = ", ".join(display(value) for value in row["median_objectives"]) or "n/a"
        lines.append(
            f"| {row['problem']} | {row['family']} | {row['checkpoint_seconds']}s | {row['solver']} | "
            f"{','.join(map(str, row['threads'])) or 'n/a'} | {row['successful_runs']}/{row['runs']} | {objectives} | "
            f"{display(row['median_best_known_gap_percent'])}% | {display(row['median_best_observed_gap_percent'])}% | "
            f"{display(row['dual_coverage_percent'])}% | "
            f"{display(row['median_relative_gap_percent'])}% | {display(row['p95_wall_seconds'], 3)}s | "
            f"{display(row['median_peak_memory_mb'], 1)} MB | {display(row['median_candidates_per_second'], 1)} |"
        )
    lines.extend((
        "",
        "Best-observed gap is computed at equal instance, seed and checkpoint using lexicographic minimization. "
        "Dual columns contain only solver-reported certified bounds. Missing bounds remain unavailable.",
        "Threads reports the effective worker count; the requested count remains in each raw JSONL record.",
        "",
    ))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", nargs="+", type=Path)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--json", dest="json_output", type=Path)
    parser.add_argument("--min-seeds", type=int, default=5)
    parser.add_argument("--require-claim-ready", action="store_true")
    args = parser.parse_args()
    if args.min_seeds <= 0:
        raise SystemExit("--min-seeds must be positive")

    records, parse_errors = load_records(args.campaign)
    provenance, provenance_errors = load_provenance(args.campaign)
    gate_errors = parse_errors + provenance_errors + audit_claims(records, provenance, args.min_seeds)
    summary = {
        "schema_version": 1,
        "record_count": len(records),
        "claim_gate": {
            "ready": not gate_errors,
            "minimum_seeds": args.min_seeds,
            "errors": sorted(set(gate_errors)),
        },
        "rows": aggregate(records),
    }
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(markdown_report(summary), encoding="utf-8")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"report: {args.markdown} ({'claim-ready' if not gate_errors else f'{len(set(gate_errors))} gate failure(s)'})")
    if args.require_claim_ready and gate_errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
