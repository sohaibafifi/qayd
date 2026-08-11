#!/usr/bin/env python3
"""Compute exact XCSP FAST COP Opt, Unsat, BB1, and BB2 scores."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    from .manifest import load_manifest, sha256_file
except ImportError:
    from manifest import load_manifest, sha256_file


SCORE_SCHEMA = "qayd.fastcop.score/v2"
RESULT_SCHEMA = "qayd.fastcop.result/v1"


class ScoreError(RuntimeError):
    """Result records are incomplete or mutually inconsistent."""


def load_records(
    paths: Iterable[Path],
    replace_solvers: Iterable[str] = (),
    audit: Optional[dict[str, Any]] = None,
) -> list[dict]:
    replacements = set(replace_solvers)
    records_by_key: dict[str, tuple[Path, int]] = {}
    records_by_pair: dict[tuple[str, str], tuple[dict, Path, int]] = {}
    inputs = []
    replaced = []
    for path_index, path in enumerate(paths):
        before_hash = sha256_file(path)
        try:
            source = path.open(encoding="utf-8")
        except OSError as error:
            raise ScoreError(f"cannot read {path}: {error}") from error
        with source:
            for number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ScoreError(f"invalid JSON at {path}:{number}: {error}") from error
                if not isinstance(record, dict):
                    raise ScoreError(f"record at {path}:{number} is not an object")
                if record.get("schema") != RESULT_SCHEMA:
                    raise ScoreError(f"unsupported result schema at {path}:{number}")
                key = record.get("run_key")
                if not isinstance(key, str) or not key:
                    raise ScoreError(f"missing run_key at {path}:{number}")
                if key in records_by_key:
                    previous_path, previous_line = records_by_key[key]
                    raise ScoreError(
                        f"duplicate run_key {key} at {path}:{number}; first seen "
                        f"at {previous_path}:{previous_line}"
                    )
                solver = record.get("solver")
                instance = record.get("instance")
                if not isinstance(solver, str) or not isinstance(instance, str):
                    raise ScoreError(
                        f"missing solver or instance at {path}:{number}"
                    )
                pair = (solver, instance)
                previous = records_by_pair.get(pair)
                if previous is not None:
                    old_record, previous_path, previous_path_index = previous
                    if solver not in replacements or previous_path_index == path_index:
                        raise ScoreError(
                            "multiple runs for the same solver/instance: "
                            f"{solver}/{instance} in {previous_path} and {path}"
                        )
                    replaced.append(
                        {
                            "solver": solver,
                            "instance": instance,
                            "old_run_key": old_record["run_key"],
                            "new_run_key": key,
                            "old_input": str(previous_path),
                            "new_input": str(path),
                        }
                    )
                records_by_key[key] = (path, number)
                records_by_pair[pair] = (record, path, path_index)
        after_hash = sha256_file(path)
        if after_hash != before_hash:
            raise ScoreError(f"result file changed while it was read: {path}")
        inputs.append({"path": str(path), "sha256": before_hash})
    if audit is not None:
        audit.update(
            {
                "inputs": inputs,
                "replace_solvers": sorted(replacements),
                "replacements": replaced,
            }
        )
    return [record for record, _path, _path_index in records_by_pair.values()]


def objective(record: dict) -> Optional[int]:
    incumbent = record.get("best_incumbent")
    if not isinstance(incumbent, dict):
        return None
    value = incumbent.get("value")
    if not isinstance(value, int) or isinstance(value, bool):
        raise ScoreError(
            f"non-integer objective for {record.get('solver')} on {record.get('instance')}"
        )
    return value


def is_invalid(record: dict) -> bool:
    validation = record.get("validation")
    rejected = isinstance(validation, dict) and validation.get("valid") is False
    return bool(record.get("invalid")) or record.get("status") == "INVALID" or rejected


def execution_completed(record: dict) -> bool:
    return (
        record.get("returncode") == 0
        and record.get("execution_error") is None
        and not record.get("timed_out", False)
        and not record.get("killed", False)
    )


def verified_objective(record: dict) -> Optional[int]:
    value = objective(record)
    validation = record.get("validation")
    if (
        value is None
        or record.get("status") not in ("SAT", "OPTIMUM")
        or record.get("execution_error") is not None
        or not isinstance(validation, dict)
        or validation.get("valid") is not True
    ):
        return None
    return value


def has_complete_proof(record: dict, status: str) -> bool:
    return (
        record.get("status") == status
        and record.get("proof") is True
        and execution_completed(record)
    )


def better(left: int, right: int, sense: str) -> bool:
    return left < right if sense == "min" else left > right


CAMPAIGN_LIMIT_FIELDS = (
    "cpu_seconds",
    "wall_seconds",
    "memory_mb",
    "grace_seconds",
    "parallel_jobs",
)


def manifest_instances(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    items = manifest.get("instances")
    if not isinstance(items, list):
        raise ScoreError("manifest has no instance list")
    indexed: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ScoreError("manifest contains an invalid instance")
        if item["id"] in indexed:
            raise ScoreError(f"manifest contains duplicate instance {item['id']}")
        indexed[item["id"]] = item
    return indexed


def execution_harness(record: dict[str, Any]) -> Optional[str]:
    provenance = record.get("provenance")
    if not isinstance(provenance, dict):
        return None
    value = provenance.get(
        "execution_harness_sha256", provenance.get("harness_sha256")
    )
    return value if isinstance(value, str) else None


def validate_campaign(
    records: list[dict],
    solvers: list[str],
    expected: dict[str, dict[str, Any]],
    manifest_hash: str,
    *,
    allow_mixed_execution_harness: bool,
) -> dict[str, Any]:
    expected_pairs = {
        (solver, instance) for solver in solvers for instance in expected
    }
    observed_pairs: set[tuple[str, str]] = set()
    campaign_signatures = set()
    harnesses = set()
    solver_configs = set()

    for record in records:
        solver = record.get("solver")
        if solver not in solvers:
            continue
        instance = record.get("instance")
        if not isinstance(instance, str) or instance not in expected:
            raise ScoreError(f"record references instance outside manifest: {instance}")
        item = expected[instance]
        for field in ("objective_sense", "family", "family_group"):
            if record.get(field) != item.get(field):
                raise ScoreError(
                    f"{field} mismatch for {solver}/{instance}: "
                    f"{record.get(field)!r} != {item.get(field)!r}"
                )
        if record.get("instance_sha256") != item.get("sha256"):
            raise ScoreError(f"instance hash mismatch for {solver}/{instance}")
        validation = record.get("validation")
        incumbent = objective(record)
        if (
            incumbent is not None
            and isinstance(validation, dict)
            and validation.get("valid") is True
            and (
                validation.get("expected_objective") != incumbent
                or validation.get("reported_objective") != incumbent
            )
        ):
            raise ScoreError(
                f"validation objective mismatch for {solver}/{instance}"
            )
        limits = record.get("limits")
        if not isinstance(limits, dict):
            raise ScoreError(f"missing limits for {solver}/{instance}")
        try:
            normalized_limits = tuple((field, limits[field]) for field in CAMPAIGN_LIMIT_FIELDS)
        except KeyError as error:
            raise ScoreError(
                f"missing campaign limit {error.args[0]} for {solver}/{instance}"
            ) from error
        seed = record.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ScoreError(f"invalid seed for {solver}/{instance}")
        provenance = record.get("provenance")
        if not isinstance(provenance, dict):
            raise ScoreError(f"missing provenance for {solver}/{instance}")
        if provenance.get("manifest_sha256") != manifest_hash:
            raise ScoreError(f"manifest provenance mismatch for {solver}/{instance}")
        solver_config = provenance.get("solver_config_sha256")
        if not isinstance(solver_config, str):
            raise ScoreError(f"missing solver configuration hash for {solver}/{instance}")
        harness = execution_harness(record)
        if harness is None:
            raise ScoreError(f"missing execution harness hash for {solver}/{instance}")
        observed_pairs.add((solver, instance))
        campaign_signatures.add((seed, normalized_limits))
        solver_configs.add(solver_config)
        harnesses.add(harness)

    missing = sorted(expected_pairs - observed_pairs)
    if missing:
        sample = ", ".join(f"{solver}/{instance}" for solver, instance in missing[:3])
        raise ScoreError(f"campaign is missing {len(missing)} run(s): {sample}")
    if len(campaign_signatures) != 1:
        raise ScoreError("records mix incompatible seeds or resource limits")
    if len(solver_configs) != 1:
        raise ScoreError("records mix incompatible solver configurations")
    if len(harnesses) != 1 and not allow_mixed_execution_harness:
        raise ScoreError(
            "records mix execution harnesses; rerun one campaign or pass "
            "--allow-mixed-execution-harness for an explicit diagnostic comparison"
        )
    seed, limits = next(iter(campaign_signatures))
    return {
        "manifest_sha256": manifest_hash,
        "instance_count": len(expected),
        "seed": seed,
        "limits": dict(limits),
        "solver_config_sha256": next(iter(solver_configs)),
        "execution_harness_sha256": sorted(harnesses),
        "mixed_execution_harness": len(harnesses) > 1,
    }


def invalid_families(
    records: Iterable[dict], family_field: str, invalidation: str
) -> dict[str, set[str]]:
    invalidated: dict[str, set[str]] = {}
    if invalidation == "none":
        return invalidated
    for record in records:
        if not is_invalid(record):
            continue
        family = record.get(family_field)
        if not isinstance(family, str):
            raise ScoreError(f"missing {family_field} on invalid record")
        invalidated.setdefault(record["solver"], set()).add(family)
    return invalidated


def score_instance(
    instance: str,
    records: list[dict],
    solvers: list[str],
    invalidated: dict[str, set[str]],
    family_field: str,
    expected: Optional[dict[str, Any]] = None,
) -> dict:
    by_solver = {record["solver"]: record for record in records}
    senses = {record.get("objective_sense") for record in records}
    if expected is not None:
        senses.add(expected.get("objective_sense"))
    if len(senses) != 1 or next(iter(senses), None) not in ("min", "max"):
        raise ScoreError(f"inconsistent objective sense on {instance}: {senses}")
    sense = next(iter(senses))

    rows: dict[str, dict[str, Any]] = {}
    active: list[dict] = []
    for solver in solvers:
        record = by_solver.get(solver)
        family = (
            record.get(family_field)
            if record is not None
            else expected.get(family_field) if expected is not None else None
        )
        invalidated_here = (
            family in invalidated.get(solver, set())
        )
        if invalidated_here:
            rows[solver] = {
                "score": 0.0,
                "class": "Invalidated",
                "objective": objective(record) if record is not None else None,
                "status": record.get("status") if record is not None else None,
            }
        elif record is None:
            rows[solver] = {
                "score": 0.0,
                "class": "Missing",
                "objective": None,
                "status": None,
            }
        elif is_invalid(record):
            rows[solver] = {
                "score": 0.0,
                "class": "Invalid",
                "objective": objective(record),
                "status": record.get("status"),
            }
        else:
            active.append(record)

    unsat = [
        record for record in active
        if has_complete_proof(record, "UNSAT")
    ]
    feasible = [
        record for record in active if verified_objective(record) is not None
    ]
    if unsat and feasible:
        raise ScoreError(f"UNSAT proof conflicts with a feasible incumbent on {instance}")
    if unsat:
        for record in active:
            proved = record in unsat
            rows[record["solver"]] = {
                "score": 1.0 if proved else 0.0,
                "class": "Unsat" if proved else "None",
                "objective": None,
                "status": record.get("status"),
            }
        return {"instance": instance, "sense": sense, "best": None, "scores": rows}

    optimum_records = [
        record for record in feasible
        if has_complete_proof(record, "OPTIMUM")
    ]
    optimum_values = {verified_objective(record) for record in optimum_records}
    if len(optimum_values) > 1:
        raise ScoreError(f"conflicting optimal proofs on {instance}: {optimum_values}")
    optimum = next(iter(optimum_values)) if optimum_values else None
    if optimum is not None:
        for record in feasible:
            value = verified_objective(record)
            assert value is not None
            if better(value, optimum, sense):
                raise ScoreError(
                    f"incumbent {value} beats proved optimum {optimum} on {instance}"
                )

    values = [verified_objective(record) for record in feasible]
    values = [value for value in values if value is not None]
    best = (min(values) if sense == "min" else max(values)) if values else None
    for record in active:
        solver = record["solver"]
        value = verified_objective(record)
        proved_optimum = record in optimum_records
        if proved_optimum:
            score, label = 1.0, "Opt"
        elif value is not None and optimum is not None and value == optimum:
            score, label = 0.5, "BB2"
        elif value is not None and optimum is None and value == best:
            score, label = 1.0, "BB1"
        else:
            score, label = 0.0, "None"
        rows[solver] = {
            "score": score,
            "class": label,
            "objective": value,
            "status": record.get("status"),
        }
    return {"instance": instance, "sense": sense, "best": best, "scores": rows}


def score_pool(
    records: list[dict],
    solvers: Optional[list[str]] = None,
    invalidation: str = "family",
    family_field: str = "family",
    expected_instances: Optional[dict[str, dict[str, Any]]] = None,
) -> dict:
    if invalidation not in ("family", "none"):
        raise ScoreError(f"unknown invalidation mode: {invalidation}")
    if not records:
        raise ScoreError("no result records")
    selected = sorted(set(solvers or [record["solver"] for record in records]))
    if not selected:
        raise ScoreError("no solvers selected")
    selected_records = [record for record in records if record.get("solver") in selected]

    # One run per solver/instance is required.  Mixing seeds or configurations
    # silently would make BB scores irreproducible.
    grouped: dict[tuple[str, str], list[dict]] = {}
    for record in selected_records:
        solver = record.get("solver")
        instance = record.get("instance")
        if not isinstance(solver, str) or not isinstance(instance, str):
            raise ScoreError("every result needs string solver and instance fields")
        grouped.setdefault((solver, instance), []).append(record)
    duplicates = [key for key, values in grouped.items() if len(values) > 1]
    if duplicates:
        example = ", ".join(f"{solver}/{instance}" for solver, instance in duplicates[:3])
        raise ScoreError(f"multiple runs for the same solver/instance: {example}")

    invalidated = invalid_families(selected_records, family_field, invalidation)
    by_instance: dict[str, list[dict]] = {
        instance: [] for instance in (expected_instances or {})
    }
    for record in selected_records:
        if expected_instances is not None and record["instance"] not in expected_instances:
            raise ScoreError(
                f"record references instance outside manifest: {record['instance']}"
            )
        by_instance.setdefault(record["instance"], []).append(record)
    rows = [
        score_instance(
            instance,
            by_instance[instance],
            selected,
            invalidated,
            family_field,
            (expected_instances or {}).get(instance),
        )
        for instance in sorted(by_instance)
    ]
    summaries = {}
    for solver in selected:
        classes = Counter(row["scores"][solver]["class"] for row in rows)
        summaries[solver] = {
            "score": sum(row["scores"][solver]["score"] for row in rows),
            "instances": len(rows),
            "classes": dict(sorted(classes.items())),
            "invalidated_families": sorted(invalidated.get(solver, set())),
        }
    return {
        "solvers": summaries,
        "instance_scores": rows,
    }


def score_records(
    records: list[dict],
    mode: str = "both",
    solvers: Optional[list[str]] = None,
    invalidation: str = "family",
    family_field: str = "family",
    manifest: Optional[dict[str, Any]] = None,
    manifest_hash: Optional[str] = None,
    allow_mixed_execution_harness: bool = False,
) -> dict:
    selected = sorted(set(solvers or [record["solver"] for record in records]))
    report: dict[str, Any] = {
        "schema": SCORE_SCHEMA,
        "mode": mode,
        "invalidation": invalidation,
        "family_field": family_field,
        "selected_solvers": selected,
    }
    expected_instances = manifest_instances(manifest) if manifest is not None else None
    if manifest is not None:
        if not isinstance(manifest_hash, str):
            raise ScoreError("manifest hash is required with manifest validation")
        report["campaign"] = validate_campaign(
            records,
            selected,
            expected_instances or {},
            manifest_hash,
            allow_mixed_execution_harness=allow_mixed_execution_harness,
        )
    if mode in ("pool", "both"):
        report["pool"] = score_pool(
            records,
            selected,
            invalidation,
            family_field,
            expected_instances,
        )
    if mode in ("pairwise", "both"):
        report["pairwise"] = {}
        for left, right in itertools.combinations(selected, 2):
            name = f"{left}__vs__{right}"
            report["pairwise"][name] = score_pool(
                records,
                [left, right],
                invalidation,
                family_field,
                expected_instances,
            )
    return report


def print_summary(report: dict, target=sys.stdout) -> None:
    if "pool" in report:
        print("Pool", file=target)
        for solver, summary in report["pool"]["solvers"].items():
            print(
                f"  {solver:12s} {summary['score']:7.1f}  {summary['classes']}",
                file=target,
            )
    for name, comparison in report.get("pairwise", {}).items():
        values = ", ".join(
            f"{solver}={summary['score']:.1f}"
            for solver, summary in comparison["solvers"].items()
        )
        print(f"{name}: {values}", file=target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--mode", choices=("pool", "pairwise", "both"), default="both")
    parser.add_argument("--solver", action="append", dest="solvers")
    parser.add_argument("--invalidation", choices=("family", "none"), default="family")
    parser.add_argument("--family-field", default="family")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="canonical manifest used to verify the complete campaign grid",
    )
    parser.add_argument(
        "--replace-solver",
        action="append",
        default=[],
        help="explicitly let a later input replace this solver's earlier runs",
    )
    parser.add_argument(
        "--allow-mixed-execution-harness",
        action="store_true",
        help="allow a diagnostic comparison across different runner hashes",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    try:
        input_audit: dict[str, Any] = {}
        records = load_records(
            (path.resolve() for path in args.results),
            replace_solvers=args.replace_solver,
            audit=input_audit,
        )
        manifest = None
        manifest_hash = None
        if args.manifest is not None:
            manifest_path = args.manifest.resolve()
            manifest = load_manifest(manifest_path, verify_files=False)
            manifest_hash = sha256_file(manifest_path)
        report = score_records(
            records,
            args.mode,
            args.solvers,
            args.invalidation,
            args.family_field,
            manifest,
            manifest_hash,
            args.allow_mixed_execution_harness,
        )
        report["input_provenance"] = input_audit
        content = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
        if args.output:
            output = args.output.resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(output.name + ".tmp")
            temporary.write_text(content, encoding="utf-8")
            os.replace(temporary, output)
        else:
            print(content, end="")
        if not args.quiet:
            print_summary(report, sys.stdout if args.output else sys.stderr)
        return 0
    except (ScoreError, OSError) as error:
        print(f"fastcop score error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
