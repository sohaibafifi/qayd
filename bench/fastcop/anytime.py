#!/usr/bin/env python3
"""Analyze incumbent trajectories and FAST COP scores at time checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

try:
    from . import score
except ImportError:
    import score  # type: ignore


ANYTIME_SCHEMA = "qayd.fastcop.anytime/v1"
DEFAULT_CHECKPOINTS = (1.0, 5.0, 10.0, 30.0, 60.0, 180.0)


class AnytimeError(RuntimeError):
    """An anytime trace cannot be interpreted without guessing."""


def _nonnegative_number(value: Any, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnytimeError(f"{description} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise AnytimeError(f"{description} must be finite and non-negative")
    return parsed


def normalize_checkpoints(checkpoints: Iterable[float]) -> list[float]:
    """Validate, sort, and deduplicate checkpoint times."""
    parsed = {
        _nonnegative_number(checkpoint, "checkpoint") for checkpoint in checkpoints
    }
    if not parsed:
        raise AnytimeError("at least one checkpoint is required")
    return sorted(parsed)


def parse_checkpoints(values: Iterable[str]) -> list[float]:
    """Parse comma-separated or whitespace-separated checkpoint arguments."""
    parsed: list[float] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                raise AnytimeError("checkpoint list contains an empty value")
            try:
                parsed.append(float(part))
            except ValueError as error:
                raise AnytimeError(f"invalid checkpoint: {part}") from error
    return normalize_checkpoints(parsed)


def _identity(record: dict) -> str:
    return f"{record.get('solver')} on {record.get('instance')}"


def _objective_sense(record: dict) -> str:
    sense = record.get("objective_sense")
    if sense not in ("min", "max"):
        raise AnytimeError(f"invalid objective sense for {_identity(record)}: {sense}")
    return sense


def incumbent_events(record: dict) -> list[dict[str, Any]]:
    """Return validated events ordered by timestamp, then original position."""
    raw_events = record.get("incumbents")
    if not isinstance(raw_events, list):
        raise AnytimeError(f"incumbents must be a list for {_identity(record)}")
    events = []
    for position, raw_event in enumerate(raw_events):
        if not isinstance(raw_event, dict):
            raise AnytimeError(
                f"incumbent {position} must be an object for {_identity(record)}"
            )
        value = raw_event.get("value")
        if not isinstance(value, int) or isinstance(value, bool):
            raise AnytimeError(
                f"incumbent {position} has a non-integer objective for "
                f"{_identity(record)}"
            )
        elapsed = _nonnegative_number(
            raw_event.get("elapsed_seconds"),
            f"incumbent {position} time for {_identity(record)}",
        )
        events.append(
            {
                "value": value,
                "elapsed_seconds": elapsed,
                "_position": position,
            }
        )
    events.sort(key=lambda event: (event["elapsed_seconds"], event["_position"]))
    return events


def _public_event(event: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if event is None:
        return None
    return {
        "value": event["value"],
        "elapsed_seconds": event["elapsed_seconds"],
    }


def _is_better(value: int, incumbent: int, sense: str) -> bool:
    return value < incumbent if sense == "min" else value > incumbent


def improvement_events(
    events: Sequence[dict[str, Any]], sense: str
) -> list[dict[str, Any]]:
    """Build the strict best-so-far envelope from a timestamp-ordered trace."""
    improvements: list[dict[str, Any]] = []
    for event in events:
        if not improvements or _is_better(
            event["value"], improvements[-1]["value"], sense
        ):
            improvements.append(event)
    return improvements


def _proof_time(record: dict) -> Optional[float]:
    raw = record.get("proof_elapsed_seconds")
    if raw is None:
        return None
    return _nonnegative_number(raw, f"proof time for {_identity(record)}")


def _proof_at(record: dict, proof_time: Optional[float], checkpoint: float) -> bool:
    return bool(
        record.get("proof") is True
        and record.get("status") in ("OPTIMUM", "UNSAT")
        and proof_time is not None
        and proof_time <= checkpoint
    )


def _status_at(
    record: dict,
    best: Optional[dict[str, Any]],
    proved: bool,
) -> str:
    if score.is_invalid(record):
        return "INVALID"
    if proved:
        return record["status"]
    return "SAT" if best is not None else "UNKNOWN"


def _observation_end(
    record: dict,
    events: Sequence[dict[str, Any]],
    proof_time: Optional[float],
) -> float:
    candidates = [0.0]
    raw_end = record.get("elapsed_wall_seconds")
    if raw_end is not None:
        candidates.append(
            _nonnegative_number(raw_end, f"elapsed wall time for {_identity(record)}")
        )
    if events:
        candidates.append(events[-1]["elapsed_seconds"])
    if proof_time is not None:
        candidates.append(proof_time)
    return max(candidates)


def record_timeline(record: dict, checkpoints: Iterable[float]) -> dict[str, Any]:
    """Summarize one record and reconstruct its state at each checkpoint."""
    sense = _objective_sense(record)
    normalized = normalize_checkpoints(checkpoints)
    events = incumbent_events(record)
    improvements = improvement_events(events, sense)
    proof_time = _proof_time(record)
    observation_end = _observation_end(record, events, proof_time)

    snapshots = []
    improvement_index = 0
    current: Optional[dict[str, Any]] = None
    for checkpoint in normalized:
        while (
            improvement_index < len(improvements)
            and improvements[improvement_index]["elapsed_seconds"] <= checkpoint
        ):
            current = improvements[improvement_index]
            improvement_index += 1
        proved = _proof_at(record, proof_time, checkpoint)
        snapshots.append(
            {
                "checkpoint_seconds": checkpoint,
                "best_incumbent": _public_event(current),
                "objective": current["value"] if current is not None else None,
                "status": _status_at(record, current, proved),
                "proof": proved,
                "proof_elapsed_seconds": proof_time if proved else None,
                "improvement_count": improvement_index,
                "plateau_seconds": (
                    checkpoint - current["elapsed_seconds"]
                    if current is not None
                    else None
                ),
            }
        )

    first = events[0] if events else None
    best = improvements[-1] if improvements else None
    last_improvement = improvements[-1] if improvements else None
    timed_proof = bool(
        record.get("proof") is True
        and record.get("status") in ("OPTIMUM", "UNSAT")
        and proof_time is not None
    )
    return {
        "run_key": record.get("run_key"),
        "solver": record.get("solver"),
        "instance": record.get("instance"),
        "family": record.get("family"),
        "family_group": record.get("family_group"),
        "objective_sense": sense,
        "incumbent_count": len(events),
        "improvement_count": len(improvements),
        "strict_improvement_count": max(0, len(improvements) - 1),
        "first_incumbent": _public_event(first),
        "best_incumbent": _public_event(best),
        "last_improvement": _public_event(last_improvement),
        "first_incumbent_seconds": (
            first["elapsed_seconds"] if first is not None else None
        ),
        "best_objective": best["value"] if best is not None else None,
        "best_incumbent_seconds": (
            best["elapsed_seconds"] if best is not None else None
        ),
        "last_improvement_seconds": (
            last_improvement["elapsed_seconds"]
            if last_improvement is not None
            else None
        ),
        "observation_end_seconds": observation_end,
        "plateau_seconds": (
            observation_end - last_improvement["elapsed_seconds"]
            if last_improvement is not None
            else None
        ),
        "proof": timed_proof,
        "proof_elapsed_seconds": proof_time if timed_proof else None,
        "checkpoints": snapshots,
    }


def _scoring_snapshot(record: dict, state: dict[str, Any]) -> dict:
    snapshot = dict(record)
    best = state["best_incumbent"]
    # A missing final best_incumbent means the runner did not consider the
    # trace feasible for scoring. Invalid traces retain their objective only
    # for diagnostic display; the scorer still applies the configured penalty.
    eligible = isinstance(record.get("best_incumbent"), dict)
    snapshot["best_incumbent"] = (
        best if eligible or score.is_invalid(record) else None
    )
    snapshot["status"] = state["status"]
    snapshot["proof"] = state["proof"]
    snapshot["proof_elapsed_seconds"] = state["proof_elapsed_seconds"]
    return snapshot


def _stats(values: Iterable[Optional[float]]) -> dict[str, Any]:
    numbers = [value for value in values if value is not None]
    if not numbers:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": len(numbers),
        "min": min(numbers),
        "median": statistics.median(numbers),
        "mean": statistics.fmean(numbers),
        "max": max(numbers),
    }


def _solver_summary(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "runs": len(runs),
        "runs_with_incumbent": sum(
            run["first_incumbent"] is not None for run in runs
        ),
        "runs_with_timed_proof": sum(run["proof"] for run in runs),
        "first_incumbent_seconds": _stats(
            run["first_incumbent_seconds"] for run in runs
        ),
        "best_incumbent_seconds": _stats(
            run["best_incumbent_seconds"] for run in runs
        ),
        "last_improvement_seconds": _stats(
            run["last_improvement_seconds"] for run in runs
        ),
        "plateau_seconds": _stats(run["plateau_seconds"] for run in runs),
    }


def _checkpoint_solver_metrics(
    runs: Sequence[dict[str, Any]], checkpoint_index: int
) -> dict[str, Any]:
    states = [run["checkpoints"][checkpoint_index] for run in runs]
    return {
        "runs": len(runs),
        "runs_with_incumbent": sum(
            state["best_incumbent"] is not None for state in states
        ),
        "runs_with_proof": sum(state["proof"] for state in states),
        "plateau_seconds": _stats(state["plateau_seconds"] for state in states),
    }


def analyze_records(
    records: list[dict],
    checkpoints: Iterable[float] = DEFAULT_CHECKPOINTS,
    mode: str = "both",
    solvers: Optional[list[str]] = None,
    invalidation: str = "family",
    family_field: str = "family",
) -> dict[str, Any]:
    """Build trajectories, aggregate timing metrics, and checkpoint scores."""
    if not records:
        raise AnytimeError("no result records")
    if mode not in ("none", "pool", "pairwise", "both"):
        raise AnytimeError(f"unknown scoring mode: {mode}")
    normalized = normalize_checkpoints(checkpoints)
    discovered = []
    for record in records:
        solver_name = record.get("solver")
        instance = record.get("instance")
        if not isinstance(solver_name, str) or not isinstance(instance, str):
            raise AnytimeError("every result needs string solver and instance fields")
        discovered.append(solver_name)
    selected = sorted(set(solvers or discovered))
    if not selected:
        raise AnytimeError("no solvers selected")

    all_runs = [record_timeline(record, normalized) for record in records]
    runs = [run for run in all_runs if run["solver"] in selected]
    runs_by_solver = {
        solver_name: [run for run in runs if run["solver"] == solver_name]
        for solver_name in selected
    }
    summaries = {
        solver_name: _solver_summary(solver_runs)
        for solver_name, solver_runs in runs_by_solver.items()
    }

    checkpoint_reports = []
    for index, checkpoint in enumerate(normalized):
        item: dict[str, Any] = {
            "checkpoint_seconds": checkpoint,
            "solvers": {
                solver_name: _checkpoint_solver_metrics(solver_runs, index)
                for solver_name, solver_runs in runs_by_solver.items()
            },
        }
        if mode != "none":
            snapshots = [
                _scoring_snapshot(record, all_runs[position]["checkpoints"][index])
                for position, record in enumerate(records)
            ]
            item["scores"] = score.score_records(
                snapshots,
                mode=mode,
                solvers=selected,
                invalidation=invalidation,
                family_field=family_field,
            )
        checkpoint_reports.append(item)

    return {
        "schema": ANYTIME_SCHEMA,
        "checkpoints_seconds": normalized,
        "mode": mode,
        "invalidation": invalidation,
        "family_field": family_field,
        "selected_solvers": selected,
        "record_count": len(runs),
        "solver_summaries": summaries,
        "runs": runs,
        "checkpoint_reports": checkpoint_reports,
    }


def _format_seconds(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3g}s"


def print_summary(report: dict, target=sys.stdout) -> None:
    """Print a compact human-readable companion to the JSON report."""
    checkpoints = ", ".join(
        _format_seconds(value) for value in report["checkpoints_seconds"]
    )
    print(f"Anytime checkpoints: {checkpoints}", file=target)
    for solver_name, summary in report["solver_summaries"].items():
        first = summary["first_incumbent_seconds"]["median"]
        best = summary["best_incumbent_seconds"]["median"]
        plateau = summary["plateau_seconds"]["median"]
        print(
            f"  {solver_name}: runs={summary['runs']} "
            f"incumbents={summary['runs_with_incumbent']} "
            f"proofs={summary['runs_with_timed_proof']} "
            f"median-first={_format_seconds(first)} "
            f"median-best={_format_seconds(best)} "
            f"median-plateau={_format_seconds(plateau)}",
            file=target,
        )
    for checkpoint in report["checkpoint_reports"]:
        print(
            f"Checkpoint {_format_seconds(checkpoint['checkpoint_seconds'])}",
            file=target,
        )
        if "scores" in checkpoint:
            score.print_summary(checkpoint["scores"], target)


def _write_report(report: dict, output: Optional[Path]) -> None:
    content = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if output is None:
        print(content, end="")
        return
    target = output.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, target)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=[",".join(str(value) for value in DEFAULT_CHECKPOINTS)],
        metavar="SECONDS",
        help="comma-separated or space-separated times",
    )
    parser.add_argument(
        "--mode", choices=("none", "pool", "pairwise", "both"), default="both"
    )
    parser.add_argument("--solver", action="append", dest="solvers")
    parser.add_argument("--invalidation", choices=("family", "none"), default="family")
    parser.add_argument("--family-field", default="family")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        checkpoints = parse_checkpoints(args.checkpoints)
        records = score.load_records(path.resolve() for path in args.results)
        report = analyze_records(
            records,
            checkpoints=checkpoints,
            mode=args.mode,
            solvers=args.solvers,
            invalidation=args.invalidation,
            family_field=args.family_field,
        )
        _write_report(report, args.output)
        if not args.quiet:
            print_summary(report, sys.stdout if args.output else sys.stderr)
        return 0
    except (AnytimeError, score.ScoreError, OSError) as error:
        print(f"fastcop anytime error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
