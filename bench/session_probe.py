#!/usr/bin/env python3
"""Probe rolling-horizon SolveSession cost.

The probe compares two public Python paths on the same sequence of epochs:

* session: one SolveSession reused across epochs, keeping learned clauses.
* cold: Model.solve() for each epoch, with the same assumptions and hints but
  without a shared clause pool.

This is a measurement harness, not a solver benchmark. It gives an empirical
proxy for the current clone plus re-propagation cost per epoch. A true live
push/pop continuation is not implemented, so there is no continuation row here.

Interpretation limit: the arms compare the two PUBLIC paths, not the pure
clause-pool effect — the cold arm re-materializes objectives every epoch
(fresh objective var + channeling constraint) while the session built them
once, so part of any session win is setup cost, not learning reuse. Isolating
the pool effect alone would need a cold arm with pre-materialized objectives.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import statistics
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

try:
    import qayd as cp
except ImportError as exc:
    raise SystemExit(
        "qayd Python extension is not installed. Run "
        "`uv run --with maturin maturin develop --features python` first."
    ) from exc


@dataclass(frozen=True)
class EpochPlan:
    epoch: int
    committed: list[int]
    assumptions: list[tuple[Any, int]]
    hints: list[tuple[Any, int]]
    branch_order: list[Any]


def build_model(n: int, conflicts: int, seed: int, capacity_ratio: float) -> tuple[Any, list[Any]]:
    rng = random.Random(seed)
    model = cp.Model()
    xs = model.int_vars(n, 0, 1)
    values = [rng.randint(10, 1000) for _ in range(n)]
    weights = [rng.randint(1, 100) for _ in range(n)]

    model.add(sum(weights[i] * xs[i] for i in range(n)) <= int(sum(weights) * capacity_ratio))

    seen: set[tuple[int, int]] = set()
    while len(seen) < conflicts:
        a = rng.randrange(n)
        b = rng.randrange(n)
        if a == b:
            continue
        if a > b:
            a, b = b, a
        if (a, b) in seen:
            continue
        seen.add((a, b))
        model.add(xs[a] + xs[b] <= 1)

    model.maximize(sum(values[i] * xs[i] for i in range(n)))
    return model, xs


def solution_values(solution: Any, xs: list[Any]) -> list[int]:
    assignment = solution.assignment()
    return [int(assignment.get(x.index, 0)) for x in xs]


def reference_values(model: Any, xs: list[Any], args: argparse.Namespace) -> tuple[list[int], dict[str, Any]]:
    t0 = time.perf_counter()
    solution = model.solve(seed=args.seed, time_limit=args.reference_time_limit)
    elapsed = time.perf_counter() - t0
    if not solution.is_sat():
        values = [0 for _ in xs]
    else:
        values = solution_values(solution, xs)
    return values, {
        "reference_status": solution.status,
        "reference_objective": solution.objective,
        "reference_time_s": elapsed,
    }


def build_epoch_plans(xs: list[Any], old_values: list[int], args: argparse.Namespace) -> list[EpochPlan]:
    rng = random.Random(args.seed ^ 0x5E5510)
    order = list(range(len(xs)))
    rng.shuffle(order)
    hints = [(xs[i], old_values[i]) for i in range(len(xs))]
    plans = []
    for epoch in range(args.epochs):
        count = min(len(xs), args.initial_commit + epoch * args.commit_step)
        committed = sorted(order[:count])
        committed_set = set(committed)
        assumptions = [(xs[i], old_values[i]) for i in committed]
        branch_order = [xs[i] for i in order if i not in committed_set]
        plans.append(EpochPlan(epoch, committed, assumptions, hints, branch_order))
    return plans


def timed_solve(mode: str, model: Any, session: Any, plan: EpochPlan, args: argparse.Namespace) -> dict[str, Any]:
    gc.collect()
    t0 = time.perf_counter()
    seed = args.seed + plan.epoch
    kwargs = {
        "assumptions": plan.assumptions,
        "hints": plan.hints,
        "branch_order": plan.branch_order,
        "seed": seed,
        "time_limit": args.epoch_time_limit,
        "conflict_budget": args.conflict_budget,
    }
    if mode == "session":
        solution = session.solve(**kwargs)
        learned_nogoods = session.learned_nogoods
    elif mode == "cold":
        solution = model.solve(**kwargs)
        learned_nogoods = None
    else:
        raise ValueError(f"unknown mode: {mode}")
    elapsed = time.perf_counter() - t0
    stats = solution.stats
    return {
        "mode": mode,
        "epoch": plan.epoch,
        "committed": len(plan.committed),
        "status": solution.status,
        "objective": solution.objective,
        "time_s": elapsed,
        "nodes": stats.nodes,
        "failures": stats.failures,
        "solutions": stats.solutions,
        "learned_lits": stats.learned_lits,
        "session_nogoods": learned_nogoods,
    }


def emit_progress(row: dict[str, Any]) -> None:
    nogoods = "" if row["session_nogoods"] is None else f" nogoods={row['session_nogoods']}"
    objective = "" if row["objective"] is None else f" obj={row['objective']}"
    print(
        f"[{row['mode']} epoch={row['epoch']} committed={row['committed']}] "
        f"{row['status']}{objective} {row['time_s']:.3f}s "
        f"nodes={row['nodes']} failures={row['failures']}{nogoods}",
        file=sys.stderr,
        flush=True,
    )


def run_probe(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_args(args)
    if not args.quiet:
        print("building model and reference incumbent", file=sys.stderr, flush=True)
    model, xs = build_model(args.vars, args.conflicts, args.seed, args.capacity_ratio)
    old_values, reference = reference_values(model, xs, args)
    plans = build_epoch_plans(xs, old_values, args)
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    invalid = sorted(set(modes) - {"cold", "session"})
    if invalid:
        raise SystemExit(f"unknown mode(s): {', '.join(invalid)}")

    rows: list[dict[str, Any]] = []
    for mode in modes:
        session = model.session() if mode == "session" else None
        for plan in plans:
            row = timed_solve(mode, model, session, plan, args)
            rows.append(row)
            if not args.quiet:
                emit_progress(row)
    summary = summarize(rows)
    summary.update(reference)
    summary.update({
        "vars": args.vars,
        "conflicts": args.conflicts,
        "epochs": args.epochs,
        "initial_commit": args.initial_commit,
        "commit_step": args.commit_step,
        "capacity_ratio": args.capacity_ratio,
        "conflict_budget": args.conflict_budget,
    })
    return rows, summary


def summarize(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_mode.setdefault(row["mode"], []).append(row)

    out: dict[str, Any] = {"modes": {}}
    for mode, mode_rows in sorted(by_mode.items()):
        times = [row["time_s"] for row in mode_rows]
        out["modes"][mode] = {
            "epochs": len(mode_rows),
            "total_time_s": sum(times),
            "mean_time_s": statistics.fmean(times) if times else 0.0,
            "median_time_s": statistics.median(times) if times else 0.0,
            "total_nodes": sum(row["nodes"] for row in mode_rows),
            "total_failures": sum(row["failures"] for row in mode_rows),
            "last_session_nogoods": mode_rows[-1]["session_nogoods"],
        }
    cold = out["modes"].get("cold")
    session = out["modes"].get("session")
    if cold and session and session["total_time_s"] > 0:
        out["cold_over_session_time"] = cold["total_time_s"] / session["total_time_s"]
    return out


def emit_table(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    header = ("mode", "epoch", "committed", "status", "objective", "time_s", "nodes", "failures", "session_nogoods")
    print(" ".join(f"{name:>16}" for name in header))
    for row in rows:
        values = [
            row["mode"],
            row["epoch"],
            row["committed"],
            row["status"],
            "" if row["objective"] is None else row["objective"],
            f"{row['time_s']:.6f}",
            row["nodes"],
            row["failures"],
            "" if row["session_nogoods"] is None else row["session_nogoods"],
        ]
        print(" ".join(f"{str(value):>16}" for value in values))
    print(json.dumps({"summary": summary}, sort_keys=True))


def emit_csv(rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "mode",
        "epoch",
        "committed",
        "status",
        "objective",
        "time_s",
        "nodes",
        "failures",
        "solutions",
        "learned_lits",
        "session_nogoods",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def emit_jsonl(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    for row in rows:
        print(json.dumps(row, sort_keys=True))
    print(json.dumps({"summary": summary}, sort_keys=True))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vars", type=int, default=56, help="number of 0/1 decisions in the synthetic instance")
    ap.add_argument("--conflicts", type=int, default=120, help="number of random pairwise conflicts")
    ap.add_argument("--epochs", type=int, default=8, help="number of rolling-horizon epochs")
    ap.add_argument("--initial-commit", type=int, default=None, help="variables fixed by assumptions in the first epoch")
    ap.add_argument("--commit-step", type=int, default=6, help="new variables fixed by assumptions at each epoch")
    ap.add_argument("--capacity-ratio", type=float, default=0.45, help="knapsack capacity as a fraction of total weight")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--reference-time-limit", type=int, default=5, help="seconds for the initial x_old solve")
    ap.add_argument("--epoch-time-limit", type=int, default=1, help="seconds per epoch, 0 means unbounded")
    ap.add_argument("--conflict-budget", type=int, default=None, help="optional LCG conflict budget per epoch")
    ap.add_argument("--modes", default="cold,session", help="comma-separated subset of cold,session")
    ap.add_argument("--format", choices=("table", "csv", "jsonl"), default="table")
    ap.add_argument("--quiet", action="store_true", help="suppress per-epoch progress on stderr")
    return ap.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.vars <= 0:
        raise SystemExit("--vars must be positive")
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive")
    if args.reference_time_limit <= 0:
        raise SystemExit("--reference-time-limit must be positive")
    if args.epoch_time_limit is not None and args.epoch_time_limit < 0:
        raise SystemExit("--epoch-time-limit must be non-negative")
    if args.epoch_time_limit == 0:
        args.epoch_time_limit = None
    if args.initial_commit is None:
        args.initial_commit = args.commit_step
    if args.initial_commit < 0:
        raise SystemExit("--initial-commit must be non-negative")
    if args.commit_step < 0:
        raise SystemExit("--commit-step must be non-negative")
    if not 0 < args.capacity_ratio <= 1:
        raise SystemExit("--capacity-ratio must be in (0, 1]")
    max_conflicts = args.vars * (args.vars - 1) // 2
    if not 0 <= args.conflicts <= max_conflicts:
        raise SystemExit(f"--conflicts must be between 0 and {max_conflicts}")
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    if not modes:
        raise SystemExit("--modes must include cold, session, or both")


def main() -> None:
    args = parse_args()
    rows, summary = run_probe(args)
    if args.format == "table":
        emit_table(rows, summary)
    elif args.format == "csv":
        emit_csv(rows)
    else:
        emit_jsonl(rows, summary)


if __name__ == "__main__":
    main()
