"""Checks for per-instance positioning aggregation and paired comparisons."""

import importlib.util
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[2] / "bench" / "positioning_report.py"


def load_report_module():
    spec = importlib.util.spec_from_file_location("qayd_positioning_report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def record(solver, seed, objectives, status="SATISFIABLE", verified=True):
    return {
        "problem": "cvrptw",
        "family": "test",
        "instance": "x",
        "solver": solver,
        "seed": seed,
        "checkpoint_seconds": 10,
        "threads": 1,
        "status": status,
        "verified": verified,
        "objectives": objectives,
        "peak_memory_mb": 10 + seed,
        "wall_seconds": 1,
        "return_code": 0,
    }


def test_actual_median_and_lexicographic_paired_wins():
    report = load_report_module()
    records = []
    qayd = ([3, 100], [2, 500], [2, 400], [2, 450], [2, 420])
    rival = ([2, 600], [2, 490], [2, 400], [2, 460], [])
    for seed in range(5):
        records.append(record("qayd-api", seed, list(qayd[seed])))
        status = "UNKNOWN" if not rival[seed] else "SATISFIABLE"
        records.append(record("rival", seed, list(rival[seed]), status=status, verified=bool(rival[seed])))

    rows = report.summarize_instance_groups(records)
    qayd_row = next(row for row in rows if row["solver"] == "qayd-api")
    assert qayd_row["median_objective"] == [2.0, 450.0]
    assert qayd_row["best_objective"] == [2.0, 400.0]

    paired = report.paired_rows(records)[0]
    assert paired["baseline_wins"] == 2
    assert paired["competitor_wins"] == 2
    assert paired["ties"] == 1
    assert paired["unresolved"] == 0


def test_unknown_on_both_sides_is_unresolved():
    report = load_report_module()
    records = [
        record("qayd-api", 0, [], status="UNKNOWN", verified=False),
        record("rival", 0, [], status="UNKNOWN", verified=False),
    ]

    paired = report.paired_rows(records)[0]
    assert paired["unresolved"] == 1
