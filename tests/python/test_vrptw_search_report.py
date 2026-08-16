"""Qualification checks for the fleet-first VRPTW report."""

import importlib.util
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[2] / "bench" / "vrptw_search_report.py"


def load_module():
    spec = importlib.util.spec_from_file_location("qayd_vrptw_search_report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def record(solver, instance, seed, objective, *, rss=32.0):
    return {
        "solver": solver,
        "problem": "cvrptw",
        "instance": instance,
        "seed": seed,
        "checkpoint_seconds": 10,
        "status": "SATISFIABLE",
        "verified": True,
        "objectives": list(objective),
        "peak_memory_mb": rss,
    }


def complete_inputs(report, tmp_path):
    report.ROOT = tmp_path
    solomon = tmp_path / "bench" / "routing" / "instances" / "solomon" / "In"
    solomon.mkdir(parents=True)
    for instance in report.TARGETS:
        (solomon / f"{instance.lower()}.txt").write_text("fixture\n", encoding="utf-8")

    qayd = []
    hexaly = []
    for instance, target in report.TARGETS.items():
        objective = tuple(target["exact"] if "exact" in target else (target["fleet"], 10_000))
        for seed in report.SEEDS:
            qayd.append(record("qayd-api", instance, seed, objective))
            hexaly.append(record("hexaly", instance, seed, objective))
    solomon_records = [record("qayd-api", instance, 0, (1, 1)) for instance in report.TARGETS]
    homberger = [
        record("qayd-api", "C1_2_1", 0, (20, 10_000)),
        record("qayd-api", "C1_4_1", 0, (40, 20_000)),
    ]
    return qayd, hexaly, solomon_records, homberger


def test_complete_qualification_passes(tmp_path):
    report = load_module()
    summary = report.build_summary(*complete_inputs(report, tmp_path))

    assert summary["pass"]
    assert summary["distance"]["same_fleet_pairs"] == 30
    assert summary["rss"]["median_mb"] == 32.0
    assert summary["solomon"]["covered_instances"] == 6
    assert summary["homberger"]["pairs"][0]["pass"]


def test_one_fleet_regression_fails_the_instance_gate(tmp_path):
    report = load_module()
    qayd, hexaly, solomon, homberger = complete_inputs(report, tmp_path)
    bad = next(run for run in qayd if run["instance"] == "RC201" and run["seed"] == 3)
    bad["objectives"] = [5, 9_000]

    summary = report.build_summary(qayd, hexaly, solomon, homberger)

    row = next(row for row in summary["sentinels"] if row["instance"] == "RC201")
    assert not row["pass"]
    assert not summary["pass"]


def test_scale_gate_reports_strict_ratio_and_one_route_integrality(tmp_path):
    report = load_module()
    records = [
        record("qayd-api", "RC2_2_1", 0, (6, 10_000)),
        record("qayd-api", "RC2_4_1", 0, (13, 20_000)),
    ]

    summary = report.homberger_summary(records, "qayd-api")
    row = summary["pairs"][0]

    assert not row["strict_pass"]
    assert row["strict_limit"] == 12
    assert row["integral_limit"] == 13
    assert row["pass"]
    assert summary["pass"]

    records[-1]["objectives"] = [14, 20_000]
    summary = report.homberger_summary(records, "qayd-api")
    assert not summary["pairs"][0]["pass"]
    assert not summary["pass"]
