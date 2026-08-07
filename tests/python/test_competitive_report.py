"""End-to-end checks for the Phase 8 campaign report and claim gate."""

import json
import importlib.util
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve().parents[2] / "bench" / "report.py"
HEXALY_ADAPTER = SCRIPT.parent / "adapters" / "hexaly.py"


def write_campaign(tmp_path, seeds):
    campaign = tmp_path / "campaign.jsonl"
    records = []
    for seed in seeds:
        records.append(
            {
                "schema_version": 1,
                "run_id": f"run-{seed}",
                "solver": "qayd-native",
                "solver_version": "qayd test",
                "problem": "cvrp",
                "family": "cvrplib-x",
                "instance_path": "x.vrp",
                "instance_sha256": "abc123",
                "checkpoint_seconds": 10,
                "seed": seed,
                "threads": 1,
                "status": "SATISFIABLE",
                "objectives": [100 + seed],
                "objective_convention": "cvrplib_unlimited_fleet_distance",
                "dual_bound": 90,
                "relative_gap": (10 + seed) / (100 + seed),
                "verified": True,
                "return_code": 0,
                "timed_out": False,
                "wall_seconds": 1.0 + seed / 10,
                "peak_memory_mb": 20.0,
                "candidates_per_second": 50_000.0,
            }
        )
    campaign.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    sidecar = campaign.with_suffix(campaign.suffix + ".provenance.json")
    sidecar.write_text(
        json.dumps(
            {
                "solvers": {"qayd-native": "qayd test"},
                "budgets": [10],
                "seeds": list(seeds),
                "host": {"commit": "0123456789", "dirty": False},
            }
        ),
        encoding="utf-8",
    )
    return campaign


def run_report(tmp_path, seeds, require=False):
    campaign = write_campaign(tmp_path, seeds)
    markdown = tmp_path / "report.md"
    summary = tmp_path / "report.json"
    command = [
        sys.executable,
        str(SCRIPT),
        str(campaign),
        "--markdown",
        str(markdown),
        "--json",
        str(summary),
    ]
    if require:
        command.append("--require-claim-ready")
    result = subprocess.run(command, capture_output=True, text=True)
    return result, markdown.read_text(encoding="utf-8"), json.loads(summary.read_text(encoding="utf-8"))


def test_five_seed_verified_campaign_is_claim_ready(tmp_path):
    result, markdown, summary = run_report(tmp_path, range(5), require=True)
    assert result.returncode == 0, result.stderr
    assert summary["claim_gate"]["ready"]
    assert summary["rows"][0]["median_candidates_per_second"] == 50_000
    assert "Claim gate: **READY**" in markdown


def test_incomplete_seed_matrix_blocks_claim(tmp_path):
    result, markdown, summary = run_report(tmp_path, range(4), require=True)
    assert result.returncode == 2
    assert not summary["claim_gate"]["ready"]
    assert "at least 5 required" in markdown


def test_hexaly_infeasible_incumbent_is_not_an_unsat_proof():
    spec = importlib.util.spec_from_file_location("qayd_hexaly_adapter", HEXALY_ADAPTER)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)

    class Status:
        OPTIMAL = 3
        FEASIBLE = 2
        INFEASIBLE = 1
        INCONSISTENT = 0

    class Solution:
        status = Status.INFEASIBLE

        def get_objective_bound(self, _index):
            raise AssertionError("an infeasible incumbent has no objective bound")

        def get_objective_gap(self, _index):
            raise AssertionError("an infeasible incumbent has no objective gap")

    optimizer = type("Optimizer", (), {"solution": Solution()})()
    module = type("Module", (), {"HxSolutionStatus": Status})()
    assert adapter.status_and_bounds(optimizer, 1, module)["status"] == "UNKNOWN"
    optimizer.solution.status = Status.INCONSISTENT
    assert adapter.status_and_bounds(optimizer, 1, module)["status"] == "UNSAT"
