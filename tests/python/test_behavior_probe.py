"""Controlled-input and paired-analysis checks for the behavior probe."""

import importlib.util
import hashlib
import json
from pathlib import Path
import sys

from qayd.datasets import read_cvrplib


SCRIPT = Path(__file__).resolve().parents[2] / "bench" / "behavior_probe.py"


def load_probe_module():
    name = "qayd_behavior_probe"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def generated_probe(tmp_path, monkeypatch):
    probe = load_probe_module()
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    suite_path, manifest_path, manifest = probe.generate_probe_suite(
        tmp_path / "probe", sizes=[8], instance_seed=17,
        distance_scale=7, edge_delta=25,
    )
    return probe, suite_path, manifest_path, manifest


def role(manifest, factor, name):
    return next(
        entry for entry in manifest["probes"]
        if entry["factor"] == factor and entry["role"] == name
    )


def instance(tmp_path, entry):
    return read_cvrplib(tmp_path / entry["instance_path"])


def solver_record(entry, solver, objective, *, budget=1, seed=0, **overrides):
    record = {
        "run_id": f"{solver}-{entry['probe_id']}-t{budget}-s{seed}",
        "solver": solver,
        "solver_version": f"{solver} test-version",
        "instance_path": entry["instance_path"],
        "instance_sha256": entry["instance_sha256"],
        "checkpoint_seconds": budget,
        "seed": seed,
        "requested_threads": 1,
        "threads": 1,
        "status": "SATISFIABLE",
        "verified": True,
        "objectives": [objective * entry["objective_scale"]],
    }
    record.update(overrides)
    return record


def test_generator_changes_exactly_the_declared_factor(tmp_path, monkeypatch):
    _probe, suite_path, _manifest_path, manifest = generated_probe(tmp_path, monkeypatch)
    assert suite_path.is_file()
    assert len(manifest["probes"]) == 8
    for entry in manifest["probes"]:
        payload = (tmp_path / entry["instance_path"]).read_bytes()
        assert entry["instance_sha256"] == hashlib.sha256(payload).hexdigest()

    scale_control = instance(tmp_path, role(manifest, "distance-scale", "control"))
    scale_treatment = instance(tmp_path, role(manifest, "distance-scale", "treatment"))
    assert scale_treatment.demands == scale_control.demands
    assert scale_treatment.capacity == scale_control.capacity
    assert scale_treatment.edge_weights == tuple(
        tuple(7 * value for value in row) for row in scale_control.edge_weights
    )

    permutation_entry = role(manifest, "index-permutation", "treatment")
    permutation_control = instance(tmp_path, role(manifest, "index-permutation", "control"))
    permutation = instance(tmp_path, permutation_entry)
    mapping = {
        int(public_id) - 1: control_id - 1
        for public_id, control_id in permutation_entry["node_map_to_control"].items()
    }
    assert sorted(permutation.demands) == sorted(permutation_control.demands)
    for left in range(permutation.dimension):
        for right in range(permutation.dimension):
            assert permutation.edge_weights[left][right] == permutation_control.edge_weights[mapping[left]][mapping[right]]

    edge_control = instance(tmp_path, role(manifest, "single-edge", "control"))
    edge_treatment = instance(tmp_path, role(manifest, "single-edge", "treatment"))
    changed = [
        (left, right)
        for left in range(edge_control.dimension)
        for right in range(edge_control.dimension)
        if edge_control.edge_weights[left][right] != edge_treatment.edge_weights[left][right]
    ]
    assert len(changed) == 2
    assert changed[0] == tuple(reversed(changed[1]))

    capacity_control = instance(tmp_path, role(manifest, "capacity-threshold", "control"))
    capacity_treatment = instance(tmp_path, role(manifest, "capacity-threshold", "treatment"))
    assert capacity_treatment.capacity < capacity_control.capacity
    assert capacity_treatment.demands == capacity_control.demands
    assert capacity_treatment.edge_weights == capacity_control.edge_weights


def test_analysis_normalizes_equivalent_objectives(tmp_path, monkeypatch):
    probe, _suite_path, manifest_path, manifest = generated_probe(tmp_path, monkeypatch)
    results_path = tmp_path / "results.jsonl"
    records = []
    for entry in manifest["probes"]:
        scale = entry["objective_scale"]
        records.append({
            "run_id": entry["probe_id"],
            "solver": "black-box",
            "instance_path": entry["instance_path"],
            "checkpoint_seconds": 3,
            "seed": 0,
            "status": "SATISFIABLE",
            "verified": True,
            "objectives": [100 * scale],
            "dual_bound": 80 * scale,
            "wall_seconds": 1.0,
            "peak_memory_mb": 10.0,
            "candidates_per_second": 1000.0,
            "candidates_evaluated": 3000,
            "routes": [],
        })
    results_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    markdown_path = tmp_path / "report.md"
    json_path = tmp_path / "report.json"
    summary = probe.analyze(
        manifest_path, results_path, markdown_path, json_path,
    )

    assert summary["record_count"] == 8
    assert summary["paired_run_count"] == 4
    equivalent = [row for row in summary["rows"] if row["semantic_equivalent"]]
    assert len(equivalent) == 2
    assert all(row["normalized_objective_agreement_percent"] == 100 for row in equivalent)
    assert all(row["median_dual_ratio"] == 1 for row in equivalent)
    assert "Elles ne prouvent aucun détail d'implémentation" in markdown_path.read_text(encoding="utf-8")
    assert json.loads(json_path.read_text(encoding="utf-8"))["paired_run_count"] == 4


def test_solver_lag_is_paired_and_reports_missing_incumbents(tmp_path, monkeypatch):
    probe, _suite_path, _manifest_path, manifest = generated_probe(tmp_path, monkeypatch)
    manifest["campaign"] = {
        "solvers": ["qayd-api", "hexaly"],
        "budgets": [1],
        "seeds": [0],
        "threads": 1,
    }
    records = []
    for index, entry in enumerate(manifest["probes"]):
        records.append(solver_record(
            entry, "qayd-api", 110,
            status="UNKNOWN" if index == 0 else "SATISFIABLE",
            verified=index != 0,
            objectives=[] if index == 0 else [110 * entry["objective_scale"]],
        ))
        records.append(solver_record(entry, "hexaly", 100))

    rows = probe.solver_lag_rows(manifest, records)

    assert rows == [{
        "solver": "qayd-api",
        "reference": "hexaly",
        "size": 8,
        "checkpoint_seconds": 1,
        "expected_pairs": 8,
        "verified_pairs": 7,
        "missing_pairs": 1,
        "median_objective_lag_percent": 10.0,
    }]


def test_solver_lag_counts_pairs_missing_from_both_solvers(tmp_path, monkeypatch):
    probe, _suite_path, _manifest_path, manifest = generated_probe(tmp_path, monkeypatch)
    manifest["campaign"] = {
        "solvers": ["qayd-api", "hexaly"],
        "budgets": [1, 5],
        "seeds": [0, 1],
        "threads": 1,
    }
    records = []
    omitted = manifest["probes"][-1]["instance_path"]
    for entry in manifest["probes"]:
        if entry["instance_path"] == omitted:
            continue
        for budget in (1, 5):
            for seed in (0, 1):
                for solver, objective in (("qayd-api", 110), ("hexaly", 100)):
                    records.append(solver_record(
                        entry, solver, objective, budget=budget, seed=seed,
                    ))

    matrix = probe.solver_lag_observations(manifest, records)
    rows = probe.aggregate_solver_lag(matrix)

    assert len(rows) == 2
    assert all(row["expected_pairs"] == 16 for row in rows)
    assert all(row["verified_pairs"] == 14 for row in rows)
    assert all(row["missing_pairs"] == 2 for row in rows)
    absent = [row for row in matrix if row["instance_path"] == omitted]
    assert len(absent) == 4
    assert all(
        row["missing_reason"] == "solver:run_missing;reference:run_missing"
        for row in absent
    )
    assert all(row["solver_run_id"] is None for row in absent)
    assert all(row["reference_run_id"] is None for row in absent)


def test_solver_lag_observations_are_a_complete_exact_matrix(tmp_path, monkeypatch):
    probe, _suite_path, _manifest_path, manifest = generated_probe(tmp_path, monkeypatch)
    manifest["campaign"] = {
        "solvers": ["qayd-api", "hexaly"],
        "budgets": [1, 5],
        "seeds": [0, 1],
        "threads": 1,
    }
    records = [
        solver_record(entry, solver, objective, budget=budget, seed=seed)
        for entry in manifest["probes"]
        for budget in (1, 5)
        for seed in (0, 1)
        for solver, objective in (("qayd-api", 110), ("hexaly", 100))
    ]

    matrix = probe.solver_lag_observations(manifest, records)
    rows = probe.aggregate_solver_lag(matrix)

    assert len(matrix) == 32
    assert len(rows) == 2
    assert all(row["expected_pairs"] == 16 for row in rows)
    assert all(row["verified_pairs"] == 16 for row in rows)
    assert all(row["missing_pairs"] == 0 for row in rows)
    assert all(row["median_objective_lag_percent"] == 10 for row in rows)
    for observation in matrix:
        assert observation["missing_reason"] is None
        assert observation["objective_lag_percent"] == 10
        assert observation["campaign_threads"] == 1
        assert observation["instance_sha256"]
        assert observation["solver_run_id"].startswith("qayd-api-")
        assert observation["reference_run_id"].startswith("hexaly-")
        assert observation["solver_version"] == "qayd-api test-version"
        assert observation["reference_version"] == "hexaly test-version"
        assert observation["solver_threads"] == 1
        assert observation["reference_threads"] == 1
        assert observation["solver_verified"] is True
        assert observation["reference_verified"] is True
        assert observation["solver_normalized_objective"] == 110
        assert observation["reference_normalized_objective"] == 100
        assert observation["solver_evidence"]["run_count"] == 1
        assert observation["reference_evidence"]["run_count"] == 1
        assert observation["solver_evidence"]["runs"][0]["normalized_objectives"] == [110]
        assert observation["reference_evidence"]["runs"][0]["normalized_objectives"] == [100]


def test_solver_lag_rejects_bad_hash_and_preserves_provenance(tmp_path, monkeypatch):
    probe, _suite_path, manifest_path, manifest = generated_probe(tmp_path, monkeypatch)
    manifest["campaign"] = {
        "solvers": ["qayd-api", "hexaly"],
        "budgets": [1],
        "seeds": [0],
        "threads": 1,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    records = []
    bad_path = manifest["probes"][0]["instance_path"]
    for entry in manifest["probes"]:
        qayd = solver_record(entry, "qayd-api", 110)
        if entry["instance_path"] == bad_path:
            qayd["instance_sha256"] = "0" * 64
        records.extend((qayd, solver_record(entry, "hexaly", 100)))
    matrix = probe.solver_lag_observations(manifest, records)
    bad = next(row for row in matrix if row["instance_path"] == bad_path)

    assert bad["objective_lag_percent"] is None
    assert bad["missing_reason"] == "solver:instance_hash_mismatch"
    solver_run = bad["solver_evidence"]["runs"][0]
    reference_run = bad["reference_evidence"]["runs"][0]
    assert solver_run["run_id"].startswith("qayd-api-")
    assert solver_run["solver_version"] == "qayd-api test-version"
    assert solver_run["instance_sha256"] == "0" * 64
    assert reference_run["solver_version"] == "hexaly test-version"

    results_path = tmp_path / "results.jsonl"
    results_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8",
    )
    provenance = {"schema_version": 1, "solvers": {"qayd-api": "test"}}
    results_path.with_suffix(".jsonl.provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8",
    )
    summary = probe.analyze(
        manifest_path, results_path, tmp_path / "report.md", tmp_path / "report.json",
    )
    assert summary["campaign"] == manifest["campaign"]
    assert summary["generator"] == manifest["generator"]
    assert summary["campaign_provenance"] == provenance
    assert summary["probes"] == manifest["probes"]
    assert summary["solver_lag_pairs"] == matrix
