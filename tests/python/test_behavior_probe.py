"""Controlled-input and paired-analysis checks for the behavior probe."""

import importlib.util
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


def test_generator_changes_exactly_the_declared_factor(tmp_path, monkeypatch):
    _probe, suite_path, _manifest_path, manifest = generated_probe(tmp_path, monkeypatch)
    assert suite_path.is_file()
    assert len(manifest["probes"]) == 8

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
