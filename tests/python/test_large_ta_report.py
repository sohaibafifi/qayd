"""Strict reporting checks for the Large-TA Hexaly comparison."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "bench" / "large_ta_report.py"


def load_module():
    spec = importlib.util.spec_from_file_location("qayd_large_ta_report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fixtures(tmp_path):
    expected = {}
    references = {}
    records = []
    offsets = (-10, 0, 10, -10, 0, 10, -10, 0, 10, 0)
    for index, offset in enumerate(offsets, 1):
        filename = f"tai_j1000_m1000_{index}.data"
        stem = Path(filename).stem
        digest = f"{index:02x}" * 32
        expected[filename] = digest
        references[stem] = 1000
        records.append(
            {
                "run_id": f"run-{index}",
                "solver": "qayd-api",
                "solver_version": "qayd 0.1.0 (synthetic source and artifact)",
                "problem": "jssp",
                "family": "large-ta-1000x1000",
                "instance_path": f"data/jssp/large-ta/{filename}",
                "instance_sha256": digest,
                "checkpoint_seconds": 600,
                "seed": 0,
                "engine": "ls",
                "threads": 8,
                "requested_threads": 8,
                "schedule_restart_work": 256,
                "status": "SATISFIABLE",
                "verified": True,
                "return_code": 0,
                "timed_out": False,
                "objectives": [1000 + offset],
                "elapsed_seconds": 599.5 + index / 100,
                "elapsed_seconds_scope": "model.solve",
                "solve_seconds": 599.5 + index / 100,
                "wall_seconds": 600.0 + index / 100,
                "grace_seconds": 600.0,
                "external_timeout_seconds": 1200.0,
                "peak_memory_mb": 1000.0 + index,
                "jobs": 1000,
                "machines": 1000,
                "operations": 1_000_000,
                "start_vector_length": 1_000_000,
                "start_vector_sha256": f"{index + 16:02x}" * 32,
                "verification_counts": {
                    "starts": 1_000_000,
                    "job_precedence_pairs": 999_000,
                    "machine_non_overlap_pairs": 999_000,
                    "objective_checks": 1,
                },
                "command": [
                    sys.executable,
                    str(
                        SCRIPT.parents[1]
                        / "examples"
                        / "python"
                        / "scheduling"
                        / "api"
                        / "jssp.py"
                    ),
                    str(
                        SCRIPT.parents[1]
                        / "data"
                        / "jssp"
                        / "large-ta"
                        / filename
                    ),
                    "--time-limit",
                    "600",
                    "--threads",
                    "8",
                    "--seed",
                    "0",
                    "--json",
                    "--engine",
                    "ls",
                    "--memory-limit-mb",
                    "0",
                    "--profile",
                    "--compact-json",
                ],
            }
        )
    manifest = {
        "name": "synthetic-large-ta",
        "minimum_external_grace_seconds": 600,
        "reference_protocol": {
            "hexaly": {
                "version": "15.0",
                "time_limit_seconds": 600,
                "reference_incumbent_10m": references,
            }
        },
        "expected_instances": expected,
    }
    manifest_path = tmp_path / "manifest.json"
    campaign_path = tmp_path / "campaign.jsonl"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    campaign_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    provenance = {
        "schema_version": 1,
        "suite": manifest,
        "suite_file": str(manifest_path.resolve()),
        "solvers": {
            "qayd-api": "qayd 0.1.0 (synthetic source and artifact)",
        },
        "budgets": [600],
        "seeds": [0],
        "threads": 8,
        "memory_limit_mb": 0,
        "grace_seconds": 600.0,
        "qayd_engine": "ls",
        "max_iterations": None,
        "profile_qayd": True,
        "qayd_prepared": True,
        "qayd_artifact": {
            "path": "/tmp/qayd/_core.abi3.so",
            "sha256": "c" * 64,
        },
        "host": {
            "commit": "a" * 40,
            "source_tree_sha256": "b" * 64,
            "dirty": True,
        },
    }
    campaign_path.with_suffix(campaign_path.suffix + ".provenance.json").write_text(
        json.dumps(provenance) + "\n", encoding="utf-8"
    )
    return manifest_path, campaign_path, records


def rewrite_campaign(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def rewrite_provenance(campaign_path, mutate):
    path = campaign_path.with_suffix(campaign_path.suffix + ".provenance.json")
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def test_complete_report_has_signed_gaps_resources_and_counts(tmp_path):
    report = load_module()
    manifest_path, campaign_path, _records = fixtures(tmp_path)

    summary = report.build_summary(campaign_path, manifest_path=manifest_path)

    assert len(summary["instances"]) == 10
    assert summary["instances"][0]["qayd_vs_hexaly_percent"] == -1.0
    assert summary["instances"][2]["qayd_vs_hexaly_percent"] == 1.0
    assert summary["instances"][0]["solve_elapsed_seconds"] == 599.51
    assert summary["instances"][0]["end_to_end_wall_seconds"] == 600.01
    assert summary["instances"][0]["peak_rss_mb"] == 1001.0
    assert summary["qayd_protocol"] == {
        "solver": "qayd-api",
        "solver_version": "qayd 0.1.0 (synthetic source and artifact)",
        "engine": "ls",
        "threads": 8,
        "requested_threads": 8,
        "schedule_restart_work": 256,
        "instance_shape": {
            "jobs": 1000,
            "machines": 1000,
            "operations": 1_000_000,
        },
        "compact_schedule_commitment": {
            "start_vector_length": 1_000_000,
            "start_vector_sha256": "required",
            "role": "commitment only, not a feasibility certificate",
            "post_hoc_replayable": False,
        },
        "schedule_validation": {
            "online_replay_by_launcher": True,
            "post_hoc_replay_from_digest": False,
        },
        "verification_counts": {
            "starts": 1_000_000,
            "job_precedence_pairs": 999_000,
            "machine_non_overlap_pairs": 999_000,
            "objective_checks": 1,
        },
    }
    assert summary["aggregate"] == {
        "instance_count": 10,
        "selected_record_count": 10,
        "record_coverage_percent": 100.0,
        "feasible_count": 10,
        "verified_feasible_count": 10,
        "incumbent_coverage_percent": 100.0,
        "claim_ready": True,
        "qayd_better_count": 3,
        "equal_count": 4,
        "qayd_worse_count": 3,
        "status_counts": {"SATISFIABLE": 10},
        "median_qayd_vs_hexaly_percent": 0.0,
        "mean_qayd_vs_hexaly_percent": 0.0,
    }
    assert summary["claim_ready"] is True
    assert summary["campaign_provenance"]["qayd_prepared"] is True
    assert summary["campaign_provenance"]["grace_seconds"] == 600.0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("instance_sha256", "0" * 64, "SHA-256 does not match"),
        ("verified", False, "feasible record verified must be true"),
        ("objectives", [1000, 1], "one scalar makespan"),
        ("elapsed_seconds", None, "elapsed_seconds must be a number"),
        ("elapsed_seconds_scope", "whole launcher", "must be model.solve"),
        ("solve_seconds", None, "solve_seconds must be a number"),
        ("wall_seconds", -1, "wall_seconds must be non-negative"),
        ("peak_memory_mb", "large", "peak_memory_mb must be a number"),
        ("return_code", 1, "feasible record return_code must be exactly 0"),
        ("timed_out", True, "feasible record timed_out must be false"),
        ("jobs", 999, "jobs must be exactly 1000"),
        ("machines", 999, "machines must be exactly 1000"),
        ("operations", 999_999, "operations must be exactly 1000000"),
        ("start_vector_length", 999_999, "start_vector_length must be exactly 1000000"),
        ("start_vector_sha256", "not-a-hash", "start_vector_sha256 must be a valid SHA-256"),
        (
            "verification_counts",
            {
                "starts": 1_000_000,
                "job_precedence_pairs": 999_000,
                "machine_non_overlap_pairs": 998_999,
                "objective_checks": 1,
            },
            "verification_counts.machine_non_overlap_pairs must be exactly 999000",
        ),
    ],
)
def test_invalid_selected_record_is_rejected(tmp_path, field, value, message):
    report = load_module()
    manifest_path, campaign_path, records = fixtures(tmp_path)
    records[0][field] = value
    rewrite_campaign(campaign_path, records)

    with pytest.raises(report.ReportError, match=message):
        report.build_summary(campaign_path, manifest_path=manifest_path)


def test_mismatched_solve_elapsed_is_rejected(tmp_path):
    report = load_module()
    manifest_path, campaign_path, records = fixtures(tmp_path)
    records[0]["solve_seconds"] += 0.1
    rewrite_campaign(campaign_path, records)

    with pytest.raises(report.ReportError, match="solve_seconds must match"):
        report.build_summary(campaign_path, manifest_path=manifest_path)


def test_missing_and_duplicate_selected_records_are_rejected(tmp_path):
    report = load_module()
    manifest_path, campaign_path, records = fixtures(tmp_path)
    rewrite_campaign(campaign_path, records[1:])
    with pytest.raises(report.ReportError, match="missing Qayd record"):
        report.build_summary(campaign_path, manifest_path=manifest_path)

    rewrite_campaign(campaign_path, records + [dict(records[0], run_id="copy")])
    with pytest.raises(report.ReportError, match="duplicate Qayd records"):
        report.build_summary(campaign_path, manifest_path=manifest_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("solver", "qayd-native", "uniform solver variant"),
        ("solver_version", "qayd different", "uniform solver version"),
        ("engine", "auto", "uniform engine"),
        ("threads", 4, "uniform threads"),
        ("requested_threads", 4, "uniform requested_threads"),
        ("schedule_restart_work", 128, "uniform schedule_restart_work"),
    ],
)
def test_mixed_qayd_protocol_is_rejected(tmp_path, field, value, message):
    report = load_module()
    manifest_path, campaign_path, records = fixtures(tmp_path)
    records[0][field] = value
    rewrite_campaign(campaign_path, records)

    with pytest.raises(report.ReportError, match=message):
        report.build_summary(campaign_path, manifest_path=manifest_path)


def test_missing_schedule_restart_work_is_rejected(tmp_path):
    report = load_module()
    manifest_path, campaign_path, records = fixtures(tmp_path)
    records[0].pop("schedule_restart_work")
    rewrite_campaign(campaign_path, records)

    with pytest.raises(report.ReportError, match="no valid schedule_restart_work"):
        report.build_summary(campaign_path, manifest_path=manifest_path)


def test_incorrect_uniform_schedule_restart_work_is_rejected(tmp_path):
    report = load_module()
    manifest_path, campaign_path, records = fixtures(tmp_path)
    for record in records:
        record["schedule_restart_work"] = 128
    rewrite_campaign(campaign_path, records)

    with pytest.raises(
        report.ReportError, match="schedule_restart_work must be exactly 256"
    ):
        report.build_summary(campaign_path, manifest_path=manifest_path)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("suite",), {}, "suite content does not match"),
        (("suite_file",), "/tmp/wrong-manifest.json", "does not identify the manifest"),
        (("budgets",), [60, 600], "budgets must contain exactly 600"),
        (("seeds",), [0, 1], "seeds must contain exactly 0"),
        (("threads",), 0, "threads must be a positive integer"),
        (("grace_seconds",), 599.0, "below the manifest minimum"),
        (("qayd_prepared",), False, "qayd_prepared must be true"),
        (("qayd_artifact", "path"), "relative/_core.so", "absolute path"),
        (("qayd_artifact", "sha256"), "bad", "valid SHA-256"),
        (("host", "commit"), "not-a-commit", "commit and source tree"),
        (("host", "source_tree_sha256"), "bad", "commit and source tree"),
        (("solvers", "qayd-api"), "", "solver versions must be non-empty"),
    ],
)
def test_campaign_provenance_is_strictly_validated(tmp_path, path, value, message):
    report = load_module()
    manifest_path, campaign_path, _records = fixtures(tmp_path)

    def mutate(provenance):
        target = provenance
        for field in path[:-1]:
            target = target[field]
        target[path[-1]] = value

    rewrite_provenance(campaign_path, mutate)
    with pytest.raises(report.ReportError, match=message):
        report.build_summary(campaign_path, manifest_path=manifest_path)


def test_campaign_provenance_sidecar_is_required(tmp_path):
    report = load_module()
    manifest_path, campaign_path, _records = fixtures(tmp_path)
    campaign_path.with_suffix(campaign_path.suffix + ".provenance.json").unlink()

    with pytest.raises(report.ReportError, match="provenance sidecar is required"):
        report.build_summary(campaign_path, manifest_path=manifest_path)


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--time-limit", "599", "does not match checkpoint_seconds"),
        ("--seed", "1", "does not match seed"),
        ("--threads", "4", "does not match threads"),
        ("--engine", "auto", "does not match engine"),
    ],
)
def test_selected_command_flags_must_match_the_record(tmp_path, flag, value, message):
    report = load_module()
    manifest_path, campaign_path, records = fixtures(tmp_path)
    command = records[0]["command"]
    command[command.index(flag) + 1] = value
    rewrite_campaign(campaign_path, records)

    with pytest.raises(report.ReportError, match=message):
        report.build_summary(campaign_path, manifest_path=manifest_path)


def test_selected_command_must_request_compact_mode(tmp_path):
    report = load_module()
    manifest_path, campaign_path, records = fixtures(tmp_path)
    records[0]["command"].remove("--compact-json")
    rewrite_campaign(campaign_path, records)

    with pytest.raises(report.ReportError, match="must request compact mode"):
        report.build_summary(campaign_path, manifest_path=manifest_path)


def make_first_record_unknown(report, records):
    record = records[0]
    record["status"] = "UNKNOWN"
    record["verified"] = False
    record["objectives"] = []
    record.pop("engine")
    for field in ("jobs", "machines", "operations"):
        record.pop(field)
    record["start_vector_length"] = 0
    record["start_vector_sha256"] = report.EMPTY_START_VECTOR_SHA256
    record["verification_counts"] = dict(report.EMPTY_VERIFICATION_COUNTS)


def test_unknown_row_is_reported_without_claiming_an_incumbent(tmp_path):
    report = load_module()
    manifest_path, campaign_path, records = fixtures(tmp_path)
    make_first_record_unknown(report, records)
    rewrite_campaign(campaign_path, records)

    summary = report.build_summary(campaign_path, manifest_path=manifest_path)

    assert len(summary["instances"]) == 10
    assert summary["instances"][0]["status"] == "UNKNOWN"
    assert summary["instances"][0]["qayd_makespan"] is None
    assert summary["instances"][0]["qayd_vs_hexaly_percent"] is None
    assert summary["aggregate"]["selected_record_count"] == 10
    assert summary["aggregate"]["feasible_count"] == 9
    assert summary["aggregate"]["incumbent_coverage_percent"] == 90.0
    assert summary["claim_ready"] is False
    markdown = report.markdown_report(summary)
    assert "| `tai_j1000_m1000_1` | n/a |" in markdown
    assert "Claim ready: no" in markdown


def test_non_feasible_row_cannot_carry_a_false_incumbent(tmp_path):
    report = load_module()
    manifest_path, campaign_path, records = fixtures(tmp_path)
    make_first_record_unknown(report, records)
    records[0]["objectives"] = [999]
    rewrite_campaign(campaign_path, records)

    with pytest.raises(report.ReportError, match="objectives must be empty"):
        report.build_summary(campaign_path, manifest_path=manifest_path)


def test_require_complete_rejects_diagnostic_table(tmp_path):
    report = load_module()
    manifest_path, campaign_path, records = fixtures(tmp_path)
    make_first_record_unknown(report, records)
    rewrite_campaign(campaign_path, records)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(campaign_path),
            "--manifest",
            str(manifest_path),
            "--markdown",
            str(tmp_path / "report.md"),
            "--require-complete",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "complete incumbent coverage is required" in completed.stderr


def test_cli_writes_markdown_and_json_outputs(tmp_path):
    manifest_path, campaign_path, _records = fixtures(tmp_path)
    markdown_path = tmp_path / "report.md"
    json_path = tmp_path / "report.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(campaign_path),
            "--manifest",
            str(manifest_path),
            "--markdown",
            str(markdown_path),
            "--json",
            str(json_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    markdown = markdown_path.read_text(encoding="utf-8")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "Positive values mean Qayd is worse." in markdown
    assert "Hexaly reference incumbent" in markdown
    assert "End-to-end wall" in markdown
    assert "## Qualified Qayd protocol" in markdown
    assert "Solver variant: `qayd-api`" in markdown
    assert "Schedule restart work: 256 local-search moves per worker boundary" in markdown
    assert "1000000 starts with a valid SHA-256" in markdown
    assert "commitment only, not a feasibility certificate" in markdown
    assert "not post-hoc replayable" in markdown
    assert payload["aggregate"]["instance_count"] == 10
    assert payload["qayd_protocol"]["schedule_restart_work"] == 256
    assert payload["claim_ready"] is True
