"""Scheduling controls must reach the real local-search campaign arm."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "bench" / "campaign.py"


def load_campaign():
    sys.path.insert(0, str(ROOT / "bench"))
    try:
        spec = importlib.util.spec_from_file_location("qayd_campaign_scheduling_controls", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def arguments(**overrides):
    values = {
        "threads": 4,
        "memory_limit_mb": 384,
        "grace_seconds": 20.0,
        "qayd_engine": "ls",
        "profile_qayd": True,
        "max_iterations": 123,
        "routing_two_way": True,
        "routing_nearest_neighbor": True,
        "routing_warm_start": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def item(problem, suffix):
    return {"problem": problem, "path": ROOT / f"instance{suffix}"}


def test_jssp_local_search_keeps_engine_threads_and_profile():
    campaign = load_campaign()
    args = arguments()
    instance = item("jssp", ".txt")

    command = campaign.command_for("qayd-api", instance, 30, 7, args)

    assert command[command.index("--engine") + 1] == "ls"
    assert command[command.index("--threads") + 1] == "4"
    assert command[command.index("--memory-limit-mb") + 1] == "384"
    assert "--profile" in command
    assert "--compact-json" in command
    assert command[command.index("--max-iterations") + 1] == "123"
    assert campaign.effective_threads("qayd-api", instance, args) == 4


def test_single_mode_rcpsp_local_search_is_not_silently_reclassified():
    campaign = load_campaign()
    args = arguments()
    instance = item("rcpsp", ".sm")

    command = campaign.command_for("qayd-api", instance, 30, 11, args)

    assert command[command.index("--engine") + 1] == "ls"
    assert "--profile" in command
    assert command[command.index("--max-iterations") + 1] == "123"
    assert campaign.effective_threads("qayd-api", instance, args) == 4


def test_auto_schedule_uses_the_requested_threads_and_profile_controls():
    campaign = load_campaign()
    args = arguments(qayd_engine="auto")
    instance = item("jssp", ".txt")

    command = campaign.command_for("qayd-api", instance, 30, 17, args)

    assert command[command.index("--engine") + 1] == "auto"
    assert command[command.index("--threads") + 1] == "4"
    assert "--profile" in command
    assert command[command.index("--max-iterations") + 1] == "123"
    assert campaign.effective_threads("qayd-api", instance, args) == 4


def test_multi_mode_rcpsp_keeps_its_explicit_exact_only_guard():
    campaign = load_campaign()
    args = arguments()
    instance = item("rcpsp", ".mm")

    command = campaign.command_for("qayd-native", instance, 30, 13, args)

    assert command[command.index("--engine") + 1] == "auto"
    assert "--profile" in command
    assert "--max-iterations" not in command
    assert "--compact-json" not in command
    assert campaign.effective_threads("qayd-native", instance, args) == 1


def test_both_qayd_jssp_launchers_request_compact_json():
    campaign = load_campaign()
    args = arguments()
    instance = item("jssp", ".txt")

    for solver in ("qayd-api", "qayd-native"):
        command = campaign.command_for(solver, instance, 30, 19, args)
        assert "--json" in command
        assert "--compact-json" in command


def test_grace_seconds_participates_in_run_identity():
    campaign = load_campaign()
    instance = {
        **item("jssp", ".txt"),
        "instance_path": "data/jssp/instance.txt",
        "instance_sha256": "abc123",
    }

    short_guard = campaign.run_key(
        "qayd-api", instance, 600, 0, arguments(grace_seconds=20.0)
    )
    long_guard = campaign.run_key(
        "qayd-api", instance, 600, 0, arguments(grace_seconds=600.0)
    )

    assert short_guard != long_guard


def test_prepare_qayd_uses_optimized_pyext_profile(monkeypatch):
    campaign = load_campaign()
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))

    monkeypatch.setattr(campaign.subprocess, "run", fake_run)
    campaign.prepare_qayd_extension()

    assert calls == [
        (
            ["maturin", "develop", "--profile", "pyext", "--features", "python"],
            {"cwd": ROOT, "check": True},
        )
    ]


def test_expected_instance_preflight_accepts_exact_basenames_and_hashes():
    campaign = load_campaign()
    suite = {
        "expected_instances": {
            "first.data": "a" * 64,
            "second.data": "b" * 64,
        }
    }
    instances = [
        {
            "path": ROOT / "data" / "one" / "first.data",
            "instance_sha256": "a" * 64,
        },
        {
            "path": ROOT / "data" / "two" / "second.data",
            "instance_sha256": "b" * 64,
        },
    ]

    campaign.preflight_expected_instances(suite, instances)


@pytest.mark.parametrize(
    ("instances", "message"),
    [
        (
            [
                {"path": ROOT / "one" / "first.data", "instance_sha256": "a" * 64},
                {"path": ROOT / "two" / "first.data", "instance_sha256": "a" * 64},
            ],
            "duplicate discovered basenames",
        ),
        (
            [{"path": ROOT / "first.data", "instance_sha256": "a" * 64}],
            "missing instances: second.data",
        ),
        (
            [
                {"path": ROOT / "first.data", "instance_sha256": "a" * 64},
                {"path": ROOT / "second.data", "instance_sha256": "b" * 64},
                {"path": ROOT / "extra.data", "instance_sha256": "c" * 64},
            ],
            "unexpected instances: extra.data",
        ),
        (
            [
                {"path": ROOT / "first.data", "instance_sha256": "0" * 64},
                {"path": ROOT / "second.data", "instance_sha256": "b" * 64},
            ],
            "SHA-256 mismatches: first.data",
        ),
    ],
)
def test_expected_instance_preflight_rejects_non_exact_sets(instances, message):
    campaign = load_campaign()
    suite = {
        "expected_instances": {
            "first.data": "a" * 64,
            "second.data": "b" * 64,
        }
    }

    with pytest.raises(SystemExit, match=message):
        campaign.preflight_expected_instances(suite, instances)


def test_suite_minimum_external_grace_is_enforced():
    campaign = load_campaign()
    suite = {"minimum_external_grace_seconds": 600}

    campaign.enforce_minimum_external_grace_seconds(suite, 600.0)
    with pytest.raises(SystemExit, match=r"--grace-seconds >= 600"):
        campaign.enforce_minimum_external_grace_seconds(suite, 599.999)
