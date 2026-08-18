"""Scheduling controls must reach the real local-search campaign arm."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys


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
    assert campaign.effective_threads("qayd-native", instance, args) == 1
