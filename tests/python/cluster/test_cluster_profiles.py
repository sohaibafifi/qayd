"""Keep category qualification, selection and OAR limits independent."""
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "bench"))
from cluster.lib import campaign, controller, hexaly, report


def test_category_configs_preserve_protocols_and_pairings_without_changing_the_base(tmp_path):
    base = {"name": "comparison", "seeds": [0, 1, 2], "memory_mb": 16384,
            "protocols": {"pb": {"seconds": 1800}, "minizinc": {"seconds": 1200}},
            "solvers": {"pb": ["qayd-pb", "sat4j"], "minizinc": ["qayd-fzn", "gecode"]},
            "datasets": [{"track": "pb", "family": "decision"}, {"track": "pb", "family": "optimization"},
                         {"track": "minizinc", "family": "mzn2025"}]}
    before = campaign.digest(base)
    profiles = controller.category_profiles(base, tmp_path)
    assert set(profiles) == {"pb", "minizinc"}
    for track, profile in profiles.items():
        config = controller.read(profile["config"])
        assert config["datasets"] == [d for d in base["datasets"] if d["track"] == track]
        assert all(config[key] == base[key] for key in ("protocols", "solvers", "seeds", "memory_mb"))
    assert campaign.digest(base) == before


def test_ready_sections_share_one_global_oar_limit_and_rotate_fairly():
    choices = []
    for name in ("pb", "cvrp", "jssp", "rcpsp", "xcsp-csp"):
        plan = {"tasks": [{"seconds": 30, "solvers": ["qayd", "baseline"]} for _ in range(10)]}
        choices.append((name, ("gate", Path(name), plan, list(range(10)))))
    first = controller.wave_allocations(choices, 4, 0)
    assert len(first) == 4 and sum(count for _, _, count in first) == 4
    assert "xcsp-csp" in {name for name, _, _ in controller.wave_allocations(choices, 4, 1)}
    smaller = controller.wave_allocations(choices[:2], 4, 0)
    assert len(smaller) == 2 and sum(count for _, _, count in smaller) == 4
    only = [("pb", ("gate", Path("pb"), choices[0][1][2], [0]))]
    assert controller.wave_allocations(only, 4, 0)[0][2] == 1


def saved_settings(tmp_path):
    worker = tmp_path / "driver.py"
    worker.write_text("# frozen deadline worker\n")
    return {"root": str(tmp_path), "out": str(tmp_path), "python": sys.executable,
            "driver": str(worker), "files": {}, "prefix": "qayd-test-", "jobs": 4,
            "workers": "auto", "property": "cpu-only", "resource": "/nodes=1", "walltime": "01:00:00",
            "java": "/java", "hgs": "/hgs", "lkh": None, "pending": [],
            "execution_policy": controller.EXECUTION_POLICY, "profiles": {}}


@pytest.mark.parametrize("failure", ["ERROR", "UNSAT", "INVALID"])
def test_solver_failures_are_recorded_and_all_categories_reach_full(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    monkeypatch.setattr(controller, "modules", lambda _: (campaign, report))
    monkeypatch.setattr(campaign, "verify_snapshot", lambda _: None)
    monkeypatch.setattr(controller, "live_jobs", lambda _: [])
    monkeypatch.setattr(controller.time, "sleep", lambda _: None)
    settings = saved_settings(tmp_path)
    for track, solvers in (("pb", ["qayd-pb", "sat4j"]), ("minizinc", ["qayd-fzn", "gecode"])):
        path = tmp_path / "config" / f"{track}.json"
        controller.write(path, {"name": track, "datasets": [{"track": track}], "seeds": [0],
                                "protocols": {track: {"seconds": 1200}}, "solvers": {track: solvers}})
        settings["profiles"][track] = {"config": str(path)}

    def discover(config):
        track = config["datasets"][0]["track"]
        item = {"track": track, "path": track, "id": track, "family": "fixture", "sense": "min"}
        return [item], {track: "frozen-input"}, []

    monkeypatch.setattr(campaign, "discover", discover)
    submissions = []

    def run(command, **kwargs):
        command = list(map(str, command))
        if command[0] == "oarsub":
            payload = shlex.split(command[-1])
            path = Path(payload[payload.index("--plan") + 1])
            taskfile = Path(payload[payload.index("--tasks") + 1])
            plan = campaign.load_plan(path)
            submissions.append((path.parent.parent.name, plan["stage"], command))
            for index in map(int, taskfile.read_text().split()):
                task = plan["tasks"][index]
                arms = []
                for solver in task["solvers"]:
                    failed = solver == "qayd-fzn" and plan["stage"] == "gate"
                    arms.append({"solver": solver, "status": failure if failed else "SAT",
                                 "objective": None if failed else 10, "solved": failed,
                                 "has_incumbent": not failed, "validation": {"valid": None if failed else True},
                                 "wall_seconds": 1, "par2": 1, "details": {}, "host": {"hostname": "cpu"},
                                 "plan_sha256": plan["sha256"], "task_key": task["key"]})
                campaign.write(path.parent / "tasks" / f"{index:07d}" / "result.json", {
                    "complete": True, "plan_sha256": plan["sha256"], "task_key": task["key"], "arms": arms})
        elif "prepare" in command:
            config = controller.read(command[command.index("--config") + 1])
            stage = command[command.index("--stage") + 1]
            directory = Path(command[command.index("--out") + 1])
            items, assets, _ = discover(config)
            plan = {"root": str(tmp_path), "stage": stage, "environment": campaign.digest(config),
                    "config": config, "assets": assets, "artifacts": {}, "overview": {},
                    "tasks": campaign.tasks_for(items, config, stage)}
            plan["sha256"] = campaign.digest(plan)
            campaign.write(directory / "plan.json", plan)
        else:
            pytest.fail(f"unexpected command: {command}")

    monkeypatch.setattr(controller, "run", run)
    controller.control(SimpleNamespace(out=tmp_path, execute=True, poll_seconds=5), settings)
    assert controller.read(tmp_path / "pb/full/qualification.json")["ready"] is True
    assert (tmp_path / "minizinc/full").exists()
    gate = controller.read(tmp_path / "minizinc/gate/report.json")
    assert gate["complete"] and not gate["ready"] and not gate["blocking_errors"]
    assert bool(gate["contradictions"]) == (failure == "UNSAT")
    row = next(row for row in gate["records"] if row["solver"] == "qayd-fzn")
    assert row["status"] == ("ERROR" if failure == "UNSAT" else failure)
    assert not row["solved"] and row["par2"] == 2 * row["budget"]
    raw = controller.read(tmp_path / "minizinc/gate/tasks/0000000/result.json")
    assert next(arm for arm in raw["arms"] if arm["solver"] == "qayd-fzn")["status"] == failure
    assert {profile for profile, stage, _ in submissions if stage == "full"} == {"pb", "minizinc"}
    assert campaign.load_plan(tmp_path / "pb/full/plan.json")["tasks"][0]["seconds"] == 1200
    waves = {}
    for _, _, command in submissions:
        wave = Path(command[command.index("-O") + 1]).parent
        waves[wave] = waves.get(wave, 0) + 1
    assert all(count <= settings["jobs"] for count in waves.values())


def test_selected_profile_ignores_other_failed_sections_without_erasing_their_errors(tmp_path, monkeypatch):
    settings = saved_settings(tmp_path)
    settings.update(profiles={"pb": {}, "minizinc": {}}, selected_profiles=["pb"])
    controller.write(tmp_path / "state.json", {"waves": 0, "profile_errors": {"minizinc": "false UNSAT"}})
    monkeypatch.setattr(controller, "live_jobs", lambda _: [])
    visited = []
    monkeypatch.setattr(controller, "next_stage", lambda _, profile: visited.append(profile))
    controller.control(SimpleNamespace(out=tmp_path, execute=False), settings)
    state = controller.read(tmp_path / "state.json")
    assert visited == ["pb"]
    assert state["status"] == "complete-selected"
    assert state["profile_errors"] == {"minizinc": "false UNSAT"}
    settings["selected_profiles"] = ["typo"]
    with pytest.raises(ValueError, match="available profiles"):
        controller.selected_profiles(settings)


def test_profile_cli_remembers_a_selection_and_all_restores_the_complete_pool(tmp_path, monkeypatch):
    out = tmp_path / "runs"
    out.mkdir()
    settings = saved_settings(out)
    settings.update(root=str(tmp_path), profiles={"pb": {}, "minizinc": {}})
    controller.write(out / "settings.json", settings)
    monkeypatch.chdir(tmp_path)
    visited = []
    monkeypatch.setattr(controller, "control", lambda _, saved: visited.append(controller.selected_profiles(saved)))
    for profile in ("pb", "all"):
        monkeypatch.setattr(sys, "argv", [controller.__file__, "--repo", str(tmp_path), "--out", str(out),
                                          "--profile", profile])
        controller.main()
        stored = controller.read(out / "settings.json")
        assert stored["selected_profiles"] == (["pb"] if profile == "pb" else None)
    assert visited[0] == ["pb"]
    assert set(visited[1]) == {"pb", "minizinc"}


def test_hexaly_profiles_are_separate_and_legacy_settings_are_preserved(tmp_path):
    settings = saved_settings(tmp_path)
    controller.register_hexaly_profiles(settings)
    assert set(settings["profiles"]) == {"hexaly-cvrp", "hexaly-jssp", "hexaly-rcpsp"}
    assert [settings["profiles"][name]["reference_track"] for name in sorted(settings["profiles"])] == ["cvrp", "jssp", "rcpsp"]
    legacy = {"out": str(tmp_path), "profiles": {"hexaly": {"config": "original"}}}
    controller.register_hexaly_profiles(legacy)
    assert legacy["profiles"] == {"hexaly": {"config": "original"}}


@pytest.mark.parametrize("track", ["cvrp", "jssp", "rcpsp"])
def test_hexaly_prepares_only_the_requested_category(tmp_path, monkeypatch, track):
    monkeypatch.setattr(hexaly, "ROOT", tmp_path)
    controller.write(tmp_path / "bench/cluster/config/hexaly-jssp-1000x1000.json", {
        "reference_protocol": {"hexaly": {"reference_incumbent_10m": {}}}})
    destination = tmp_path / "data/cluster-hexaly/rg300/catalog.json"
    conversions = []
    monkeypatch.setattr(hexaly, "convert_rg300", lambda *args: conversions.append(args) or destination)
    out = tmp_path / track
    monkeypatch.setattr(sys, "argv", ["hexaly.py", "--track", track, "--out", str(out)])
    hexaly.main()
    config = controller.read(out / "config.json")
    assert {d["track"] for d in config["datasets"]} == {track}
    assert set(config["published_reference"]["pages"]) == {track}
    assert bool(conversions) == (track == "rcpsp")
    assert config["protocols"][track]["seconds"] == (600 if track == "jssp" else 60)
    assert config["protocols"][track]["seeds"] == list(range(10))
