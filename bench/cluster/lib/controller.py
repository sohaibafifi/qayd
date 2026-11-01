#!/usr/bin/env python3
"""Continue the Ritchie campaign with a bounded OAR pool and the validated build."""
from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import types
import uuid
import zlib

EXECUTION_POLICY = {
    "version": "qayd.cluster.deadline/v2",
    "smoke_seconds": 30,
    "gate_seconds": 30,
    "large_jssp_gate_seconds": 600,
    "publication_fraction": 0.05,
    "publication_minimum_seconds": 0.1,
    "publication_maximum_seconds": 30,
    "search_quantization": "floor-seconds",
}

CORPUS_TOOL = None
AUDITOR_SOURCE = None
_BUNDLED_REPORT = None
TERMINAL = {"Terminated", "Error"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run(command, **kwargs):
    return subprocess.run([str(v) for v in command], check=True, text=True, **kwargs)


def modules(root):
    global _BUNDLED_REPORT
    sys.path.insert(0, str(root / "bench"))
    from cluster.lib import campaign, report
    if campaign.ROOT != root:
        raise ValueError("use a separate process for each checkout")
    if AUDITOR_SOURCE is not None:
        if _BUNDLED_REPORT is None:
            _BUNDLED_REPORT = types.ModuleType("qayd_continuation_report")
            _BUNDLED_REPORT.__file__ = __file__
            exec(compile(AUDITOR_SOURCE, __file__ + ":audit", "exec"), _BUNDLED_REPORT.__dict__)
        return campaign, _BUNDLED_REPORT
    return campaign, report


def export_continuation(destination):
    """Bundle only the controller and auditor, leaving a running checkout frozen."""
    source = Path(__file__).read_text()
    marker = "AUDITOR_SOURCE = None\n"
    if source.count(marker) != 1:
        raise ValueError("export from the repository copy of controller.py")
    auditor = Path(__file__).with_name("report.py").read_bytes()
    payload = base64.b64encode(zlib.compress(auditor, 9)).decode()
    source = source.replace(marker, f"AUDITOR_SOURCE = zlib.decompress(base64.b64decode({payload!r})).decode()\n")
    destination = Path(destination)
    compile(source, str(destination), "exec")
    if destination.exists() and destination.read_text() != source:
        raise ValueError(f"refusing to replace a different continuation: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source)
    return destination


def dictionaries(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from dictionaries(child)
    elif isinstance(value, list):
        for child in value:
            yield from dictionaries(child)


def cpu_property(value):
    rows = [row for row in dictionaries(value) if "network_address" in row and "gpumodel" in row]
    if not rows or not any(row["gpumodel"] not in (None, "", "None") for row in rows):
        raise ValueError("cannot establish the GPU-host exclusion; provide --property")
    hosts = {row["network_address"] for row in rows}
    gpu_hosts = {row["network_address"] for row in rows if row["gpumodel"] not in (None, "", "None")}
    if not hosts - gpu_hosts:
        raise ValueError("no CPU-only host found")
    return "network_address NOT IN (SELECT network_address FROM resources WHERE gpumodel IS NOT NULL)"


def find_minizinc(root):
    candidates = [p for p in (root / "bench/cluster/runs/tools").glob(
        "MiniZincIDE-*-bundle-linux-*/bin/minizinc") if os.access(p, os.X_OK)]
    if len(candidates) == 1:
        return str(candidates[0].absolute())
    if not candidates and shutil.which("minizinc"):
        return shutil.which("minizinc")
    raise ValueError("cannot select MiniZinc; provide --minizinc <executable>")


def shared_java(root, executable):
    directory = root / "bench/cluster/runs/tools/shared-java"
    manifest = directory.with_suffix(".json")
    if not manifest.exists():
        result = run([executable, "-XshowSettings:properties", "-version"], capture_output=True)
        match = re.search(r"^\s*java.home\s*=\s*(.+)$", result.stderr + result.stdout, re.M)
        if not match:
            raise ValueError("Java did not report java.home")
        source = Path(match.group(1).strip())
        if directory.exists():
            raise ValueError(f"incomplete Java copy at {directory}; choose another --out after resolving it")
        print(f"Copy shared Java from {source}", flush=True)
        shutil.copytree(source, directory, symlinks=False, ignore_dangling_symlinks=True)
        run([directory / "bin/java", "-version"])
        write(manifest, {str(p): sha(p) for p in sorted(directory.rglob("*")) if p.is_file()})
    verify_files(read(manifest))
    return str(directory / "bin/java"), {str(manifest): sha(manifest)}


def verify_files(files):
    for path, expected in files.items():
        if not Path(path).is_file() or sha(path) != expected:
            raise ValueError(f"frozen runtime changed: {path}")


def worker_count(requested, memory_mb, threads, affinity, available_mb):
    capacity = min(len(affinity) // threads, max(0, available_mb - 2048) // memory_mb)
    workers = min(16, capacity) if requested == "auto" else int(requested)
    if workers < 1 or workers > capacity:
        raise ValueError(f"insufficient allocation for {workers} workers: capacity={capacity}")
    return workers


def worker(args):
    root = args.repo.resolve()
    campaign, _report = modules(root)
    settings = read(args.out / "settings.json")
    verify_files(settings["files"])
    verify_files(read(settings["java_manifest"]))
    plan = campaign.load_plan(args.plan)
    memory = {key: int(value.split()[0]) // 1024 for key, value in
              (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())}
    lanes = worker_count(settings["workers"], plan["memory_mb"], plan["threads"],
                         os.sched_getaffinity(0), memory["MemAvailable"])
    print(f"OAR {os.environ.get('OAR_JOB_ID')}: {lanes} workers, {plan['memory_mb']} MiB per solver", flush=True)
    os.environ.update(QAYD_CAMPAIGN_WORKERS=str(lanes), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                      MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    if plan.get("execution_policy"):
        payload = [plan["runtime"]["python"], settings["driver"], "--repo", root,
                   "--out", args.out, "--plan", args.plan, "--task", "{}"]
    else:
        payload = [plan["runtime"]["python"], root / "bench/cluster/lib/campaign.py", "run",
                   "--plan", args.plan, "--task", "{}"]
    run(["parallel", "--quote", "--jobs", lanes, "--halt", "soon,fail=1", "--line-buffer",
         *payload, "::::", args.tasks], cwd=root)


def remaining_search(seconds, started):
    reserve = min(EXECUTION_POLICY["publication_maximum_seconds"], max(
        EXECUTION_POLICY["publication_minimum_seconds"], seconds * EXECUTION_POLICY["publication_fraction"]))
    return max(0.0, seconds - (time.monotonic() - started) - reserve)


def model_adapter(argv):
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    def remaining():
        value = math.floor(remaining_search(args.seconds, started))
        if value <= 0:
            print(json.dumps({"status": "UNKNOWN", "objectives": [], "verified": False,
                              "diagnostic": "preparation exhausted the end-to-end deadline"}))
            raise SystemExit(0)
        return value
    if args.solver.startswith("qayd-"):
        import qayd
        original = qayd.Model.solve
        def solve(model, *values, **kwargs):
            kwargs["time_limit"] = remaining()
            return original(model, *values, **kwargs)
        qayd.Model.solve = solve
    elif args.solver == "ortools-cp-sat":
        from ortools.sat.python import cp_model
        original = cp_model.CpSolver.solve
        def solve(solver, *values, **kwargs):
            solver.parameters.max_time_in_seconds = remaining()
            return original(solver, *values, **kwargs)
        cp_model.CpSolver.solve = solve
    else:
        original = argparse.ArgumentParser.parse_args
        def parse(adapter_parser, *values, **kwargs):
            result = original(adapter_parser, *values, **kwargs)
            if hasattr(result, "time_limit"):
                result.time_limit = max(1, math.floor(remaining()))
            return result
        argparse.ArgumentParser.parse_args = parse
    import runpy
    tail = args.script_args[1:] if args.script_args[:1] == ["--"] else args.script_args
    sys.argv = [str(args.script), *tail]
    runpy.run_path(str(args.script), run_name="__main__")


def normalize_deadline(record):
    if (record.get("status") == "ERROR" and record.get("timed_out") is True
            and record.get("returncode") in (-9, -15) and not record.get("has_incumbent")
            and record.get("validation", {}).get("valid") is None
            and record.get("details", {}).get("parse_error") == "solver did not emit a JSON object"
            and all(record.get(key) and not Path(record[key]).read_text().strip() for key in ("stdout", "stderr"))):
        record["raw_status"] = "ERROR"
        record["status"] = "UNKNOWN"
        record["status_reason"] = "external deadline before any solver output"
    return record


def run_task(args):
    campaign, _report = modules(args.repo)
    plan = campaign.load_plan(args.plan)
    if plan.get("execution_policy") != EXECUTION_POLICY:
        raise ValueError("task requires a fresh v2 plan; preserve the old campaign and choose a new --out")
    from cluster.lib import runners
    original_command, original_arm = runners.command, runners.run_arm
    driver = read(args.out / "settings.json")["driver"]
    def command(solver, item, task, frozen, instance):
        result = original_command(solver, item, task, frozen, instance)
        if item["track"] in runners.MODEL_TRACKS:
            return [result[0], driver, "model", "--solver", solver, "--seconds", str(task["seconds"]),
                    "--script", result[1], "--", *result[2:]]
        return result
    def arm(*values, **kwargs):
        result = normalize_deadline(original_arm(*values, **kwargs))
        result["execution_policy"] = EXECUTION_POLICY["version"]
        return result
    runners.command, runners.run_arm = command, arm
    campaign.run_task(args)


def live_jobs(prefix):
    result = run(["oarstat", "-f", "-J", "--sql", f"job_name LIKE '{prefix}%'"], capture_output=True)
    output = result.stdout.strip()
    diagnostic = (getattr(result, "stderr", "") or "").strip()
    # A no-match OAR query can succeed without printing a JSON document.
    # Keep command failures and stderr-only responses distinct from that case.
    if not output:
        if diagnostic:
            raise ValueError(f"oarstat returned no JSON: stderr={diagnostic[:500]!r}")
        return []
    try:
        value = json.loads(output)
    except json.JSONDecodeError as error:
        raise ValueError(f"oarstat returned invalid JSON: stdout={output[:500]!r}; "
                         f"stderr={diagnostic[:500]!r}") from error
    rows = [row for row in dictionaries(value) if "state" in row and
            any(key in row for key in ("job_id", "id", "name", "job_name"))]
    if value and not rows:
        raise ValueError("unexpected oarstat JSON; refusing to submit another pool")
    return [row for row in rows if row["state"] not in TERMINAL]


def command_for(settings, plan, tasks, number):
    directory = Path(tasks).parent
    name = f"{settings['prefix']}{number}"
    command = ["oarsub", "-t", "idempotent", "-t", "besteffort", "-n", name,
               "-l", f"{settings['resource']},walltime={settings['walltime']}", "-p", settings["property"],
               "-O", str(directory / f"{number}.%jobid%.stdout"),
               "-E", str(directory / f"{number}.%jobid%.stderr")]
    if settings.get("queue"):
        command += ["-q", settings["queue"]]
    payload = [settings["python"], settings["driver"], "--worker", "--repo", settings["root"],
               "--out", settings["out"], "--plan", str(plan), "--tasks", str(tasks)]
    return [*command, shlex.join(payload)]


def category_profiles(config, out):
    profiles = {}
    for track in sorted({dataset["track"] for dataset in config["datasets"]}):
        selected = json.loads(json.dumps(config))
        selected["name"] = f"{config['name']}-{track}"
        selected["datasets"] = [d for d in selected["datasets"] if d["track"] == track]
        target = out / "config" / f"{track}.json"
        write(target, selected)
        profiles[track] = {"config": str(target)}
    return profiles


def register_hexaly_profiles(settings):
    # Keep already frozen legacy profiles usable on resume.
    if "hexaly" in settings["profiles"]:
        return
    for track in ("cvrp", "jssp", "rcpsp"):
        settings["profiles"].setdefault(f"hexaly-{track}", {
            "config": str(Path(settings["out"]) / "hexaly-reference" / track / "config.json"),
            "reference_track": track,
        })


def selected_profiles(settings):
    selected = settings.get("selected_profiles")
    if selected is None:
        return list(settings["profiles"])
    unknown = set(selected) - set(settings["profiles"])
    if not selected or unknown:
        raise ValueError("unknown or empty profile selection; available profiles: " +
                         ", ".join(sorted(settings["profiles"])))
    return list(dict.fromkeys(selected))


def wave_allocations(choices, capacity, wave):
    """Share one global job limit across several ready sections."""
    offset = wave % len(choices)
    selected = (choices[offset:] + choices[:offset])[:capacity]
    allocations = [1] * len(selected)
    while sum(allocations) < capacity:
        candidates = [i for i, (_, (_, _, _, pending)) in enumerate(selected)
                      if allocations[i] < len(pending)]
        if not candidates:
            break
        def weight(index):
            _, (_, _, plan, pending) = selected[index]
            work = sum(plan["tasks"][i]["seconds"] * len(plan["tasks"][i]["solvers"]) for i in pending)
            return work / allocations[index]
        allocations[max(candidates, key=weight)] += 1
    return [(profile, candidate, count) for (profile, candidate), count in zip(selected, allocations)]


def initial_settings(args, root, out):
    campaign, _report = modules(root)
    config_path = (args.config or root / "bench/cluster/runs/corpus-ready/config.json").absolute()
    python = str(root / ".venv/bin/python")
    if not config_path.is_file():
        if args.config:
            raise ValueError(f"missing --config: {config_path}")
        if CORPUS_TOOL is None:
            tool = root / "bench/cluster/lib/corpus.py"
        else:
            tool = out / "corpus.py"
            tool.write_bytes(zlib.decompress(base64.b64decode(CORPUS_TOOL)))
        run([python, tool, "--repo", root], cwd=root)
    base = read(config_path)
    campaign.validate_config(base)
    if any(d["track"] == "sat" for d in base["datasets"]):
        raise ValueError("SAT is deferred; select a configuration without SAT")
    receipt = read(root / "bench/cluster/runs/build.json")
    if campaign.source_identity()["sha256"] != receipt["source_sha256"]:
        raise ValueError("the validated build does not match the checkout")
    if not all("lp-relaxation" in receipt.get("features", {}).get(key, []) for key in ("binaries", "python")):
        raise ValueError("the build receipt does not confirm LP for both binaries and Python")
    verify_files(receipt["artifacts"])
    property_value = args.property or cpu_property(json.loads(run(["oarnodes", "-J"], capture_output=True).stdout))
    java, files = shared_java(root, args.java)
    manifest = Path(java).parents[1].with_suffix(".json")
    driver = out / "driver.py"
    driver.write_bytes(Path(__file__).read_bytes())
    files.update({str(driver): sha(driver), str(config_path): sha(config_path)})
    tracks = {dataset["track"] for dataset in base["datasets"]}
    needs_cosoco = any("cosoco" in base["solvers"].get(track, []) for track in tracks)
    if needs_cosoco:
        binary = campaign.install_cosoco.verify(root / f"bench/cluster/runs/tools/cosoco-{campaign.install_cosoco.VERSION}")
        files.update(campaign.cosoco_artifacts(binary))
    lkh = args.lkh or os.environ.get("LKH_BINARY") or root / "bench/cluster/runs/tools/solvers/LKH-3.0.13/LKH"
    hgs = args.hgs or os.environ.get("HGS_BINARY") or root / "bench/cluster/runs/tools/solvers/HGS-CVRP/build/hgs"
    if not Path(hgs).is_file() or not os.access(hgs, os.X_OK):
        raise ValueError(f"missing HGS baseline: {hgs}")
    lkh = str(Path(lkh).absolute()) if Path(lkh).is_file() and os.access(lkh, os.X_OK) else None
    main = json.loads(json.dumps(base))
    pending = []
    if lkh is None:
        main["datasets"] = [d for d in base["datasets"] if d["track"] != "cvrptw"]
        pending.append("VRPTW: LKH missing; all 356 instances and its comparator remain pending")
    profiles = category_profiles(main, out)
    for profile in profiles.values():
        if Path(profile["config"]).is_file():
            files[profile["config"]] = sha(profile["config"])
    settings = {"root": str(root), "out": str(out), "python": python, "driver": str(driver), "files": files,
                "java": java, "java_manifest": str(manifest), "property": property_value,
                "workers": args.workers, "jobs": args.jobs, "resource": args.resource, "walltime": args.walltime,
                "queue": args.queue, "hgs": str(Path(hgs).absolute()), "lkh": lkh,
                "profiles": profiles, "pending": pending, "prefix": f"qayd-{uuid.uuid4().hex[:10]}-",
                "source_sha256": receipt["source_sha256"], "base_config": str(config_path)}
    settings["profile_layout"] = "categories/v1"
    if args.hexaly:
        register_hexaly_profiles(settings)
    settings["execution_policy"] = EXECUTION_POLICY
    if any(dataset["track"] == "minizinc" for dataset in base["datasets"]):
        settings["minizinc"] = getattr(args, "minizinc", None) or find_minizinc(root)
    write(out / "settings.json", settings)
    return settings


def add_vrptw(settings, value):
    if not value or settings.get("lkh"):
        return
    path = Path(value).absolute()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"LKH is not executable: {path}")
    config = read(settings["base_config"])
    config["datasets"] = [d for d in config["datasets"] if d["track"] == "cvrptw"]
    name = "cvrptw" if settings.get("profile_layout") == "categories/v1" else "vrptw"
    target = Path(settings["out"]) / "config" / f"{name}.json"
    write(target, config)
    settings["profiles"][name] = {"config": str(target)}
    settings["files"][str(target)] = sha(target)
    settings["lkh"] = str(path)
    settings["pending"] = []
    write(Path(settings["out"]) / "settings.json", settings)


def prepare_hexaly(settings, profile="hexaly"):
    config = Path(settings["profiles"][profile]["config"])
    reference = config.parent
    if not config.exists():
        command = [settings["python"], Path(settings["root"]) / "bench/cluster/lib/hexaly.py",
                   "--out", reference]
        if settings["profiles"][profile].get("reference_track"):
            command += ["--track", settings["profiles"][profile]["reference_track"]]
        run(command, cwd=settings["root"])
    snapshots = reference / "sources.json"
    if not snapshots.exists():
        sources = {}
        for track, url in read(config)["published_reference"]["pages"].items():
            path = reference / (track + ".html")
            if not path.exists():
                temporary = path.with_suffix(".html.part")
                run(["curl", "--fail", "--location", "--retry", "1", "--connect-timeout", "10",
                     "--max-time", "45", "--output", temporary, url])
                temporary.replace(path)
            sources[track] = {"url": url, "sha256": sha(path)}
        write(snapshots, sources)
    verify_files({str(reference / (track + ".html")): value["sha256"]
                  for track, value in read(snapshots).items()})
    settings["files"].update({str(p): sha(p) for p in reference.iterdir() if p.is_file() and p.suffix != ".part"})
    write(Path(settings["out"]) / "settings.json", settings)


def freeze_deadline_plan(settings, profile, stage, directory):
    root = Path(settings["root"])
    campaign, _report = modules(root)
    template = Path(settings["out"]) / "templates" / f"{profile}-{stage}-{uuid.uuid4().hex[:8]}"
    source_stage = "gate" if stage == "full" else stage
    command = [settings["python"], root / "bench/cluster/lib/campaign.py", "prepare", "--stage", source_stage,
               "--config", settings["profiles"][profile]["config"], "--java", settings["java"],
               "--hgs", settings["hgs"], "--out", template]
    if settings.get("lkh"):
        command += ["--lkh", settings["lkh"]]
    if settings.get("minizinc"):
        command += ["--minizinc", settings["minizinc"]]
    run(command, cwd=root)
    base = campaign.load_plan(template / "plan.json")
    plan = json.loads(json.dumps(base))
    plan.pop("sha256")
    driver_sha = sha(settings["driver"])
    plan["artifacts"][settings["driver"]] = driver_sha
    plan["environment"] = campaign.digest({"base_environment": base["environment"],
                                           "driver_sha256": driver_sha, "execution_policy": EXECUTION_POLICY})
    plan["execution_policy"] = EXECUTION_POLICY
    if AUDITOR_SOURCE is not None:
        policy_path = str(Path(__file__).resolve())
        plan["continuation"] = {"failure_handling": "record-errors/v1", "path": policy_path,
                                "sha256": sha(policy_path)}
        plan["artifacts"][policy_path] = plan["continuation"]["sha256"]
    plan["stage"] = stage
    if stage == "full":
        gate_dir = Path(settings["out"]) / profile / "gate"
        gate = read(gate_dir / "qualification.json")
        if (not gate.get("complete", gate.get("ready")) or gate.get("blocking_errors")
                or gate.get("stage") != "gate" or gate.get("environment") != plan["environment"]
                or gate.get("plan_sha256") != campaign.load_plan(gate_dir / "plan.json")["sha256"]):
            raise ValueError("full campaign requires a complete v2 gate for the same source, runtime and corpus")
        items, assets, missing = campaign.discover(plan["config"])
        if missing or assets != plan["assets"]:
            raise ValueError("corpus changed between template and full preparation")
        plan["tasks"] = campaign.tasks_for(items, plan["config"], "full")
    else:
        for task in plan["tasks"]:
            task["seconds"] = (EXECUTION_POLICY["smoke_seconds"] if stage == "smoke" else
                               EXECUTION_POLICY["large_jssp_gate_seconds"] if
                               (profile == "hexaly" or settings["profiles"][profile].get("reference_track")) and
                               task["item"]["track"] == "jssp" else EXECUTION_POLICY["gate_seconds"])
            task.pop("key")
            task["key"] = campaign.digest(task)
    plan["derivation"] = {"template": str(template / "plan.json"), "template_sha256": base["sha256"],
                          "note": "Original template and prior results preserved; full literature budgets unchanged."}
    plan["overview"].update(paired_tasks=len(plan["tasks"]),
                            solver_runs=sum(len(task["solvers"]) for task in plan["tasks"]),
                            maximum_solver_hours=round(sum(task["seconds"] * len(task["solvers"])
                                                           for task in plan["tasks"]) / 3600, 2))
    plan["sha256"] = campaign.digest(plan)
    directory.mkdir(parents=True, exist_ok=False)
    campaign.write(directory / "plan.json", plan)
    print(f"Frozen {stage} v2 plan: {directory / 'plan.json'}", flush=True)


def next_stage(settings, profile):
    root, out = Path(settings["root"]), Path(settings["out"])
    campaign, report = modules(root)
    reference = profile == "hexaly" or settings["profiles"][profile].get("reference_track")
    if reference:
        prepare_hexaly(settings, profile)
    for stage in ("smoke", "gate", "full"):
        directory = out / profile / stage
        path = directory / "plan.json"
        if not path.exists():
            if settings.get("execution_policy"):
                freeze_deadline_plan(settings, profile, stage, directory)
            else:
                prepare_legacy_plan(settings, profile, stage, directory)
        plan = campaign.load_plan(path)
        campaign.verify_snapshot(plan)
        result = report.audit(plan, directory)
        campaign.write(directory / "report.json", result)
        campaign.write(directory / "qualification.json", {key: result[key] for key in (
            "ready", "complete", "blocking_errors", "environment", "plan_sha256", "stage", "expected_tasks", "completed_tasks")})
        (directory / "report.md").write_text(report.markdown(result))
        if result["blocking_errors"]:
            raise ValueError(f"{profile}/{stage}: incompatible results; see {directory / 'report.md'}")
        if not result["complete"]:
            return stage, path, plan, result["missing_tasks"]
        if reference and stage == "full":
            run([settings["python"], root / "bench/cluster/lib/hexaly_report.py", "--plan", path], cwd=root)
    return None


def prepare_legacy_plan(settings, profile, stage, directory):
    root, out = Path(settings["root"]), Path(settings["out"])
    command = [settings["python"], root / "bench/cluster/lib/campaign.py", "prepare", "--stage", stage,
               "--config", settings["profiles"][profile]["config"], "--java", settings["java"],
               "--hgs", settings["hgs"], "--out", directory]
    if settings.get("lkh"):
        command += ["--lkh", settings["lkh"]]
    if stage == "full":
        command += ["--gate", out / profile / "gate/qualification.json"]
    run(command, cwd=root)


def control(args, settings):
    out = args.out
    state_path = out / "state.json"
    state = read(state_path) if state_path.exists() else {"waves": 0, "empty_checks": 0}
    state.pop("error", None)
    while True:
        verify_files(settings["files"])
        jobs = live_jobs(settings["prefix"])
        if jobs:
            state.update(status="running", active_jobs=jobs, empty_checks=0)
            write(state_path, state)
            if not args.execute:
                print(f"{len(jobs)} existing OAR jobs; no new submission.")
                return
            time.sleep(args.poll_seconds)
            continue
        if state.get("inflight") and state.get("empty_checks", 0) < 2:
            state["empty_checks"] = state.get("empty_checks", 0) + 1
            write(state_path, state)
            if args.execute:
                time.sleep(args.poll_seconds)
                continue
            print("Waiting for OAR resubmission reconciliation; no new submission.")
            return
        active = selected_profiles(settings)
        choices, profile_errors = [], dict(state.get("profile_errors", {}))
        for profile in active:
            profile_errors.pop(profile, None)
            try:
                candidate = next_stage(settings, profile)
            except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
                profile_errors[profile] = str(error)
                continue
            if candidate:
                choices.append((profile, candidate))
        state["profile_errors"] = profile_errors
        write(state_path, state)
        if not choices:
            selected_errors = {name: profile_errors[name] for name in active if name in profile_errors}
            if selected_errors:
                raise ValueError(f"blocked profiles: {selected_errors}")
            partial = set(active) != set(settings["profiles"])
            status = "complete-selected" if partial else "complete" if not settings["pending"] else "pending-baseline"
            state.update(status=status, pending=settings["pending"], active_jobs=[])
            write(state_path, state)
            print(json.dumps(state, indent=2), flush=True)
            return
        priority = {"smoke": 0, "gate": 1, "full": 2}
        minimum = min(priority[candidate[0]] for _profile, candidate in choices)
        choices = [(p, c) for p, c in choices if priority[c[0]] == minimum]
        campaign, _report = modules(Path(settings["root"]))
        commands = []
        allocations = wave_allocations(choices, settings["jobs"], state["waves"])
        for profile, (stage, plan_path, plan, pending), count in allocations:
            completed = len(plan["tasks"]) - len(pending)
            checkpoints = state.setdefault("progress_by_stage", {})
            previous = checkpoints.get(f"{profile}/{stage}")
            stalled = previous["stalled"] + 1 if previous and previous["completed"] == completed else 0
            checkpoints[f"{profile}/{stage}"] = {"completed": completed, "stalled": stalled}
            if stalled >= 3:
                write(state_path, state)
                raise ValueError(f"three pools made no task progress; inspect {out / 'waves'} before retrying")
            groups = campaign.partition_tasks([plan["tasks"][i] for i in pending], count)
            for indexes in groups:
                group = len(commands)
                taskfile = out / "waves" / f"{state['waves']:05d}" / f"group-{group}.txt"
                taskfile.parent.mkdir(parents=True, exist_ok=True)
                taskfile.write_text("".join(f"{pending[i]}\n" for i in indexes))
                commands.append(command_for(settings, plan_path, taskfile, f"{state['waves']}-{group}"))
            print(f"{profile}/{stage}: {len(pending)} remaining paired tasks; {len(groups)} OAR jobs in this pool.", flush=True)
        active_profiles = [name for name, _, _ in allocations]
        stage = allocations[0][1][0]
        state.update(status="submitting", profile=active_profiles[0] if len(active_profiles) == 1 else "multiple",
                     active_profiles=active_profiles, stage=stage, inflight=True, empty_checks=0, active_jobs=[])
        write(state_path, state)
        print(f"Pool total: at most {len(commands)}/{settings['jobs']} OAR jobs across all selected sections.", flush=True)
        if profile_errors:
            print(f"Profiles awaiting correction: {profile_errors}", flush=True)
        if settings["pending"]:
            print("Pending: " + "; ".join(settings["pending"]), flush=True)
        for command in commands:
            print(shlex.join(command), flush=True)
        if not args.execute:
            state["inflight"] = False
            write(state_path, state)
            return
        state["waves"] += 1
        write(state_path, state)
        for command in commands:
            run(command, cwd=settings["root"])
        time.sleep(args.poll_seconds)


def main():
    if sys.argv[1:2] == ["model"]:
        model_adapter(sys.argv[2:])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, default=Path("bench/cluster/runs/next-v2"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--resource", default="/nodes=1")
    parser.add_argument("--walltime", default="12:00:00")
    parser.add_argument("--property")
    parser.add_argument("--queue")
    parser.add_argument("--java", default="java")
    parser.add_argument("--minizinc")
    parser.add_argument("--hgs")
    parser.add_argument("--lkh")
    parser.add_argument("--hexaly", action="store_true", help="also run the published-table quality profile")
    parser.add_argument("--profile", action="append", help="run only this profile; repeat for several, or use 'all'")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--plan", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--tasks", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--task", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    args.out = (args.repo / args.out).absolute() if not args.out.is_absolute() else args.out
    os.chdir(args.repo)
    if args.status:
        print(json.dumps(read(args.out / "state.json"), indent=2))
        return
    if args.worker:
        worker(args)
        return
    if args.task is not None:
        run_task(args)
        return
    if args.jobs < 1 or args.poll_seconds < 5 or args.workers != "auto" and (not args.workers.isdigit() or int(args.workers) < 1):
        parser.error("positive jobs/workers and --poll-seconds >= 5 required")
    if "walltime" in args.resource or not re.fullmatch(r"\d+:[0-5]\d:[0-5]\d", args.walltime):
        parser.error("provide --resource without walltime, and --walltime HHH:MM:SS")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.detach:
        if not args.execute:
            parser.error("--detach requires --execute")
        command = [str(args.repo / ".venv/bin/python"), str(Path(__file__).absolute()),
                   *[v for v in sys.argv[1:] if v != "--detach"]]
        with (args.out / "controller.log").open("ab") as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                       cwd=args.repo, start_new_session=True)
        print(f"Controller PID {process.pid}; log: {args.out / 'controller.log'}")
        return
    with (args.out / "controller.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("A controller already owns this campaign; use --status.")
        if not (args.out / "state.json").exists():
            write(args.out / "state.json", {"waves": 0, "empty_checks": 0, "status": "preparing"})
        try:
            settings_path = args.out / "settings.json"
            settings = read(settings_path) if settings_path.exists() else initial_settings(args, args.repo, args.out)
            verify_files(settings["files"])
            if AUDITOR_SOURCE is not None:
                policy_path = str(Path(__file__).resolve())
                settings["files"][policy_path] = sha(policy_path)
                settings["failure_handling"] = {"version": "record-errors/v1", "controller": policy_path,
                                                "sha256": sha(policy_path)}
            if args.hexaly:
                register_hexaly_profiles(settings)
            for name in ("jobs", "workers", "resource", "walltime", "queue", "property"):
                flag = "--" + name.replace("_", "-")
                if any(value == flag or value.startswith(flag + "=") for value in sys.argv[1:]):
                    settings[name] = getattr(args, name)
            add_vrptw(settings, args.lkh)
            if args.profile is not None:
                settings["selected_profiles"] = None if args.profile == ["all"] else args.profile
            selected_profiles(settings)
            write(settings_path, settings)
            control(args, settings)
        except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
            state = read(args.out / "state.json")
            state.update(status="failed", error=str(error))
            write(args.out / "state.json", state)
            raise


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        raise SystemExit(f"Campaign continuation failed: {error}") from error
