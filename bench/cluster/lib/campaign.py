#!/usr/bin/env python3
"""Prepare, execute and submit a small pool of paired benchmark workers."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "bench"))
from cluster.lib.adapters.records import sha256_file
from cluster.lib.adapters.instances import opener
from cluster.lib.adapters.xcsp import ManifestError, detect_objective_sense, family_for
from cluster.lib import minizinc

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config/campaign.v1.json"
MODEL_TRACKS = {"cvrp", "cvrptw", "jssp", "rcpsp"}


def validate_config(config: dict) -> None:
    if config.get("schema") != "qayd.cluster.config/v1":
        raise ValueError("unsupported config schema")
    if not config.get("datasets"):
        raise ValueError("configuration must declare datasets")
    allowed = {"xcsp-csp": {"qayd", "ace", "ace-rr", "choco"},
               "xcsp-cop": {"qayd", "ace", "ace-rr", "choco"},
               "xcsp-fastcop": {"qayd", "ace", "ace-rr", "choco"},
               "sat": {"qayd-sat", "cadical"}, "pb": {"qayd-pb", "sat4j"},
               "minizinc": {"qayd-fzn", "gecode", "chuffed"},
               "cvrp": {"qayd-api", "qayd-native", "hgs", "lkh"},
               "cvrptw": {"qayd-api", "qayd-native", "lkh"},
               "jssp": {"qayd-api", "qayd-native", "ortools-cp-sat"},
               "rcpsp": {"qayd-api", "qayd-native", "ortools-cp-sat"}}
    for dataset in config["datasets"]:
        track = dataset["track"]
        solvers = config.get("solvers", {}).get(track, [])
        if track not in allowed or not solvers or not set(solvers) <= allowed[track] or len(solvers) != len(set(solvers)):
            raise ValueError(f"unknown, missing or duplicate solver for track: {track}")
        protocol = config["protocols"][track]
        limit = protocol.get("seconds", protocol.get("seconds_per_customer", 0))
        if not isinstance(limit, (int, float)) or not math.isfinite(limit) or limit <= 0:
            raise ValueError(f"invalid budget for {track}")
        seeds = protocol.get("seeds", config["seeds"])
        if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
            raise ValueError(f"invalid or duplicate seeds for {track}")
    for key in ("memory_mb", "checker_memory_mb", "checker_seconds"):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    grace = config.get("grace_seconds")
    if not isinstance(grace, (int, float)) or not math.isfinite(grace) or grace < 0:
        raise ValueError("grace_seconds must be finite and non-negative")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def positive(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def checked(argv: list[str], **kwargs) -> str:
    return subprocess.run(argv, text=True, capture_output=True, check=True, **kwargs).stdout.strip()


def source_files() -> list[Path]:
    files = [ROOT / name for name in ("Cargo.toml", "Cargo.lock", "pyproject.toml")]
    observers = {"bench/cluster/status.py", "bench/cluster/collect.py"}
    excluded = {"results", "instances", "runs", "__pycache__", "target", ".git"}
    for name in ("src", "frontends", "examples/python", "bench/cluster", "tools", "integrations/amthal"):
        for directory, children, names in os.walk(ROOT / name):
            children[:] = [child for child in children if child not in excluded]
            for filename in names:
                path = Path(directory) / filename
                if path.relative_to(ROOT).as_posix() in observers:
                    continue
                if path.suffix in {".rs", ".py", ".pyi", ".toml", ".lock", ".sh", ".oar", ".json", ".mzn", ".msc"}:
                    files.append(path)
    return sorted(set(path for path in files if path.is_file()))


def source_identity() -> dict:
    files = {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in source_files()}
    try:
        if Path(checked(["git", "rev-parse", "--show-toplevel"], cwd=ROOT)).resolve() != ROOT:
            raise OSError("not the source repository")
        revision = checked(["git", "rev-parse", "HEAD"], cwd=ROOT)
        dirty = bool(checked(["git", "status", "--porcelain"], cwd=ROOT))
    except (OSError, subprocess.SubprocessError):
        metadata = ROOT / "CAMPAIGN_SOURCE.json"
        snapshot = read(metadata) if metadata.is_file() else {}
        revision, dirty = snapshot.get("base_revision"), snapshot.get("working_tree_dirty")
    return {"files": files, "sha256": digest(files), "revision": revision, "dirty": dirty}


def executable(value: str) -> str:
    path = shutil.which(value)
    if path is None:
        raise ValueError(f"missing executable: {value}")
    return str(Path(path).resolve())


def package_files(name: str) -> dict[str, str]:
    distribution = importlib.metadata.distribution(name)
    return {str(Path(distribution.locate_file(path)).resolve()): sha256_file(Path(distribution.locate_file(path)))
            for path in distribution.files or []
            if str(path).endswith((".py", ".so", ".dylib", ".dll", ".json"))
            and Path(distribution.locate_file(path)).is_file()}


def runtime_identity(config: dict, items: list[dict], directory: Path, args) -> dict:
    solvers = {solver for item in items for solver in config["solvers"][item["track"]]}
    runtime = {"python": str(Path(sys.executable).absolute())}
    paths = {runtime["python"]}
    for solver in ("qayd", "qayd-fzn", "qayd-sat", "qayd-pb"):
        if solver in solvers:
            runtime[solver] = executable(str(ROOT / "target/release" / solver))
            paths.add(runtime[solver])
    if any(solver in solvers for solver in ("qayd", "ace", "ace-rr", "choco", "sat4j")):
        runtime["java"] = executable(args.java)
        paths.add(runtime["java"])
    jars = {"ace": "ACE-2.5.jar", "choco": "choco-xcsp25.jar",
            "sat4j-pb": "s4j-pb.jar", "sat4j-core": "s4j-core.jar"}
    for key, name in jars.items():
        needed = (key == "ace" and any(item["track"].startswith("xcsp") for item in items)
                  or key == "choco" and "choco" in solvers
                  or key.startswith("sat4j") and "sat4j" in solvers)
        if needed:
            path = ROOT / "bench/cluster/runs/tools/solvers" / name
            if not path.is_file():
                raise ValueError(f"missing artifact: {path}")
            runtime[key] = str(path.resolve())
            paths.add(runtime[key])
    if "cadical" in solvers:
        runtime["cadical"] = executable(args.cadical)
        paths.add(runtime["cadical"])
    if "qayd-fzn" in solvers or "gecode" in solvers or "chuffed" in solvers:
        runtime["minizinc"] = executable(args.minizinc)
        paths.add(runtime["minizinc"])
        # Explicit config prevents picking up an old globally installed Qayd.
        msc = directory / "qayd.msc"
        version = re.search(r'\[workspace.package\][\s\S]*?version\s*=\s*"([^"]+)"',
                            (ROOT / "Cargo.toml").read_text()).group(1)
        write(msc, {"id": "org.qayd.cluster", "name": "qayd-cluster", "version": version,
                    "executable": runtime["qayd-fzn"],
                    "mznlib": str(ROOT / "frontends/flatzinc/minizinc/mznlib"),
                    "stdFlags": ["-t"], "supportsMzn": False,
                    "supportsFzn": True, "needsSolns2Out": True})
        runtime["qayd_msc"] = str(msc)
        paths.add(str(msc))
        for solver_id in ("org.gecode.gecode", "org.chuffed.chuffed"):
            details = json.loads(checked([runtime["minizinc"], "--solver-json", solver_id], timeout=30))
            runtime[solver_id] = details
            resolved = details.get("extraInfo", details)
            solver_path = Path(resolved["executable"])
            if not solver_path.is_absolute():
                solver_path = Path(executable(str(solver_path)))
            paths.add(str(solver_path.resolve()))
            library = Path(resolved["mznlib"])
            if not library.is_dir():
                raise ValueError(f"MiniZinc did not resolve its mznlib: {details}")
            paths.update(str(p.resolve()) for p in library.rglob("*") if p.is_file())
        # Include the compiler standard library as well as solver libraries.
        directories = json.loads(checked([runtime["minizinc"], "--config-dirs"], timeout=30))
        standard = Path(directories["mznStdlibDir"])
        paths.update(str(p.resolve()) for p in standard.rglob("*") if p.is_file())
    if solvers & {"qayd-api", "qayd-native"}:
        spec = importlib.util.find_spec("qayd._core")
        if spec is None or not spec.origin:
            raise ValueError("missing qayd._core; build with this Python interpreter")
        runtime["qayd_extension"] = spec.origin
        paths.add(spec.origin)
    for key in ("hgs", "lkh"):
        if key in solvers:
            value = getattr(args, key)
            if not value:
                raise ValueError(f"--{key} must name the baseline executable")
            runtime[key] = executable(value)
            paths.add(runtime[key])
    if "ortools-cp-sat" in solvers:
        runtime["ortools_version"] = importlib.metadata.version("ortools")
        paths.update(package_files("ortools"))
    if "hexaly" in solvers:
        raise ValueError("Hexaly must first be configured and qualified separately; default campaign uses open baselines")
    return {"runtime": runtime, "artifacts": {path: sha256_file(Path(path)) for path in sorted(paths)}}


def objective_sense(path: Path, track: str) -> str:
    if track == "xcsp-csp":
        return "satisfy"
    if track.startswith("xcsp"):
        return detect_objective_sense(path)
    if track == "sat":
        return "satisfy"
    if track == "pb":
        with opener(str(path))() as stream:
            for raw in stream:
                line = raw.decode("utf-8", "replace").strip()
                if not line or line.startswith("*"):
                    continue
                return "min" if line.startswith("min:") else "max" if line.startswith("max:") else "satisfy"
    return "min"


def discover(config: dict) -> tuple[list[dict], dict, list[str]]:
    items, assets, missing = [], {}, []
    seen = set()
    for dataset in config["datasets"]:
        track, family = dataset["track"], dataset["family"]
        found = []
        if "catalog" in dataset:
            catalog_path = ROOT / dataset["catalog"]
            if catalog_path.is_file():
                catalog = read(catalog_path)
                for entry in catalog["instances"]:
                    found.append({**entry, "family": family + "/" + entry["family"]})
                for path, expected in catalog["assets"].items():
                    candidate = ROOT / path
                    if not candidate.is_file() or sha256_file(candidate) != expected:
                        raise ValueError(f"catalog is stale: {path}")
                    assets[path] = expected
                assets[dataset["catalog"]] = sha256_file(catalog_path)
        else:
            for path in sorted(ROOT.glob(dataset["glob"])):
                if not path.is_file():
                    continue
                relative = path.relative_to(ROOT).as_posix()
                # Derive group labels from input structure, before running Qayd.
                subfamily = path.name.split("-", 1)[0] if track.startswith("xcsp") else path.parent.name
                if track in {"xcsp-cop", "xcsp-fastcop"}:
                    try:
                        subfamily = family_for(path.name)[0]
                    except ManifestError:
                        pass
                found.append({"path": relative, "family": family + "/" + subfamily,
                              "sense": objective_sense(path, track), "data": []})
        if not found:
            missing.append(family)
        if "expected_count" in dataset and len(found) != dataset["expected_count"]:
            missing.append(f"{family}: {len(found)}/{dataset['expected_count']} expected instances")
        if dataset.get("receipt"):
            receipt_path = ROOT / dataset["receipt"]
            expected = read(receipt_path).get(family) if receipt_path.is_file() else None
            observed = {entry["path"]: sha256_file(ROOT / entry["path"]) for entry in found}
            if not expected or expected != observed:
                missing.append(f"{family}: complete download receipt missing or stale")
        if dataset.get("manifest"):
            pinned = read(ROOT / dataset["manifest"])["expected_instances"]
            observed = {Path(entry["path"]).name: sha256_file(ROOT / entry["path"]) for entry in found}
            if observed != pinned:
                missing.append(f"{family}: pinned instance set differs")
            assets[dataset["manifest"]] = sha256_file(ROOT / dataset["manifest"])
        for item in found:
            path = item["path"]
            identity = (track, path, tuple(item.get("data", [])))
            if identity in seen:
                raise ValueError(f"overlapping datasets: {identity}")
            seen.add(identity)
            assets.setdefault(path, sha256_file(ROOT / path))
            if track == "cvrp":
                match = re.search(r"DIMENSION\s*:?\s*(\d+)", (ROOT / path).read_text())
                if match is None:
                    raise ValueError(f"no CVRP dimension in {path}")
                item["customers"] = int(match.group(1)) - 1
            reference = (ROOT / path).with_suffix(".sol")
            if track in MODEL_TRACKS and reference.is_file():
                relative_reference = reference.relative_to(ROOT).as_posix()
                assets[relative_reference] = sha256_file(reference)
                item.setdefault("dependencies", []).append(relative_reference)
                if track == "cvrp":
                    for line in reference.read_text().splitlines():
                        fields = line.replace(":", " ").split()
                        if len(fields) == 2 and fields[0].lower() in {"cost", "objective", "value"}:
                            item["reference_bks"] = float(fields[1])
            item.update(track=track, sha256=assets[path])
            item["id"] = digest(identity)[:24]
            items.append(item)
    return items, assets, missing


def smoke_config(config: dict) -> dict:
    fixtures = ROOT / "bench/cluster/fixtures"
    active_tracks = {dataset["track"] for dataset in config["datasets"]}
    datasets = [
        ("xcsp-csp", "bench/cluster/fixtures/xcsp_minimize.xml"),
        ("xcsp-cop", "bench/cluster/fixtures/xcsp_minimize.xml"),
        ("xcsp-fastcop", "bench/cluster/fixtures/xcsp_minimize.xml"),
        ("sat", "bench/cluster/fixtures/sat_unique.cnf"),
        ("sat", "bench/cluster/fixtures/sat_unsat.cnf"),
        ("pb", "bench/cluster/fixtures/pb_minimize.opb"),
        ("pb", "bench/cluster/fixtures/unsat.opb"),
        ("xcsp-cop", "bench/cluster/fixtures/maximize.xml"),
        ("xcsp-csp", "bench/cluster/fixtures/unsat.xml"),
        ("cvrp", "examples/instances/routing/tiny-cvrp.vrp"),
        ("cvrptw", "examples/instances/routing/tiny-vrptw.txt"),
        ("jssp", "examples/instances/scheduling/tiny-jssp.txt"),
        ("rcpsp", "examples/instances/scheduling/tiny-rcpsp.sm"),
    ]
    # This decision fixture is independent of the optimization smoke fixture.
    datasets[0] = ("xcsp-csp", "bench/cluster/fixtures/decision.xml")
    smoke_datasets = [
        {"track": track, "family": "smoke-" + track, "glob": path}
        for track, path in datasets if track in active_tracks
    ]
    if "minizinc" in active_tracks:
        smoke_datasets.append({"track": "minizinc", "family": "smoke-mzn", "catalog":
                               str((fixtures / "catalog.json").relative_to(ROOT))})
    return {**config, "datasets": smoke_datasets}


def budget(item: dict, config: dict, stage: str) -> int:
    if stage == "smoke":
        return 2
    if stage == "gate":
        return 10
    protocol = config["protocols"][item["track"]]
    if "seconds_per_customer" in protocol:
        if "customers" in item:
            return max(1, math.ceil(item["customers"] * protocol["seconds_per_customer"]))
        with (ROOT / item["path"]).open() as stream:
            for line in stream:
                match = re.match(r"\s*DIMENSION\s*:?\s*(\d+)", line)
                if match:
                    return max(1, math.ceil((int(match.group(1)) - 1) * protocol["seconds_per_customer"]))
        raise ValueError(f"no CVRP dimension: {item['path']}")
    return protocol["seconds"]


def tasks_for(items: list[dict], config: dict, stage: str) -> list[dict]:
    if stage == "gate":
        grouped = defaultdict(list)
        for item in items:
            grouped[(item["track"], item["family"])].append(item)
        # Stable hash sampling avoids an alphabetic easy-instance prefix.
        items = [min(group, key=lambda item: digest(item["id"])) for group in grouped.values()]
    tasks = []
    for item in items:
        seeds = [None] if item["track"] in {"pb", "minizinc"} else (
            [0] if stage != "full" else config["protocols"][item["track"]].get("seeds", config["seeds"]))
        for seed in seeds:
            task = {"item": item, "seconds": budget(item, config, stage), "seed": seed,
                    "solvers": config["solvers"][item["track"]]}
            task["key"] = digest(task)
            tasks.append(task)
    return sorted(tasks, key=lambda task: task["key"])


def load_plan(path: Path) -> dict:
    plan = read(path)
    expected = plan.pop("sha256")
    if digest(plan) != expected:
        raise ValueError("plan hash mismatch")
    plan["sha256"] = expected
    if plan["root"] != str(ROOT):
        raise ValueError("plan belongs to another checkout; prepare it on the execution host")
    return plan


def verify_snapshot(plan: dict) -> None:
    if source_identity()["sha256"] != plan["source"]["sha256"]:
        raise ValueError("source changed after preparation; rebuild and use a new plan")
    for raw, expected in plan["artifacts"].items():
        if not Path(raw).is_file() or sha256_file(Path(raw)) != expected:
            raise ValueError(f"runtime artifact changed: {raw}")


def lp_backend_identity():
    """Record the authorized backend sources used by the private workspace."""
    backend = ROOT.parent / "amthal"
    if not (ROOT / "integrations/amthal/Cargo.toml").is_file() or not (backend / "Cargo.toml").is_file():
        raise ValueError("LP requires integrations/amthal/Cargo.toml and an authorized sibling ../amthal checkout")
    files = {}
    for directory, children, names in os.walk(backend):
        children[:] = [child for child in children if child not in {"target", ".git", "__pycache__"}]
        for name in names:
            path = Path(directory) / name
            if path.suffix in {".rs", ".toml", ".lock"}:
                files[path.relative_to(backend).as_posix()] = sha256_file(path)
    return {"root": str(backend), "files": files, "sha256": digest(files)}


def build(args) -> None:
    source = source_identity()
    backend = lp_backend_identity() if args.lp_feature else None
    manifest = ROOT / ("integrations/amthal/Cargo.toml" if args.lp_feature else "Cargo.toml")
    target_dir = Path(getattr(args, "target_dir", None) or os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))
    if not target_dir.is_absolute():
        target_dir = ROOT / target_dir
    target_dir = target_dir.resolve()
    binary_features = ["lp-relaxation"] if args.lp_feature else []
    python_features = ["python", *binary_features]
    # The private manifest enables LP by default. The public workspace has no LP feature.
    binary_command = ["cargo", "build", "--manifest-path", str(manifest), "--locked", "--release",
                      "--workspace", "--target-dir", str(target_dir)]
    python_command = [sys.executable, "-m", "maturin", "develop", "--profile", "pyext", "--locked",
                      "--manifest-path", str(manifest), "--target-dir", str(target_dir), "--features", "python"]
    subprocess.run(binary_command, cwd=ROOT, check=True)
    env = {**os.environ, "VIRTUAL_ENV": sys.prefix}
    subprocess.run(python_command, cwd=ROOT, env=env, check=True)
    if source_identity()["sha256"] != source["sha256"]:
        raise ValueError("source changed during build")
    if backend is not None and lp_backend_identity() != backend:
        raise ValueError("Amthal sources changed during build")
    spec = importlib.util.find_spec("qayd._core")
    artifacts = [ROOT / "target/release" / name for name in ("qayd", "qayd-fzn", "qayd-sat", "qayd-pb")]
    # Only the runnable binaries must survive node-local scratch cleanup.
    for artifact in artifacts:
        compiled = target_dir / "release" / artifact.name
        if compiled.resolve() != artifact.resolve():
            artifact.parent.mkdir(parents=True, exist_ok=True)
            temporary = artifact.with_name(artifact.name + f".{os.getpid()}.tmp")
            shutil.copy2(compiled, temporary)
            os.replace(temporary, artifact)
    artifacts.append(Path(spec.origin))
    write(args.receipt, {"source_sha256": source["sha256"], "source": source,
                         "workspace_manifest": str(manifest), "lp_backend": backend,
                         "features": {"binaries": binary_features, "python": python_features},
                         "commands": {"binaries": binary_command, "python": python_command},
                         "python": str(Path(sys.executable).absolute()),
                         "artifacts": {str(p): sha256_file(p) for p in artifacts},
                         "rustc": checked(["rustc", "-Vv"]),
                         "build_environment": {name: os.environ.get(name) for name in (
                             "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "RUSTC_WRAPPER", "CARGO_BUILD_TARGET",
                             "CARGO_TARGET_DIR", "CARGO_INCREMENTAL", "CARGO_PROFILE_DEV_DEBUG", "CARGO_PROFILE_TEST_DEBUG")},
                         "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})


def prepare(args) -> None:
    config = read(args.config)
    validate_config(config)
    if args.stage == "smoke":
        config = smoke_config(config)
    items, assets, missing = discover(config)
    tasks = tasks_for(items, config, args.stage)
    counts = Counter(item["track"] for item in items)
    estimate = sum(task["seconds"] * len(task["solvers"]) for task in tasks)
    overview = {"instances": dict(counts), "missing_datasets": missing, "paired_tasks": len(tasks),
                "inventory_complete": not missing,
                "estimate_scope": "downloaded instances only; lower bound when the corpus is incomplete; excludes checking and overhead",
                "solver_runs": sum(len(task["solvers"]) for task in tasks),
                "maximum_solver_hours": round(estimate / 3600, 2),
                "asset_bytes": sum((ROOT / path).stat().st_size for path in assets)}
    print(json.dumps(overview, indent=2))
    if args.inventory:
        if args.out:
            write(args.out / "inventory.json", overview)
        return
    if missing or not tasks:
        raise ValueError(f"incomplete corpus: {missing}; fetch it or explicitly edit the configuration")
    if args.out is None:
        raise ValueError("--out is required to freeze a plan")
    directory = args.out.resolve()
    if directory.exists():
        raise ValueError(f"refusing to replace existing campaign directory: {directory}")
    source = source_identity()
    receipt = read(args.receipt)
    if receipt["source_sha256"] != source["sha256"]:
        raise ValueError("build receipt does not match the current source")
    if receipt["python"] != str(Path(sys.executable).absolute()):
        raise ValueError("build receipt uses a different Python interpreter")
    for path, expected in receipt["artifacts"].items():
        if sha256_file(Path(path)) != expected:
            raise ValueError(f"binary no longer matches the build receipt: {path}")
    try:
        runtime = runtime_identity(config, items, directory, args)
    except (ValueError, OSError, KeyError, subprocess.SubprocessError):
        # Only our newly generated solver config can exist here at this point.
        shutil.rmtree(directory, ignore_errors=True)
        raise
    extension = runtime["runtime"].get("qayd_extension")
    if extension and extension not in receipt["artifacts"]:
        raise ValueError("imported Qayd extension differs from the build receipt")
    environment = digest({"source": source["sha256"], "artifacts": {
        path: value for path, value in runtime["artifacts"].items()
        if path != runtime["runtime"].get("qayd_msc")}, "config": config,
        "assets": assets, "memory_mb": args.memory_mb or config["memory_mb"], "threads": args.threads})
    if args.threads != 1 and any(item["track"] not in MODEL_TRACKS for item in items):
        raise ValueError("multi-thread runs require a model-only configuration")
    if args.stage == "full":
        if not args.gate:
            shutil.rmtree(directory, ignore_errors=True)
            raise ValueError("full campaign requires --gate <gate>/qualification.json")
        gate = read(args.gate)
        if gate.get("stage") != "gate" or not gate.get("ready") or gate.get("environment") != environment:
            shutil.rmtree(directory, ignore_errors=True)
            raise ValueError("gate is not complete and clean for this source, corpus and runtime")
    plan = {"schema": "qayd.cluster.plan/v1", "root": str(ROOT), "stage": args.stage,
            "source": source, **runtime, "assets": assets, "tasks": tasks,
            "environment": environment, "config": config, "build_receipt": receipt,
            "threads": args.threads, "memory_mb": args.memory_mb or config["memory_mb"],
            "checker_memory_mb": config["checker_memory_mb"], "checker_seconds": config["checker_seconds"],
            "grace_seconds": config["grace_seconds"], "overview": overview}
    plan["sha256"] = digest(plan)
    write(directory / "plan.json", plan)
    print(f"Frozen plan: {directory / 'plan.json'}")


def run_task(args) -> None:
    from cluster.lib.runners import STOP_EVENT, run_arm
    def stop(_signal, _frame):
        STOP_EVENT.set()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    path = args.plan.resolve()
    plan = load_plan(path)
    if args.task < 0 or args.task >= len(plan["tasks"]):
        raise ValueError("task index outside the frozen plan")
    task = plan["tasks"][args.task]
    directory = path.parent / "tasks" / f"{args.task:07d}"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        verify_snapshot(plan)
        item = task["item"]
        relevant = [item["path"], *item.get("data", []), *item.get("dependencies", [])]
        for relative in relevant:
            if sha256_file(ROOT / relative) != plan["assets"][relative]:
                raise ValueError(f"instance or model changed: {relative}")
        completed = directory / "result.json"
        if completed.exists():
            if read(completed)["plan_sha256"] != plan["sha256"]:
                raise ValueError("result belongs to another plan")
            return
        # Rotate arms across paired tasks to balance warm-cache and order effects.
        solvers = task["solvers"]
        offset = args.task % len(solvers)
        order = solvers[offset:] + solvers[:offset]
        records = []
        host = {"hostname": socket.gethostname(), "platform": platform.platform(),
                "cpu": platform.processor(), "oar_job_id": os.environ.get("OAR_JOB_ID"),
                "workers_per_job": os.environ.get("QAYD_CAMPAIGN_WORKERS", "1"),
                "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None}
        cpu_info = Path("/proc/cpuinfo")
        if cpu_info.is_file():
            host["cpu"] = next((line.split(":", 1)[1].strip() for line in cpu_info.read_text().splitlines()
                                if line.startswith("model name")), host["cpu"])
        for solver in order:
            saved = directory / f"{solver}.json"
            if saved.exists():
                record = read(saved)
                if record["plan_sha256"] != plan["sha256"]:
                    raise ValueError("arm result belongs to another plan")
            else:
                record = run_arm(solver, item, task, plan, directory)
                record.update(plan_sha256=plan["sha256"], task_key=task["key"], host=host)
                verify_snapshot(plan)
                for relative in relevant:
                    if sha256_file(ROOT / relative) != plan["assets"][relative]:
                        raise ValueError(f"instance or model changed during execution: {relative}")
                write(saved, record)
            records.append(record)
        write(completed, {"complete": True, "plan_sha256": plan["sha256"], "task_key": task["key"],
                          "arms": records, "item": item, "seed": task["seed"], "seconds": task["seconds"]})


def partition_tasks(tasks: list[dict], jobs: int) -> list[list[int]]:
    groups = [[] for _ in range(min(jobs, len(tasks)))]
    loads = [0] * len(groups)
    for index in sorted(range(len(tasks)), key=lambda i: -tasks[i]["seconds"] * len(tasks[i]["solvers"])):
        group = min(range(len(groups)), key=lambda i: loads[i])
        groups[group].append(index)
        loads[group] += tasks[index]["seconds"] * len(tasks[index]["solvers"])
    return groups


def submit(args) -> None:
    if not args.property or not args.property.strip():
        raise ValueError("an explicit CPU-only OAR --property is required")
    path = args.plan.resolve()
    plan = load_plan(path)
    verify_snapshot(plan)
    if "walltime" in args.resource or not re.fullmatch(r"\d+(?::[0-5]\d){0,2}", args.walltime):
        raise ValueError("provide resource without walltime, and a valid HHH:MM:SS walltime")
    directory = path.parent / "oar"
    directory.mkdir(exist_ok=True)
    groups = partition_tasks(plan["tasks"], args.jobs)
    commands = []
    for index, ids in enumerate(groups):
        pending = [i for i in ids if not (path.parent / "tasks" / f"{i:07d}" / "result.json").exists()]
        if not pending:
            continue
        task_list = directory / f"group-{index:03d}.txt"
        task_list.write_text("".join(f"{i}\n" for i in pending))
        script = Path(__file__).resolve().parents[1] / "scripts/worker.sh"
        command = ["oarsub", "-t", "idempotent", "-t", "besteffort",
                   "-n", f"qayd-{plan['stage']}-{index:03d}",
                   "-l", args.resource + ",walltime=" + args.walltime,
                   "-O", str(directory / f"group-{index:03d}.%jobid%.out"),
                   "-E", str(directory / f"group-{index:03d}.%jobid%.err")]
        if args.queue:
            command += ["-q", args.queue]
        command += ["-p", args.property]
        payload = ["bash", str(script), str(path), str(task_list), str(args.workers_per_job)]
        command.append(shlex.join(payload))
        commands.append(command)
    write(directory / "submission.json", {"commands": commands, "jobs": len(commands),
          "workers_per_job": args.workers_per_job,
          "required_memory_mb_per_job": args.workers_per_job * plan["memory_mb"],
          "required_cores_per_job": args.workers_per_job * plan["threads"],
          "types": ["idempotent", "besteffort"], "resource_property": args.property,
          "plan_sha256": plan["sha256"], "resource": args.resource, "walltime": args.walltime})
    for command in commands:
        print(shlex.join(command))
    print(f"{len(commands)} OAR jobs, {args.workers_per_job} sequential-solver lanes per job; "
          f"allow at least {args.workers_per_job * plan['memory_mb']} MiB and "
          f"{args.workers_per_job * plan['threads']} cores per job plus harness overhead.")
    if args.execute:
        for command in commands:
            subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    b = subs.add_parser("build", help="build current Rust binaries and Python extension, write receipt")
    b.add_argument("--receipt", type=Path, required=True)
    b.add_argument("--lp", "--lp-feature", dest="lp_feature", action="store_true",
                   help="compile all Qayd binaries and the Python extension with lp-relaxation")
    b.add_argument("--target-dir", type=Path,
                   help="temporary Cargo build directory; runnable binaries are copied to target/release")
    p = subs.add_parser("prepare", help="inventory or freeze a campaign on the execution host")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--stage", choices=("smoke", "gate", "full"), default="gate")
    p.add_argument("--inventory", action="store_true")
    p.add_argument("--out", type=Path)
    p.add_argument("--receipt", type=Path, default=ROOT / "bench/cluster/runs/build.json")
    p.add_argument("--gate", type=Path)
    p.add_argument("--memory-mb", type=positive)
    p.add_argument("--threads", type=positive, default=1)
    p.add_argument("--java", default="java")
    p.add_argument("--cadical", default="cadical")
    p.add_argument("--minizinc", default="minizinc")
    p.add_argument("--hgs")
    p.add_argument("--lkh")
    r = subs.add_parser("run", help="run one paired task, resumably")
    r.add_argument("--plan", type=Path, required=True)
    r.add_argument("--task", type=int, required=True)
    s = subs.add_parser("oar", help="print a bounded group of OAR submissions; submit only with --execute")
    s.add_argument("--plan", type=Path, required=True)
    s.add_argument("--jobs", type=positive, required=True, help="total number of OAR worker jobs")
    s.add_argument("--workers-per-job", type=positive, required=True)
    s.add_argument("--resource", required=True, help="OAR resource expression, without walltime")
    s.add_argument("--walltime", required=True)
    s.add_argument("--queue")
    s.add_argument("--property", required=True, help="explicit OAR predicate excluding all GPU nodes")
    s.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        {"build": build, "prepare": prepare, "run": run_task, "oar": submit}[args.action](args)
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        print(f"campaign error: {error}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
