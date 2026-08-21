#!/usr/bin/env python3
"""Run reproducible multi-seed routing and scheduling campaigns.

Examples:

  uv run bench/campaign.py --suite smoke --solver qayd-native \
      --budget 1 --seed 0 --out bench/results/smoke.jsonl

  uv run bench/campaign.py --suite competitive \
      --solver qayd-native --solver hexaly --solver hgs --solver lkh \
      --solver ortools-cp-sat --budget 1,10,60,600 --seed 0,1,2,3,4 \
      --threads 4 --hexaly-home /opt/hexaly_14_5 \
      --hexaly-license-path /opt/hexaly_14_5/license.dat \
      --hgs-binary bench/solvers/HGS-CVRP/build/hgs \
      --lkh-binary bench/solvers/LKH-3.0.13/LKH \
      --out bench/results/competitive.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable

from common.competitive import (
    complete_record,
    json_record_from_output,
    machine_provenance,
    run_measured,
    sha256_file,
    source_tree_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = ROOT / "bench" / "adapters"
QAYD_SCRIPTS = {
    ("api", "cvrp"): "examples/python/routing/api/vrp.py",
    ("native", "cvrp"): "examples/python/routing/native/vrp.py",
    ("api", "cvrptw"): "examples/python/routing/api/cvrptw.py",
    ("native", "cvrptw"): "examples/python/routing/native/cvrptw.py",
    ("api", "jssp"): "examples/python/scheduling/api/jssp.py",
    ("native", "jssp"): "examples/python/scheduling/native/jssp.py",
    ("api", "rcpsp"): "examples/python/scheduling/api/rcpsp.py",
    ("native", "rcpsp"): "examples/python/scheduling/native/rcpsp.py",
}
SOLVERS = ("qayd-api", "qayd-native", "hexaly", "hgs", "lkh", "ortools-cp-sat")
SOLVER_PROBLEMS = {
    "qayd-api": {"cvrp", "cvrptw", "jssp", "rcpsp"},
    "qayd-native": {"cvrp", "cvrptw", "jssp", "rcpsp"},
    "hexaly": {"cvrp", "cvrptw", "jssp", "rcpsp"},
    "hgs": {"cvrp"},
    "lkh": {"cvrp", "cvrptw"},
    "ortools-cp-sat": {"jssp", "rcpsp"},
}


def number_list(values: Iterable[str], *, positive: bool) -> list[int]:
    parsed = []
    for value in values:
        for field in value.split(","):
            if not field.strip():
                continue
            number = int(field)
            if (positive and number <= 0) or (not positive and number < 0):
                raise argparse.ArgumentTypeError("values must be positive" if positive else "values must be non-negative")
            parsed.append(number)
    return sorted(set(parsed))


def load_suite(value: str) -> tuple[Path, dict[str, Any]]:
    candidate = Path(value)
    if not candidate.exists():
        candidate = ROOT / "bench" / "suites" / f"{value}.json"
    if not candidate.is_file():
        raise SystemExit(f"suite not found: {value}")
    with candidate.open(encoding="utf-8") as stream:
        suite = json.load(stream)
    if not isinstance(suite.get("datasets"), list):
        raise SystemExit(f"invalid suite: {candidate}")
    return candidate.resolve(), suite


def discover_instances(
    suite: dict[str, Any], families: set[str], limit_per_family: int,
) -> list[dict[str, Any]]:
    found = []
    for dataset in suite["datasets"]:
        family = dataset["family"]
        if families and family not in families:
            continue
        pattern = dataset["glob"]
        paths = sorted(path for path in ROOT.glob(pattern) if path.is_file())
        if limit_per_family:
            paths = paths[:limit_per_family]
        for path in paths:
            found.append({
                "family": family,
                "problem": dataset["problem"],
                "path": path.resolve(),
                "instance_path": str(path.relative_to(ROOT)),
                "instance_sha256": sha256_file(path),
                "best_known": adjacent_best_known(path),
            })
    return found


def preflight_expected_instances(
    suite: dict[str, Any], instances: list[dict[str, Any]],
) -> None:
    """Validate a manifest-pinned instance set before any solver is built or run."""
    expected = suite.get("expected_instances")
    if expected is None:
        return
    if not isinstance(expected, dict) or not expected:
        raise SystemExit("suite expected_instances must be a non-empty JSON object")

    expected_by_name: dict[str, str] = {}
    malformed = []
    duplicate_expected = []
    for raw_name, raw_digest in expected.items():
        if not isinstance(raw_name, str) or not raw_name:
            malformed.append(repr(raw_name))
            continue
        name = raw_name.replace("\\", "/").rsplit("/", 1)[-1]
        if name in expected_by_name:
            duplicate_expected.append(name)
            continue
        if (
            not isinstance(raw_digest, str)
            or len(raw_digest) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in raw_digest)
        ):
            malformed.append(name)
            continue
        expected_by_name[name] = raw_digest.lower()

    discovered_by_name: dict[str, list[dict[str, Any]]] = {}
    for item in instances:
        discovered_by_name.setdefault(item["path"].name, []).append(item)

    errors = []
    if malformed:
        errors.append("invalid expected instance entries: " + ", ".join(sorted(malformed)))
    if duplicate_expected:
        errors.append(
            "duplicate expected basenames: " + ", ".join(sorted(set(duplicate_expected)))
        )
    duplicate_discovered = sorted(
        name for name, matches in discovered_by_name.items() if len(matches) != 1
    )
    if duplicate_discovered:
        errors.append("duplicate discovered basenames: " + ", ".join(duplicate_discovered))

    expected_names = set(expected_by_name)
    discovered_names = set(discovered_by_name)
    missing = sorted(expected_names - discovered_names)
    extra = sorted(discovered_names - expected_names)
    if missing:
        errors.append("missing instances: " + ", ".join(missing))
    if extra:
        errors.append("unexpected instances: " + ", ".join(extra))

    mismatches = sorted(
        name
        for name in expected_names & discovered_names
        if len(discovered_by_name[name]) == 1
        and discovered_by_name[name][0].get("instance_sha256") != expected_by_name[name]
    )
    if mismatches:
        errors.append("SHA-256 mismatches: " + ", ".join(mismatches))
    if errors:
        raise SystemExit("suite instance preflight failed: " + "; ".join(errors))


def minimum_external_grace_seconds(suite: dict[str, Any]) -> float:
    raw_value = suite.get("minimum_external_grace_seconds", 0)
    if (
        isinstance(raw_value, bool)
        or not isinstance(raw_value, (int, float))
        or not math.isfinite(float(raw_value))
        or float(raw_value) < 0
    ):
        raise SystemExit(
            "suite minimum_external_grace_seconds must be a finite non-negative number"
        )
    return float(raw_value)


def enforce_minimum_external_grace_seconds(
    suite: dict[str, Any], configured_grace_seconds: float,
) -> None:
    minimum_grace = minimum_external_grace_seconds(suite)
    if configured_grace_seconds < minimum_grace:
        raise SystemExit(
            f"suite requires --grace-seconds >= {minimum_grace:g}; "
            f"received {configured_grace_seconds:g}"
        )


def adjacent_best_known(path: Path) -> float | None:
    solution = path.with_suffix(".sol")
    if solution.is_file():
        for raw in reversed(solution.read_text(encoding="utf-8", errors="replace").splitlines()):
            fields = raw.replace(":", " ").split()
            if len(fields) >= 2 and fields[0].lower() in {"cost", "objective", "value"}:
                try:
                    return float(fields[1])
                except ValueError:
                    return None
    for parent in path.parents:
        metadata = parent / "instances.json"
        if metadata.is_file():
            try:
                entries = json.loads(metadata.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                entries = []
            for entry in entries if isinstance(entries, list) else []:
                if entry.get("name") == path.name and isinstance(entry.get("optimum"), (int, float)):
                    return float(entry["optimum"])
        summary_path = parent / "summary.json"
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                summary = {}
            prefix = str(summary.get("instance_set", ""))
            suffix = path.stem.removeprefix(prefix)
            fields = suffix.split("_")
            if prefix and len(fields) == 2 and all(field.isdigit() for field in fields):
                parameter, instance = map(int, fields)
                for entry in summary.get("solutions", []):
                    if entry.get("parameter") == parameter and entry.get("instance") == instance:
                        value = entry.get("optimal_makespan") if entry.get("is_proven_optimal") else entry.get("upper_bound")
                        return float(value) if isinstance(value, (int, float)) else None
    return None


def binary_version(path: Path) -> str:
    if not path.is_file():
        return f"missing:{path}"
    return f"sha256:{sha256_file(path)[:16]}"


def solver_version(
    name: str, args: argparse.Namespace, provenance: dict[str, Any],
    qayd_artifact: dict[str, str] | None,
    hexaly_artifact: dict[str, str] | None = None,
) -> str:
    if name.startswith("qayd-"):
        cargo = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
        workspace = cargo.split("[workspace.package]", 1)[-1]
        match = re.search(r'^version\s*=\s*"([^"]+)"', workspace, re.MULTILINE)
        version = match.group(1) if match else "unknown"
        commit = (provenance.get("commit") or "unknown")[:12]
        source = (provenance.get("source_tree_sha256") or "unknown")[:12]
        artifact = ((qayd_artifact or {}).get("sha256") or "unknown")[:12]
        return f"qayd {version} ({commit}, source {source}, artifact {artifact})"
    if name == "ortools-cp-sat":
        try:
            import ortools
            return f"OR-Tools {ortools.__version__} CP-SAT"
        except ImportError:
            return "OR-Tools missing"
    if name == "hexaly":
        if hexaly_artifact is None:
            return "Hexaly runtime not inspected"
        return (
            f"Hexaly Optimizer {hexaly_artifact['hx_version']} "
            f"(Python API {hexaly_artifact['python_api_version']}, "
            f"{hexaly_artifact['runtime_source']}, libhexaly "
            f"sha256:{hexaly_artifact['native_library_sha256']})"
        )
    if name == "hgs":
        return f"HGS-CVRP {binary_version(Path(args.hgs_binary or 'missing'))}"
    if name == "lkh":
        return f"LKH-3 {binary_version(Path(args.lkh_binary or 'missing'))}"
    return "unknown"


def command_for(
    solver: str, item: dict[str, Any], budget: int, seed: int,
    args: argparse.Namespace,
) -> list[str]:
    problem = item["problem"]
    instance = str(item["path"])
    threads = effective_threads(solver, item, args)
    common = [instance, "--time-limit", str(budget), "--threads", str(threads), "--seed", str(seed), "--json"]
    if solver.startswith("qayd-"):
        variant = solver.removeprefix("qayd-")
        engine = args.qayd_engine
        if problem == "rcpsp" and item["path"].suffix.lower() == ".mm" and engine == "ls":
            engine = "auto"
        command = [
            sys.executable,
            str(ROOT / QAYD_SCRIPTS[(variant, problem)]),
            *common,
            "--engine",
            engine,
            "--memory-limit-mb",
            str(args.memory_limit_mb),
        ]
        list_controls = problem in {"cvrp", "cvrptw"}
        if args.profile_qayd:
            command.append("--profile")
        scheduling_ls_controls = problem in {"jssp", "rcpsp"} and engine in {"auto", "ls"} and item["path"].suffix.lower() != ".mm"
        if args.max_iterations is not None and (engine == "ls" or scheduling_ls_controls):
            command.extend(("--max-iterations", str(args.max_iterations)))
        if problem in {"cvrp", "cvrptw"}:
            command.append("--routing-two-way" if args.routing_two_way else "--no-routing-two-way")
            command.append("--routing-nearest-neighbor" if args.routing_nearest_neighbor else "--no-routing-nearest-neighbor")
            command.append("--routing-warm-start" if args.routing_warm_start else "--no-routing-warm-start")
        if problem == "jssp":
            command.append("--compact-json")
        return command
    if solver == "ortools-cp-sat":
        return [sys.executable, str(ADAPTERS / "ortools_cp_sat.py"), problem, *common]
    if solver == "hexaly":
        command = [
            sys.executable, str(ADAPTERS / "hexaly.py"), problem, *common,
            "--hexaly-license-path", str(Path(args.hexaly_license_path).expanduser().resolve()),
        ]
        if args.hexaly_home:
            command.extend((
                "--hexaly-home", str(Path(args.hexaly_home).expanduser().resolve()),
            ))
        return command
    if solver == "hgs":
        return [
            sys.executable, str(ADAPTERS / "hgs.py"), instance,
            "--time-limit", str(budget), "--seed", str(seed), "--threads", str(threads),
            "--binary", str(Path(args.hgs_binary).resolve()), "--json",
        ]
    if solver == "lkh":
        return [
            sys.executable, str(ADAPTERS / "lkh.py"), problem, instance,
            "--time-limit", str(budget), "--seed", str(seed), "--threads", str(threads),
            "--binary", str(Path(args.lkh_binary).resolve()), "--json",
        ]
    raise AssertionError(solver)


def effective_threads(solver: str, item: dict[str, Any], args: argparse.Namespace) -> int:
    if solver in {"hgs", "lkh"}:
        return 1
    if solver.startswith("qayd-"):
        if args.qayd_engine == "exact":
            return 1
        if item["problem"] == "rcpsp" and item["path"].suffix.lower() == ".mm":
            return 1
    return args.threads


def run_key(solver: str, item: dict[str, Any], budget: int, seed: int, args: argparse.Namespace) -> str:
    raw = json.dumps(
        [
            solver,
            item["instance_path"],
            item["instance_sha256"],
            item["problem"],
            budget,
            seed,
            args.threads,
            args.grace_seconds,
            args.qayd_engine if solver.startswith("qayd-") else None,
            args.max_iterations if solver.startswith("qayd-") else None,
            args.profile_qayd if solver.startswith("qayd-") else None,
            args.routing_two_way if solver.startswith("qayd-") else None,
            args.routing_nearest_neighbor if solver.startswith("qayd-") else None,
            args.routing_warm_start if solver.startswith("qayd-") else None,
        ],
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def existing_keys(path: Path) -> set[str]:
    keys = set()
    if not path.is_file():
        return keys
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("run_id"):
                keys.add(record["run_id"])
    return keys


def check_solver_arguments(solvers: list[str], args: argparse.Namespace) -> None:
    requirements = {
        "hgs": (args.hgs_binary, "--hgs-binary"),
        "lkh": (args.lkh_binary, "--lkh-binary"),
    }
    for solver, (value, flag) in requirements.items():
        if solver in solvers and not value:
            raise SystemExit(f"{solver} requires {flag}")
    if "hexaly" in solvers:
        if not args.hexaly_license_path:
            if not args.hexaly_home:
                raise SystemExit("hexaly requires --hexaly-license-path without --hexaly-home")
            args.hexaly_license_path = (
                Path(args.hexaly_home).expanduser().resolve() / "license.dat"
            )
        license_path = Path(args.hexaly_license_path).expanduser().resolve()
        if not license_path.is_file():
            raise SystemExit(f"Hexaly license file not found: {license_path}")


def hexaly_runtime_provenance(args: argparse.Namespace) -> dict[str, str]:
    """Inspect the imported runtime without constructing a HexalyOptimizer."""
    command = [sys.executable, str(ADAPTERS / "hexaly.py"), "--runtime-info"]
    if args.hexaly_home:
        command.extend((
            "--hexaly-home", str(Path(args.hexaly_home).expanduser().resolve()),
        ))
    try:
        completed = subprocess.run(
            command, cwd=ROOT, text=True, capture_output=True, check=True, timeout=30,
        )
        artifact = json_record_from_output(completed.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        diagnostic = ""
        if isinstance(error, subprocess.CalledProcessError):
            diagnostic = (error.stderr or error.stdout or "").strip()
        suffix = f": {diagnostic}" if diagnostic else f": {error}"
        raise SystemExit(f"cannot inspect the Hexaly runtime{suffix}") from error
    required = {
        "runtime_source", "hx_version", "python_api_version", "module_path",
        "native_library_path", "native_library_sha256",
    }
    missing = sorted(required - artifact.keys())
    digest = artifact.get("native_library_sha256")
    if missing or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        detail = f"missing {', '.join(missing)}" if missing else "invalid native library SHA-256"
        raise SystemExit(f"invalid Hexaly runtime provenance: {detail}")
    return {key: str(artifact[key]) for key in sorted(required)}


def hexaly_artifact_unchanged(artifact: dict[str, str]) -> bool:
    path = Path(artifact.get("native_library_path", ""))
    return path.is_file() and sha256_file(path) == artifact.get("native_library_sha256")


def prepare_qayd_extension() -> None:
    try:
        subprocess.run(
            ["maturin", "develop", "--profile", "pyext", "--features", "python"], cwd=ROOT,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise SystemExit(f"cannot build the qayd Python extension: {error}") from error


def qayd_artifact_provenance() -> dict[str, str]:
    try:
        completed = subprocess.run(
            [sys.executable, "-c", "import qayd._core; print(qayd._core.__file__)"],
            cwd=ROOT, text=True, capture_output=True, check=True, timeout=30,
        )
        location = completed.stdout.strip()
        if not location:
            raise OSError("the extension did not report its path")
        path = Path(location).resolve(strict=True)
        if not path.is_file():
            raise OSError(f"the extension path is not a file: {path}")
    except (OSError, subprocess.SubprocessError) as error:
        raise SystemExit(f"cannot locate the installed qayd extension: {error}") from error
    return {"path": str(path), "sha256": sha256_file(path)}


def qayd_artifact_unchanged(artifact: dict[str, str]) -> bool:
    path = Path(artifact.get("path", ""))
    return path.is_file() and sha256_file(path) == artifact.get("sha256")


def ensure_campaign_artifacts_unchanged(
    provenance: dict[str, Any],
    qayd_artifact: dict[str, str] | None,
    hexaly_artifact: dict[str, str] | None,
) -> None:
    if source_tree_sha256(ROOT) != provenance.get("source_tree_sha256"):
        raise SystemExit("source tree changed during the campaign; restart with a new output")
    if qayd_artifact is not None and not qayd_artifact_unchanged(qayd_artifact):
        raise SystemExit("qayd extension changed during the campaign; restart with a new output")
    if hexaly_artifact is not None and not hexaly_artifact_unchanged(hexaly_artifact):
        raise SystemExit("Hexaly native library changed during the campaign; restart with a new output")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="smoke", help="suite name or JSON file")
    parser.add_argument("--solver", action="append", choices=SOLVERS, required=True)
    parser.add_argument("--budget", action="append", default=[], metavar="SECONDS", help="repeat or use comma-separated checkpoints")
    parser.add_argument("--seed", action="append", default=[], help="repeat or use comma-separated seeds")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--family", action="append", default=[], help="only selected dataset families")
    parser.add_argument("--limit-per-family", type=int, default=0)
    parser.add_argument("--memory-limit-mb", type=int, default=0)
    parser.add_argument("--grace-seconds", type=float, default=20.0, help="wrapper grace beyond solver budget")
    parser.add_argument("--qayd-engine", choices=("auto", "exact", "ls"), default="ls")
    parser.add_argument("--max-iterations", type=int, help="explicit LS iteration cap for qayd")
    parser.add_argument(
        "--profile-qayd", action=argparse.BooleanOptionalAction, default=True,
        help="collect candidate throughput for qayd LS runs",
    )
    parser.add_argument(
        "--prepare-qayd", action=argparse.BooleanOptionalAction, default=False,
        help="build and install the current source tree before running qayd",
    )
    parser.add_argument("--routing-two-way", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--routing-nearest-neighbor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--routing-warm-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--hexaly-home",
        help="optional Hexaly installation root; omit for an importable Python distribution",
    )
    parser.add_argument(
        "--hexaly-license-path",
        help="explicit Hexaly license file used by the adapter",
    )
    parser.add_argument("--hgs-binary", help="compiled HGS-CVRP executable")
    parser.add_argument("--lkh-binary", help="compiled LKH-3 executable")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--restart", action="store_true", help="ignore completed run IDs instead of resuming")
    args = parser.parse_args()

    if (
        args.threads <= 0
        or args.limit_per_family < 0
        or args.memory_limit_mb < 0
        or not math.isfinite(args.grace_seconds)
        or args.grace_seconds < 0
        or (args.max_iterations is not None and args.max_iterations < 0)
    ):
        raise SystemExit("threads must be positive; limits and grace must be non-negative")
    budgets = number_list(args.budget or ["1,10,60,600"], positive=True)
    seeds = number_list(args.seed or ["0,1,2,3,4"], positive=False)
    solvers = list(dict.fromkeys(args.solver))
    check_solver_arguments(solvers, args)
    suite_path, suite = load_suite(args.suite)
    instances = discover_instances(suite, set(args.family), args.limit_per_family)
    preflight_expected_instances(suite, instances)
    enforce_minimum_external_grace_seconds(suite, args.grace_seconds)
    if not instances:
        raise SystemExit("no matching instances; run bench/fetch_collections.py for the competitive suite")

    has_qayd = any(solver.startswith("qayd-") for solver in solvers)
    if has_qayd and args.prepare_qayd:
        prepare_qayd_extension()
    qayd_artifact = qayd_artifact_provenance() if has_qayd else None

    args.out.parent.mkdir(parents=True, exist_ok=True)
    completed = set() if args.restart else existing_keys(args.out)
    provenance = machine_provenance(ROOT)
    hexaly_artifact = hexaly_runtime_provenance(args) if "hexaly" in solvers else None
    versions = {
        solver: solver_version(
            solver, args, provenance, qayd_artifact, hexaly_artifact,
        )
        for solver in solvers
    }
    sidecar = args.out.with_suffix(args.out.suffix + ".provenance.json")
    sidecar_payload = {
        "schema_version": 1,
        "suite": suite,
        "suite_file": str(suite_path),
        "solvers": versions,
        "budgets": budgets,
        "seeds": seeds,
        "threads": args.threads,
        "memory_limit_mb": args.memory_limit_mb,
        "grace_seconds": args.grace_seconds,
        "qayd_engine": args.qayd_engine,
        "max_iterations": args.max_iterations,
        "profile_qayd": args.profile_qayd,
        "qayd_prepared": args.prepare_qayd,
        "qayd_artifact": qayd_artifact,
        "hexaly_artifact": hexaly_artifact,
        "hexaly_license_path": (
            str(Path(args.hexaly_license_path).expanduser().resolve())
            if args.hexaly_license_path else None
        ),
        "routing_two_way": args.routing_two_way,
        "routing_nearest_neighbor": args.routing_nearest_neighbor,
        "routing_warm_start": args.routing_warm_start,
        "host": provenance,
    }
    compatibility_fields = (
        "suite", "suite_file", "solvers", "budgets", "seeds", "threads",
        "memory_limit_mb", "grace_seconds", "qayd_engine", "max_iterations", "profile_qayd",
        "qayd_prepared", "qayd_artifact",
        "hexaly_artifact", "hexaly_license_path",
        "routing_two_way", "routing_nearest_neighbor", "routing_warm_start",
    )
    if args.out.exists() and not args.restart:
        if not sidecar.is_file():
            raise SystemExit(f"cannot resume without provenance sidecar: {sidecar}")
        previous = json.loads(sidecar.read_text(encoding="utf-8"))
        changed = [field for field in compatibility_fields if previous.get(field) != sidecar_payload.get(field)]
        if changed:
            fields = ", ".join(changed)
            raise SystemExit(f"campaign configuration changed ({fields}); use a new --out or --restart")
        previous_host = previous.get("host", {})
        if (
            previous_host.get("commit") != provenance.get("commit")
            or previous_host.get("source_tree_sha256") != provenance.get("source_tree_sha256")
        ):
            raise SystemExit("qayd source tree changed; use a new --out or --restart")
    sidecar.write_text(json.dumps(sidecar_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    jobs = [
        (solver, item, budget, seed)
        for item in instances
        for budget in budgets
        for seed in seeds
        for solver in solvers if item["problem"] in SOLVER_PROBLEMS[solver]
    ]
    pending = [job for job in jobs if run_key(job[0], job[1], job[2], job[3], args) not in completed]
    print(f"campaign: {len(instances)} instances, {len(jobs)} runs, {len(pending)} pending", file=sys.stderr)
    mode = "w" if args.restart else "a"
    with args.out.open(mode, encoding="utf-8") as output:
        for index, (solver, item, budget, seed) in enumerate(pending, 1):
            ensure_campaign_artifacts_unchanged(
                provenance, qayd_artifact, hexaly_artifact,
            )
            identifier = run_key(solver, item, budget, seed, args)
            argv = command_for(solver, item, budget, seed, args)
            measured = run_measured(
                argv, timeout=budget + args.grace_seconds, cwd=ROOT,
                memory_limit_mb=args.memory_limit_mb,
            )
            ensure_campaign_artifacts_unchanged(
                provenance, qayd_artifact, hexaly_artifact,
            )
            try:
                record = json_record_from_output(measured["stdout"])
            except ValueError as error:
                record = {"status": "ERROR", "objectives": [], "verified": False, "error": str(error)}
            record.update({
                "run_id": identifier,
                "solver": solver,
                "solver_version": versions[solver],
                "problem": item["problem"],
                "family": item["family"],
                "instance_path": item["instance_path"],
                "instance_sha256": item["instance_sha256"],
                "checkpoint_seconds": budget,
                "grace_seconds": args.grace_seconds,
                "external_timeout_seconds": budget + args.grace_seconds,
                "seed": seed,
                "requested_threads": args.threads,
                "threads": effective_threads(solver, item, args),
                "wall_seconds": measured["wall_seconds"],
                "peak_memory_mb": measured["peak_memory_mb"],
                "return_code": measured["return_code"],
                "timed_out": measured["timed_out"],
                "command": measured["argv"],
            })
            if record.get("best_known") is None and item["best_known"] is not None:
                record["best_known"] = item["best_known"]
            if measured["return_code"] != 0:
                record["status"] = "ERROR"
                record["verified"] = False
                record["diagnostic"] = measured["stderr"][-4000:]
            record = complete_record(record)
            output.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            output.flush()
            objective = record["objectives"] or "-"
            print(
                f"[{index}/{len(pending)}] {solver} {item['family']}/{item['path'].name} "
                f"t={budget} seed={seed}: {record['status']} obj={objective} "
                f"rss={record['peak_memory_mb']:.1f}MB",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
