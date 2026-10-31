"""Adapters for frozen cluster tasks, reusing the repository's validators."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import threading

from cluster.lib.adapters.records import complete_record, json_record_from_output
from cluster.lib.adapters.instances import materialize
from cluster.lib.adapters.satpb import verify_solution
from cluster.lib.adapters.measurement import execute_streaming, validate_solution
from cluster.lib import minizinc
from cluster.lib.adapters import domains as domain_campaign


MODEL_TRACKS = {"cvrp", "cvrptw", "jssp", "rcpsp"}
STOP_EVENT = threading.Event()
SUMMARY_FIELDS = {
    "status", "objectives", "verified", "dual_bound", "absolute_gap", "relative_gap",
    "bound_method", "objective_convention", "best_known", "anytime", "profile",
    "parse_error", "diagnostic", "elapsed_seconds", "elapsed_total_seconds",
    "preparation_seconds", "solver_engine", "solver_library_version", "validation",
    "engine", "engine_requested", "schedule_jssp_search", "schedule_cdcl_requested",
    "search_engine", "schedule_execution_phase",
}


def measure(argv: list[str], seconds: int, memory: int, grace: float, stem: Path, sense="min") -> dict:
    result = execute_streaming(
        argv, sense, seconds + grace, 0, memory, grace,
        stem.with_suffix(".stdout.log"), stem.with_suffix(".stderr.log"),
        stop_event=STOP_EVENT,
    )
    if result["cancelled"]:
        raise InterruptedError("worker interrupted; completed arms remain resumable")
    result["stdout"] = stem.with_suffix(".stdout.log").read_text(errors="replace")
    result["stderr"] = stem.with_suffix(".stderr.log").read_text(errors="replace")
    return result


def domain_arguments(runtime: dict, plan: dict, track: str | None = None) -> argparse.Namespace:
    # Missing options preserve the semantics of old frozen plans. Fresh plans
    # declare their JSSP policy explicitly in the hashed configuration.
    options = plan.get("config", {}).get("model_options", {}).get(track, {})
    threads = options.get("threads", plan["threads"])
    if track == "jssp" and options and threads != plan["threads"]:
        raise ValueError("plan threads differ from its JSSP policy")
    return argparse.Namespace(
        threads=threads, qayd_engine=options.get("engine", "auto"), memory_limit_mb=plan["memory_mb"],
        max_iterations=None, profile_qayd=False, routing_two_way=True,
        routing_nearest_neighbor=True, routing_warm_start=True,
        qayd_schedule_path_relink=False, qayd_schedule_lns_shadow=False,
        qayd_schedule_constructor_workers=0, qayd_schedule_jssp_search=options.get("schedule_jssp_search", "legacy"),
        hexaly_home=runtime.get("hexaly_home"),
        hexaly_license_path=runtime.get("hexaly_license_path"),
        hgs_binary=runtime.get("hgs"), lkh_binary=runtime.get("lkh"),
    )


def command(solver: str, item: dict, task: dict, plan: dict, instance: Path) -> list[str]:
    runtime = plan["runtime"]
    seconds = task["seconds"]
    seed = task.get("solver_seeds", {}).get(solver, task["seed"])
    root = Path(plan["root"])
    memory = plan["memory_mb"]
    java = [runtime.get("java", "java"), "-XX:ActiveProcessorCount=1",
            f"-Xmx{max(128, memory // 2)}m", "-XX:CompressedClassSpaceSize=256m",
            "-XX:ReservedCodeCacheSize=128m"]
    if item["track"].startswith("xcsp"):
        if solver == "cosoco":
            if seed is not None:
                raise ValueError("CoSoCo requires an explicitly unseeded arm")
            return [runtime["cosoco"], str(instance), f"-cpu_lim={seconds}",
                    "-nbcores=1", "-model=2", "-colors=0"]
        if solver == "qayd":
            return [runtime["qayd"], "-t", str(seconds), "--threads", "1",
                    "--seed", str(seed), "--mem-limit", str(memory), "--verbose", str(instance)]
        if solver in ("ace", "ace-rr"):
            return [*java, "-jar", runtime["ace"], str(instance), f"-t={seconds}s",
                    f"-seed={seed}", "-npc", "-v=-1", *(["-rr"] if solver == "ace-rr" else [])]
        return [*java, "-cp", runtime["choco"], "org.chocosolver.parser.xcsp.ChocoXCSP",
                f"-limit={seconds}s", "-p", "1", "-seed", str(seed), "-lvl", "COMPET", str(instance)]
    if item["track"] == "sat":
        if solver == "qayd-sat":
            return [runtime[solver], "-t", str(seconds), "--seed", str(seed),
                    "--memory-mb", str(memory), str(instance)]
        return [runtime[solver], "-t", str(seconds), f"--seed={seed}", str(instance)]
    if item["track"] == "pb":
        if solver == "qayd-pb":
            return [runtime[solver], "-t", str(seconds), "--memory-mb", str(memory), str(instance)]
        return [*java, "-cp", runtime["sat4j-pb"] + ":" + runtime["sat4j-core"],
                "org.sat4j.pb.LanceurPseudo2007", "Default", str(seconds), str(instance)]
    if item["track"] == "minizinc":
        solver_id = {"qayd-fzn": runtime["qayd_msc"], "gecode": "org.gecode.gecode",
                     "chuffed": "org.chuffed.chuffed"}[solver]
        argv = [runtime["minizinc"], "--solver", solver_id, "--time-limit", str(seconds * 1000),
                "--json-stream", "--output-mode", "json", "--output-objective", "--statistics"]
        if solver != "qayd-fzn":
            argv += ["-p", "1", "-r", "0"]
        return [*argv, str(instance), *[str(root / p) for p in item.get("data", [])]]
    argv = domain_campaign.command_for(
        solver, {"problem": item["track"], "path": instance}, seconds, seed,
        domain_arguments(runtime, plan, item["track"]),
    )
    argv[0] = runtime["python"]
    if solver.startswith("qayd-") and item["track"] in {"cvrp", "cvrptw"}:
        # The shared builder predates the current routing CLI, which has no
        # memory option. The same outer RLIMIT_AS still applies to every arm.
        index = argv.index("--memory-limit-mb")
        del argv[index:index+2]
    # Preserve the independently checkable schedule in the raw log.
    if "--compact-json" in argv:
        argv.remove("--compact-json")
    return argv


def run_arm(solver: str, item: dict, task: dict, plan: dict, directory: Path) -> dict:
    root = Path(plan["root"])
    stem = directory / solver
    with tempfile.TemporaryDirectory(prefix="qayd-cluster-") as scratch_name:
        scratch = Path(scratch_name)
        instance = root / item["path"]
        if item["track"].startswith("xcsp") or item["track"] in ("sat", "pb"):
            instance = Path(materialize(str(instance), str(scratch), solver))
        argv = command(solver, item, task, plan, instance)
        measured = measure(argv, task["seconds"], plan["memory_mb"], plan["grace_seconds"], stem,
                           item["sense"] if item["sense"] in ("min", "max") else "min")
        validation = {"valid": None, "reason": "no-witness"}
        objective = None
        if item["track"] in MODEL_TRACKS:
            try:
                raw = json_record_from_output(measured["stdout"])
                result = complete_record(raw)
                status = {"SATISFIABLE": "SAT", "OPTIMAL": "OPTIMUM"}.get(result["status"], result["status"])
                if str(raw.get("status", "")).upper() == "UNSUPPORTED":
                    status = "UNSUPPORTED"
                objective = result["objectives"] or None
                if result.get("best_known") is None:
                    result["best_known"] = domain_campaign.adjacent_best_known(root / item["path"])
                validation = {"valid": True if result["verified"] else None,
                              "reason": "dataset-replay" if result["verified"] else "no-verified-witness"}
                requested = plan.get("config", {}).get("model_options", {}).get(item["track"], {})
                if solver.startswith("qayd-") and requested.get("schedule_jssp_search") == "tsab-adaptive":
                    selectors_reported = "engine" in raw or "schedule_jssp_search" in raw
                    if (validation["valid"] is True or selectors_reported) and (
                            raw.get("engine") != "ls" or raw.get("schedule_jssp_search") != "tsab-adaptive"):
                        status = "ERROR"
                        result["parse_error"] = "returned JSSP selectors differ from the frozen adaptive policy"
                        validation = {"valid": None, "reason": "model-policy-mismatch"}
            except (ValueError, KeyError, TypeError) as error:
                result = {"parse_error": str(error)}
                status = "ERROR"
        elif item["track"] == "minizinc":
            result = minizinc.summary(measured["stdout"])
            status, objective = result["status"], result["objective"]
            def checker_measure(args, timeout, _stem):
                return measure(args, timeout, plan["memory_mb"], plan["grace_seconds"], directory / (solver + "-checker"))
            validation = minizinc.replay(
                plan["runtime"]["minizinc"], item, result, root, scratch,
                plan["checker_seconds"], checker_measure,
            )
        else:
            result = {}
            status = measured["claimed_status"]
            best = measured["best_incumbent"]
            objective = best["value"] if best else None
            if item["track"] in ("sat", "pb"):
                validation = verify_solution(item["track"], instance, measured["stdout"], status, objective)
            elif measured["has_solution"]:
                validation = validate_solution(
                    [plan["runtime"]["java"], "-XX:ActiveProcessorCount=1",
                     f"-Xmx{plan['checker_memory_mb']}m", "-cp", plan["runtime"]["ace"],
                     "org.xcsp.parser.callbacks.SolutionChecker", str(instance), "-cm"],
                    stem.with_suffix(".stdout.log"), directory / f"{solver}.checker.log",
                    plan["checker_seconds"], expected_objective=objective,
                    stop_event=STOP_EVENT,
                )
        if validation.get("valid") is False:
            status = "INVALID"
        error_text = measured["stderr"].lower() + json.dumps(result.get("errors", [])).lower()
        if measured["returncode"] not in (0, 10, 20, 30) and not measured["timed_out"]:
            status = "UNSUPPORTED" if any(word in error_text for word in (
                "unsupported", "not supported", "unknown predicate", "undefined predicate",
            )) else "ERROR"
        if measured["timed_out"] or measured["returncode"] not in (0, 10, 20, 30):
            # A truncated process cannot publish a proof. A checked witness survives.
            if status in ("UNSAT", "OPTIMUM"):
                status = "SAT" if validation.get("valid") is True else "UNKNOWN"
        if status in ("SAT", "OPTIMUM") and validation.get("valid") is not True:
            status = "UNKNOWN"
        valid = validation.get("valid") is True
        decision = item["sense"] == "satisfy"
        solved = status == "UNSAT" or (valid and (decision or status == "OPTIMUM"))
        return {
            "solver": solver, "status": status, "claimed_status": measured["claimed_status"],
            "seed": task.get("solver_seeds", {}).get(solver, task["seed"]),
            "objective": objective if valid else None, "validation": validation,
            "solved": solved, "has_incumbent": valid,
            "proof_checked": False, "proof_claimed": status in ("UNSAT", "OPTIMUM"),
            "wall_seconds": measured["elapsed_wall_seconds"], "cpu_seconds": measured["cpu_seconds"],
            "peak_rss_kb": measured["peak_rss_kb"], "timed_out": measured["timed_out"],
            "returncode": measured["returncode"], "command": argv,
            "first_incumbent": measured["first_incumbent"],
            # Large-TA schedules contain one million operations. Keep the full
            # witness once in stdout, rather than duplicating it in each report.
            "details": ({key: value for key, value in result.items() if key in SUMMARY_FIELDS}
                        if item["track"] in MODEL_TRACKS else result),
            "stdout": str(stem.with_suffix(".stdout.log")),
            "stderr": str(stem.with_suffix(".stderr.log")),
            "par2": min(task["seconds"], measured["elapsed_wall_seconds"]) if solved else 2 * task["seconds"],
        }
