#!/usr/bin/env python3
"""Add a CoSoCo-only FAST COP block to a running, immutable cluster campaign."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

PROFILE = "fastcop-cosoco"
PARENT = "xcsp-fastcop"


def frozen_driver(settings):
    spec = importlib.util.spec_from_file_location("qayd_frozen_continuation", settings["driver"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.verify_files(settings["files"])
    return module


def prepare(args, settings, driver):
    campaign, _ = driver.modules(args.repo)
    directory = args.out / "extensions" / PROFILE
    descriptor = directory / "extension.json"
    if descriptor.exists():
        extension = driver.read(descriptor)
        driver.verify_files(extension["artifacts"])
        return extension
    from cluster.lib.install_cosoco import VERSION, verify
    binary = verify(args.repo / f"bench/cluster/runs/tools/cosoco-{VERSION}").resolve()
    parents = {stage: args.out / PARENT / stage / "plan.json" for stage in ("smoke", "gate", "full")}
    plans = {stage: campaign.load_plan(path) for stage, path in parents.items()}
    for plan in plans.values():
        campaign.verify_snapshot(plan)
        if plan["source"]["sha256"] != plans["full"]["source"]["sha256"]:
            raise ValueError("parent stages use different sources")
        if any(t["item"]["track"] != "xcsp-fastcop" for t in plan["tasks"]):
            raise ValueError("this extension accepts only a FAST COP parent")
    directory.mkdir(parents=True, exist_ok=True)
    worker_driver = directory / "driver.py"
    source = Path(__file__).read_bytes()
    if worker_driver.exists() and worker_driver.read_bytes() != source:
        raise ValueError("refusing to overwrite a different frozen supplement driver")
    worker_driver.write_bytes(source)
    artifacts = {str(p): driver.sha(p) for p in [worker_driver, binary, *parents.values()]}
    # Preserve all files shipped beside the official executable, including its license.
    runtime = binary.parents[1]
    if binary.parent.name == "cosoco_bin" and runtime.name == "runtime":
        artifacts.update({str(p): driver.sha(p) for p in runtime.rglob("*") if p.is_file()})
        receipt = runtime.parent / "download.json"
        if receipt.is_file():
            artifacts[str(receipt)] = driver.sha(receipt)
    extension = {"schema": "qayd.cluster.cosoco/v1", "profile": PROFILE,
                 "parent_plans": {k: str(v) for k, v in parents.items()},
                 "parent_full_sha256": plans["full"]["sha256"], "binary": str(binary),
                 "driver": str(worker_driver), "artifacts": artifacts,
                 "note": "Additional experimental block; existing results are never rewritten or rerun."}
    extension["environment"] = campaign.digest(extension)
    driver.write(descriptor, extension)
    return extension


def derive_plan(parent, original, extension, stage, campaign):
    plan = copy.deepcopy(parent)
    plan.pop("sha256")
    tasks = []
    instances = {}
    for task in original["tasks"]:
        instances.setdefault(campaign.digest(task["item"]), []).append(task)
    for repetitions in instances.values():
        # The published executable has no seed option. Do not invent five seeds
        # or count one unseeded run five times in the comparison.
        reference = next((t for t in repetitions if t["seed"] == 0), repetitions[0])
        if any(t["seconds"] != reference["seconds"] for t in repetitions):
            raise ValueError("parent repetitions use different budgets")
        task = copy.deepcopy(reference)
        task["parent_task_key"] = task.pop("key")
        task["parent_seed"] = task["seed"]
        task["parent_task_keys"] = [t["key"] for t in repetitions]
        task["seed"] = None
        task["seed_policy"] = "unseeded: CoSoCo 2.6.1 exposes no seed option"
        task["solvers"] = ["cosoco"]
        task["key"] = campaign.digest(task)
        tasks.append(task)
    plan.update(stage=stage, tasks=tasks, environment=extension["environment"], supplement=extension)
    plan["runtime"]["cosoco"] = extension["binary"]
    plan["artifacts"].update(extension["artifacts"])
    plan["assets"].update(original["assets"])
    plan["config"]["solvers"]["xcsp-fastcop"] = ["cosoco"]
    plan["config"]["seeds"] = [None]
    plan["overview"].update(paired_tasks=len(tasks), solver_runs=len(tasks),
                            maximum_solver_hours=round(sum(t["seconds"] for t in tasks) / 3600, 2))
    plan["sha256"] = campaign.digest(plan)
    return plan


def require_gate(directory, plan, campaign, report):
    path = directory / "gate/plan.json"
    gate = campaign.load_plan(path)
    qualification = campaign.read(directory / "gate/qualification.json")
    if (not qualification.get("ready") or qualification.get("stage") != "gate"
            or qualification.get("plan_sha256") != gate["sha256"]
            or qualification.get("environment") != plan["environment"]
            or gate["environment"] != plan["environment"]
            or not report.audit(gate, path.parent)["ready"]):
        raise ValueError("CoSoCo full runs require their own complete, clean gate")


def next_stage(settings, extension, driver):
    campaign, report = driver.modules(Path(settings["root"]))
    directory = Path(settings["out"]) / PROFILE
    driver.verify_files(extension["artifacts"])
    parent = campaign.load_plan(Path(extension["parent_plans"]["full"]))
    for stage in ("smoke", "gate", "full"):
        path = directory / stage / "plan.json"
        if not path.exists():
            original = campaign.load_plan(Path(extension["parent_plans"][stage]))
            plan = derive_plan(parent, original, extension, stage, campaign)
            if stage == "full":
                require_gate(directory, plan, campaign, report)
            path.parent.mkdir(parents=True, exist_ok=True)
            campaign.write(path, plan)
        plan = campaign.load_plan(path)
        campaign.verify_snapshot(plan)
        audit = report.audit(plan, path.parent)
        campaign.write(path.parent / "report.json", audit)
        campaign.write(path.parent / "qualification.json", {k: audit[k] for k in (
            "ready", "complete", "environment", "plan_sha256", "stage", "expected_tasks", "completed_tasks")})
        (path.parent / "report.md").write_text(report.markdown(audit))
        if stage == "full":
            combined = combined_report(parent, plan, audit, path.parent, campaign, report)
            if combined["cross_block_contradictions"]:
                raise ValueError(f"CoSoCo contradicts an existing result; see {path.parent / 'comparison.md'}")
        if audit["errors"] or audit["contradictions"]:
            raise ValueError(f"{PROFILE}/{stage}: audit failed; see {path.parent / 'report.md'}")
        if not audit["ready"]:
            return stage, path, plan, audit["missing_tasks"]
    return None


def cosoco_command(binary, task, instance):
    # Instance first, a full XCSP witness, one sequential solver process.
    if task["seed"] is not None:
        raise ValueError("CoSoCo 2.6.1 cannot accept a configured seed")
    return [binary, str(instance), f"-cpu_lim={task['seconds']}", "-nbcores=1", "-model=2", "-colors=0"]


def run_task(args, settings, driver):
    campaign, _ = driver.modules(args.repo)
    plan = campaign.load_plan(args.plan)
    if any(t["solvers"] != ["cosoco"] for t in plan["tasks"]):
        raise ValueError("supplement worker refuses to run an existing solver arm")
    from cluster.lib import runners
    def command(solver, _item, task, frozen, instance):
        if solver != "cosoco":
            raise ValueError("only CoSoCo is allowed in this supplement")
        return cosoco_command(frozen["runtime"]["cosoco"], task, instance)
    runners.command = command
    campaign.run_task(args)


def worker(args, settings, driver):
    campaign, _ = driver.modules(args.repo)
    driver.verify_files(driver.read(settings["java_manifest"]))
    plan = campaign.load_plan(args.plan)
    memory = {key: int(value.split()[0]) // 1024 for key, value in
              (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())}
    lanes = driver.worker_count(settings["workers"], plan["memory_mb"], plan["threads"],
                                os.sched_getaffinity(0), memory["MemAvailable"])
    print(f"OAR {os.environ.get('OAR_JOB_ID')}: {lanes} workers, {plan['memory_mb']} MiB per solver", flush=True)
    os.environ.update(QAYD_CAMPAIGN_WORKERS=str(lanes), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                      MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    payload = [settings["python"], plan["supplement"]["driver"], "--repo", args.repo, "--out", args.out,
               "--plan", args.plan, "--task", "{}"]
    driver.run(["parallel", "--quote", "--jobs", lanes, "--halt", "soon,fail=1", "--line-buffer",
                *payload, "::::", args.tasks], cwd=args.repo)


def combined_report(parent, plan, supplement_audit, directory, campaign, report):
    """Join by frozen parent keys; retain every original error and provenance."""
    origin = Path(plan["supplement"]["parent_plans"]["full"]).parent
    parent_audit = report.audit(parent, origin)
    groups = {t["key"]: {"item": t["item"], "seed": t["seed"], "seconds": t["seconds"], "arms": []}
              for t in parent["tasks"]}
    for audit, source, extra in ((parent_audit, parent, False), (supplement_audit, plan, True)):
        for row in audit["records"]:
            task = source["tasks"][row["task"]]
            key = task["parent_task_key"] if extra else task["key"]
            expected = groups[key]
            if (any(task[name] != expected[name] for name in ("item", "seconds"))
                    or (extra and (task["seed"] is not None or task["parent_seed"] != expected["seed"]))
                    or (not extra and task["seed"] != expected["seed"])):
                raise ValueError("supplement differs from the parent instance, reference or budget")
            expected["arms"].append({**row, "source_plan_sha256": source["sha256"]})
    comparisons = []
    contradictions = []
    for key, group in groups.items():
        arms = group["arms"]
        if any(a["has_incumbent"] for a in arms) and any(a["status"] == "UNSAT" for a in arms):
            contradictions.append({"parent_task_key": key, "reason": "UNSAT-versus-verified-witness"})
        cosoco = next((a for a in arms if a["solver"] == "cosoco" and a["has_incumbent"]), None)
        for other in arms:
            if not cosoco or other["solver"] == "cosoco" or not other["has_incumbent"]:
                continue
            a, b = report.vector(cosoco["objective"]), report.vector(other["objective"])
            if a is None or b is None:
                continue
            better = a > b if group["item"]["sense"] == "max" else a < b
            outcome = "tie" if a == b else "cosoco-win" if better else "cosoco-loss"
            comparisons.append({"parent_task_key": key, "other": other["solver"], "outcome": outcome})
            if ((cosoco["status"] == "OPTIMUM" and outcome == "cosoco-loss")
                    or (other["status"] == "OPTIMUM" and outcome == "cosoco-win")):
                contradictions.append({"parent_task_key": key, "reason": "better-witness-than-claimed-optimum"})
    # Correctness contradictions also matter against the other four original
    # seeds, even though the primary quality table counts only one reference.
    by_instance = {}
    for group in groups.values():
        by_instance.setdefault(campaign.digest(group["item"]), []).extend(group["arms"])
    cross_block = []
    for task in plan["tasks"]:
        arms = by_instance[campaign.digest(task["item"])]
        extra = next((a for a in arms if a["solver"] == "cosoco"), None)
        if not extra:
            continue
        for other in arms:
            if other["solver"] == "cosoco":
                continue
            reason = None
            if ((extra["status"] == "UNSAT" and other["has_incumbent"])
                    or (other["status"] == "UNSAT" and extra["has_incumbent"])):
                reason = "UNSAT-versus-verified-witness"
            if extra["has_incumbent"] and other["has_incumbent"]:
                a, b = report.vector(extra["objective"]), report.vector(other["objective"])
                if a is not None and b is not None:
                    better = a > b if task["item"]["sense"] == "max" else a < b
                    if ((extra["status"] == "OPTIMUM" and a != b and not better)
                            or (other["status"] == "OPTIMUM" and better)):
                        reason = "better-witness-than-claimed-optimum"
            if reason:
                cross_block.append({"instance": task["item"]["id"], "other": other["solver"],
                                    "other_seed": other["seed"], "reason": reason})
    result = {"parent_plan_sha256": parent["sha256"], "supplement_plan_sha256": plan["sha256"],
              "parent_errors": parent_audit["errors"], "supplement_errors": supplement_audit["errors"],
              "contradictions": [*parent_audit["contradictions"], *supplement_audit["contradictions"],
                                 *contradictions, *cross_block],
              "cross_block_contradictions": cross_block,
              "parent_complete": parent_audit["complete"], "supplement_complete": supplement_audit["complete"],
              "timing_comparable": False,
              "timing_note": "CoSoCo is an additional block; keep host, CPU, concurrency and order effects explicit.",
              "seed_policy": "One unseeded CoSoCo run per instance, compared once to the parent's seed 0 "
                             "(or its first available seed). All original repetitions and errors are retained; "
                             "the runs are not matched by seed.",
              "parent_solver_runs": len(parent_audit["records"]),
              "supplement_solver_runs": len(supplement_audit["records"]),
              "pairs": groups, "comparisons": comparisons}
    campaign.write(directory / "comparison.json", result)
    lines = ["# FAST COP with the CoSoCo supplement", "",
             "Original results are retained. The comparisons below require two accepted witnesses.",
             "One unseeded CoSoCo run per instance, compared once to the original seed 0 or first available seed. "
             "Other repetitions remain in comparison.json. These are not matched-seed runs.", "",
             f"Original errors: {len(result['parent_errors'])}. CoSoCo errors: {len(result['supplement_errors'])}.",
             f"Contradictions: {len(result['contradictions'])}. Timings come from separate experimental blocks.", "",
             "| Comparator | CoSoCo wins | Ties | CoSoCo losses |", "|---|---:|---:|---:|"]
    for solver in sorted({r["other"] for r in comparisons}):
        counts = Counter(r["outcome"] for r in comparisons if r["other"] == solver)
        lines.append(f"| {solver} | {counts['cosoco-win']} | {counts['tie']} | {counts['cosoco-loss']} |")
    (directory / "comparison.md").write_text("\n".join(lines) + "\n")
    return result


def install_hooks(driver, extension):
    original_next, original_command = driver.next_stage, driver.command_for
    def choose(settings, profile):
        return next_stage(settings, extension, driver) if profile == PROFILE else original_next(settings, profile)
    def command(settings, plan, tasks, number):
        result = original_command(settings, plan, tasks, number)
        if Path(plan).parent.parent.name == PROFILE:
            result[-1] = shlex.join([settings["python"], extension["driver"], "--worker", "--repo", settings["root"],
                                     "--out", settings["out"], "--plan", str(plan), "--tasks", str(tasks)])
        return result
    driver.next_stage, driver.command_for = choose, command


def register(settings, extension, driver):
    descriptor = Path(extension["driver"]).with_name("extension.json")
    updated = copy.deepcopy(settings)
    updated["profiles"][PROFILE] = {"supplement": str(descriptor), "config": str(descriptor)}
    updated["files"].update(extension["artifacts"])
    updated["files"][str(descriptor)] = driver.sha(descriptor)
    return updated


def controller(args, settings, extension, driver):
    with (args.out / "controller.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if not args.replace_controller:
                raise ValueError("controller already active; provide its verified PID with --replace-controller")
            pid = args.replace_controller
            command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            allowed = {Path(settings["driver"]).resolve(), Path.home() / "qayd-ritchie-next-v2-20261030.py",
                       args.repo / "bench/cluster/lib/controller.py", args.repo / "bench/cluster",
                       args.repo / "bench/cluster/__main__.py"}
            if (b"--execute" not in command or b"--worker" in command or b"--task" in command
                    or not any(Path(os.fsdecode(v)).resolve() in allowed for v in command if v)):
                raise ValueError("PID does not identify this campaign's original controller")
            if Path(f"/proc/{pid}/cwd").resolve() != args.repo:
                raise ValueError("controller PID belongs to another working directory")
            os.kill(pid, signal.SIGTERM)
            for _ in range(50):
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(0.1)
            else:
                raise ValueError("original controller still owns the lock; no profile registered")
            print(f"Controller {pid} reloaded; no OAR job stopped.", flush=True)
        current = driver.read(args.out / "settings.json")
        driver.verify_files(current["files"])
        backup = args.out / "settings.before-cosoco.json"
        if not backup.exists():
            backup.write_bytes((args.out / "settings.json").read_bytes())
        current = register(current, extension, driver)
        driver.write(args.out / "settings.json", current)
        install_hooks(driver, extension)
        (args.out / "controller.pid").write_text(str(os.getpid()) + "\n")
        driver.control(args, current)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, default=Path("bench/cluster/runs/next-v2"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--replace-controller", type=int)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--plan", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--tasks", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--task", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    args.out = (args.repo / args.out).absolute()
    os.chdir(args.repo)
    settings = json.loads((args.out / "settings.json").read_text())
    driver = frozen_driver(settings)
    if args.worker:
        worker(args, settings, driver)
        return
    if args.task is not None:
        run_task(args, settings, driver)
        return
    if args.poll_seconds < 5:
        parser.error("--poll-seconds must be >= 5")
    extension = prepare(args, settings, driver)
    if not args.execute:
        print(f"Prepared CoSoCo supplement: {extension['driver']}. Parent campaign unchanged.")
        return
    if args.detach:
        command = [settings["python"], str(Path(__file__).absolute()), *[a for a in sys.argv[1:] if a != "--detach"]]
        with (args.out / "controller.log").open("ab") as log:
            process = subprocess.Popen(command, cwd=args.repo, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                       start_new_session=True)
        print(f"Controller PID {process.pid}; log: {args.out / 'controller.log'}")
        return
    controller(args, settings, extension, driver)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        raise SystemExit(f"CoSoCo supplement failed: {error}") from error
