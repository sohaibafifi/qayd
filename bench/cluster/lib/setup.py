#!/usr/bin/env python3
"""Submit one compilation and validation job, without starting the main campaign."""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess


ROOT = Path(__file__).resolve().parents[3]


def positive(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--cores", type=positive, required=True)
    result.add_argument("--walltime", required=True)
    result.add_argument("--property", required=True, help="OAR predicate excluding entire GPU hosts")
    result.add_argument("--resource", help="override the OAR resource expression, without walltime")
    result.add_argument("--queue")
    result.add_argument("--scratch-dir", type=Path, help="temporary build storage on the compute node")
    result.add_argument("--lp", "--lp-feature", dest="lp_feature", action="store_true",
                        help="select the private integrations/amthal workspace (LP enabled there by default)")
    result.add_argument("--execute", action="store_true", help="submit the displayed job")
    return result


def submit(args) -> list[str]:
    if not args.property.strip():
        raise ValueError("an explicit CPU-only OAR property is required")
    for path in (ROOT / ".venv/bin/python", ROOT / "bench/cluster/runs/tools/cargo/bin/rustup"):
        if not path.is_file():
            raise ValueError("Frontend preparation is incomplete. Run: "
                             "bash bench/cluster/scripts/bootstrap.oar --prepare")
    logs = ROOT / "bench/cluster/runs/setup"
    logs.mkdir(parents=True, exist_ok=True)
    payload = ["bash", str(ROOT / "bench/cluster/scripts/bootstrap.oar"), "--repo", str(ROOT),
               "--jobs", str(args.cores)]
    if args.lp_feature:
        payload.append("--lp")
    if args.scratch_dir:
        payload += ["--scratch-dir", str(args.scratch_dir)]
    command = ["oarsub", "-n", "qayd-setup", "-t", "idempotent", "-t", "besteffort",
               "-l", f"{args.resource or f'/core={args.cores}'},walltime={args.walltime}",
               "-p", args.property,
               "-O", str(logs / "%jobid%.stdout"), "-E", str(logs / "%jobid%.stderr")]
    if args.queue:
        command += ["-q", args.queue]
    command.append(shlex.join(payload))
    print("One compilation and validation job. The main campaign is not started.", flush=True)
    print(f"lp-relaxation: {'enabled' if args.lp_feature else 'disabled'}", flush=True)
    print(shlex.join(command), flush=True)
    if args.execute:
        subprocess.run(command, cwd=ROOT, check=True)
    return command


def main() -> None:
    args = parser().parse_args()
    try:
        submit(args)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        raise SystemExit(f"Setup error: {error}") from error


if __name__ == "__main__":
    main()
