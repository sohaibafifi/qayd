"""Prepare, run, monitor and collect a resumable cluster campaign."""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import subprocess
import sys

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))


def parser():
    result = argparse.ArgumentParser(description=__doc__, prog="python bench/cluster")
    actions = result.add_subparsers(dest="action", required=True)
    preparation = actions.add_parser("prepare", help="prepare tools, build, datasets or storage")
    steps = preparation.add_subparsers(dest="step", required=True)
    tools = steps.add_parser("tools", help="download the tools on the frontend")
    tools.add_argument("--repo", type=Path, default=BASE.parents[1])
    tools.add_argument("--jobs", type=int, default=1)
    tools.add_argument("--lp", action="store_true", help="fetch dependencies for the private Amthal workspace")
    for name, target, prefix, help_text in (
        ("build", "lib.setup", [], "submit one validation and build job; --lp selects the private workspace"),
        ("corpus", "lib.corpus", ["--download"], "download and verify the declared competition corpus"),
        ("inventory", "lib.campaign", ["prepare", "--inventory", "--stage", "full"], "inspect corpus coverage and budgets"),
        ("storage", "lib.storage", [], "configure shared storage for datasets and results"),
    ):
        command = steps.add_parser(name, add_help=False, help=help_text)
        command.set_defaults(target=target, prefix=prefix)
    run = actions.add_parser("run", add_help=False, help="run or resume the bounded OAR controller")
    run.add_argument("--add-cosoco", action="store_true", help="add a CoSoCo-only FAST COP supplement")
    run.set_defaults(target="lib.controller", prefix=[])
    for name, target, help_text in (
        ("status", "status", "show live progress, results and remaining budgets"),
        ("collect", "collect", "audit and export results across profiles to CSV and JSONL"),
    ):
        command = actions.add_parser(name, add_help=False, help=help_text)
        command.set_defaults(target=target, prefix=[])
    return result


def main(argv=None):
    args, remainder = parser().parse_known_args(argv)
    if args.action == "prepare" and args.step == "tools":
        if remainder:
            parser().error("unrecognized tools arguments: " + " ".join(remainder))
        if args.jobs < 1:
            parser().error("--jobs must be positive")
        command = ["bash", str(BASE / "scripts/bootstrap.oar"), "--repo", str(args.repo),
                   "--prepare", "--jobs", str(args.jobs)]
        if args.lp:
            command.append("--lp")
        subprocess.run(command, check=True)
        return
    target = "lib.cosoco" if getattr(args, "add_cosoco", False) else args.target
    module = importlib.import_module("cluster." + target)
    # Existing workers freeze and re-execute their own module. Preserve their
    # argument contract without adding a second controller or shell wrapper.
    previous = sys.argv
    try:
        sys.argv = [module.__file__, *args.prefix, *remainder]
        module.main()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        raise SystemExit(f"Cluster command failed: {error}") from error
