#!/usr/bin/env python3
"""Apply per-run resource limits, then replace this process with a solver."""

from __future__ import annotations

import argparse
import math
import os
import resource
import sys
from typing import Optional, Sequence


def apply_limits(cpu_seconds: float, memory_mb: int) -> None:
    if cpu_seconds > 0:
        # The solver option is the primary CPU limit. RLIMIT_CPU is a fail-safe
        # with room for JVM startup and helper threads.
        soft = max(1, int(math.ceil(max(cpu_seconds + 5, cpu_seconds * 1.25))))
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (soft, soft + 1))
        except (OSError, ValueError):
            pass
    if memory_mb > 0:
        memory = memory_mb * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        except (OSError, ValueError):
            pass


def parse_args(argv: Sequence[str]) -> tuple[float, int, list[str]]:
    try:
        separator = argv.index("--")
    except ValueError as error:
        raise ValueError("missing '--' before the solver command") from error

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-seconds", type=float, required=True)
    parser.add_argument("--memory-mb", type=int, required=True)
    options = parser.parse_args(list(argv[:separator]))
    command = list(argv[separator + 1 :])
    if not command:
        raise ValueError("solver command is empty")
    if options.cpu_seconds < 0 or options.memory_mb < 0:
        raise ValueError("resource limits must be non-negative")
    return options.cpu_seconds, options.memory_mb, command


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        cpu_seconds, memory_mb, command = parse_args(
            sys.argv[1:] if argv is None else argv
        )
        apply_limits(cpu_seconds, memory_mb)
        os.execvp(command[0], command)
    except (OSError, ValueError) as error:
        print(f"fastcop exec error: {error}", file=sys.stderr)
        return 127
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
