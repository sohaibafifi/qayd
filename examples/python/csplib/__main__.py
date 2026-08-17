"""List or run CSPLib models implemented with Qayd."""

from __future__ import annotations

import importlib
import sys

from .catalog import (
    ALL_PROBLEM_IDS,
    IMPLEMENTATIONS,
    coverage_counts,
    normalize_problem_id,
)


def _print_coverage() -> None:
    complete, partial, total = coverage_counts()
    print(f"CSPLib coverage: {complete} complete, {partial} partial, {total} total")
    for problem_id in ALL_PROBLEM_IDS:
        implementation = IMPLEMENTATIONS.get(problem_id)
        if implementation is None:
            print(f"{problem_id}  pending")
            continue
        suffix = f" ({implementation.scope})" if implementation.scope else ""
        print(
            f"{problem_id}  {implementation.status:8}  {implementation.title}{suffix}"
        )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"list", "--list", "-l"}:
        _print_coverage()
        return 0

    try:
        problem_id = normalize_problem_id(args.pop(0))
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2

    implementation = IMPLEMENTATIONS.get(problem_id)
    if implementation is None:
        print(f"{problem_id} is not implemented yet", file=sys.stderr)
        return 2

    module = importlib.import_module(implementation.module)
    return int(module.main(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
