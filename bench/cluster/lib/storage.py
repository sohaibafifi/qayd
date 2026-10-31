#!/usr/bin/env python3
"""Place campaign data and logs on shared data storage, preserving repository paths."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[3]
DIRECTORIES = ("data", "bench/cluster/runs")


def configure(data_root: Path) -> dict[str, str]:
    data_root = data_root.expanduser().resolve()
    if data_root == ROOT or ROOT in data_root.parents or data_root in ROOT.parents:
        raise ValueError("data storage must be outside the source checkout")
    if data_root.is_relative_to(Path.home().resolve()):
        raise ValueError("choose shared data storage outside HOME, for example the data-unprotected link")
    planned = []
    for name in DIRECTORIES:
        source, destination = ROOT / name, data_root / name
        if source.is_symlink():
            if source.resolve() != destination:
                raise ValueError(f"already linked to another location: {source}")
            continue
        if source.exists() and not source.is_dir():
            raise ValueError(f"expected a directory: {source}")
        if source.exists() and destination.exists():
            raise ValueError(f"refusing to merge existing directories: {source} and {destination}")
        if destination.exists() and not destination.is_dir():
            raise ValueError(f"expected data directory: {destination}")
        planned.append((source, destination))
    # Validate every path before moving anything. Existing results are preserved.
    for source, destination in planned:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.exists():
            shutil.move(str(source), str(destination))
        else:
            destination.mkdir(parents=True, exist_ok=True)
        source.parent.mkdir(parents=True, exist_ok=True)
        source.symlink_to(destination, target_is_directory=True)
    mapping = {name: str(data_root / name) for name in DIRECTORIES}
    receipt = ROOT / "bench/cluster/runs/storage.json"
    receipt.write_text(json.dumps({"data_root": str(data_root), "paths": mapping}, indent=2) + "\n")
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(configure(args.data_root), indent=2))
    except (ValueError, OSError, shutil.Error) as error:
        raise SystemExit(f"Storage error: {error}") from error


if __name__ == "__main__":
    main()
