#!/usr/bin/env python3
"""Fetch the declared corpus explicitly, including original MiniZinc models."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cluster.lib.campaign import DEFAULT_CONFIG, ROOT, read, sha256_file, write
from cluster.lib.minizinc import interface


def corpus_commands(config: dict) -> list[list[str]]:
    tracks = {dataset["track"] for dataset in config["datasets"]}
    commands = []
    if "sat" in tracks:
        raise ValueError("SAT downloads are deferred in this campaign")
    # XCSP and PB are fetched against the official member hashes by corpus.py.
    if tracks & {"cvrp", "cvrptw", "jssp", "rcpsp"}:
        commands.append(["bench/cluster/lib/downloads/fetch_collections.py", "--collection", "all"])
    return commands


def fetch_minizinc(year: int, executable: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / f"mznc{year}_probs.tar.gz"
    url = f"https://www.minizinc.org/challenge/{year}/mznc{year}_probs.tar.gz"
    if not archive.exists():
        partial = archive.with_suffix(".part")
        subprocess.run(["curl", "--fail", "--location", "--retry", "3", "--output", str(partial), url], check=True)
        partial.rename(archive)
    sources = destination / "sources"
    if not sources.exists():
        sources.mkdir()
        with tarfile.open(archive) as stream:
            stream.extractall(sources, filter="data")
    models = defaultdict(list)
    for path in sorted(sources.rglob("*.mzn")):
        if not path.name.endswith(".mzc.mzn"):
            models[path.parent].append(path)
    entries, skipped, normalizations = [], [], []
    for parent, candidates in models.items():
        candidates = [p for p in candidates if re.search(r'\bsolve\b', re.sub(
            r'%[^\n]*|/\*[\s\S]*?\*/', '', p.read_text(errors='replace')))]
        data_files = sorted([*parent.rglob("*.dzn"), *parent.rglob("*.json")])
        # Do not guess which alternative model belongs to a data file.
        if data_files and len(candidates) != 1:
            skipped.append({"directory": str(parent.relative_to(ROOT)), "reason": "ambiguous-model",
                            "models": [p.name for p in candidates]})
            continue
        pairs = [(candidates[0], [data]) for data in data_files] if data_files else [(model, []) for model in candidates]
        for model, data in pairs:
            if year == 2024 and parent.name == "train-scheduling":
                text = model.read_text()
                # The official archive has a tuple interpolation syntax error
                # in an assertion's diagnostic string. Preserve the original;
                # normalize just its parentheses for every compared solver.
                needle = r'\(s1,s2)'
                if text.count(needle) == 1:
                    normalized = destination / "normalized/train-scheduling/trains.mzn"
                    normalized.parent.mkdir(parents=True, exist_ok=True)
                    normalized.write_text(text.replace(needle, r'\((s1,s2))'))
                    normalizations.append({"original": model.relative_to(ROOT).as_posix(),
                                           "normalized": normalized.relative_to(ROOT).as_posix(),
                                           "change": "parenthesize tuple in assertion diagnostic interpolation"})
                    model = normalized
            try:
                info = interface(executable, model, data, 120)
            except (ValueError, subprocess.SubprocessError) as error:
                skipped.append({"model": str(model.relative_to(ROOT)),
                                "data": [p.relative_to(ROOT).as_posix() for p in data], "reason": str(error)})
                continue
            entries.append({"path": model.relative_to(ROOT).as_posix(),
                            "data": [p.relative_to(ROOT).as_posix() for p in data],
                            "family": parent.name, "sense": {"sat": "satisfy"}.get(info["method"], info["method"]),
                            "outputs": sorted(info["output"]),
                            "dependencies": [p.relative_to(ROOT).as_posix() for p in sources.rglob("*")
                                             if p.is_file() and p.suffix not in (".dzn", ".json")]})
    assets = {path.relative_to(ROOT).as_posix(): sha256_file(path)
              for path in sorted(sources.rglob("*")) if path.is_file()}
    assets.update({p.relative_to(ROOT).as_posix(): sha256_file(p)
                   for p in (destination / "normalized").rglob("*") if p.is_file()})
    write(destination / "catalog.json", {"source": url, "archive_sha256": sha256_file(archive),
          "minizinc_version": subprocess.check_output([executable, "--version"], text=True).strip(),
          "instances": entries, "assets": assets, "skipped": skipped, "normalizations": normalizations})
    print(f"MiniZinc {year}: {len(entries)} model/data pairs; {len(skipped)} require review")
    if skipped:
        raise SystemExit("Unresolved MiniZinc models are recorded in catalog.json; review before qualification")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="download all datasets in the default campaign, possibly many GB")
    parser.add_argument("--minizinc-year", type=int, action="append", default=[])
    parser.add_argument("--minizinc", default="minizinc")
    args = parser.parse_args()
    years = [2024, 2025] if args.all else args.minizinc_year
    if not args.all and not years:
        parser.error("select --all or --minizinc-year")
    if args.all:
        config = read(DEFAULT_CONFIG)
        for command in corpus_commands(config):
            subprocess.run([sys.executable, *command], cwd=ROOT, check=True)
    executable = shutil.which(args.minizinc)
    if executable is None:
        executable = next((str(p) for p in (ROOT / "bench/cluster/runs/tools").glob(
            "MiniZincIDE-*-bundle-linux-x86_64/bin/minizinc") if p.is_file()), None)
    if executable is None:
        raise SystemExit("MiniZinc executable not found")
    for year in years:
        fetch_minizinc(year, executable, ROOT / "data/minizinc" / str(year))


if __name__ == "__main__":
    main()
