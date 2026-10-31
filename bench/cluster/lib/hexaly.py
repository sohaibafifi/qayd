#!/usr/bin/env python3
"""Prepare the exact corpora and budgets needed beside Hexaly's public tables."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cluster.lib.campaign import DEFAULT_CONFIG, ROOT, read, sha256_file, write

PAGES = {
    "cvrp": "https://www.hexaly.com/benchmarks/hexaly-vs-gurobi-capacitated-vehicle-routing-problem-cvrp",
    "cvrptw": "https://www.hexaly.com/benchmarks/hexaly-gurobi-or-tools-capacitated-vehicle-routing-problem-with-time-windows-cvrptw",
    "jssp": "https://www.hexaly.com/benchmarks/hexaly-vs-cp-optimizer-vs-or-tools-on-the-job-shop-scheduling-problem-jssp",
    "rcpsp": "https://www.hexaly.com/benchmarks/hexaly-vs-or-tools-on-the-resource-constrained-project-scheduling-problem-rcpsp",
}
def parse_rcp(text: str) -> tuple[list[int], list[tuple[int, list[int], list[int]]]]:
    """RCPLIB Patterson format: whitespace tokens, including wrapped rows."""
    tokens = iter(map(int, text.split()))
    try:
        jobs, resources = next(tokens), next(tokens)
        if jobs <= 0 or resources <= 0:
            raise ValueError("invalid RCP dimensions")
        capacities = [next(tokens) for _ in range(resources)]
        rows = []
        for _ in range(jobs):
            duration = next(tokens)
            demands = [next(tokens) for _ in range(resources)]
            count = next(tokens)
            if count < 0:
                raise ValueError("negative successor count")
            successors = [next(tokens) for _ in range(count)]
            if duration < 0 or any(v < 0 for v in demands) or any(not 1 <= v <= jobs for v in successors):
                raise ValueError("invalid RCP row")
            rows.append((duration, demands, successors))
        if any(v < 0 for v in capacities) or next(tokens, None) is not None:
            raise ValueError("negative capacity or trailing RCP tokens")
    except StopIteration as error:
        raise ValueError("truncated RCP instance") from error
    return capacities, rows


def psplib_text(capacities, rows) -> str:
    labels = " ".join(f"R {i+1}" for i in range(len(capacities)))
    lines = [f"jobs : {len(rows)}", f"horizon : {max(1, sum(row[0] for row in rows))}",
             "PRECEDENCE RELATIONS:", "jobnr. #modes #successors successors"]
    for i, (_, _, successors) in enumerate(rows, 1):
        lines.append(" ".join(map(str, [i, 1, len(successors), *successors])))
    lines += ["REQUESTS/DURATIONS:", "jobnr. mode duration " + labels]
    for i, (duration, demands, _) in enumerate(rows, 1):
        lines.append(" ".join(map(str, [i, 1, duration, *demands])))
    lines += ["RESOURCEAVAILABILITIES:", labels, " ".join(map(str, capacities))]
    return "\n".join(lines) + "\n"


def convert_rg300(directory: Path, destination: Path) -> Path:
    # The independent existing PSPLIB parser provides a structural round trip.
    from qayd.datasets import read_psplib
    corpus = read(directory / "manifest.json")
    references_path = directory / "rg300-references.json"
    references = read(references_path)["instances"]
    sources = sorted((directory / "RG300").glob("*.rcp"))
    if len(sources) != 480:
        raise ValueError(f"RG300 corpus incomplete: {len(sources)}/480")
    entries, assets = [], {}
    for source in sources:
        if sha256_file(source) != corpus["instances"][source.name]:
            raise ValueError(f"RG300 original instance changed: {source}")
        capacities, rows = parse_rcp(source.read_text())
        target = destination / (source.stem + ".sm")
        target.parent.mkdir(parents=True, exist_ok=True)
        text = psplib_text(capacities, rows)
        if target.exists() and target.read_text() != text:
            raise ValueError(f"refusing to replace a different normalization: {target}")
        target.write_text(text)
        parsed = read_psplib(target)
        if tuple(capacities) != tuple(parsed.capacities) or len(rows) != len(parsed.jobs):
            raise ValueError(f"RCP normalization changed dimensions: {source}")
        for row, job in zip(rows, parsed.jobs):
            mode = job.modes[0]
            if (row[0] != mode.duration or tuple(row[1]) != tuple(mode.demands)
                    or tuple(row[2]) != tuple(job.successors)):
                raise ValueError(f"RCP normalization changed a task: {source}")
        paths = [source, target, references_path, directory / "manifest.json"]
        for path in paths:
            assets[path.relative_to(ROOT).as_posix()] = sha256_file(path)
        entries.append({"path": target.relative_to(ROOT).as_posix(), "data": [],
                        "family": "RG300", "sense": "min", "outputs": [],
                        "dependencies": [p.relative_to(ROOT).as_posix() for p in paths if p != target],
                        "reference_bks": references[source.stem]["best_known_makespan"],
                        "reference_bks_kind": "RCPLIB v2.4 UB-lit, not a pinned Hexaly BKS snapshot"})
    catalog = destination / "catalog.json"
    write(catalog, {"instances": entries, "assets": assets, "normalization": "RCP to single-mode PSPLIB; structural round trip"})
    return catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new directory under bench/cluster/runs")
    parser.add_argument("--rg300", type=Path, default=ROOT / "data/benchmarks/scheduling/rcplib/v2.4")
    parser.add_argument("--track", choices=("cvrp", "jssp", "rcpsp"),
                        help="prepare one independent published-table comparison")
    parser.add_argument("--snapshot-pages", action="store_true")
    args = parser.parse_args()
    directory = args.out.resolve()
    if directory.exists():
        raise SystemExit("choose a new --out directory")
    # Preserve the repository alias when instances live on shared data storage.
    config = read(DEFAULT_CONFIG)
    config["name"] = "hexaly-published-quality" + (f"-{args.track}" if args.track else "")
    tracks = {args.track} if args.track else {"cvrp", "jssp", "rcpsp"}
    config["datasets"] = [d for d in config["datasets"] if d["track"] == "cvrp" and "cvrp" in tracks]
    if "jssp" in tracks:
        config["datasets"].append({
            "track": "jssp", "family": "large-ta-1000x1000", "glob": "data/jssp/large-ta/tai_j1000_m1000_*.data",
            "expected_count": 10, "manifest": "bench/cluster/config/hexaly-jssp-1000x1000.json"})
    if "rcpsp" in tracks:
        catalog = convert_rg300(args.rg300.absolute(), ROOT / "data/cluster-hexaly/rg300")
        config["datasets"].append({"track": "rcpsp", "family": "rg300",
                                   "catalog": catalog.relative_to(ROOT).as_posix(), "expected_count": 480})
    # These runs compare Qayd's verified quality with fixed published values.
    # A direct Hexaly installation is not required for this arm.
    for track in ("cvrp", "jssp", "rcpsp"):
        config["solvers"][track] = ["qayd-native"]
        config["protocols"][track] = {"seconds": 600 if track == "jssp" else 60,
                                       "seeds": list(range(10)), "reference": PAGES[track]}
    config["published_reference"] = {
        "version": "Hexaly 15.0", "hardware": "AMD Ryzen 7 7700, 8 cores, 3.8 GHz, 32 GB RAM",
        "threads": None, "seeds": None, "pages": {args.track: PAGES[args.track]} if args.track else PAGES,
        "vrptw_status": "not matched: fleet-first/trunc1 differs from the published distance objective",
        "rcpsp_bks_status": "RCPLIB v2.4 snapshot; equality to Hexaly's BKS snapshot is not established",
        "jssp_incumbents": read(ROOT / "bench/cluster/config/hexaly-jssp-1000x1000.json")["reference_protocol"]["hexaly"]["reference_incumbent_10m"],
    }
    directory.mkdir(parents=True)
    write(directory / "config.json", config)
    if args.snapshot_pages:
        snapshots = {}
        for track, url in config["published_reference"]["pages"].items():
            path = directory / (track + ".html")
            subprocess.run(["curl", "--fail", "--location", "--retry", "3", "--output", str(path), url], check=True)
            snapshots[track] = {"url": url, "sha256": sha256_file(path)}
        write(directory / "sources.json", snapshots)
    print(directory / "config.json")


if __name__ == "__main__":
    main()
