#!/usr/bin/env python3
"""Fetch and build the optional open routing baselines.

HGS-CVRP is MIT licensed.  LKH-3 is restricted to academic and non-commercial
use, so downloading it requires an explicit acknowledgement flag.
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
from urllib.request import Request, urlopen


HERE = Path(__file__).resolve().parent
HGS_REPOSITORY = "https://github.com/vidalt/HGS-CVRP.git"
HGS_REF = "v2.0.0"
LKH_URL = "https://webhotel4.ruc.dk/~keld/research/LKH-3/LKH-3.0.13.tgz"


def command(argv: list[str], cwd: Path) -> None:
    print("+", " ".join(argv))
    subprocess.run(argv, cwd=cwd, check=True)


def fetch_hgs(jobs: int) -> None:
    destination = HERE / "HGS-CVRP"
    if not destination.exists():
        command(["git", "clone", "--branch", HGS_REF, "--depth", "1", HGS_REPOSITORY, str(destination)], HERE)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=destination, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    expected = subprocess.run(
        ["git", "rev-list", "-n", "1", HGS_REF], cwd=destination, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    if head != expected:
        raise SystemExit(f"existing HGS-CVRP checkout is {head}, expected {HGS_REF} ({expected})")
    build = destination / "build"
    build.mkdir(exist_ok=True)
    command(["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"], build)
    command(["cmake", "--build", ".", "--target", "bin", "--parallel", str(jobs)], build)
    print(f"HGS-CVRP: {build / 'hgs'}")


def fetch_lkh(jobs: int) -> None:
    destination = HERE / "LKH-3.0.13"
    if not destination.exists():
        request = Request(LKH_URL, headers={"User-Agent": "qayd-benchmark/1"})
        with urlopen(request, timeout=120) as response:
            data = response.read()
        with tarfile.open(fileobj=BytesIO(data), mode="r:gz") as archive:
            for member in archive.getmembers():
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts or (not member.isfile() and not member.isdir()):
                    continue
                relative = Path(*path.parts[1:])
                if not relative.parts:
                    continue
                target = destination / relative
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                source = archive.extractfile(member)
                if source is None:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
    command(["make", "-j", str(jobs)], destination)
    print(f"LKH-3: {destination / 'LKH'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hgs", action="store_true")
    parser.add_argument("--lkh", action="store_true")
    parser.add_argument("--solver", action="append", choices=("hgs", "lkh"), default=[])
    parser.add_argument("--accept-lkh-academic-license", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    selected = set(args.solver)
    if args.hgs:
        selected.add("hgs")
    if args.lkh:
        selected.add("lkh")
    if not selected:
        raise SystemExit("select --solver hgs and/or --solver lkh")
    if args.jobs <= 0:
        raise SystemExit("jobs must be positive")
    if "lkh" in selected and not args.accept_lkh_academic_license:
        raise SystemExit("LKH-3 requires --accept-lkh-academic-license after reviewing its terms")
    if "hgs" in selected:
        fetch_hgs(args.jobs)
    if "lkh" in selected:
        fetch_lkh(args.jobs)


if __name__ == "__main__":
    main()
