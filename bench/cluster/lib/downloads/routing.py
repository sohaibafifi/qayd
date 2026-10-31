#!/usr/bin/env python3
"""Fetch and build the optional open routing baselines.

HGS-CVRP is MIT licensed.  LKH-3 is restricted to academic and non-commercial
use, so downloading it requires an explicit acknowledgement flag.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen


HERE = Path(__file__).resolve().parents[4] / "bench/cluster/runs/tools/solvers"
HGS_REPOSITORY = "https://github.com/vidalt/HGS-CVRP.git"
HGS_REF = "v2.0.0"
LKH_URL = "https://webhotel4.ruc.dk/~keld/research/LKH-3/LKH-3.0.13.tgz"
# Pin the archived tarball, rather than a fork's current source tree. This is a
# project checksum of the mirror, not a publisher-authenticated checksum.
LKH_MIRROR_URL = (
    "https://raw.githubusercontent.com/yuhengli01/vitsp/"
    "58bb0e763d425a5d7fae05d6a5bff34e97ffcccf/LKH-3.0.13.tgz"
)
LKH_SHA256 = "8e4b70064e963e1bd24d44a6f9492fe793323ad4f0f063a6652d697cf28ad15d"
LKH_NAME = "LKH-3.0.13"


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


def check_lkh_sources(destination: Path) -> None:
    required = ("README.txt", "Makefile", "SRC/Makefile", "SRC/LKHmain.c", "SRC/INCLUDE/LKH.h")
    if any(not (destination / name).is_file() for name in required):
        raise ValueError(f"Incomplete LKH sources at {destination}; move this directory aside and retry")
    if "Version 3.0.13" not in (destination / "README.txt").read_text():
        raise ValueError(f"Expected LKH 3.0.13 sources at {destination}")


def prepare_lkh() -> Path:
    """Cache and unpack LKH without compiling on the cluster frontend."""
    destination = HERE / LKH_NAME
    if destination.exists():
        check_lkh_sources(destination)
        return destination
    downloads = HERE.parent / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    cached = downloads / f"{LKH_NAME}.tgz"
    download_url = None
    if cached.exists():
        digest = hashlib.sha256(cached.read_bytes()).hexdigest()
        if digest != LKH_SHA256:
            raise ValueError(f"LKH archive checksum mismatch at {cached}; move it aside and retry")
    else:
        errors = []
        for url in (LKH_URL, LKH_MIRROR_URL):
            print(f"Download LKH: {url}", flush=True)
            try:
                request = Request(url, headers={"User-Agent": "qayd-benchmark/1"})
                with urlopen(request, timeout=20) as response:
                    data = response.read()
                if hashlib.sha256(data).hexdigest() != LKH_SHA256:
                    raise ValueError("archive checksum mismatch")
            except (OSError, ValueError) as error:
                errors.append(f"{url}: {error}")
                print(f"LKH source unavailable: {error}", flush=True)
                continue
            with TemporaryDirectory(prefix="lkh-download-", dir=downloads) as temporary:
                part = Path(temporary) / cached.name
                part.write_bytes(data)
                part.replace(cached)
            download_url = url
            break
        else:
            raise ValueError("Cannot download pinned LKH archive: " + "; ".join(errors))
    HERE.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="lkh-extract-", dir=HERE) as temporary:
        with tarfile.open(cached, mode="r:gz") as archive:
            for member in archive.getmembers():
                path = PurePosixPath(member.name)
                if (path.is_absolute() or ".." in path.parts or not path.parts
                        or path.parts[0] != LKH_NAME or not (member.isfile() or member.isdir())):
                    raise ValueError(f"Unexpected LKH archive entry: {member.name}")
            archive.extractall(temporary, filter="data")
        staged = Path(temporary) / LKH_NAME
        check_lkh_sources(staged)
        (staged / "qayd-source.json").write_text(json.dumps({
            "version": "3.0.13",
            "archive_sha256": LKH_SHA256,
            "download_url": download_url,
            "pinned_mirror_url": LKH_MIRROR_URL,
            "publisher_checksum_verified": False,
        }, indent=2) + "\n")
        staged.rename(destination)
    print(f"LKH sources ready: {destination}")
    return destination


def fetch_lkh(jobs: int) -> None:
    destination = prepare_lkh()
    # The top-level Makefile launches an unbounded nested `make -j`.
    command(["make", "-j", str(jobs)], destination / "SRC")
    print(f"LKH-3: {destination / 'LKH'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hgs", action="store_true")
    parser.add_argument("--lkh", action="store_true")
    parser.add_argument("--solver", action="append", choices=("hgs", "lkh"), default=[])
    parser.add_argument("--accept-lkh-academic-license", action="store_true")
    parser.add_argument("--download-only", action="store_true", help="prepare LKH sources without compiling")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    selected = set(args.solver)
    if args.hgs:
        selected.add("hgs")
    if args.lkh:
        selected.add("lkh")
    if not selected:
        raise SystemExit("select --solver hgs and/or --solver lkh")
    if args.jobs <= 0:
        raise SystemExit("jobs must be positive")
    if args.download_only and selected != {"lkh"}:
        raise SystemExit("--download-only requires selecting only LKH")
    if "lkh" in selected and not args.accept_lkh_academic_license:
        raise SystemExit("LKH-3 requires --accept-lkh-academic-license after reviewing its terms")
    if "hgs" in selected:
        fetch_hgs(args.jobs)
    if "lkh" in selected:
        if args.download_only:
            prepare_lkh()
        else:
            fetch_lkh(args.jobs)


if __name__ == "__main__":
    main()
