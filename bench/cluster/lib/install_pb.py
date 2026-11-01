#!/usr/bin/env python3
"""Prepare pinned PB baselines; compile NaPS offline inside the build allocation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DIRECTORY = ROOT / "bench/cluster/runs/tools/solvers/pb"
PYSCIPOPT_VERSION = "6.0.0"
SOURCES = {
    "naps": {
        "filename": "naps-1.03a6.tar.gz", "version": "1.03a6",
        "url": "https://gitlab.com/api/v4/projects/84546807/repository/archive.tar.gz?sha=f6af1531b9ce436de9aecb23231a63bc58d043e9",
        "sha256": "bd11a8361e9c391535d415bafe292cc4f2576f8ab2351a6d662c857a420c1e25",
    },
    "gmp": {
        "filename": "gmp-6.3.0.tar.xz", "version": "6.3.0",
        "url": "https://ftp.gnu.org/gnu/gmp/gmp-6.3.0.tar.xz",
        "sha256": "a3c2b80201b89e68616f4ad30bc66aee4927c3ce50e33929ca819d5c43538898",
    },
    "zlib": {
        "filename": "zlib-1.3.1.tar.gz", "version": "1.3.1",
        "url": "https://zlib.net/fossils/zlib-1.3.1.tar.gz",
        "sha256": "9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23",
    },
    "uwrmaxsat-scip": {
        "filename": "solver-UWrMaxSAT-2026.zip", "version": "MSE2026",
        "url": "https://zenodo.org/records/21358014/files/solver-UWrMaxSAT-2026.zip?download=1",
        "sha256": "dd0b700121c2d642fac343b12957868ae9a84a8773c826bbd2492896ecdc58aa",
    },
}
INSTALLATIONS = {
    "scip-naps": "scip-naps-1.00a2-naps-1.03a6",
    "uwrmaxsat-scip": "uwrmaxsat-scip-mse2026",
}
SOURCE_KEYS = {"scip-naps": ("naps", "gmp", "zlib"), "uwrmaxsat-scip": ("uwrmaxsat-scip",)}
UWR_BINARY_SHA256 = "706a32b0e43d5f2300487e799882ebbc4858a1dfd1b32fd262b9fc5dd3335283"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sources(solver: str) -> dict:
    return {key: SOURCES[key] for key in SOURCE_KEYS[solver]}


def cached(directory: Path, key: str) -> Path:
    spec = SOURCES[key]
    path = directory / "downloads" / spec["filename"]
    if not path.is_file() or sha(path) != spec["sha256"]:
        raise ValueError(f"missing or changed PB source: {path}; run prepare tools")
    return path


def verify(directory: Path, solver: str) -> dict[str, Path]:
    destination = directory / INSTALLATIONS[solver]
    try:
        receipt = json.loads((destination / "install.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{solver} is not built; run prepare tools, then prepare build") from error
    if receipt.get("schema") != "qayd.cluster.pb-install/v1" or receipt.get("sources") != sources(solver):
        raise ValueError(f"{solver} installation differs from the pinned sources")
    expected = receipt.get("files")
    actual = {path.relative_to(destination).as_posix(): sha(path)
              for path in (destination / "runtime").rglob("*") if path.is_file()}
    if not expected or actual != expected:
        raise ValueError(f"{solver} runtime differs from its installation receipt")
    for key in SOURCE_KEYS[solver]:
        cached(directory, key)
    if solver == "uwrmaxsat-scip":
        binaries = {solver: destination / "runtime/uwrmaxsat"}
        if sha(binaries[solver]) != UWR_BINARY_SHA256:
            raise ValueError("UWrMaxSat executable differs from the official evaluation binary")
    else:
        binaries = {solver: destination / "runtime/scip-naps.py", "naps": destination / "runtime/naps"}
    for path in binaries.values():
        if not path.is_file() or not os.access(path, os.X_OK):
            raise ValueError(f"PB executable is missing or not executable: {path}")
    return binaries


def artifacts(directory: Path, solver: str) -> list[Path]:
    verify(directory, solver)
    destination = directory / INSTALLATIONS[solver]
    return [destination / "install.json", *sorted(path for path in (destination / "runtime").rglob("*")
                                                if path.is_file())]


def publish(directory: Path, solver: str, staging: Path, **build) -> None:
    destination = directory / INSTALLATIONS[solver]
    files = {path.relative_to(staging).as_posix(): sha(path)
             for path in (staging / "runtime").rglob("*") if path.is_file()}
    (staging / "install.json").write_text(json.dumps({
        "schema": "qayd.cluster.pb-install/v1", "solver": solver,
        "sources": sources(solver), "files": files, "build": build,
    }, indent=2, sort_keys=True) + "\n")
    # Never replace a runtime that another frozen campaign might be using.
    if destination.exists():
        raise ValueError(f"refusing to replace PB installation: {destination}")
    staging.rename(destination)
    verify(directory, solver)


def prepare(directory: Path) -> None:
    downloads = directory / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    for key, spec in SOURCES.items():
        target = downloads / spec["filename"]
        if not target.exists():
            with tempfile.NamedTemporaryFile(dir=downloads, delete=False) as output:
                temporary = Path(output.name)
                try:
                    with urllib.request.urlopen(spec["url"], timeout=180) as response:
                        shutil.copyfileobj(response, output)
                    output.flush()
                    if sha(temporary) != spec["sha256"]:
                        raise ValueError(f"PB download failed SHA-256 verification: {key}")
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
        cached(directory, key)
    solver = "uwrmaxsat-scip"
    if (directory / INSTALLATIONS[solver]).exists():
        verify(directory, solver)
        return
    with tempfile.TemporaryDirectory(prefix=".uwrmaxsat-", dir=directory) as temporary:
        staging = Path(temporary) / "install"
        runtime = staging / "runtime"
        runtime.mkdir(parents=True)
        with zipfile.ZipFile(cached(directory, solver)) as stream:
            for source, target in (("bin/uwrmaxsat", "uwrmaxsat"),
                                   ("code/uwrmaxsat/LICENSE", "LICENSE"),
                                   ("code/uwrmaxsat/README.md", "README.md")):
                (runtime / target).write_bytes(stream.read("UWrMaxSAT-2026/" + source))
        (runtime / "uwrmaxsat").chmod(0o755)
        publish(directory, solver, staging, mode="official-static-binary")


def unpack(archive: Path, destination: Path) -> Path:
    destination.mkdir()
    with tarfile.open(archive) as stream:
        for member in stream.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("unsafe path in PB source archive")
        stream.extractall(destination, filter="data")
    roots = list(destination.iterdir())
    if len(roots) != 1 or not roots[0].is_dir():
        raise ValueError("PB source archive must contain one root directory")
    return roots[0]


def build(directory: Path, scratch: Path, jobs: int) -> None:
    solver = "scip-naps"
    if (directory / INSTALLATIONS[solver]).exists():
        verify(directory, solver)
        return
    if not os.environ.get("OAR_JOB_ID"):
        raise ValueError("NaPS compilation requires an OAR allocation")
    for key in SOURCE_KEYS[solver]:
        cached(directory, key)
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="qayd-pb-build-", dir=scratch) as temporary:
        work = Path(temporary)
        prefix = work / "dependencies"
        env = {**os.environ, "CFLAGS": "-O3 -DNDEBUG", "CXXFLAGS": "-O3 -DNDEBUG", "MAKEFLAGS": "", "MFLAGS": ""}

        def run(argv: list[str], cwd: Path) -> None:
            subprocess.run(argv, cwd=cwd, env=env, check=True)

        gmp = unpack(cached(directory, "gmp"), work / "gmp")
        run(["./configure", f"--prefix={prefix}", "--disable-shared", "--enable-static",
             "--host=x86_64-pc-linux-gnu", "--build=x86_64-pc-linux-gnu"], gmp)
        run(["make", "-j", str(jobs)], gmp)
        run(["make", "install"], gmp)
        zlib = unpack(cached(directory, "zlib"), work / "zlib")
        run(["./configure", "--static", f"--prefix={prefix}"], zlib)
        run(["make", "-j", str(jobs)], zlib)
        run(["make", "install"], zlib)
        naps = unpack(cached(directory, "naps"), work / "naps")
        # NaPS has its own target-specific CFLAGS. Exporting CFLAGS here would
        # leak its forced Main.h include into the bundled SAT solvers' makefiles.
        env.pop("CFLAGS", None)
        env.pop("CXXFLAGS", None)
        env["CPATH"] = str(prefix / "include")
        command = ["make", "r", "CXX=c++", "COPTIMIZE=-O3 -fomit-frame-pointer",
                   f"LDFLAGS={prefix / 'lib/libz.a'} {prefix / 'lib/libgmp.a'} -Wall"]
        # SCIP-NaPS is the official Python wrapper, followed by this NaPS backend.
        # Its original Makefile has repeated sub-make recipes for shared object
        # files. Keep this phase serial; GMP and zlib use the allocated jobs.
        run(command, naps)
        with tempfile.TemporaryDirectory(prefix=".scip-naps-", dir=directory) as publishing:
            staging = Path(publishing) / "install"
            runtime = staging / "runtime"
            runtime.mkdir(parents=True)
            for source, target in ((naps / "naps_release", "naps"), (naps / "scip-naps.py", "scip-naps.py"),
                                   (naps / "LICENSE", "LICENSE")):
                shutil.copy2(source, runtime / target)
            for name in ("naps", "scip-naps.py"):
                (runtime / name).chmod(0o755)
            publish(directory, solver, staging, mode="Release", command=command,
                    compiler=subprocess.check_output(["c++", "--version"], text=True).splitlines()[0],
                    pyscipopt=PYSCIPOPT_VERSION)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--build", action="store_true", help="build cached sources offline inside OAR")
    modes.add_argument("--verify-only", action="store_true")
    parser.add_argument("--scratch-dir", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")))
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    try:
        if platform.system() != "Linux" or platform.machine() != "x86_64":
            raise ValueError("the pinned PB baselines require Linux x86_64")
        if args.jobs <= 0 or args.jobs > len(os.sched_getaffinity(0)):
            raise ValueError("PB build jobs must be positive and fit the allocated CPU affinity")
        directory = args.directory.resolve()
        if args.build:
            build(directory, args.scratch_dir.resolve(), args.jobs)
        elif not args.verify_only:
            prepare(directory)
        for solver in INSTALLATIONS:
            if args.build or args.verify_only or solver == "uwrmaxsat-scip":
                print(solver, {key: str(path) for key, path in verify(directory, solver).items()})
    except (ValueError, OSError, subprocess.SubprocessError, tarfile.TarError, zipfile.BadZipFile) as error:
        raise SystemExit(f"PB preparation failed: {error}") from error


if __name__ == "__main__":
    main()
