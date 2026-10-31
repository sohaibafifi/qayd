#!/usr/bin/env python3
"""Install or verify the pinned official CoSoCo executable for Linux cluster nodes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import tempfile
import urllib.request
import zipfile


VERSION = "2.6.1"
WHEEL = "cosoco-2.6.1-py3-none-manylinux_2_17_x86_64.whl"
URL = ("https://files.pythonhosted.org/packages/f9/f5/"
       "f49f309a37e0034de075f4847b5d6e082a1638453658f6898e1459226dd5/" + WHEEL)
SHA256 = "be829cf12d71f1be83bec6e4fdb7d9fce03d3cee620b47206e2861b9954d6ec5"
ROOT = Path(__file__).resolve().parents[3]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(destination: Path) -> Path:
    receipt_path = destination / "download.json"
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("CoSoCo is not prepared; run python bench/cluster prepare tools") from error
    if any(receipt.get(key) != expected for key, expected in
           (("version", VERSION), ("url", URL), ("wheel_sha256", SHA256), ("wheel", WHEEL))):
        raise ValueError("CoSoCo receipt differs from the pinned official wheel")
    expected_files = receipt.get("files")
    if not isinstance(expected_files, dict) or not expected_files:
        raise ValueError("CoSoCo receipt has no shipped-file hashes")
    archive = destination / WHEEL
    if not archive.is_file() or sha(archive) != SHA256:
        raise ValueError("CoSoCo cached wheel is missing or its hash differs from the pin")
    with zipfile.ZipFile(archive) as stream:
        pinned_files = {"runtime/" + member.filename: hashlib.sha256(stream.read(member)).hexdigest()
                        for member in stream.infolist() if not member.is_dir()}
    if expected_files != pinned_files:
        raise ValueError("CoSoCo receipt files differ from the pinned wheel")
    actual_files = {path.relative_to(destination).as_posix(): sha(path)
                    for path in (destination / "runtime").rglob("*") if path.is_file()}
    if actual_files != expected_files:
        raise ValueError("CoSoCo runtime files differ from the download receipt")
    binary = destination / "runtime/cosoco_bin/cosoco"
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError("CoSoCo executable is missing or not executable")
    return binary


def install(destination: Path, *, verify_only: bool = False) -> Path:
    destination = destination.resolve()
    if verify_only or (destination / "download.json").exists():
        return verify(destination)
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("the pinned cluster CoSoCo wheel requires Linux x86_64")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("refusing to replace an unreceipted CoSoCo installation")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Finish and verify in a sibling temporary directory before publishing it.
    with tempfile.TemporaryDirectory(prefix=".cosoco-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        archive = staging / WHEEL
        with urllib.request.urlopen(URL, timeout=120) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
        if sha(archive) != SHA256:
            raise ValueError("downloaded CoSoCo wheel failed SHA-256 verification")
        runtime = staging / "runtime"
        runtime.mkdir()
        with zipfile.ZipFile(archive) as stream:
            for member in stream.infolist():
                path = PurePosixPath(member.filename)
                if path.is_absolute() or ".." in path.parts or "\\" in member.filename:
                    raise ValueError("unsafe path in CoSoCo wheel")
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("symlinks are not accepted in the CoSoCo wheel")
            stream.extractall(runtime)
        binary = runtime / "cosoco_bin/cosoco"
        if not binary.is_file():
            raise ValueError("official CoSoCo wheel has no expected executable")
        binary.chmod(0o755)
        files = {path.relative_to(staging).as_posix(): sha(path)
                 for path in runtime.rglob("*") if path.is_file()}
        (staging / "download.json").write_text(json.dumps({
            "schema": "qayd.cluster.cosoco-install/v1", "version": VERSION,
            "source": f"https://pypi.org/project/cosoco/{VERSION}/", "url": URL,
            "wheel": WHEEL, "wheel_sha256": SHA256, "files": files,
        }, indent=2, sort_keys=True) + "\n")
        verify(staging)
        if destination.exists():
            destination.rmdir()
        staging.rename(destination)
    return verify(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path,
                        default=ROOT / f"bench/cluster/runs/tools/cosoco-{VERSION}")
    parser.add_argument("--verify-only", action="store_true", help="check cached files without network access")
    args = parser.parse_args()
    try:
        print(install(args.destination, verify_only=args.verify_only))
    except (ValueError, OSError, zipfile.BadZipFile) as error:
        raise SystemExit(f"CoSoCo preparation failed: {error}") from error


if __name__ == "__main__":
    main()
