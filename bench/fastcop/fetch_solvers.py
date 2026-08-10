#!/usr/bin/env python3
"""Fetch the exact ACE and Choco binaries used by the FAST COP harness."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import ssl
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


WHEEL_URL = (
    "https://files.pythonhosted.org/packages/94/cb/"
    "7f10b909decf1ae1918a850715017daae8c39b3ff0a03c0daa4ca6fda2f0/"
    "pycsp3-2.5-py3-none-any.whl"
)
WHEEL_SHA256 = "602a1ed893e0485cce28c5d5125a79b70f0932e9412e035c32e912ce9d736560"
ACE_MEMBER = "pycsp3/solvers/ace/ACE-2.5.jar"
ACE_SHA256 = "cb0d0741d7f626c5166371efa2ce8cbe5dc62c49f10180e3ebf4be008106dcc7"
CHOCO_URL = (
    "https://repo1.maven.org/maven2/org/choco-solver/choco-parsers/"
    "5.0.0-beta.1/choco-parsers-5.0.0-beta.1-jar-with-dependencies.jar"
)
CHOCO_SHA256 = "6dec5d54b335a18170314539e6c5ecc779778600154db28b59cffacdca898600"
DEFAULT_DIRECTORY = Path(__file__).resolve().parents[1] / "solvers"


class FetchError(RuntimeError):
    """A pinned solver artifact could not be obtained safely."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, expected: str) -> bool:
    return path.is_file() and sha256_file(path) == expected


def download(url: str, destination: Path) -> None:
    """Download with TLS verification, using curl only for local CA issues."""
    request = urllib.request.Request(url, headers={"User-Agent": "qayd-fastcop/1"})
    try:
        with urllib.request.urlopen(
            request, timeout=120, context=ssl.create_default_context()
        ) as source:
            with destination.open("wb") as target:
                shutil.copyfileobj(source, target)
        return
    except (OSError, urllib.error.URLError) as error:
        curl = shutil.which("curl")
        if curl is None:
            raise FetchError(f"download failed for {url}: {error}") from error
        completed = subprocess.run(
            [
                curl, "--fail", "--location", "--silent", "--show-error",
                "--output", str(destination), url,
            ],
            check=False,
        )
        if completed.returncode != 0:
            raise FetchError(f"download failed for {url} (curl exit {completed.returncode})")


def require_hash(path: Path, expected: str, description: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise FetchError(
            f"{description} SHA-256 mismatch: expected {expected}, got {observed}"
        )


def install_bytes(data: bytes, destination: Path, expected: str, force: bool) -> None:
    if destination.exists() and not force and not verify(destination, expected):
        raise FetchError(
            f"refusing to replace unrecognized artifact {destination}; pass --force"
        )
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
        temporary.write(data)
        temporary_path = Path(temporary.name)
    try:
        require_hash(temporary_path, expected, destination.name)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def fetch_ace(directory: Path, force: bool) -> Path:
    destination = directory / "ACE-2.5.jar"
    if verify(destination, ACE_SHA256):
        return destination
    if destination.exists() and not force:
        raise FetchError(f"unexpected ACE artifact at {destination}; pass --force")
    with tempfile.TemporaryDirectory(prefix="qayd-fastcop-wheel-") as scratch:
        wheel = Path(scratch) / "pycsp3-2.5.whl"
        download(WHEEL_URL, wheel)
        require_hash(wheel, WHEEL_SHA256, "PyCSP3 2.5 wheel")
        try:
            with zipfile.ZipFile(wheel) as archive:
                data = archive.read(ACE_MEMBER)
        except (KeyError, zipfile.BadZipFile) as error:
            raise FetchError(f"ACE jar missing from pinned PyCSP3 wheel: {error}") from error
    install_bytes(data, destination, ACE_SHA256, force=True)
    return destination


def fetch_choco(directory: Path, force: bool) -> Path:
    destination = directory / "choco-xcsp25.jar"
    if verify(destination, CHOCO_SHA256):
        return destination
    if destination.exists() and not force:
        raise FetchError(f"unexpected Choco artifact at {destination}; pass --force")
    with tempfile.TemporaryDirectory(prefix="qayd-fastcop-choco-") as scratch:
        downloaded = Path(scratch) / destination.name
        download(CHOCO_URL, downloaded)
        require_hash(downloaded, CHOCO_SHA256, "Choco XCSP25 jar")
        data = downloaded.read_bytes()
    install_bytes(data, destination, CHOCO_SHA256, force=True)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)

    try:
        ace = directory / "ACE-2.5.jar"
        choco = directory / "choco-xcsp25.jar"
        if args.verify_only:
            require_hash(ace, ACE_SHA256, "ACE 2.5 jar")
            require_hash(choco, CHOCO_SHA256, "Choco XCSP25 jar")
        else:
            ace = fetch_ace(directory, args.force)
            choco = fetch_choco(directory, args.force)
        report = {
            "ace": {"path": str(ace), "sha256": sha256_file(ace), "version": "2.5"},
            "choco": {
                "path": str(choco),
                "sha256": sha256_file(choco),
                "version": "5.0.0-beta.1",
            },
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (FetchError, OSError) as error:
        print(f"fastcop solver fetch error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
