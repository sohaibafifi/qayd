#!/usr/bin/env python3
"""Fetch the public instance collections used by the competitive campaign.

Downloads are explicit and idempotent.  Existing files are preserved.  Use
``--collection`` repeatedly to select only the desired families.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
import ssl
import tarfile
from urllib.request import Request, urlopen
import zipfile


ROOT = Path(__file__).resolve().parents[1]
COLLECTIONS = (
    "cvrplib-x", "solomon", "homberger-200", "homberger-400",
    "homberger-600", "homberger-800", "homberger-1000", "jsplib",
    "psplib-j30", "psplib-j60", "psplib-j90", "psplib-j120", "psplib-mm-j30",
)
CVRPLIB_ROOT = "https://galgos.inf.puc-rio.br/cvrplib"
SOLOMON_URL = "https://www.sintef.no/globalassets/project/top/vrptw/solomon/solomon-100.zip"
HOMBERGER_URL = "https://www.sintef.no/globalassets/project/top/vrptw/homberger/{size}/homberger_{size}_customer_instances.zip"
JSPLIB_URL = "https://codeload.github.com/tamy0612/JSPLIB/zip/refs/heads/master"
PSPLIB_ARCHIVE = "https://www.om-db.wi.tum.de/psplib/download_dataset.php?format=tgz&mode={mode}&set={name}"
PSPLIB_SUMMARY = "https://www.om-db.wi.tum.de/psplib/download_merged.php?mode={mode}&set={name}"


def context(insecure: bool) -> ssl.SSLContext:
    return ssl._create_unverified_context() if insecure else ssl.create_default_context()


def download(url: str, *, insecure: bool) -> bytes:
    print(f"download {url}")
    request = Request(url, headers={"User-Agent": "qayd-benchmark/1"})
    with urlopen(request, context=context(insecure), timeout=120) as response:
        return response.read()


def safe_name(name: str, *, strip_first: bool = False) -> Path | None:
    pure = PurePosixPath(name)
    fields = pure.parts[1:] if strip_first else pure.parts
    if not fields or pure.is_absolute() or any(field in {"", ".", ".."} for field in fields):
        return None
    return Path(*fields)


def extract_zip(data: bytes, destination: Path, *, strip_first: bool = False) -> int:
    count = 0
    with zipfile.ZipFile(BytesIO(data)) as archive:
        for member in archive.infolist():
            relative = safe_name(member.filename, strip_first=strip_first)
            if relative is None or member.is_dir():
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                continue
            with archive.open(member) as source, target.open("wb") as output:
                while block := source.read(1 << 20):
                    output.write(block)
            count += 1
    return count


def extract_tar(data: bytes, destination: Path) -> int:
    count = 0
    with tarfile.open(fileobj=BytesIO(data), mode="r:*") as archive:
        for member in archive.getmembers():
            relative = safe_name(member.name)
            if relative is None or not member.isfile():
                continue
            source = archive.extractfile(member)
            if source is None:
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                continue
            with source, target.open("wb") as output:
                while block := source.read(1 << 20):
                    output.write(block)
            count += 1
    return count


def fetch_cvrplib_x(insecure: bool, limit: int) -> int:
    html = download(f"{CVRPLIB_ROOT}/en/instances/1", insecure=insecure).decode("utf-8", "replace")
    start = html.find('data-bs-target="#set-17"')
    end = html.find('data-bs-target="#set-18"', start + 1)
    if start < 0:
        raise RuntimeError("CVRPLIB X set not found on the instances page")
    section = html[start:end if end >= 0 else None]
    matches = re.findall(
        r'href="/cvrplib/en/download/instance/(\d+)"[^>]*>\s*(X-n\d+-k\d+)\s*</a>',
        section,
    )
    if limit:
        matches = matches[:limit]
    destination = ROOT / "bench" / "routing" / "instances" / "cvrplib-x"
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    for identifier, name in matches:
        for kind, suffix in (("instance", ".vrp"), ("bks", ".sol")):
            target = destination / f"{name}{suffix}"
            if target.exists():
                continue
            target.write_bytes(download(f"{CVRPLIB_ROOT}/en/download/{kind}/{identifier}", insecure=insecure))
            count += 1
    return count


def fetch_zip_collection(url: str, destination: Path, insecure: bool, *, strip_first: bool = False) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    return extract_zip(download(url, insecure=insecure), destination, strip_first=strip_first)


def fetch_psplib(collection: str, insecure: bool) -> int:
    multi = collection == "psplib-mm-j30"
    mode = "mm" if multi else "sm"
    name = "j30" if multi else collection.removeprefix("psplib-")
    destination = ROOT / "bench" / "scheduling" / "instances" / "psplib" / mode / name
    destination.mkdir(parents=True, exist_ok=True)
    count = extract_tar(download(PSPLIB_ARCHIVE.format(mode=mode, name=name), insecure=insecure), destination)
    summary = destination / "summary.json"
    if not summary.exists():
        raw = download(PSPLIB_SUMMARY.format(mode=mode, name=name), insecure=insecure)
        json.loads(raw)
        summary.write_bytes(raw)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", action="append", choices=("all", *COLLECTIONS), required=True)
    parser.add_argument("--limit", type=int, default=0, help="limit CVRPLIB X downloads only; 0 means all")
    parser.add_argument("--insecure", action="store_true", help="disable TLS verification for legacy mirrors")
    args = parser.parse_args()
    if args.limit < 0:
        raise SystemExit("limit must be non-negative")
    selected = list(COLLECTIONS) if "all" in args.collection else list(dict.fromkeys(args.collection))
    total = 0
    for collection in selected:
        if collection == "cvrplib-x":
            count = fetch_cvrplib_x(args.insecure, args.limit)
        elif collection == "solomon":
            count = fetch_zip_collection(
                SOLOMON_URL,
                ROOT / "bench" / "routing" / "instances" / "solomon",
                args.insecure,
            )
        elif collection.startswith("homberger-"):
            size = int(collection.split("-")[1])
            count = fetch_zip_collection(
                HOMBERGER_URL.format(size=size),
                ROOT / "bench" / "routing" / "instances" / "homberger" / str(size),
                args.insecure,
            )
        elif collection == "jsplib":
            count = fetch_zip_collection(
                JSPLIB_URL,
                ROOT / "bench" / "scheduling" / "instances" / "jsplib",
                args.insecure,
                strip_first=True,
            )
        else:
            count = fetch_psplib(collection, args.insecure)
        total += count
        print(f"{collection}: {count} new files")
    print(f"complete: {total} new files")


if __name__ == "__main__":
    main()
