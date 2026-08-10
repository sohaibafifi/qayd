#!/usr/bin/env python3
"""Build and verify the deterministic XCSP25 FAST COP instance manifest."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import lzma
import os
import re
import sys
import xml.sax
import xml.sax.handler
from pathlib import Path
from typing import BinaryIO, Iterator


SCHEMA = "qayd.fastcop.manifest/v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INSTANCES = REPO_ROOT / "data" / "XCSP25" / "COP25"
DEFAULT_MANIFEST = Path(__file__).with_name("manifest.v1.json")
INSTANCE_SUFFIXES = (".xml", ".xml.lzma", ".xml.xz", ".xml.gz")

# XCSP models sometimes have several deliberately distinct variants.  Keeping
# the variant in `family` matches competition invalidation granularity, while
# `family_group` lets reports aggregate related variants.
FAMILIES = (
    ("FlexibleJobshopScen", "FlexibleJobshopScen"),
    ("ChampionsLeague-strict", "ChampionsLeague"),
    ("LowAutocorrelation", "LowAutocorrelation"),
    ("MetabolicNetwork", "MetabolicNetwork"),
    ("BlockModeling", "BlockModeling"),
    ("ButtonsScissors", "ButtonsScissors"),
    ("ChampionsLeague", "ChampionsLeague"),
    ("AlteredStates-bis", "AlteredStates"),
    ("AlteredStates", "AlteredStates"),
    ("BusScheduling", "BusScheduling"),
    ("RoadefPlaning2", "RoadefPlaning2"),
    ("TankAllocation1", "TankAllocation"),
    ("TankAllocation2", "TankAllocation"),
    ("SchedulingOS", "SchedulingOS"),
    ("RollerSplat", "RollerSplat"),
    ("Fortress1", "Fortress"),
    ("Fortress2", "Fortress"),
    ("Cutstock", "Cutstock"),
    ("Coprime", "Coprime"),
    ("FAPP", "FAPP"),
    ("IHTC", "IHTC"),
)


class ManifestError(RuntimeError):
    """The corpus or manifest is incomplete or inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def open_instance(path: Path) -> BinaryIO:
    """Open an XCSP instance, detecting compression from magic bytes."""
    with path.open("rb") as source:
        magic = source.read(6)
    if magic.startswith((b"\xfd7zXZ", b"\x5d\x00\x00")):
        return lzma.open(path, "rb")
    if magic.startswith(b"\x1f\x8b"):
        return gzip.open(path, "rb")
    return path.open("rb")


def detect_objective_sense(path: Path) -> str:
    """Return min or max after streaming the complete XML objective section."""
    class ObjectiveFound(Exception):
        def __init__(self, sense: str) -> None:
            self.sense = sense

    class ObjectiveHandler(xml.sax.handler.ContentHandler):
        def startElement(self, name: str, attrs) -> None:
            del attrs
            local_name = name.rsplit(":", 1)[-1].lower()
            if local_name == "minimize":
                raise ObjectiveFound("min")
            if local_name == "maximize":
                raise ObjectiveFound("max")

    try:
        with open_instance(path) as source:
            parser = xml.sax.make_parser()
            parser.setFeature(xml.sax.handler.feature_external_ges, False)
            parser.setContentHandler(ObjectiveHandler())
            parser.parse(source)
    except ObjectiveFound as found:
        return found.sense
    except (xml.sax.SAXException, EOFError, lzma.LZMAError, OSError) as error:
        raise ManifestError(f"cannot parse objective in {path}: {error}") from error
    raise ManifestError(f"no minimize/maximize objective in {path}")


def family_for(filename: str) -> tuple[str, str]:
    for family, group in FAMILIES:
        if filename == family or filename.startswith(family + "-"):
            return family, group
    raise ManifestError(f"unknown XCSP25 FAST COP family: {filename}")


def instance_id(path: Path) -> str:
    name = path.name
    for suffix in (".lzma", ".xz", ".gz", ".xml"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return re.sub(r"_c25$", "", name)


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ManifestError(f"instance path is outside repository: {path}") from error


def iter_instances(directory: Path) -> Iterator[Path]:
    if not directory.is_dir():
        raise ManifestError(f"instance directory does not exist: {directory}")
    paths = sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.name.endswith(INSTANCE_SUFFIXES)
    )
    if not paths:
        raise ManifestError(f"no XCSP instances under {directory}")
    yield from paths


def build_manifest(directory: Path) -> dict:
    instances = []
    seen_ids = set()
    for path in iter_instances(directory):
        item_id = instance_id(path)
        if item_id in seen_ids:
            raise ManifestError(f"duplicate instance id: {item_id}")
        seen_ids.add(item_id)
        family, family_group = family_for(item_id)
        instances.append(
            {
                "id": item_id,
                "family": family,
                "family_group": family_group,
                "objective_sense": detect_objective_sense(path),
                "path": repo_relative(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "schema": SCHEMA,
        "competition": "XCSP25",
        "track": "FAST COP",
        "source": "https://www.cril.univ-artois.fr/~lecoutre/compets/instancesXCSP25.zip",
        "instance_count": len(instances),
        "instances": instances,
    }


def canonical_json(value: dict) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def write_if_changed(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def load_manifest(path: Path, verify_files: bool = True) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot read manifest {path}: {error}") from error
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ManifestError(f"unsupported manifest schema in {path}")
    instances = value.get("instances")
    if not isinstance(instances, list) or value.get("instance_count") != len(instances):
        raise ManifestError("manifest instance_count is inconsistent")
    seen = set()
    previous = ""
    for item in instances:
        if not isinstance(item, dict):
            raise ManifestError("manifest instance must be an object")
        item_id = item.get("id")
        if not isinstance(item_id, str) or item_id in seen or item_id < previous:
            raise ManifestError("manifest ids must be unique and sorted")
        seen.add(item_id)
        previous = item_id
        if item.get("objective_sense") not in ("min", "max"):
            raise ManifestError(f"invalid objective sense for {item_id}")
        digest = item.get("sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ManifestError(f"invalid SHA-256 for {item_id}")
        if verify_files:
            instance_path = (REPO_ROOT / str(item.get("path"))).resolve()
            try:
                instance_path.relative_to(REPO_ROOT.resolve())
            except ValueError as error:
                raise ManifestError(f"path escapes repository for {item_id}") from error
            if not instance_path.is_file():
                raise ManifestError(f"missing instance for {item_id}: {instance_path}")
            observed = sha256_file(instance_path)
            if observed != digest:
                raise ManifestError(
                    f"hash mismatch for {item_id}: expected {digest}, got {observed}"
                )
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, default=DEFAULT_INSTANCES)
    parser.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--check", action="store_true", help="verify output without rewriting it")
    parser.add_argument("--expect-count", type=int, default=250)
    args = parser.parse_args()

    try:
        generated = build_manifest(args.instances.resolve())
        if args.expect_count and generated["instance_count"] != args.expect_count:
            raise ManifestError(
                f"expected {args.expect_count} instances, found {generated['instance_count']}"
            )
        content = canonical_json(generated)
        if args.check:
            existing = args.output.read_text(encoding="utf-8")
            if existing != content:
                raise ManifestError(f"manifest is stale: regenerate {args.output}")
            load_manifest(args.output)
        else:
            write_if_changed(args.output, content)
            load_manifest(args.output)
        print(f"{generated['instance_count']} instances: {args.output}")
        return 0
    except (ManifestError, OSError) as error:
        print(f"fastcop manifest error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
