#!/usr/bin/env python3
"""Repair the corpus against frozen official member hashes, without rebuilding Qayd."""
from __future__ import annotations

import argparse
import base64
import bz2
import copy
import gzip
import hashlib
import json
import lzma
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import zipfile
import zlib


REFERENCE = None
SUFFIXES = (".opb", ".opb.gz", ".opb.bz2", ".opb.xz")


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def key(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def write_once(path: Path, value) -> None:
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"refusing to replace another frozen reference: {path}; choose a new --out")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_bytes(data)
    temporary.replace(path)


def safe_member(name: str) -> Path:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe archive member: {name}")
    return Path(*path.parts)


def download(url: str, path: Path, offline: bool) -> None:
    if offline:
        raise ValueError(f"offline mode: missing archive {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    print(f"Download {url}", flush=True)
    subprocess.run(["curl", "--fail", "--location", "--retry", "3", "--connect-timeout", "30",
                    "--max-time", "1800", "--output", str(temporary), url], check=True)
    temporary.replace(path)


def select_archive(candidates: list[Path], destination: Path, reference: dict, offline: bool) -> Path:
    expected = reference["archive_sha256"]
    for path in [*candidates, destination]:
        if path.is_file() and digest_file(path) == expected:
            return path
    if destination.exists():
        raise ValueError(f"cached download differs from the official reference; it was preserved: {destination}")
    download(reference["source"], destination, offline)
    if digest_file(destination) != expected:
        raise ValueError(f"download differs from the frozen official archive: {destination}")
    return destination


def write_input(path: Path, data: bytes, expected: str) -> None:
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"archive member differs from the frozen official reference: {path}")
    if path.exists():
        if digest_file(path) != expected:
            raise ValueError(f"existing instance differs; it was preserved: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_bytes(data)
    temporary.replace(path)


def existing_matches(path: Path, expected: str) -> bool:
    if not path.exists():
        return False
    if digest_file(path) != expected:
        raise ValueError(f"existing instance differs; it was preserved: {path}")
    return True


def xcsp_inputs(root: Path, reference: dict, offline: bool) -> None:
    tracks = reference["tracks"]
    missing = {}
    for track, expected in tracks.items():
        if not expected:
            raise ValueError(f"empty XCSP reference: {track}")
        for name, sha in expected.items():
            target = root / "data/XCSP25" / track / name
            if existing_matches(target, sha):
                continue
            for base in (root / "data/benchmarks/csp", root / "data/benchmarks/cop"):
                source = base / track / name
                if source.is_file() and digest_file(source) == sha:
                    write_input(target, source.read_bytes(), sha)
                    break
            else:
                missing[(track, name)] = (target, sha)
    if missing:
        caches = [root / "data/XCSP25/instancesXCSP25.zip",
                  root / "data/benchmarks/csp/instancesXCSP25.zip",
                  root / "data/benchmarks/cop/instancesXCSP25.zip"]
        destination = root / "data/cluster-downloads" / f"instancesXCSP25-{reference['archive_sha256'][:12]}.zip"
        archive = select_archive(caches, destination, reference, offline)
        with zipfile.ZipFile(archive) as stream:
            members = {}
            for member in stream.infolist():
                if member.is_dir():
                    continue
                parts = PurePosixPath(member.filename).parts
                for track in tracks:
                    if track in parts and member.filename.endswith((".xml", ".lzma", ".xz")):
                        identity = (track, Path(member.filename).name)
                        if identity in members:
                            raise ValueError(f"duplicate XCSP archive identity: {identity}")
                        members[identity] = member
            expected_keys = {(track, name) for track, expected in tracks.items() for name in expected}
            if set(members) != expected_keys:
                raise ValueError("XCSP archive membership differs from the frozen reference")
            for identity, (target, sha) in missing.items():
                write_input(target, stream.read(members[identity]), sha)
    print("XCSP25:", {track: len(names) for track, names in tracks.items()}, flush=True)


def opb_sense(path: Path) -> str:
    opener = {".xz": lzma.open, ".bz2": bz2.open, ".gz": gzip.open}.get(path.suffix, open)
    with opener(path, "rb") as stream:
        for raw in stream:
            line = raw.decode("utf-8", "replace").strip()
            if line and not line.startswith("*"):
                return "min" if line.startswith("min:") else "max" if line.startswith("max:") else "satisfy"
    raise ValueError(f"empty OPB instance: {path}")


def pb_inputs(root: Path, reference: dict, offline: bool) -> dict:
    expected = {name: sha for group in reference["tracks"].values() for name, sha in group.items()}
    base = root / "data/benchmarks/pb/selected-PB24"
    paths = {name: base / safe_member(name) for name in expected}
    missing = {name for name, path in paths.items() if not existing_matches(path, expected[name])}
    if missing:
        destination = root / "data/cluster-downloads" / f"selected-PB24-{reference['archive_sha256'][:12]}.tar"
        archive = select_archive([root / "data/benchmarks/pb/selected-PB24.tar"], destination, reference, offline)
        with tarfile.open(archive) as stream:
            members = {}
            for member in stream:
                if member.isfile() and member.name.lower().endswith(SUFFIXES) and any(
                    track in PurePosixPath(member.name).parts for track in reference["tracks"]
                ):
                    safe_member(member.name)
                    if member.name in members:
                        raise ValueError(f"duplicate PB archive identity: {member.name}")
                    members[member.name] = member
            if set(members) != set(expected):
                raise ValueError("PB archive membership differs from the frozen official reference")
            for name in sorted(missing):
                with stream.extractfile(members[name]) as source:
                    write_input(paths[name], source.read(), expected[name])
    print("PB24:", {track: len(names) for track, names in reference["tracks"].items()}, flush=True)
    return {track: {key(paths[name], root): expected[name] for name in group}
            for track, group in reference["tracks"].items()}


def repair(root: Path, reference: dict, out: Path, offline: bool = False) -> Path:
    root = root.resolve()
    if not reference["mm_j30"]["expected_instances"]:
        raise ValueError("empty MRCPSP reference")
    config_path = root / "bench/cluster/config/campaign.v1.json"
    config = copy.deepcopy(json.loads(config_path.read_text()))
    out = out if out.is_absolute() else root / out
    mm = next(dataset for dataset in config["datasets"] if dataset["family"] == "psplib-mm-j30")
    expected = reference["mm_j30"]["expected_instances"]
    observed = {}
    for path in root.glob(mm["glob"]):
        if path.name in observed:
            raise ValueError(f"duplicate multi-mode instance name: {path.name}")
        if path.is_file():
            observed[path.name] = digest_file(path)
    if observed != expected:
        absent = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        changed = sorted(name for name in expected.keys() & observed.keys() if expected[name] != observed[name])
        raise ValueError(f"MRCPSP differs from the official archive: missing={absent[:5]}, extra={extra[:5]}, changed={changed[:5]}")
    xcsp_inputs(root, reference["xcsp"], offline)
    pb = pb_inputs(root, reference["pb"], offline)
    reference_path = out / "reference.json"
    write_once(reference_path, reference)
    receipt = {"pb24-decision": pb["DEC-LIN"], "pb24-optimization": pb["OPT-LIN"]}
    receipt_path = out / "pb24-receipt.json"
    write_once(receipt_path, receipt)
    mm_manifest = out / "psplib-mm-j30.json"
    write_once(mm_manifest, reference["mm_j30"])
    for dataset in config["datasets"]:
        if dataset["track"].startswith("xcsp"):
            track = "CSP25" if dataset["track"] == "xcsp-csp" else "COP25"
            manifest = out / (track + ".json")
            write_once(manifest, {"source": reference["xcsp"]["source"],
                                 "expected_instances": reference["xcsp"]["tracks"][track]})
            dataset["manifest"] = key(manifest, root)
        if dataset["track"] == "pb":
            track = "DEC-LIN" if dataset["family"] == "pb24-decision" else "OPT-LIN"
            assets = {**pb[track], key(reference_path, root): digest_file(reference_path)}
            catalog = out / (track + ".json")
            write_once(catalog, {"assets": assets, "instances": [
                {"path": path, "family": Path(path).parent.name, "data": [], "sense": opb_sense(root / path)}
                for path in sorted(pb[track])
            ]})
            dataset.pop("glob", None)
            dataset.update(catalog=key(catalog, root), receipt=key(receipt_path, root), expected_count=len(pb[track]))
    mm.update(expected_count=len(expected), manifest=key(mm_manifest, root))
    config["name"] += "-verified-corpus"
    config["corpus_reference"] = key(reference_path, root)
    result = out / "config.json"
    write_once(result, config)
    print(f"MRCPSP: all {len(expected)} official instances retained.", flush=True)
    print(f"Configuration: {result}", flush=True)
    return result


def export_standalone(destination: Path, reference: dict) -> None:
    source = Path(__file__).read_text()
    marker = "REFERENCE = None\n"
    if source.count(marker) != 1:
        raise ValueError("export from the repository copy of corpus.py")
    payload = base64.b64encode(zlib.compress(json.dumps(reference, sort_keys=True).encode(), 9)).decode()
    embedded = f"REFERENCE = json.loads(zlib.decompress(base64.b64decode({payload!r})))\n"
    data = source.replace(marker, embedded).encode()
    compile(data, str(destination), "exec")
    if destination.exists() and destination.read_bytes() != data:
        raise ValueError(f"refusing to replace another delivery: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    print(f"Standalone corpus repair: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, default=Path("bench/cluster/runs/corpus-ready"))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--download", action="store_true", help="fetch native and MiniZinc datasets before verifying the full corpus")
    parser.add_argument("--export-standalone", type=Path, help="export this tool with its frozen reference, then exit")
    args = parser.parse_args()
    root = args.repo.resolve()
    if not (root / "Cargo.toml").is_file() or not (root / "bench/cluster/lib/campaign.py").is_file():
        raise SystemExit("Run from the Qayd repository root, or pass --repo.")
    reference = REFERENCE if REFERENCE is not None else json.loads((Path(__file__).resolve().parents[1] / "config/corpus-reference.v1.json").read_text())
    try:
        if args.export_standalone:
            export_standalone(args.export_standalone, reference)
            return
        if args.download and not args.offline:
            subprocess.run([str(root / ".venv/bin/python"), str(root / "bench/cluster/lib/fetch.py"), "--all"],
                           cwd=root, check=True)
        config = repair(root, reference, args.out, args.offline)
        subprocess.run([str(root / ".venv/bin/python"), str(root / "bench/cluster/lib/campaign.py"),
                        "prepare", "--stage", "full", "--inventory", "--config", str(config),
                        "--out", str(config.parent / "inventory")], cwd=root, check=True)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise SystemExit(f"Corpus repair failed: {error}") from error


if __name__ == "__main__":
    main()
