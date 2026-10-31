"""Check official membership, offline cache reuse and build-receipt preservation."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
from cluster.lib import campaign, corpus


def sha(data):
    return hashlib.sha256(data).hexdigest()


def prepared(tmp_path):
    root, shared = tmp_path / "source", tmp_path / "shared"
    root.mkdir(); shared.mkdir()
    (root / "data").symlink_to(shared, target_is_directory=True)
    (root / "Cargo.toml").write_text("[package]\n")
    (root / "bench/cluster/lib").mkdir(parents=True)
    (root / "bench/cluster/lib/campaign.py").write_text("# installed runner\n")
    xmldata = {
        "CSP25": {"csp.xml": b'<instance format="XCSP3" type="CSP"><variables><var id="x">0..1</var></variables><constraints/></instance>'},
        "COP25": {"cop.xml": b'<instance format="XCSP3" type="COP"><variables><var id="x">0..1</var></variables><objectives><minimize>x</minimize></objectives></instance>'},
    }
    archive = root / "data/benchmarks/csp/instancesXCSP25.zip"
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as stream:
        for track, entries in xmldata.items():
            for name, data in entries.items():
                stream.writestr(f"XCSP25/{track}/{name}", data)
    pbdata = {"DEC-LIN": {"PB24/DEC-LIN/decision.opb": b"* comment\n+1 x1 >= 1;\n"},
              "OPT-LIN": {"PB24/OPT-LIN/minimize.opb.gz": gzip.compress(b"min: +1 x1;\n+1 x1 >= 1;\n")}}
    pbarchive = root / "data/benchmarks/pb/selected-PB24.tar"
    pbarchive.parent.mkdir(parents=True)
    with tarfile.open(pbarchive, "w") as stream:
        for entries in pbdata.values():
            for name, data in entries.items():
                member = tarfile.TarInfo(name); member.size = len(data)
                stream.addfile(member, io.BytesIO(data))
    for name, data in pbdata["DEC-LIN"].items():
        path = root / "data/benchmarks/pb/selected-PB24" / name
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    extra = root / "data/benchmarks/pb/selected-PB24/PB24/DEC-LIN/unrelated.opb"
    extra.write_bytes(b"+1 x2 >= 0;\n")
    mm = root / "data/benchmarks/scheduling/psplib/mm/j30"
    mm.mkdir(parents=True)
    (mm / "a.mm").write_bytes(b"a-model")
    (mm / "b.mm").write_bytes(b"b-model")
    config = {
        "schema": "qayd.cluster.config/v1", "name": "test-corpus", "seeds": [0],
        "memory_mb": 1024, "checker_memory_mb": 512, "checker_seconds": 2, "grace_seconds": 1,
        "protocols": {track: {"seconds": 1} for track in ["xcsp-csp", "xcsp-cop", "pb", "rcpsp"]},
        "solvers": {"xcsp-csp": ["qayd"], "xcsp-cop": ["qayd"], "pb": ["qayd-pb"], "rcpsp": ["qayd-native"]},
        "datasets": [
            {"track": "xcsp-csp", "family": "xcsp25-csp", "glob": "data/XCSP25/CSP25/*.xml", "expected_count": 1},
            {"track": "xcsp-cop", "family": "xcsp25-cop", "glob": "data/XCSP25/COP25/*.xml", "expected_count": 1},
            {"track": "pb", "family": "pb24-decision", "glob": "data/benchmarks/pb/**/*.opb*", "receipt": "data/old.json"},
            {"track": "pb", "family": "pb24-optimization", "glob": "data/benchmarks/pb/**/*.opb*", "receipt": "data/old.json"},
            {"track": "rcpsp", "family": "psplib-mm-j30", "glob": "data/benchmarks/scheduling/psplib/mm/j30/*.mm", "expected_count": 552},
        ],
    }
    campaign.write(root / "bench/cluster/config/campaign.v1.json", config)
    reference = {
        "xcsp": {"source": "https://example.invalid/xcsp.zip", "archive_sha256": corpus.digest_file(archive), "tracks": {
            track: {name: sha(data) for name, data in entries.items()} for track, entries in xmldata.items()}},
        "pb": {"source": "https://example.invalid/pb.tar", "archive_sha256": corpus.digest_file(pbarchive), "tracks": {
            track: {name: sha(data) for name, data in entries.items()} for track, entries in pbdata.items()}},
        "mm_j30": {"source": "https://example.invalid/mm.tgz", "expected_instances": {
            path.name: sha(path.read_bytes()) for path in mm.glob("*.mm")}},
    }
    return root, reference, archive, pbarchive


def test_repair_reuses_caches_verifies_membership_and_preserves_build_source(tmp_path, monkeypatch):
    root, reference, zip_cache, tar_cache = prepared(tmp_path)
    monkeypatch.setattr(campaign, "ROOT", root)
    before = campaign.source_identity()["sha256"]
    original = (root / "bench/cluster/config/campaign.v1.json").read_bytes()
    out = Path("bench/cluster/runs/corpus-ready")
    path = corpus.repair(root, reference, out, offline=True)
    config = json.loads(path.read_text())
    campaign.validate_config(config)
    items, _assets, missing = campaign.discover(config)
    assert missing == []
    assert len(items) == 6
    assert all("unrelated" not in item["path"] for item in items)
    assert (root / "data/benchmarks/pb/selected-PB24/PB24/DEC-LIN/unrelated.opb").exists()
    assert next(item for item in items if item["path"].endswith(".gz"))["sense"] == "min"
    assert (root / "bench/cluster/config/campaign.v1.json").read_bytes() == original
    assert campaign.source_identity()["sha256"] == before
    zip_cache.unlink(); tar_cache.unlink()
    assert corpus.repair(root, reference, out, offline=True) == path


def test_mrcpsp_mismatch_stops_before_other_corpora_are_modified(tmp_path):
    root, reference, _zip, _tar = prepared(tmp_path)
    path = root / "data/benchmarks/scheduling/psplib/mm/j30/a.mm"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="MRCPSP differs"):
        corpus.repair(root, reference, Path("bench/cluster/runs/repair"), offline=True)
    assert path.read_bytes() == b"changed"
    assert not (root / "data/XCSP25").exists()
    assert not (root / "bench/cluster/runs/repair").exists()


def test_corrupted_pb_is_preserved_and_no_complete_receipt_is_created(tmp_path):
    root, reference, _zip, _tar = prepared(tmp_path)
    path = root / "data/benchmarks/pb/selected-PB24/PB24/DEC-LIN/decision.opb"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="instance differs"):
        corpus.repair(root, reference, Path("bench/cluster/runs/repair"), offline=True)
    assert path.read_bytes() == b"changed"
    assert not (root / "bench/cluster/runs/repair/config.json").exists()


def test_an_old_pb_archive_is_preserved_and_replaced_by_a_separate_verified_download(tmp_path, monkeypatch):
    root, reference, _zip, archive = prepared(tmp_path)
    official = archive.read_bytes()
    archive.write_bytes(b"old incomplete archive")
    calls = []

    def fake_download(url, destination, offline):
        calls.append(url)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(official)

    monkeypatch.setattr(corpus, "download", fake_download)
    corpus.repair(root, reference, Path("bench/cluster/runs/corpus-ready"))
    assert calls == [reference["pb"]["source"]]
    assert archive.read_bytes() == b"old incomplete archive"
    monkeypatch.setattr(campaign, "ROOT", root)
    config = json.loads((root / "bench/cluster/runs/corpus-ready/config.json").read_text())
    assert campaign.discover(config)[2] == []


def test_reference_counts_come_from_complete_official_archives():
    reference = json.loads((ROOT / "bench/cluster/config/corpus-reference.v1.json").read_text())
    assert {name: len(entries) for name, entries in reference["xcsp"]["tracks"].items()} == {"CSP25": 200, "COP25": 250}
    assert {name: len(entries) for name, entries in reference["pb"]["tracks"].items()} == {"DEC-LIN": 397, "OPT-LIN": 478}
    assert len(reference["mm_j30"]["expected_instances"]) == 640


def test_standalone_delivery_runs_outside_the_source_tree_without_a_reference_sidecar(tmp_path, monkeypatch):
    root, reference, _zip, _tar = prepared(tmp_path)
    monkeypatch.setattr(campaign, "ROOT", root)
    (root / ".venv/bin").mkdir(parents=True)
    (root / ".venv/bin/python").symlink_to(sys.executable)
    (root / "bench/cluster/lib/campaign.py").write_text(
        "import json, pathlib, sys\n"
        "path = pathlib.Path('bench/cluster/runs/inventory-invocation.json')\n"
        "path.write_text(json.dumps(sys.argv[1:]))\n"
    )
    before = campaign.source_identity()["sha256"]
    delivery = tmp_path / "qayd-corpus.py"
    corpus.export_standalone(delivery, reference)
    result = subprocess.run([sys.executable, str(delivery), "--repo", str(root), "--offline"],
                            cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    invocation = json.loads((root / "bench/cluster/runs/inventory-invocation.json").read_text())
    assert invocation[:4] == ["prepare", "--stage", "full", "--inventory"]
    assert invocation[invocation.index("--config") + 1] == str(root / "bench/cluster/runs/corpus-ready/config.json")
    assert campaign.source_identity()["sha256"] == before
    assert not (tmp_path / "corpus-reference.v1.json").exists()


@pytest.mark.parametrize("partial", [False, True])
def test_full_xcsp_fetch_handles_the_wrapping_directory_and_an_empty_or_partial_cache(tmp_path, partial):
    root = tmp_path / "source"
    output = root / "data/XCSP25"
    canonical = output / "CSP25"
    canonical.mkdir(parents=True)
    if partial:
        (canonical / "a.xml").write_bytes(b"a")
    archive = output / "instancesXCSP25.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("XCSP25/CSP25/a.xml", b"a")
        stream.writestr("XCSP25/CSP25/b.xml", b"b")
        stream.writestr("XCSP25/MiniCSP25/not-standard.xml", b"mini")
    reference = {"source": "https://example.invalid/xcsp.zip", "archive_sha256": corpus.digest_file(archive),
                 "tracks": {"CSP25": {"a.xml": sha(b"a"), "b.xml": sha(b"b")}}}
    corpus.xcsp_inputs(root, reference, offline=True)
    assert sorted(path.name for path in canonical.glob("*.xml")) == ["a.xml", "b.xml"]


def test_missing_xcsp_track_is_an_error_instead_of_a_zero_instance_success(tmp_path):
    output = tmp_path / "data/XCSP25"
    output.mkdir(parents=True)
    archive = output / "instancesXCSP25.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("XCSP25/MiniCSP25/a.xml", b"a")
    reference = {"source": "https://example.invalid/xcsp.zip", "archive_sha256": corpus.digest_file(archive),
                 "tracks": {"CSP25": {"a.xml": sha(b"a")}}}
    with pytest.raises(ValueError, match="XCSP archive membership differs"):
        corpus.xcsp_inputs(tmp_path, reference, offline=True)
