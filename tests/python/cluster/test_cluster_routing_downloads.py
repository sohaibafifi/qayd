"""Exercise LKH outage recovery without network access or a compiler."""
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import tarfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "bench"))
from cluster.lib.downloads import routing


def source_archive(extra=None):
    files = {
        "README.txt": b"Version 3.0.13 - November 2024\n",
        "Makefile": b"all:\n\tmake -j -C SRC\n",
        "SRC/Makefile": b"all:\n\tcc -o ../LKH LKHmain.c\n",
        "SRC/LKHmain.c": b"int main(void) { return 0; }\n",
        "SRC/INCLUDE/LKH.h": b"/* LKH */\n",
    }
    if extra:
        files.update(extra)
    output = BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, data in files.items():
            member = tarfile.TarInfo(f"{routing.LKH_NAME}/{name}")
            member.size = len(data)
            archive.addfile(member, BytesIO(data))
    return output.getvalue()


def configure(tmp_path, monkeypatch, data):
    monkeypatch.setattr(routing, "HERE", tmp_path / "solvers")
    monkeypatch.setattr(routing, "LKH_SHA256", hashlib.sha256(data).hexdigest())
    return tmp_path / "downloads" / f"{routing.LKH_NAME}.tgz"


def test_lkh_official_outage_falls_back_to_pinned_archive(tmp_path, monkeypatch):
    data = source_archive()
    cached = configure(tmp_path, monkeypatch, data)
    requests = []

    def download(request, timeout):
        requests.append((request.full_url, timeout))
        if request.full_url == routing.LKH_URL:
            raise TimeoutError("official server unavailable")
        return BytesIO(data)

    monkeypatch.setattr(routing, "urlopen", download)
    destination = routing.prepare_lkh()
    assert [url for url, _ in requests] == [routing.LKH_URL, routing.LKH_MIRROR_URL]
    assert all(timeout <= 20 for _, timeout in requests)
    assert cached.read_bytes() == data
    receipt = json.loads((destination / "qayd-source.json").read_text())
    assert receipt["archive_sha256"] == hashlib.sha256(data).hexdigest()
    assert receipt["download_url"] == routing.LKH_MIRROR_URL
    assert receipt["publisher_checksum_verified"] is False
    assert (destination / "SRC/LKHmain.c").is_file()
    assert sorted(path.name for path in destination.parent.iterdir()) == [routing.LKH_NAME]


def test_lkh_cached_archive_needs_no_network(tmp_path, monkeypatch):
    data = source_archive()
    cached = configure(tmp_path, monkeypatch, data)
    cached.parent.mkdir()
    cached.write_bytes(data)

    def unexpected_network(*args, **kwargs):
        pytest.fail("a cached archive must work offline")

    monkeypatch.setattr(routing, "urlopen", unexpected_network)
    destination = routing.prepare_lkh()
    assert routing.prepare_lkh() == destination
    assert json.loads((destination / "qayd-source.json").read_text())["download_url"] is None


def test_lkh_rejects_bad_downloads_without_publishing_a_cache(tmp_path, monkeypatch):
    cached = configure(tmp_path, monkeypatch, source_archive())
    monkeypatch.setattr(routing, "urlopen", lambda *a, **kw: BytesIO(b"truncated archive"))
    with pytest.raises(ValueError, match="checksum mismatch"):
        routing.prepare_lkh()
    assert not cached.exists()
    assert not (routing.HERE / routing.LKH_NAME).exists()


def test_lkh_preserves_a_corrupt_existing_cache(tmp_path, monkeypatch):
    cached = configure(tmp_path, monkeypatch, source_archive())
    cached.parent.mkdir()
    cached.write_bytes(b"keep for inspection")
    with pytest.raises(ValueError, match="move it aside"):
        routing.prepare_lkh()
    assert cached.read_bytes() == b"keep for inspection"
    assert not (routing.HERE / routing.LKH_NAME).exists()


def test_lkh_rejects_unsafe_archive_before_installing_any_sources(tmp_path, monkeypatch):
    data = source_archive({"../escaped": b"outside the source directory"})
    cached = configure(tmp_path, monkeypatch, data)
    cached.parent.mkdir()
    cached.write_bytes(data)
    with pytest.raises(ValueError, match="Unexpected LKH archive entry"):
        routing.prepare_lkh()
    assert not (routing.HERE / routing.LKH_NAME).exists()
    assert not (routing.HERE / "escaped").exists()


def test_lkh_download_only_never_builds_on_the_frontend(tmp_path, monkeypatch):
    data = source_archive()
    cached = configure(tmp_path, monkeypatch, data)
    cached.parent.mkdir()
    cached.write_bytes(data)
    monkeypatch.setattr(sys, "argv", ["routing.py", "--solver", "lkh", "--download-only",
                                      "--accept-lkh-academic-license"])
    monkeypatch.setattr(routing, "command", lambda *a: pytest.fail("unexpected build"))
    routing.main()
    assert (routing.HERE / routing.LKH_NAME / "qayd-source.json").is_file()


def test_lkh_still_requires_license_acknowledgement(tmp_path, monkeypatch):
    configure(tmp_path, monkeypatch, source_archive())
    monkeypatch.setattr(sys, "argv", ["routing.py", "--solver", "lkh", "--download-only"])
    with pytest.raises(SystemExit, match="requires --accept-lkh-academic-license"):
        routing.main()
    assert not (routing.HERE / routing.LKH_NAME).exists()


def test_lkh_build_bypasses_the_unbounded_top_level_makefile(tmp_path, monkeypatch):
    data = source_archive()
    cached = configure(tmp_path, monkeypatch, data)
    cached.parent.mkdir()
    cached.write_bytes(data)
    calls = []
    monkeypatch.setattr(routing, "command", lambda argv, cwd: calls.append((argv, cwd)))
    routing.fetch_lkh(4)
    assert calls == [(["make", "-j", "4"], routing.HERE / routing.LKH_NAME / "SRC")]
