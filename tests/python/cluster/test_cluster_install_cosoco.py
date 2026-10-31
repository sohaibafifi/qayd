"""Verify the pinned executable installer without network or solver execution."""
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "bench"))
from cluster.lib import install_cosoco as installer


def wheel(monkeypatch, files=None):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, body in (files or {"cosoco_bin/cosoco": b"official executable",
                                    "cosoco-2.6.1.dist-info/licenses/LICENSE": b"license"}).items():
            archive.writestr(name, body)
    payload = output.getvalue()
    monkeypatch.setattr(installer, "SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(installer.platform, "system", lambda: "Linux")
    monkeypatch.setattr(installer.platform, "machine", lambda: "x86_64")
    calls = []
    def download(url, *, timeout):
        calls.append((url, timeout))
        return io.BytesIO(payload)
    monkeypatch.setattr(installer.urllib.request, "urlopen", download)
    return calls


def test_install_preserves_license_hashes_runtime_and_reuses_it_offline(tmp_path, monkeypatch):
    calls = wheel(monkeypatch)
    destination = tmp_path / "tools/cosoco-2.6.1"
    binary = installer.install(destination)
    assert binary.read_bytes() == b"official executable"
    assert binary.stat().st_mode & 0o111
    receipt = json.loads((destination / "download.json").read_text())
    assert any(name.endswith("/LICENSE") for name in receipt["files"])
    assert len(receipt["files"]) == 2
    assert receipt["wheel_sha256"] == installer.SHA256
    monkeypatch.setattr(installer.urllib.request, "urlopen", lambda *_a, **_k: pytest.fail("offline verification"))
    assert installer.install(destination) == binary
    assert installer.install(destination, verify_only=True) == binary
    assert len(calls) == 1
    assert not list(destination.parent.glob(".cosoco-*"))


@pytest.mark.parametrize("change", ["binary", "license", "extra", "wheel", "receipt", "executable"])
def test_verification_rejects_changed_or_unreceipted_files(tmp_path, monkeypatch, change):
    wheel(monkeypatch)
    destination = tmp_path / "cosoco"
    binary = installer.install(destination)
    if change == "binary":
        binary.write_bytes(b"different solver")
    elif change == "license":
        (destination / "runtime/cosoco-2.6.1.dist-info/licenses/LICENSE").unlink()
    elif change == "extra":
        (destination / "runtime/unreceipted.so").write_bytes(b"extra")
    elif change == "wheel":
        (destination / installer.WHEEL).write_bytes(b"corrupt archive")
    elif change == "receipt":
        receipt = json.loads((destination / "download.json").read_text())
        receipt["version"] = "other"
        (destination / "download.json").write_text(json.dumps(receipt))
    else:
        binary.chmod(0o644)
    with pytest.raises(ValueError):
        installer.install(destination, verify_only=True)


def test_wrong_download_hash_does_not_publish_an_installation(tmp_path, monkeypatch):
    wheel(monkeypatch)
    monkeypatch.setattr(installer, "SHA256", "0" * 64)
    destination = tmp_path / "cosoco"
    with pytest.raises(ValueError, match="SHA-256"):
        installer.install(destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".cosoco-*"))


def test_runtime_cannot_be_replaced_by_editing_its_local_receipt(tmp_path, monkeypatch):
    wheel(monkeypatch)
    destination = tmp_path / "cosoco"
    binary = installer.install(destination)
    binary.write_bytes(b"different solver with a matching local receipt")
    receipt_path = destination / "download.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["files"]["runtime/cosoco_bin/cosoco"] = hashlib.sha256(binary.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="pinned wheel"):
        installer.verify(destination)


def test_official_verification_requires_the_cached_pinned_wheel(tmp_path, monkeypatch):
    wheel(monkeypatch)
    destination = tmp_path / "cosoco"
    installer.install(destination)
    (destination / installer.WHEEL).unlink()
    with pytest.raises(ValueError, match="wheel"):
        installer.verify(destination)


@pytest.mark.parametrize("files", [{"../escaped": b"bad"}, {"other": b"no executable"}])
def test_bad_wheel_layout_cannot_escape_or_publish(tmp_path, monkeypatch, files):
    wheel(monkeypatch, files)
    destination = tmp_path / "cosoco"
    with pytest.raises(ValueError):
        installer.install(destination)
    assert not destination.exists()
    assert not (tmp_path / "escaped").exists()


def test_unreceipted_installation_is_preserved_and_offline_missing_is_clear(tmp_path, monkeypatch):
    wheel(monkeypatch)
    destination = tmp_path / "cosoco"
    destination.mkdir()
    keep = destination / "keep"
    keep.write_text("previous local files")
    with pytest.raises(ValueError, match="unreceipted"):
        installer.install(destination)
    assert keep.read_text() == "previous local files"
    with pytest.raises(ValueError, match="not prepared"):
        installer.install(tmp_path / "missing", verify_only=True)


def test_unsupported_prepare_platform_is_rejected_before_download(tmp_path, monkeypatch):
    calls = wheel(monkeypatch)
    monkeypatch.setattr(installer.platform, "system", lambda: "Darwin")
    with pytest.raises(ValueError, match="Linux x86_64"):
        installer.install(tmp_path / "cosoco")
    assert not calls
