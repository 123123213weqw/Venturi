"""Explicit staging copies preserve every byte and never overwrite a cache."""

import hashlib
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from test_ninfer import write_artifact
from venturi.ninfer import NInferArtifact
from venturi.staging import stage_artifact


@pytest.mark.parametrize("split", [None, 2])
def test_stage_verify_reuse_and_private_files(tmp_path, split):
    source = write_artifact(tmp_path, split=split)
    report = stage_artifact(source, tmp_path / "cache")
    assert report["reused"] is False
    staged = NInferArtifact(report["path"])
    assert staged.read_object("w") == NInferArtifact(source).read_object("w")
    for file in staged.files:
        original = source.parent / file.path.name
        assert file.path.read_bytes() == original.read_bytes()
        assert (
            report["sha256"][file.path.name]
            == hashlib.sha256(original.read_bytes()).hexdigest()
        )
        assert stat.S_IMODE(file.path.stat().st_mode) == 0o600
    with pytest.raises(ValueError, match="already exists"):
        stage_artifact(source, tmp_path / "cache")
    reused = stage_artifact(source, tmp_path / "cache", reuse=True)
    assert reused["reused"] and reused["path"] == report["path"]
    assert reused["validation_seconds"] > 0
    path = Path(report["path"])
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="checksum"):
        stage_artifact(source, tmp_path / "cache", reuse=True)


def test_insufficient_space_does_not_create_partial_copy(tmp_path, monkeypatch):
    source = write_artifact(tmp_path)
    monkeypatch.setattr(
        "venturi.staging.shutil.disk_usage", lambda _: SimpleNamespace(free=0)
    )
    with pytest.raises(ValueError, match="headroom"):
        stage_artifact(source, tmp_path / "cache")
    assert not list((tmp_path / "cache").iterdir())


def test_staging_report_cannot_overwrite_an_artifact_volume(tmp_path):
    source = write_artifact(tmp_path)
    collision = tmp_path / "staging.json"
    source.rename(collision)
    with pytest.raises(ValueError, match="reserved"):
        stage_artifact(collision, tmp_path / "cache")
    assert NInferArtifact(collision).read_object("w") == b"\x00\x00\x80?"


def test_source_change_aborts_and_removes_only_owned_partial(tmp_path, monkeypatch):
    source = write_artifact(tmp_path)
    import venturi.staging as staging

    original = staging._identity
    calls = 0

    def changed(artifact):
        nonlocal calls
        calls += 1
        result = original(artifact)
        if calls > 1:
            result[0]["mtime_ns"] += 1
        return result

    monkeypatch.setattr(staging, "_identity", changed)
    cache = tmp_path / "cache"
    cache.mkdir()
    unrelated = cache / "keep-me"
    unrelated.write_text("not owned by staging")
    with pytest.raises(ValueError, match="changed during"):
        stage_artifact(source, cache)
    assert list(cache.iterdir()) == [unrelated]
