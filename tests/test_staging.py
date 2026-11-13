"""Standard checkpoint copies are byte-exact and never overwrite caller files."""

import hashlib, stat
from pathlib import Path
from types import SimpleNamespace
import pytest, torch
from safetensors.torch import save_file
from venturi.container.staging import stage_checkpoint
from venturi.container.safetensors import SafetensorsContainer


def source(tmp_path):
    p = tmp_path / "source.safetensors"
    save_file({"w": torch.arange(4).float()}, str(p))
    return p


def test_stage_verify_reuse_and_private_files(tmp_path):
    p = source(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    report = stage_checkpoint(p, cache)
    target = Path(report["path"])
    assert not report["reused"] and target.read_bytes() == p.read_bytes()
    assert report["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        stage_checkpoint(p, cache)
    assert stage_checkpoint(p, cache, reuse=True)["reused"]
    data = bytearray(target.read_bytes())
    data[-1] ^= 1
    target.write_bytes(data)
    with pytest.raises(ValueError, match="checksum"):
        stage_checkpoint(p, cache, reuse=True)


def test_insufficient_space_does_not_touch_unrelated_files(tmp_path, monkeypatch):
    p = source(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    other = cache / "keep"
    other.write_text("unrelated")
    monkeypatch.setattr(
        "venturi.container.staging.shutil.disk_usage", lambda _: SimpleNamespace(free=0)
    )
    with pytest.raises(ValueError):
        stage_checkpoint(p, cache)
    assert list(cache.iterdir()) == [other]


def test_source_change_aborts_and_removes_only_owned_partial(tmp_path, monkeypatch):
    p = source(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    other = cache / "keep"
    other.write_text("unrelated")
    original = SafetensorsContainer.check_source
    calls = 0

    def changed(self):
        nonlocal calls
        calls += 1
        if calls >= 3:
            raise RuntimeError("source changed during copy")
        return original(self)

    monkeypatch.setattr(SafetensorsContainer, "check_source", changed)
    with pytest.raises(RuntimeError, match="changed"):
        stage_checkpoint(p, cache)
    assert list(cache.iterdir()) == [other]
