"""Loaded descriptors are immutable, detached and guarded without per-step JSON."""
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

from venturi.models.qwen_metadata import LoadedQwenSource, source_digest
from venturi.ninfer import PayloadFile


def source(tmp_path):
    path = tmp_path / "source.data"
    path.write_bytes(b"unchanged")
    manifest = dict(objects=[dict(id="weight", shape=[2, 3], format="q8")],
                    bindings={"role": {"object": "weight"}}, metadata={"unicode": "中文"})
    artifact = SimpleNamespace(path=path, manifest=manifest, artifact_id="fixture",
                               payload_bytes=9, files=[PayloadFile(path, 0, 0, 9)])
    return SimpleNamespace(config=dict(layers=["gdn", "attention"], epsilon=1e-6), artifact=artifact)


def capture(weights):
    return LoadedQwenSource.capture(weights, source_digest(weights.config, weights.artifact.manifest))


def test_capture_preserves_strict_digest_and_detaches_every_nested_container(tmp_path):
    weights = source(tmp_path)
    sealed = capture(weights)
    digest = sealed.digest
    weights.config["layers"].clear()
    weights.artifact.manifest["objects"][0]["shape"].append(90)
    weights.artifact.manifest["bindings"].clear()
    weights.artifact.files.clear()
    assert sealed.config["layers"] == ("gdn", "attention")
    assert sealed.artifact.objects["weight"]["shape"] == (2, 3)
    assert sealed.artifact.manifest["bindings"]["role"]["object"] == "weight"
    assert sealed.artifact.manifest["metadata"]["unicode"] == "中文"
    assert len(sealed.artifact.files) == 1 and sealed.digest == digest
    with pytest.raises(TypeError): sealed.config["epsilon"] = 1
    with pytest.raises(TypeError): sealed.artifact.objects["weight"]["format"] = "q4"
    with pytest.raises(AttributeError): sealed.config["layers"].append("different")
    with pytest.raises(FrozenInstanceError): sealed.artifact.path = tmp_path / "other"
    with pytest.raises(FrozenInstanceError): sealed.artifact.files[0].payload_bytes = 1
    with pytest.raises(FrozenInstanceError): sealed.config = {}


def test_creation_digest_rejects_changed_metadata_and_distinguishes_json_scalar_types(tmp_path):
    weights = source(tmp_path)
    expected = source_digest(weights.config, weights.artifact.manifest)
    weights.config["epsilon"] = True
    with pytest.raises(RuntimeError, match="metadata changed"):
        LoadedQwenSource.capture(weights, expected)
    assert source_digest(dict(x=True), {}) != source_digest(dict(x=1), {})
    assert source_digest(dict(x=1.0), {}) != source_digest(dict(x=1), {})


def test_native_preflight_keeps_file_guards_checks_identity_and_avoids_reserialization(tmp_path, monkeypatch):
    import venturi.models.qwen_native as native
    weights = source(tmp_path)
    sealed = capture(weights)
    model = native.NativeQwenModel.__new__(native.NativeQwenModel)
    model._metadata_seal = model._weights = sealed
    model._device = "cpu"
    stat = sealed.artifact.path.stat()
    model._stamps = ((stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns),)
    captures, calls = [], []
    monkeypatch.setattr(native, "_outside_capture", lambda device: captures.append(device))
    monkeypatch.setattr(native, "_digest", lambda *args: pytest.fail("Loaded metadata must not be reserialized"))
    original = Path.stat

    def tracked(path, *args, **kwargs):
        calls.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", tracked)
    model._metadata()
    assert calls == [sealed.artifact.path] and captures == ["cpu"]
    model._weights = capture(weights)
    with pytest.raises(RuntimeError, match="metadata changed"):
        model._metadata()
    model._weights = sealed
    sealed.artifact.path.write_bytes(b"changed source size")
    with pytest.raises(RuntimeError, match="source changed"):
        model._metadata()


def test_mtp_binding_uses_detached_reader_with_original_source_provenance(tmp_path, monkeypatch):
    import torch
    import venturi.models.qwen_native as native
    from test_qwen_mtp import mtp_fixture
    from venturi.models.qwen_weights import QwenQ8Weights

    weights, draft, _, _ = mtp_fixture(tmp_path)
    sealed = capture(weights)
    model = native.NativeQwenModel.__new__(native.NativeQwenModel)
    model._metadata_seal = model._weights = sealed
    model._device = torch.device("cpu")
    model._stamps = tuple((s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
                          for f in sealed.artifact.files for s in [f.path.stat()])
    monkeypatch.setattr(native, "_outside_capture", lambda device: None)
    reopened = model._binding_weights()
    assert isinstance(reopened, QwenQ8Weights)
    draft.validate_target(reopened)
    reopened.artifact.manifest["bindings"].clear()
    reopened.config["layer_types"].clear()
    model._metadata()
    draft.validate_target(model._binding_weights())
