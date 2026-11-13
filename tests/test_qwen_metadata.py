"""Loaded configuration and source identity stay immutable and independently guarded."""

from dataclasses import FrozenInstanceError
from pathlib import Path
import pytest
from venturi.models.qwen.qwen_metadata import LoadedQwenSource, source_digest
from venturi.models.qwen.qwen_weights import QwenWeights
from qwen_safetensors import write_quantized_checkpoint


def source(tmp_path):
    return QwenWeights.open(
        write_quantized_checkpoint(tmp_path / "model.safetensors")[0]
    )


def capture(weights):
    return LoadedQwenSource.capture(weights, source_digest(weights))


def test_capture_detaches_configuration_and_source_identity(tmp_path):
    weights = source(tmp_path)
    sealed = capture(weights)
    digest = sealed.digest
    weights.config["layer_types"].clear()
    assert sealed.config["layer_types"] == ("linear_attention", "full_attention")
    assert sealed.digest == digest
    with pytest.raises(TypeError):
        sealed.config["rms_norm_eps"] = 1
    with pytest.raises(AttributeError):
        sealed.config["layer_types"].append("other")
    with pytest.raises(FrozenInstanceError):
        sealed.source.path = tmp_path / "other"
    with pytest.raises(FrozenInstanceError):
        sealed.config = {}


def test_creation_rejects_changed_metadata(tmp_path):
    weights = source(tmp_path)
    expected = source_digest(weights)
    weights.config["rms_norm_eps"] = True
    with pytest.raises(RuntimeError):
        LoadedQwenSource.capture(weights, expected)


def test_source_guard_and_no_per_token_reserialization(tmp_path, monkeypatch):
    import venturi.models.qwen.qwen_native as native

    weights = source(tmp_path)
    sealed = capture(weights)
    model = native.NativeQwenModel.__new__(native.NativeQwenModel)
    model._metadata_seal = model._weights = sealed
    model._device = "cpu"
    monkeypatch.setattr(native, "_outside_capture", lambda _: None)
    monkeypatch.setattr(
        native,
        "source_digest",
        lambda *_: pytest.fail("Sealed metadata serialized during step"),
    )
    model._metadata()
    model._weights = capture(weights)
    with pytest.raises(RuntimeError, match="metadata changed"):
        model._metadata()
    model._weights = sealed
    sealed.source.path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="source changed"):
        model._metadata()


def test_mtp_reopen_is_detached_and_same_lineage(tmp_path, monkeypatch):
    import torch
    import venturi.models.qwen.qwen_native as native
    from test_qwen_mtp import mtp_fixture

    weights, draft, _, _ = mtp_fixture(tmp_path)
    sealed = capture(weights)
    model = native.NativeQwenModel.__new__(native.NativeQwenModel)
    model._metadata_seal = model._weights = sealed
    model._device = torch.device("cpu")
    monkeypatch.setattr(native, "_outside_capture", lambda _: None)
    reopened = model._binding_weights()
    draft.validate_target(reopened)
    reopened.config["layer_types"].clear()
    model._metadata()
    draft.validate_target(model._binding_weights())
