"""Strict trained-constant bindings without model downloads or CUDA."""

import copy

import pytest
import torch

from test_ninfer import write_artifact
from venturi.models import GDNFrontendWeights
from venturi.ninfer import NInferArtifact
from venturi.ops import TorchBackend


def frontend_artifact(directory):
    cfg = dict(
        model_type="qwen3_5_text",
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=64,
        linear_value_head_dim=13,
        linear_conv_kernel_dim=4,
        hidden_size=32,
    )
    prefix = "text/layers/0/"
    values = {
        "gdn/convolution": torch.full((4, 308), 0.1, dtype=torch.bfloat16),
        "gdn/a_log": torch.zeros(4),
        "gdn/dt_bias": torch.ones(4),
        "gdn/norm": torch.ones(13, dtype=torch.bfloat16),
        "input_norm": torch.full((32,), 0.25, dtype=torch.bfloat16),
    }
    payload, objects, bindings = bytearray(), [], {}
    for name, tensor in values.items():
        payload.extend(bytes((-len(payload)) % 256))
        data = tensor.view(torch.uint8).numpy().tobytes()
        objects.append(
            dict(
                id=name,
                kind="tensor",
                shape=list(tensor.shape),
                format="fp32" if tensor.dtype == torch.float32 else "bf16",
                layout="contiguous_le_v1",
                offset=len(payload),
                bytes=len(data),
            )
        )
        bindings[prefix + name] = {"object": name}
        payload.extend(data)
    manifest = dict(
        components={"text": {"config": cfg}},
        objects=objects,
        bindings=bindings,
        uses=[],
    )
    return NInferArtifact(
        write_artifact(directory, manifest=manifest, payload=bytes(payload))
    ), values


def test_bind_constants_and_prepare_reference(tmp_path):
    artifact, values = frontend_artifact(tmp_path)
    weights = GDNFrontendWeights.from_ninfer(artifact, layer=0)
    assert weights.qk_heads == 2 and weights.value_heads == 4
    assert torch.equal(weights.conv, values["gdn/convolution"])
    resident = weights.upload(TorchBackend(), dtype=torch.float64, device="cpu")
    assert resident.storage_nbytes == sum(
        t.numel() * t.element_size() for t in resident.storage_tensors
    )
    assert torch.equal(
        resident.input_norm._weight, torch.full((32,), 1.25, dtype=torch.float64)
    )
    assert torch.equal(resident.gates._rate, torch.ones(4, dtype=torch.float64))
    for layer in (1, -1, True, 2):
        with pytest.raises(ValueError, match="GDN layer"):
            GDNFrontendWeights.from_ninfer(artifact, layer=layer)


@pytest.mark.parametrize("bad", ["missing", "shape", "format", "config"])
def test_preflight_all_metadata_before_payload_read(tmp_path, monkeypatch, bad):
    artifact, _ = frontend_artifact(tmp_path)
    if bad == "missing":
        artifact.manifest["bindings"].pop("text/layers/0/gdn/norm")
    elif bad in ("shape", "format"):
        obj = copy.deepcopy(artifact.objects["gdn/norm"])
        obj[bad] = [14] if bad == "shape" else "fp32"
        artifact.objects["gdn/norm"] = obj
    else:
        artifact.manifest["components"]["text"]["config"]["linear_num_value_heads"] = 3

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid metadata triggered a payload read")

    monkeypatch.setattr(artifact, "read_object", forbidden)
    with pytest.raises(ValueError):
        GDNFrontendWeights.from_ninfer(artifact, layer=0)
