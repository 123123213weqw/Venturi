"""Codec-driven loading: admission, geometry, byte round-trip, execution gate.

Synthetic NInfer containers exercise the load path per codec (q8/q4/q5/q6
row-split and fp16 dense). Loading qualification must not depend on the
execution tier: non-q8 codecs admit and verify, and from_q8 refuses them with
an explicit pending-execution error. No native backend required.
"""

import json
import struct

import pytest

import torch

from venturi.ninfer import MAGIC, NInferArtifact, row_geometry
from venturi.codecs import CODECS, codec_for
from venturi.storage_verify import verify_artifact
from venturi.models.qwen_weights import QwenQ8Weights


def _write_artifact(tmp_path, objects, bindings, config):
    payload = bytearray()
    manifest_objects = []
    for name, (fmt, shape, blob) in objects.items():
        payload += bytes((-len(payload)) % 256)  # object offsets are 256-aligned
        manifest_objects.append(
            {
                "id": name,
                "kind": "tensor",
                "shape": list(shape),
                "format": fmt,
                "layout": codec_for(fmt).layout,
                "offset": len(payload),
                "bytes": len(blob),
            }
        )
        payload += blob
    manifest = {
        "components": {"text": {"config": config}},
        "objects": manifest_objects,
        "bindings": bindings,
        "uses": [],
        "files": [{"path": None, "payload_bytes": len(payload)}],
    }
    encoded = json.dumps(manifest).encode()
    header = MAGIC + struct.pack("<Q", len(encoded)) + bytes(range(16)) + encoded
    path = tmp_path / "model.ninfer"
    path.write_bytes(header + bytes((-len(header)) % 4096) + payload)
    return path


def _quant_payload(shape, fmt, *, forbid_minus_128=False):
    n, k = shape
    codec = codec_for(fmt)
    bits, group, groups, high_offset, scales_offset, total = row_geometry(
        shape, fmt
    )
    row_codes = groups * group * bits // 8
    codes, high, scales = bytearray(), bytearray(), bytearray()
    for _ in range(n):
        codes += bytes(
            (0x7F if forbid_minus_128 else 0x11) % 256 for _ in range(row_codes)
        )
        high += bytes(row_high) if (row_high := groups * (bits - 4) // 8 if bits in (5, 6) else 0) else b""
        scales += b"\x3c\x00" * groups  # finite non-negative FP16
    return bytes(codes) + bytes((-len(codes)) % 256) + bytes(high) + bytes(
        (-len(high)) % 256
    ) + bytes(scales)


def _dense_payload(shape, fmt):
    import math

    import torch

    value = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[fmt]
    if fmt == "fp32":
        return value.numpy().tobytes()
    raw = value.to(dtype).view(torch.uint8).flatten().tolist()
    return bytes(raw)


def _full_artifact(tmp_path, matrix_format):
    """Minimal complete Qwen text binding set with the requested matrix codec."""
    h, vocab, layers = 64, 32, 1
    config = {
        "architectures": ["Qwen3_5ForCausalLM"],
        "model_type": "qwen3_5_text",
        "dtype": "bfloat16",
        "hidden_act": "silu",
        "attention_bias": False,
        "attn_output_gate": True,
        "mlp_only_layers": [],
        "rms_norm_eps": 1e-6,
        "hidden_size": h,
        "intermediate_size": 128,
        "num_hidden_layers": layers,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 64,
        "linear_value_head_dim": 8,
        "linear_conv_kernel_dim": 4,
        "layer_types": ["linear_attention"],
        "vocab_size": vocab,
        "max_position_embeddings": 64,
        "tie_word_embeddings": True,
    }
    plan = _plan(config, matrix_format)
    objects, bindings = {}, {}
    for spec in plan.bindings:
        shape = spec.shape
        if spec.format in CODECS and codec_for(spec.format).quantized:
            blob = _quant_payload(shape, spec.format, forbid_minus_128=True)
        else:
            blob = _dense_payload(shape, spec.format)
        objects[spec.binding] = (spec.format, shape, blob)
        bindings[spec.binding] = {"object": spec.binding}
    if plan.tied_embeddings:
        bindings["text/output_head"] = bindings["text/token_embedding"]
    return _write_artifact(tmp_path, objects, bindings, config)


def _plan(config, matrix_format):
    from venturi.models.qwen_source import QwenSourcePlan

    return QwenSourcePlan.from_config(config, matrix_format=matrix_format)


@pytest.mark.parametrize("fmt", ["q8_g32_fp16", "q4_g64_fp16", "q6_g64_fp16", "fp16"])
def test_full_artifact_admits_and_verifies(tmp_path, fmt):
    path = _full_artifact(tmp_path, fmt)
    artifact = NInferArtifact(path)
    weights = QwenQ8Weights.from_ninfer(artifact, matrix_format=fmt)
    assert weights.plan.tied_embeddings
    report = verify_artifact(artifact, sample_objects=5)
    assert report["status"] == "storage-verified"
    assert fmt in report["codecs_present"]


def test_q8_upload_bytes_formula_unchanged():
    codec = codec_for("q8_g32_fp16")
    n, k = 512, 320
    padded = (k + 127) // 128 * 128
    assert codec.upload_bytes(n, k) == n * (padded + padded // 32 * 2)
    assert codec.encoded_bytes(n, k) == row_geometry((n, k), "q8_g32_fp16")[-1]


@pytest.mark.parametrize("fmt,bits,group", [("q4_g64_fp16", 4, 64), ("q8_g32_fp16", 8, 32), ("q5_g64_fp16", 5, 64), ("q6_g64_fp16", 6, 64)])
def test_codec_geometry_matches_container(fmt, bits, group):
    codec = codec_for(fmt)
    assert (codec.bits, codec.group) == (bits, group)
    n, k = 6, 130
    assert codec.encoded_bytes(n, k) == row_geometry((n, k), fmt)[-1]
    padded = (k + 127) // 128 * 128
    assert codec.upload_bytes(n, k) == n * (
        padded * bits // 8 + (padded // group) * 2
    )


def test_unqualified_matrix_refused_for_execution(tmp_path):
    from venturi.models.qwen_load.spec import ensure_execution_qualified

    # q6 is registered for loading but never execution-qualified; q4 was
    # promoted on SM89 on 2026-11-11 and must now pass the union gate.
    path = _full_artifact(tmp_path, "q6_g64_fp16")
    weights = QwenQ8Weights.from_ninfer(
        NInferArtifact(path), matrix_format="q6_g64_fp16"
    )
    with pytest.raises(ValueError, match="execution qualification pending"):
        ensure_execution_qualified(weights)
    path = _full_artifact(tmp_path, "fp16")
    weights = QwenQ8Weights.from_ninfer(NInferArtifact(path), matrix_format="fp16")
    with pytest.raises(ValueError, match="execution qualification pending"):
        ensure_execution_qualified(weights)
    path = _full_artifact(tmp_path, "q4_g64_fp16")
    weights = QwenQ8Weights.from_ninfer(
        NInferArtifact(path), matrix_format="q4_g64_fp16"
    )
    ensure_execution_qualified(weights)  # qualified on SM89 since 2026-11-11


def test_q8_execution_gate_passes(tmp_path):
    from venturi.models.qwen_load.spec import ensure_execution_qualified

    path = _full_artifact(tmp_path, "q8_g32_fp16")
    weights = QwenQ8Weights.from_ninfer(NInferArtifact(path))
    ensure_execution_qualified(weights)  # must not raise


def test_read_linear_q4_reference_dequantization(tmp_path):
    # read_linear performs the explicit reference dequantization for q4
    # (reference/qualification paths); the execution gate lives in spec.
    path = _full_artifact(tmp_path, "q4_g64_fp16")
    weights = QwenQ8Weights.from_ninfer(
        NInferArtifact(path), matrix_format="q4_g64_fp16"
    )
    weight = weights.read_linear("text/token_embedding")
    assert tuple(weight.data.shape) == (32, 64)
    assert weight.data.dtype == torch.float16


def test_read_linear_dense_fp16_round_trip(tmp_path):
    path = _full_artifact(tmp_path, "fp16")
    weights = QwenQ8Weights.from_ninfer(
        NInferArtifact(path), matrix_format="fp16"
    )
    assert tuple(weights.read_float("text/token_embedding").shape) == (32, 64)


def test_unknown_format_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown weight format"):
        codec_for("q3_g17_int4")


def test_default_plan_is_still_q8(tmp_path):
    path = _full_artifact(tmp_path, "q8_g32_fp16")
    weights = QwenQ8Weights.from_ninfer(NInferArtifact(path))
    formats = {s.format for s in weights.plan.bindings}
    assert "q8_g32_fp16" in formats
    assert formats <= {"q8_g32_fp16", "bf16", "fp32"}
