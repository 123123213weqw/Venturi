"""GLM5 tensor directory validation and core-tensor loading.

Synthetic small GGUF (via the quant fixture builder) exercises the directory
contract; a gated real-weight test runs only with VENTURI_GLM5_PATH.
"""

import json
import os
import struct
from pathlib import Path

import pytest
import torch

from venturi.models.glm5.binding import Glm5TensorDirectory
from venturi.models.glm5.geometry import (
    DENSE_MLP_TENSORS,
    KDA_TENSORS,
    MLA_TENSORS,
    PER_LAYER_SHARED,
    SPARSE_ONLY,
    SPARSE_MLP_TENSORS,
    GLOBAL_TENSORS,
    glm5_text_geometry_from_metadata,
)
from venturi.models.glm5.loader import (
    load_core_tensors,
    plan_glm5_text,
)

FIXTURE = Path(__file__).parent / "fixtures" / "glm5-metadata.json"


@pytest.fixture(scope="module")
def metadata():
    return json.loads(FIXTURE.read_text())


def _build_directory(tmp_path, metadata, *, drop=None, dtype_map=None):
    """Hand-built GGUF shards carrying every expected tensor name."""
    import hashlib

    from venturi.container.gguf.gguf_inventory import inspect_gguf
    from venturi.container.model_assets import ModelAsset

    geometry = glm5_text_geometry_from_metadata(metadata)
    dtype_map = dtype_map or {}
    tensors = {}
    # Tiny shapes: experts (2, 3, hidden), everything else collapsed rows.
    for name in GLOBAL_TENSORS:
        tensors[name] = (0, (8, 4))
    from venturi.models.glm5.geometry import NEXTN_TENSORS
    for layer in range(geometry.num_layers + geometry.nextn_layers):
        is_nextn = layer >= geometry.num_layers
        if is_nextn:
            suffixes = list(MLA_TENSORS) + ["attn_norm.weight", "ffn_norm.weight"] + list(NEXTN_TENSORS)
            for suffix in suffixes:
                tensors[f"blk.{layer}.{suffix}"] = (0, (4,))
            for suffix in SPARSE_MLP_TENSORS + SPARSE_ONLY:
                name = f"blk.{layer}.{suffix}"
                tensors[name] = (10, (2, 1, 256)) if suffix.endswith("_exps.weight") else (0, (4,))
            continue
        attention = MLA_TENSORS if geometry.attention_kind(layer) == "sparse_mla" else KDA_TENSORS
        for suffix in attention + PER_LAYER_SHARED:
            name = f"blk.{layer}.{suffix}"
            tensors[name] = (0, (4,))
        if geometry.mlp_kind(layer) == "sparse":
            for suffix in SPARSE_MLP_TENSORS + SPARSE_ONLY:
                name = f"blk.{layer}.{suffix}"
                if suffix.endswith("_exps.weight"):
                    tensors[name] = (10, (2, 1, 256))   # Q2_K fused bank, width 256-aligned
                else:
                    tensors[name] = (0, (4, 3))
        else:
            for suffix in DENSE_MLP_TENSORS:
                tensors[f"blk.{layer}.{suffix}"] = (0, (4, 4))
    if drop:
        tensors.pop(drop, None)

    # Single GGUF file with F32 (type 0) 4-byte and Q2_K/BF16 blocks.
    payload = bytearray()
    entries = []
    import math
    for name, (type_id, shape) in sorted(tensors.items()):
        offset = len(payload)
        count = math.prod(shape)
        if type_id == 0:
            payload += struct.pack(f"<{count}f", *([0.5] * count))
        elif type_id == 10:
            for _ in range(count // 256):
                payload += bytes([17] * 16) + bytes(range(64)) + struct.pack("<e", 1.0) + struct.pack("<e", 0.25)
        payload += bytes((-len(payload)) % 32)
        entries.append((name, shape, type_id, offset))

    header = b"GGUF" + struct.pack("<IQQ", 3, len(entries), 0)
    for name, shape, type_id, offset in entries:
        raw = name.encode()
        header += struct.pack("<Q", len(raw)) + raw
        header += struct.pack("<I", len(shape))
        for dimension in shape[::-1]:
            header += struct.pack("<Q", dimension)
        header += struct.pack("<IQ", type_id, offset)
    blob = header + bytes((-len(header)) % 32) + bytes(payload)
    path = tmp_path / "glm5-mini.gguf"
    path.write_bytes(blob)
    inventory = inspect_gguf(path)
    revision = "c" * 40
    asset = ModelAsset(
        "glm5-mini.gguf", len(blob), hashlib.sha256(blob).hexdigest(),
        f"https://huggingface.co/venturi-fixtures/glm5/resolve/{revision}/glm5-mini.gguf",
        "venturi-fixtures/glm5", revision, "glm5-mini.gguf", "backbone",
    )
    directory = Glm5TensorDirectory(
        geometry, [(path, inventory.tensors)])
    return geometry, directory, path, asset


def test_directory_validates_and_plans(tmp_path, metadata):
    geometry, directory, path, asset = _build_directory(tmp_path, metadata)
    directory.validate()
    plan = plan_glm5_text(directory)
    summary = plan.summary()
    assert summary["blocks"] == 45 and summary["nextn_blocks"] == 1
    assert summary["layer_kinds"] == {
        "kda/dense": 3, "kda/sparse": 32, "sparse_mla/sparse": 11,
    }  # MTP draft block (layer 45) also counts as kda/sparse
    assert summary["core_bytes"] > 0 and summary["expert_bank_bytes"] > 0


def test_missing_tensor_fails_closed(tmp_path, metadata):
    geometry, directory, path, asset = _build_directory(
        tmp_path, metadata, drop="blk.4.ssm_a")
    with pytest.raises(ValueError, match="missing"):
        directory.validate()


def test_load_core_tensors_decodes_finite(tmp_path, metadata):
    geometry, directory, path, asset = _build_directory(tmp_path, metadata)
    directory.validate()
    tensors = load_core_tensors(directory, device="cpu", layers=(4,))
    assert "blk.4.attn_norm.weight" in tensors
    assert "blk.4.ssm_a" in tensors
    assert tensors["blk.4.ssm_beta.weight"].dtype == torch.float32
    assert all(bool(torch.isfinite(t).all()) for t in tensors.values())
    assert "blk.4.ffn_gate_exps.weight" not in tensors  # banks stay on disk


@pytest.mark.cuda
def test_real_weights_directory_and_plan(tmp_path):
    root = os.environ.get("VENTURI_GLM5_PATH")
    if not root:
        pytest.skip("Set VENTURI_GLM5_PATH to the GLM-5.3-Flash GGUF directory")
    from venturi.container.gguf.gguf_inventory import inspect_gguf

    shards = []
    metadata = None
    for path in sorted(Path(root).glob("*.gguf")):
        inventory = inspect_gguf(path)
        shards.append((path, inventory.tensors))
        if metadata is None:
            metadata = dict(inventory.metadata)
    geometry = glm5_text_geometry_from_metadata(metadata)
    directory = Glm5TensorDirectory(geometry, shards).validate()
    plan = plan_glm5_text(directory)
    summary = plan.summary()
    print(summary)
    tensors = load_core_tensors(
        directory, device="cuda:0", dtype=torch.float16, layers=(4,), include_globals=False)
    assert tensors["blk.4.ffn_gate_inp.weight"].shape == (288, 4096)
    assert bool(torch.isfinite(tensors["blk.4.ffn_gate_inp.weight"]).all())
