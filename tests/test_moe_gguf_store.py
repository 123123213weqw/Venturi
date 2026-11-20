"""GGUFExpertStore over fused 3-D expert banks: per-expert slicing, chunked
bounded reads, bank layout enforcement. The GGUF fixture is hand-built with
row-identifiable Q2_K payloads so any wrong stride or chunk scatter shows up
as a value mismatch, not just a shape error.
"""

import hashlib
import struct

import pytest
import torch

from venturi.container.gguf import gguf_quantization as quant
from venturi.container.model_assets import ModelAsset
from venturi.models.moe.geometry import ExpertGeometry
from venturi.models.moe.gguf_store import GGUFExpertStore

# Q2_K: one 84-byte block covers 256 values; whole rows are self-contained.
Q2_K_BLOCK_VALUES = 256


def _q2_k_row(seed):
    """Deterministic Q2_K payload whose decode depends on the row bytes."""
    out = bytearray()
    out += bytes(((seed + 1) | ((seed + 3) << 4)) & 0xFF for _ in range(16))
    out += bytes((seed + k) % 256 for k in range(64))
    out += struct.pack("<e", 1.0 + (seed % 7)) + struct.pack("<e", 0.25)
    return bytes(out)


def _bank_body(experts, rows_per_expert):
    payload = bytearray()
    for seed in range(experts * rows_per_expert):
        payload += _q2_k_row(seed)
    return bytes(payload)


def _tensor_info(name, torch_shape, type_id, offset):
    """GGUF stores dims ggml-order (fastest varying first): reversed."""
    out = struct.pack("<Q", len(name)) + name.encode()
    out += struct.pack("<I", len(torch_shape))
    out += b"".join(struct.pack("<Q", d) for d in reversed(torch_shape))
    return out + struct.pack("<IQ", type_id, offset)


def expert_store_fixture(root, geometry, *, max_values):
    """One GGUF holding gate/up/down fused banks with distinguishable rows."""
    g = geometry
    tensors = [
        ("blk.7.ffn_gate_exps.weight", (g.num_experts, g.intermediate_size, g.hidden_size)),
        ("blk.7.ffn_up_exps.weight", (g.num_experts, g.intermediate_size, g.hidden_size)),
        ("blk.7.ffn_down_exps.weight", (g.num_experts, g.hidden_size, g.intermediate_size)),
    ]
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 0)
    bodies, decoded_banks = [], {}
    for _, shape in tensors:
        bodies.append(_bank_body(shape[0], shape[1]))
    offsets, cursor = [], 0
    for body in bodies:
        offsets.append(cursor)
        cursor += len(body) + (-len(body)) % 32
    for (name, shape), offset in zip(tensors, offsets):
        header += _tensor_info(name, shape, 10, offset)
    header += bytes((-len(header)) % 32)
    raw = header + b"".join(body + bytes((-len(body)) % 32) for body in bodies)
    for (name, shape), body in zip(tensors, bodies):
        flat = quant.decode_q2_k_rows(body, shape[2])
        decoded_banks[name] = flat.reshape(shape)
    path = root / "experts.gguf"
    path.write_bytes(raw)
    revision = "c" * 40
    asset = ModelAsset(
        "experts.gguf", len(raw), hashlib.sha256(raw).hexdigest(),
        f"https://huggingface.co/venturi-fixtures/moe/resolve/{revision}/experts.gguf",
        "venturi-fixtures/moe", revision, "experts.gguf", "sparse_lookup",
    )
    shard_assets = {name: (asset, 10) for name, _ in tensors}
    store = GGUFExpertStore(
        geometry, root, shard_assets, layer=7,
        gate_name=tensors[0][0], up_name=tensors[1][0], down_name=tensors[2][0],
        max_values=max_values,
    )
    return store, decoded_banks


GEOMETRY = ExpertGeometry(hidden_size=Q2_K_BLOCK_VALUES, num_experts=5,
                          top_k=2, intermediate_size=Q2_K_BLOCK_VALUES)


@pytest.mark.parametrize("max_values", [2**20, 8192])  # single-shot and forced chunking
def test_read_returns_the_experts_slice(tmp_path, max_values):
    store, banks = expert_store_fixture(tmp_path, GEOMETRY, max_values=max_values)
    with store:
        for expert in (0, 2, GEOMETRY.num_experts - 1):
            gate_up, down = store.read(expert)
            torch.testing.assert_close(
                gate_up,
                torch.cat([banks["blk.7.ffn_gate_exps.weight"][expert],
                           banks["blk.7.ffn_up_exps.weight"][expert]], dim=0),
                rtol=0, atol=0)
            torch.testing.assert_close(
                down, banks["blk.7.ffn_down_exps.weight"][expert], rtol=0, atol=0)


def test_expert_bounds_enforced(tmp_path):
    store, _ = expert_store_fixture(tmp_path, GEOMETRY, max_values=2**20)
    with store:
        for bad in (-1, GEOMETRY.num_experts, 1.5):
            with pytest.raises(ValueError):
                store.read(bad)


def test_missing_bank_rejected(tmp_path):
    expert_store_fixture(tmp_path, GEOMETRY, max_values=2**20)
    with pytest.raises(ValueError, match="absent from shard assets"):
        GGUFExpertStore(GEOMETRY, tmp_path, {}, layer=0,
                        gate_name="x", up_name="y", down_name="z")
