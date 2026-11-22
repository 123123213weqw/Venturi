"""GLM5 sparse MLA + DSA indexer correctness: query bottleneck, compressed
KV expansion (3D per-head), NoPE (no RoPE), indexer selection semantics."""

import pytest
import torch
from torch import nn

from venturi.models.glm5.mla import (
    Glm5Indexer,
    Glm5MLA,
    build_sparse_mask,
)

H, QK, VL, QL, KL = 4, 32, 32, 64, 48  # small geometry


def make_mla():
    torch.manual_seed(7)
    return Glm5MLA(hidden_size=128, num_heads=H, q_lora_rank=QL,
                   kv_lora_rank=KL, qk_head_dim=QK, v_head_dim=VL)


def make_indexer():
    torch.manual_seed(11)
    return Glm5Indexer(hidden_size=128, q_lora_rank=QL, num_heads=2,
                       head_dim=16, top_k=8, kpool=2)


class TestMLA:
    def test_no_rope_anywhere(self):
        mla = make_mla()
        x = torch.randn(1, 4, 128)
        with torch.no_grad():
            out, state, _ = mla(x)
        assert out.shape == (1, 4, 128)
        assert state.latent.shape == (1, 4, KL)
        # No rope table, no cos/sin parameters
        assert not hasattr(mla, 'rope') and not hasattr(mla, 'cos')

    def test_compressed_kv_expansion_per_head(self):
        mla = make_mla().eval()
        latent = torch.randn(1, 5, KL)
        with torch.no_grad():
            keys, values = mla._expand_kv(latent)
        assert keys.shape == (1, H, 5, QK)
        assert values.shape == (1, H, 5, VL)
        # Verify per-head matmul: keys[h] = latent @ k_b[h]
        h = 2
        expected_k = torch.matmul(latent[0], mla.k_b[h])  # [5, QK]
        torch.testing.assert_close(keys[0, h], expected_k, rtol=0, atol=0)
        # values[h] = latent @ v_b[h].T
        expected_v = torch.matmul(latent[0], mla.v_b[h].T)  # [5, VL]
        torch.testing.assert_close(values[0, h], expected_v, rtol=0, atol=0)

    def test_forward_matches_manual_attention(self):
        mla = make_mla().eval()
        x = torch.randn(1, 4, 128)
        with torch.no_grad():
            output, state, _ = mla(x)
            # Manual: q path → expand → attention → output
            q_latent = mla.q_a_proj(x)
            from venturi.models.glm5.mla import _rms_norm
            q_resid = _rms_norm(q_latent, mla.q_a_norm_weight, 1e-5)
            q = mla.q_b_proj(q_resid).view(1, 4, H, QK).transpose(1, 2)
            k_lat = _rms_norm(mla.kv_a_proj(x), mla.kv_a_norm_weight, 1e-5)
            keys, values = mla._expand_kv(k_lat)
            scores = torch.matmul(q, keys.transpose(2, 3)) * (QK ** -0.5)
            causal = torch.tril(torch.ones(4, 4, dtype=torch.bool))
            scores = scores.masked_fill(~causal, float("-inf"))
            probs = torch.softmax(scores.float(), dim=-1)
            attended = torch.matmul(probs.to(values.dtype), values)
            manual = mla.o_proj(attended.transpose(1, 2).reshape(1, 4, -1))
        torch.testing.assert_close(output, manual, rtol=1e-5, atol=1e-5)

    def test_state_latent_grows_correctly(self):
        mla = make_mla().eval()
        x = torch.randn(1, 3, 128)
        with torch.no_grad():
            _, s1, _ = mla(x)
            _, s2, _ = mla(x[:, :1], past=s1)
        assert s1.latent.shape == (1, 3, KL) and s1.tokens_seen == 3
        assert s2.latent.shape == (1, 4, KL) and s2.tokens_seen == 4

    def test_decode_matches_prefill(self):
        mla = make_mla().eval()
        x = torch.randn(1, 5, 128)
        with torch.no_grad():
            out_full, state, _ = mla(x)
            # Prefill 4, decode 1
            out_4, s4, _ = mla(x[:, :4])
            out_1, _, _ = mla(x[:, 4:5], past=s4)
        torch.testing.assert_close(out_full[:, 4:5], out_1, rtol=0.02, atol=0.02)


class TestIndexer:
    def test_key_computation(self):
        idx = make_indexer()
        x = torch.randn(1, 4, 128)
        key = idx.compute_key(x)
        assert key.shape == (1, 4, 16)  # head_dim=16

    def test_topk_selection_and_causality(self):
        idx = make_indexer().eval()
        x = torch.randn(1, 3, 128)
        q_resid = torch.randn(1, 3, QL)
        indexer_key = idx.compute_key(torch.randn(1, 16, 128))  # 16 total tokens
        with torch.no_grad():
            indices = idx(x, q_resid, indexer_key, position=10)
        assert indices.shape == (1, 3, 8)  # top_k=8
        assert indices.dtype == torch.int32
        # All valid indices < 16 and >= 0
        valid = indices >= 0
        assert bool((indices[valid] < 16).all())

    def test_always_select_tail(self):
        idx = make_indexer().eval()
        x = torch.randn(1, 1, 128)
        q_resid = torch.randn(1, 1, QL)
        indexer_key = idx.compute_key(torch.randn(1, 20, 128))
        with torch.no_grad():
            indices = idx(x, q_resid, indexer_key, position=10)
        # Position 10, so the last visible is token 9, pool 9//2=4, tokens 8,9
        # The tail pool should be in the selection
        selected = set(indices[0, 0].tolist())
        assert 8 in selected or 9 in selected  # tail pool included

    def test_shared_indexer_passthrough(self):
        """MLA with prev_topk skips indexer computation."""
        mla = make_mla().eval()
        x = torch.randn(1, 3, 128)
        fake_topk = torch.randint(0, 3, (1, 3, 4)).to(torch.int32)
        with torch.no_grad():
            out, _, topk = mla(x, prev_topk=fake_topk)
        assert torch.equal(topk, fake_topk)


class TestSparseMask:
    def test_mask_zero_at_selected(self):
        indices = torch.tensor([[[0, 2, -1]]], dtype=torch.int32)
        mask = build_sparse_mask(indices, 1, 1, 4, torch.float32)
        assert mask[0, 0, 0, 0] == 0.0
        assert mask[0, 0, 0, 2] == 0.0
        assert mask[0, 0, 0, 1] < 0  # not selected → -inf


@pytest.mark.cuda
def test_real_weights_mla_forward(tmp_path=None):
    import os, sys
    sys.path.insert(0, os.environ.get("VENTURI_REPO", "/data/wangyue/venturi-staged") + "/src")
    root = os.environ.get("VENTURI_GLM5_PATH")
    if not root or os.environ.get("VENTURI_TEST_CUDA") != "1":
        pytest.skip("Set VENTURI_GLM5_PATH + VENTURI_TEST_CUDA=1")
    from pathlib import Path
    from venturi.container.gguf.gguf_inventory import inspect_gguf
    from venturi.container.gguf.gguf_quant_rows import QUANT_TABLES, quant_row_bytes
    from venturi.container.gguf import gguf_quantization as quant

    tensors = {}
    for shard in sorted(Path(root).glob("*.gguf")):
        for t in inspect_gguf(shard).tensors:
            if t.name.startswith("blk.3."):
                tensors[t.name.split(".", 2)[2]] = (t, shard)
    print(f"blk.3 tensors: {len(tensors)}")

    mla = Glm5MLA(hidden_size=4096, num_heads=64, q_lora_rank=1536,
                  kv_lora_rank=512, qk_head_dim=256, v_head_dim=256).to(torch.float16).cuda(0).eval()

    def load(short, target):
        t, shard = tensors[short]
        decoder = getattr(quant, QUANT_TABLES[t.type_id][0])
        width = t.shape[-1]
        rb = quant_row_bytes(t.type_id, width)
        rows = 1
        for d in t.shape[:-1]: rows *= d
        import os as os_mod
        fd = os_mod.open(str(shard), os_mod.O_RDONLY)
        try:
            rem, pieces = rows * rb, []
            while rem:
                data = os_mod.pread(fd, rem, t.offset + (rows * rb - rem))
                pieces.append(data); rem -= len(data)
        finally: os_mod.close(fd)
        raw = decoder(b"".join(pieces), width, max_values=2**28).reshape(-1).to(torch.float16)
        target.copy_(raw.reshape(target.shape))

    with torch.no_grad():
        load("attn_q_a.weight", mla.q_a_proj.weight)
        load("attn_q_a_norm.weight", mla.q_a_norm_weight)
        load("attn_q_b.weight", mla.q_b_proj.weight)
        load("attn_kv_a_mqa.weight", mla.kv_a_proj.weight)
        load("attn_kv_a_norm.weight", mla.kv_a_norm_weight)
        load("attn_k_b.weight", mla.k_b)
        load("attn_v_b.weight", mla.v_b)
        load("attn_output.weight", mla.o_proj.weight)

    x = torch.randn(1, 4, 4096, dtype=torch.float16, device="cuda:0") * 0.1
    with torch.no_grad():
        output, state, _ = mla(x)
        print(f"output: {tuple(output.shape)} finite={bool(torch.isfinite(output).all())} "
              f"std={output.float().std():.4f}")
        print(f"state.latent: {tuple(state.latent.shape)} finite={bool(torch.isfinite(state.latent).all())}")
        out2, _, _ = mla(x[:, :1], past=state)
        print(f"decode: {tuple(out2.shape)} finite={bool(torch.isfinite(out2).all())} std={out2.float().std():.4f}")
    assert bool(torch.isfinite(output).all())
    print("MLA SMOKE OK")
