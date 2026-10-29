"""Main QSA reference/state contracts; no model speed or native claim."""

from dataclasses import replace
import json

import pytest
from safetensors.torch import save_file
import torch

from venturi.checkpoint import inspect_checkpoint
from venturi.models.flash_next_attention import FlashNextQSAAttention, QSAAttentionGeometry, QSAState
from venturi.models.flash_next_qsa import QSAIndexerGeometry


def module(dtype=torch.float32):
    torch.manual_seed(116)
    g = QSAAttentionGeometry(QSAIndexerGeometry(8, 2, 4, 4, 2), 4, 2, 4)
    m = FlashNextQSAAttention(g).to(dtype).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn_like(p) * 0.15)
    return m


def inputs(batch, length, dtype=torch.float32, past=0):
    x = torch.randn(batch, length, 8, dtype=dtype)
    angle = torch.arange(past + length, dtype=torch.float32).view(1, -1, 1).expand(batch, -1, 2) * 0.3
    pos = (angle.cos().to(dtype), angle.sin().to(dtype))
    visible = torch.arange(past + length).view(1, 1, 1, -1) <= torch.arange(past, past + length).view(1, 1, -1, 1)
    mask = torch.where(visible.expand(batch, 1, -1, -1), torch.zeros((), dtype=dtype), torch.finfo(dtype).min)
    return x, pos, mask


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_prefill_chunk_state_ownership_and_accepted_prefix(dtype):
    m = module(dtype)
    x, pos, mask = inputs(2, 9, dtype)
    full, state, _ = m(x, pos, mask)
    first, past, _ = m(x[:, :3], tuple(p[:, :3] for p in pos), mask[:, :, :3, :3])
    retained = tuple(t.clone() for t in (past.raw_indexer_keys, past.keys, past.values))
    tail, proposal, _ = m(x[:, 3:], pos, mask[:, :, 3:], past=past)
    tol = 1e-6 if dtype == torch.float32 else 3e-3
    torch.testing.assert_close(first, full[:, :3], atol=tol, rtol=tol)
    torch.testing.assert_close(tail, full[:, 3:], atol=tol, rtol=tol)
    for a, b in zip((past.raw_indexer_keys, past.keys, past.values), retained):
        assert torch.equal(a, b)
    for t in (proposal.raw_indexer_keys, proposal.keys, proposal.values):
        assert t.untyped_storage().nbytes() == t.numel() * t.element_size()
    accepted = proposal.truncate(4)
    one, next_state, _ = m(x[:, 4:5], tuple(p[:, :5] for p in pos), mask[:, :, 4:5, :5], past=accepted)
    torch.testing.assert_close(one, full[:, 4:5], atol=tol, rtol=tol)
    assert next_state.keys.shape[2] == 5 and accepted.keys.shape[2] == 4
    assert accepted.keys.data_ptr() != proposal.keys.data_ptr()
    assert state.truncate(0).keys.shape[2] == 0
    with pytest.raises(ValueError, match="prefix"):
        state.truncate(True)


def test_per_head_query_gate_not_global_half_and_gqa_manual_result():
    m = module()
    x, pos, mask = inputs(1, 3)
    output, state, p = m(x, pos, mask)
    # Independent plain formula for no-pruning short case with zero rotation.
    pos = (torch.ones_like(pos[0]), torch.zeros_like(pos[1]))
    output, state, p = m(x, pos, mask)
    qgate = x @ m.q_proj.weight.T
    q, gate = qgate.view(1, 3, 4, 8).chunk(2, -1)
    def norm(t, w):
        return (t.float() * torch.rsqrt(t.float().square().mean(-1, keepdim=True) + 1e-6) * (1 + w.float())).to(t.dtype)
    q = norm(q, m.q_norm.weight).transpose(1, 2)
    k = norm((x @ m.k_proj.weight.T).view(1, 3, 2, 4), m.k_norm.weight).transpose(1, 2)
    v = (x @ m.v_proj.weight.T).view(1, 3, 2, 4).transpose(1, 2)
    scores = q @ k.repeat_interleave(2, 1).transpose(2, 3) * 0.5 + mask
    probs = scores.softmax(-1)
    y = (probs @ v.repeat_interleave(2, 1)).transpose(1, 2).reshape(1, 3, -1)
    expected = (y * gate.reshape(1, 3, -1).sigmoid()) @ m.o_proj.weight.T
    torch.testing.assert_close(output, expected)
    assert torch.equal(p, probs) and torch.equal(state.keys, k)
    assert not torch.equal(gate.reshape(1, 3, -1), qgate.chunk(2, -1)[1])


def test_fully_masked_eager_nan_is_preserved_not_silently_fixed():
    m = module()
    x, pos, mask = inputs(1, 2)
    mask[:, :, -1] = torch.finfo(mask.dtype).min
    output, _, probabilities = m(x, pos, mask)
    assert torch.isfinite(output[:, 0]).all()
    assert torch.isnan(output[:, -1]).all() and torch.isnan(probabilities[:, :, -1]).all()


def test_budget_admission_precedes_projections(monkeypatch):
    m = module()
    x, pos, mask = inputs(1, 2)
    def forbidden(*args, **kwargs):
        raise AssertionError("must reject before indexer allocation/projection")
    monkeypatch.setattr(m.indexer, "forward", forbidden)
    m.max_score_elements = 1
    with pytest.raises(ValueError, match="budget"):
        m(x, pos, mask)
    m.max_score_elements = 4096
    m.max_state_bytes = 4096
    m.max_expanded_kv_bytes = 1
    with pytest.raises(ValueError, match="expanded-KV budget"):
        m(x, pos, mask)
    m.max_score_elements = 4096
    m.max_state_bytes = 1
    with pytest.raises(ValueError, match="budget"):
        m(x, pos, mask)


def test_reported_b8_long_history_gqa_expansion_is_rejected_without_allocation():
    with torch.device("meta"):
        g = QSAAttentionGeometry(QSAIndexerGeometry(2560, 4, 128, 2048, 4), 24, 2, 256)
        m = FlashNextQSAAttention(g, max_state_bytes=256 * 2**20).eval()
        x = torch.empty(8, 1, 2560)
        past = QSAState(torch.empty(8, 4100, 128), torch.empty(8, 2, 4100, 256), torch.empty(8, 2, 4100, 256))
        mask = torch.empty(8, 1, 1, 4101)
        positions = torch.empty(8, 4101, 64), torch.empty(8, 4101, 64)
        with pytest.raises(ValueError, match="expanded-KV budget"):
            m(x, positions, mask, past=past)


def test_mask_cache_training_and_geometry_errors():
    m = module()
    x, pos, mask = inputs(1, 2)
    with pytest.raises(ValueError, match="additive"):
        m(x, pos, mask == 0)
    with pytest.raises(ValueError, match="QSAState"):
        m(x, pos, mask, past={})
    bad = QSAState(torch.zeros(1, 3, 4), torch.zeros(1, 2, 2, 4), torch.zeros(1, 2, 3, 4))
    with pytest.raises(ValueError, match="past geometry"):
        m(x, pos, mask, past=bad)
    m.train()
    with pytest.raises(ValueError, match="inference"):
        m(x, pos, mask)
    with pytest.raises(ValueError):
        replace(m.geometry, query_heads=3)
    with pytest.raises(ValueError):
        replace(m.geometry, kv_heads=True)


def test_bounded_floating_checkpoint_binding(tmp_path):
    m = module(torch.bfloat16)
    prefix = "model.language_model.layers.3.self_attn"
    save_file({prefix + "." + k: v.detach().clone() for k, v in m.state_dict().items()}, tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "fixture"}))
    cp = inspect_checkpoint(tmp_path)
    loaded = FlashNextQSAAttention.from_checkpoint(cp, m.geometry, prefix)
    assert not loaded.training
    for name, value in loaded.state_dict().items():
        assert torch.equal(value.view(torch.int16), m.state_dict()[name].view(torch.int16))
    with pytest.raises(ValueError, match="budget"):
        FlashNextQSAAttention.from_checkpoint(cp, m.geometry, prefix, max_weight_bytes=1)
