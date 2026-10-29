"""Small complete floating decoder assembly/private state, synthetic weights."""

from dataclasses import replace

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from test_flash_next_attention import inputs
from venturi.models.flash_next_attention import FlashNextQSAAttention, QSAAttentionGeometry
from venturi.models.flash_next_gdn import FlashNextGDN, GDNGeometry
from venturi.models.flash_next_layer import FlashNextDecoderLayer
from venturi.models.flash_next_moe import FlashNextMoE, MoEGeometry, TensorExpertStore
from venturi.models.flash_next_ple import FlashNextPLE, PLEGeometry
from venturi.models.flash_next_qsa import QSAIndexerGeometry
from venturi.models.flash_next_residual import FlashNextGatedResidual


def module(kind="gdn", ple=True, dtype=torch.float32):
    torch.manual_seed(213)
    mg = MoEGeometry(8, 4, 2, 6, 6)
    bank = TensorExpertStore(mg, {i: (torch.randn(12, 8)*0.1, torch.randn(8, 6)*0.1) for i in range(4)})
    attention = FlashNextGDN(GDNGeometry(8, 2, 6, 4, 4, 4)) if kind == "gdn" else FlashNextQSAAttention(
        QSAAttentionGeometry(QSAIndexerGeometry(8, 2, 4, 4, 2), 4, 2, 4), attention_backend="sdpa")
    m = FlashNextDecoderLayer(attention, FlashNextMoE(mg, bank), FlashNextGatedResidual(8, 2, 3),
        FlashNextGatedResidual(8, 2, 3), ple=FlashNextPLE(PLEGeometry(8, 2, 16, 4, 3)) if ple else None).to(dtype).eval()
    with torch.no_grad():
        for name, p in m.named_parameters():
            p.copy_(torch.randn_like(p)*0.1 + (1 if name == "attention.norm.weight" else 0))
    return m


@pytest.mark.parametrize("kind", ["gdn", "qsa"])
@pytest.mark.parametrize("has_ple", [False, True])
def test_layer_manual_assembly_order_and_no_input_mutation(kind, has_ple):
    m = module(kind, has_ple)
    x, rows = torch.randn(2, 7, 16), torch.randn(2, 7, 16)
    before = x.clone()
    _, pos, mask = inputs(2, 7)
    kwargs = dict(position_embeddings=pos, attention_mask=mask == 0, ple_rows=rows if has_ple else None)
    with sdpa_kernel(SDPBackend.MATH), torch.inference_mode():
        actual, state = m(x, **kwargs)
        conditioned = x + m.ple(x, rows)[0] if has_ple else x
        mixed, residual, gate = m.attn_hyper_connection(conditioned)
        block = m.attention(mixed)[0] if kind == "gdn" else m.attention(mixed, pos, mask == 0)[0]
        hidden = residual + (block.unsqueeze(-2)*gate.unsqueeze(-1)).flatten(-2)
        mixed, residual, gate = m.mlp_hyper_connection(hidden)
        expected = residual + (m.mlp(mixed).unsqueeze(-2)*gate.unsqueeze(-1)).flatten(-2)
    assert torch.equal(actual, expected) and torch.equal(before, x)
    assert (state.ple is not None) == has_ple


def test_layer_proposals_replay_accepted_prefix_and_failure_does_not_commit(monkeypatch):
    m = module()
    x, rows = torch.randn(2, 7, 16), torch.randn(2, 7, 16)
    _, base = m(x[:, :3], ple_rows=rows[:, :3])
    saved = base.snapshot()
    _, proposal = m(x[:, 3:], ple_rows=rows[:, 3:], past=base)
    accepted, state = m(x[:, 3:4], ple_rows=rows[:, 3:4], past=saved)
    whole, expected = m(x[:, :4], ple_rows=rows[:, :4])
    torch.testing.assert_close(accepted, whole[:, -1:], atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(state.attention.recurrent, expected.attention.recurrent, atol=3e-6, rtol=3e-6)
    assert proposal.attention.tokens_seen == 7 and state.attention.tokens_seen == 4
    assert torch.equal(base.attention.recurrent, saved.attention.recurrent) and torch.equal(base.ple.conv, saved.ple.conv)
    def fail(*args):
        raise ValueError("Selected expert is absent; cannot prune/substitute")
    monkeypatch.setattr(m.mlp, "forward", fail)
    with pytest.raises(ValueError, match="Selected expert"):
        m(x[:, 3:], ple_rows=rows[:, 3:], past=base)
    assert torch.equal(base.attention.recurrent, saved.attention.recurrent) and torch.equal(base.ple.conv, saved.ple.conv)
    with pytest.raises(ValueError, match="together"):
        m(x[:, :1], ple_rows=rows[:, :1], past=replace(base, ple=replace(base.ple, tokens_seen=4)))


def test_layer_requires_matching_explicit_components_rows_and_masks():
    m = module("qsa")
    x = torch.zeros(1, 1, 16)
    with pytest.raises(ValueError, match="PLE rows"):
        m(x)
    with pytest.raises(ValueError, match="positions"):
        m(x, ple_rows=x)
    with pytest.raises(ValueError, match="PLE rows"):
        module(ple=False)(x, ple_rows=x)
