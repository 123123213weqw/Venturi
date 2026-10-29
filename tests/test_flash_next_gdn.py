"""Floating GDN block geometry, private state and padding/checkpoint contracts."""

from dataclasses import replace
import json

import pytest
from safetensors.torch import save_file
import torch
from torch.nn import functional as F

from venturi.checkpoint import inspect_checkpoint
from venturi.models.flash_next_gdn import FlashNextGDN, GDNGeometry, GDNState


def module(dtype=torch.float32):
    torch.manual_seed(351)
    m = FlashNextGDN(GDNGeometry(8, 2, 6, 4, 3, 4)).to(dtype).eval()
    with torch.no_grad():
        for name, p in m.named_parameters():
            p.copy_(torch.randn_like(p) * 0.1 + (1 if name == "norm.weight" else 0))
    return m


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_initial_and_cached_conv_history_is_compact_owned_chronological(dtype):
    m = module(dtype)
    x = torch.randn(2, 7, 8, dtype=dtype)
    _, first = m(x[:, :2])
    raw = m.in_proj_qkv(x).transpose(1, 2)
    torch.testing.assert_close(first.conv, F.pad(raw[..., :2], (2, 0)), rtol=0, atol=0)
    retained = first.snapshot()
    _, final = m(x[:, 2:], past=first)
    torch.testing.assert_close(final.conv, raw[..., -4:], rtol=0, atol=0)
    assert torch.equal(first.conv, retained.conv) and torch.equal(first.recurrent, retained.recurrent)
    assert final.conv.untyped_storage().nbytes() == final.conv.numel() * final.conv.element_size()
    assert first.tokens_seen == 2 and final.tokens_seen == 7


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_prefill_vs_cached_decode_and_rollback_replay(dtype):
    m = module(dtype)
    x = torch.randn(2, 7, 8, dtype=dtype)
    with torch.inference_mode():
        whole, state = m(x)
        outputs, past = [], None
        for t in range(7):
            output, past = m(x[:, t:t+1], past=past)
            outputs.append(output)
        tolerance = 2e-6 if dtype == torch.float32 else 8e-3
        torch.testing.assert_close(torch.cat(outputs, 1), whole, atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(past.recurrent, state.recurrent, atol=3e-6, rtol=3e-6)
        _, base = m(x[:, :3])
        snapshot = base.snapshot()
        _, proposed = m(x[:, 3:], past=base)
        accepted_output, accepted = m(x[:, 3:4], past=snapshot)
        direct_output, direct = m(x[:, :4])
        torch.testing.assert_close(accepted_output, direct_output[:, -1:], atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(accepted.recurrent, direct.recurrent, atol=3e-6, rtol=3e-6)
        assert accepted.tokens_seen == 4 and proposed.tokens_seen == 7
        assert torch.equal(snapshot.conv, base.conv) and torch.equal(snapshot.recurrent, base.recurrent)
        assert snapshot.conv.data_ptr() != base.conv.data_ptr()


def test_padding_masks_input_but_does_not_skip_recurrent_or_conv_update():
    m = module()
    x = torch.randn(2, 3, 8)
    _, base = m(x)
    masked, proposed = m(x[:, :1], past=base, attention_mask=torch.zeros(2, 1, dtype=torch.bool))
    zeroed, expected = m(torch.zeros_like(x[:, :1]), past=base)
    assert torch.equal(masked, zeroed) and torch.equal(proposed.recurrent, expected.recurrent)
    assert torch.equal(proposed.conv, expected.conv)
    assert not torch.equal(proposed.recurrent, base.recurrent) and not torch.equal(proposed.conv, base.conv)
    assert proposed.tokens_seen == base.tokens_seen + 1


def test_cached_single_slices_conv_before_silu_matching_source_rounding(monkeypatch):
    m = module()
    x = torch.randn(2, 3, 8)
    _, state = m(x)
    original, shapes = F.silu, []
    def record(tensor, *args, **kwargs):
        shapes.append(tuple(tensor.shape))
        return original(tensor, *args, **kwargs)
    monkeypatch.setattr(F, "silu", record)
    m(x[:, :1], past=state)
    assert shapes == [(2, m.geometry.conv_channels, 1)]


def test_budget_rejection_before_projection_and_invalid_state_mask_training(monkeypatch):
    m, x = module(), torch.zeros(1, 2, 8)
    with pytest.raises(ValueError, match="2D bool"):
        m(x, attention_mask=torch.ones(1, 2))
    bad = GDNState(torch.zeros(1, 34, 4), torch.zeros(1, 6, 3, 4), 1)
    with pytest.raises(ValueError, match="geometry"):
        m(x, past=bad)
    with pytest.raises(ValueError, match="tokens_seen"):
        m(x, past=replace(bad, tokens_seen=True))
    m.train()
    with pytest.raises(ValueError, match="inference"):
        m(x)
    m.eval()
    def forbidden(*args):
        raise AssertionError("rejected work must not reach projection")
    monkeypatch.setattr(m.in_proj_qkv, "forward", forbidden)
    m.max_work_elements = 1
    with pytest.raises(ValueError, match="budget"):
        m(x)


@pytest.mark.parametrize("field,value", [("key_heads", True), ("value_heads", 7), ("conv_kernel", 17), ("eps", True)])
def test_geometry_rejects_unsupported(field, value):
    with pytest.raises(ValueError):
        replace(module().geometry, **{field: value})


def test_explicit_flashnext_config_gate_contract():
    raw = dict(model_type="qwen4_exp", text_config=dict(model_type="qwen4_exp_text", hidden_size=8,
        linear_num_key_heads=2, linear_num_value_heads=6, linear_key_head_dim=4,
        linear_value_head_dim=3, linear_conv_kernel_dim=4, hidden_act="silu", output_gate_type="sigmoid"))
    assert GDNGeometry.from_dict(raw) == module().geometry
    raw["text_config"]["output_gate_type"] = "silu"
    with pytest.raises(ValueError, match="sigmoid"):
        GDNGeometry.from_dict(raw)
    raw["model_type"] = "qwen3_5"
    with pytest.raises(ValueError, match="qwen4_exp"):
        GDNGeometry.from_dict(raw)


def test_small_original_floating_checkpoint_binding_and_budget(tmp_path):
    m = module(torch.bfloat16)
    prefix = "model.language_model.layers.0.linear_attn"
    save_file({prefix + "." + k: v.detach().clone() for k, v in m.state_dict().items()}, tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "fixture"}))
    cp = inspect_checkpoint(tmp_path)
    loaded = FlashNextGDN.from_checkpoint(cp, m.geometry, prefix)
    assert len(loaded.state_dict()) == 9 and not loaded.training
    for name, value in loaded.state_dict().items():
        assert torch.equal(value.view(torch.int16), m.state_dict()[name].view(torch.int16))
    with pytest.raises(ValueError, match="budget"):
        FlashNextGDN.from_checkpoint(cp, m.geometry, prefix, max_weight_bytes=1)
