"""PLE gate/dilated history and explicit CPU-row bridge contracts."""

from dataclasses import replace
import json

import pytest
from safetensors.torch import save_file
import torch

from test_flash_next import fixture as checkpoint_fixture  # noqa: F401
from venturi.checkpoint import inspect_checkpoint
from venturi.models.flash_next_ngram import NGramRequest, PreparedNGramTable, ShardedNGramTable
from venturi.models.flash_next_ple import FlashNextPLE, PLEGeometry, PLEState, gather_ple_rows


def module(dtype=torch.float32):
    torch.manual_seed(1251)
    m = FlashNextPLE(PLEGeometry(8, 2, 16, 4, 3)).to(dtype).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn_like(p) * 0.1)
    return m


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gate_zero_query_is_half_value_per_stream(dtype):
    m = module(dtype)
    x, rows = torch.zeros(2, 3, 16, dtype=dtype), torch.randn(2, 3, 16, dtype=dtype)
    with torch.no_grad():
        m.conv1d.weight.zero_()
    output, state = m(x, rows)
    expected = (m.value_proj(rows).unsqueeze(-2).expand(-1, -1, 2, -1) * 0.5).flatten(-2)
    assert torch.equal(output, expected)
    assert state.conv.shape == (2, 16, 9) and state.tokens_seen == 3


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cached_dilated_history_prefill_and_rejected_prefix_replay(dtype):
    m = module(dtype)
    x, rows = torch.randn(2, 13, 16, dtype=dtype), torch.randn(2, 13, 16, dtype=dtype)
    output, full = m(x, rows)
    first, past = m(x[:, :4], rows[:, :4])
    snapshot = past.snapshot()
    tail, proposed = m(x[:, 4:], rows[:, 4:], past=past)
    tol = 3e-6 if dtype == torch.float32 else 0.03
    torch.testing.assert_close(torch.cat((first, tail), 1), output, atol=tol, rtol=tol)
    torch.testing.assert_close(proposed.conv, full.conv, atol=tol, rtol=tol)
    assert torch.equal(past.conv, snapshot.conv) and past.conv.data_ptr() != snapshot.conv.data_ptr()
    accepted, accepted_state = m(x[:, 4:5], rows[:, 4:5], past=snapshot)
    direct, direct_state = m(x[:, :5], rows[:, :5])
    torch.testing.assert_close(accepted, direct[:, -1:], atol=tol, rtol=tol)
    torch.testing.assert_close(accepted_state.conv, direct_state.conv, atol=tol, rtol=tol)
    assert accepted_state.tokens_seen == 5 and proposed.tokens_seen == 13
    assert proposed.conv.untyped_storage().nbytes() == proposed.conv.numel() * proposed.conv.element_size()


def test_masked_current_token_can_still_read_past_conv_and_advances_history():
    m = module()
    with torch.no_grad():
        m.conv1d.weight.fill_(0.1)
    past = PLEState(torch.ones(1, 16, 9), 9)
    output, state = m(torch.zeros(1, 1, 16), torch.ones(1, 1, 16), past=past,
                      conv_mask=torch.zeros(1, 1, dtype=torch.bool))
    assert output.count_nonzero() == output.numel()
    assert state.tokens_seen == 10 and not state.conv[..., -1].any()
    assert past.conv.eq(1).all()


def test_cpu_row_bridge_preserves_private_eos_history_and_owned_data(checkpoint_fixture):  # noqa: F811
    path, _, c, table, _ = checkpoint_fixture
    store = ShardedNGramTable(inspect_checkpoint(path), c)
    requests = [NGramRequest(c, store.multipliers) for _ in range(2)]
    plans = [requests[0].preview([2, c.eos_token_id, 5]), requests[1].preview([7, c.eos_token_id, 5])]
    reader = PreparedNGramTable(store).prepare()
    rows = gather_ple_rows(reader, plans, device="cpu", dtype=torch.bfloat16)
    reader.close()
    expected = torch.stack([table[torch.tensor(p.rows)].flatten(-2) for p in plans])
    assert torch.equal(rows, expected) and all(r.history == () for r in requests)
    assert torch.equal(rows[0, -1], rows[1, -1])  # EOS prevents earlier token leakage.
    output, state = module(torch.bfloat16)(torch.zeros(2, 3, 16, dtype=torch.bfloat16), rows)
    assert torch.isfinite(output).all() and state.tokens_seen == 3
    requests[0].commit(plans[0], 2)
    assert requests[0].history == () and requests[1].history == ()
    with pytest.raises(ValueError, match="equal-length"):
        gather_ple_rows(store, [requests[0].preview([2]), requests[1].preview([2, 3])], device="cpu", dtype=torch.float32)


def test_geometry_mask_budget_and_binding_rejections(tmp_path, monkeypatch):
    m = module(torch.bfloat16)
    prefix = "model.language_model.layers.1.ple"
    save_file({prefix+"."+k: v.detach().clone() for k, v in m.state_dict().items()}, tmp_path/"model.safetensors")
    (tmp_path/"config.json").write_text(json.dumps({"model_type": "fixture"}))
    cp = inspect_checkpoint(tmp_path)
    loaded = FlashNextPLE.from_checkpoint(cp, m.geometry, prefix)
    assert len(loaded.state_dict()) == 6
    for k, v in loaded.state_dict().items():
        assert torch.equal(v.view(torch.int16), m.state_dict()[k].view(torch.int16))
    with pytest.raises(ValueError, match="budget"):
        FlashNextPLE.from_checkpoint(cp, m.geometry, prefix, max_weight_bytes=1)
    with pytest.raises(ValueError):
        replace(m.geometry, conv_kernel=1)
    with pytest.raises(ValueError):
        replace(m.geometry, dilation=True)
    x = torch.zeros(1, 1, 16, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="2D bool"):
        m(x, x, conv_mask=torch.ones(1, 1))
    def forbidden(*args):
        raise AssertionError("work admission must precede projection")
    monkeypatch.setattr(m.key_proj, "forward", forbidden)
    m.max_work_elements = 1
    with pytest.raises(ValueError, match="budget"):
        m(x, x)
