"""Small QSA indexer mask/state contracts, not a full attention/model test."""

from dataclasses import replace
import json

import pytest
from safetensors.torch import save_file
import torch

from venturi.checkpoint import inspect_checkpoint
from venturi.models.flash_next_qsa import FlashNextQSAIndexer, QSAIndexerGeometry, partial_rope


def make_module(dtype=torch.float32):
    torch.manual_seed(516)
    g = QSAIndexerGeometry(8, 2, 4, 4, 2)
    m = FlashNextQSAIndexer(g).to(dtype)
    with torch.no_grad():
        for parameter in m.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.15)
    return m.eval()


def positions(batch, length, dtype):
    angle = torch.arange(length, dtype=torch.float32).view(1, length, 1) * 0.3
    # Only two of the four indexer dimensions rotate: supplied width matters.
    return angle.cos().expand(batch, length, 2).to(dtype), angle.sin().expand(batch, length, 2).to(dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_visible_blocks_tail_mask_format_and_compact_owned_state(dtype):
    m = make_module(dtype)
    x = torch.randn(2, 9, 8, dtype=dtype)
    mask = torch.ones(2, 1, 9, 9, dtype=torch.bool).tril()
    mask[0, 0, -1] = torch.tensor([True, False, True, True, False, True, True, False, True])
    mask[1, 0, -1] = False
    selected, keys = m(x, positions(2, 9, dtype), mask)
    assert keys.shape == (2, 9, 4)
    assert keys.untyped_storage().nbytes() == keys.numel() * keys.element_size()
    assert torch.all(~selected | mask)
    assert not selected[1, 0, -1].any()
    # Six visible tokens form three blocks. Budget4 selects exactly two blocks.
    assert selected[0, 0, -1].sum() == 4
    for b in range(2):
        for t in range(9):
            visible = torch.where(mask[b, 0, t])[0]
            tail_start = visible.numel() // 2 * 2
            assert selected[b, 0, t, visible[tail_start:]].all()
            if visible.numel() <= 4:
                assert torch.equal(selected[b, 0, t], mask[b, 0, t])
    additive = torch.where(mask, torch.zeros((), dtype=dtype), torch.finfo(dtype).min)
    floating, again = m(x, positions(2, 9, dtype), additive)
    assert torch.equal(floating == 0, selected)
    assert torch.equal(keys, again)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_chunked_history_is_explicit_private_and_rollback_can_reject_suffix(dtype):
    m = make_module(dtype)
    x = torch.randn(2, 7, 8, dtype=dtype)
    full_positions = positions(2, 7, dtype)
    full_mask = torch.ones(2, 1, 7, 7, dtype=torch.bool).tril()
    whole, whole_keys = m(x, full_positions, full_mask)
    _, past = m(x[:, :3], positions(2, 3, dtype), full_mask[:, :, :3, :3])
    retained = past.clone()
    tail, proposed = m(x[:, 3:], full_positions, full_mask[:, :, 3:], raw_past=past)
    assert torch.equal(tail, whole[:, :, 3:])
    assert torch.equal(proposed, whole_keys)
    assert torch.equal(past, retained)
    # Accept only one of four proposed tokens. The other3 never enter past.
    accepted = proposed[:, :4].clone()
    one, state = m(x[:, 4:5], positions(2, 5, dtype), full_mask[:, :, 4:5, :5], raw_past=accepted)
    assert torch.equal(one, whole[:, :, 4:5, :5])
    assert state.shape == (2, 5, 4) and torch.equal(accepted, proposed[:, :4])


def test_partial_rope_leaves_nonrotary_dimensions_untouched():
    x = torch.tensor([[[1., 2., 3., 4.]]])
    cos, sin = torch.zeros(1, 2), torch.ones(1, 2)
    assert torch.equal(partial_rope(x, cos, sin, unsqueeze_dim=1), torch.tensor([[[-2., 1., 3., 4.]]]))
    with pytest.raises(ValueError, match="geometry"):
        partial_rope(x, torch.zeros(1, 3), torch.ones(1, 3), unsqueeze_dim=1)


@pytest.mark.parametrize("field,value", [("query_heads", True), ("head_dim", 3), ("token_budget", 3),
                                        ("compress_ratio", 0), ("eps", float("nan"))])
def test_geometry_rejects_unsupported(field, value):
    with pytest.raises(ValueError):
        replace(make_module().geometry, **{field: value})


def test_explicit_mask_state_position_and_allocation_validation():
    m = make_module()
    x = torch.zeros(1, 2, 8)
    mask = torch.ones(1, 1, 2, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match="4D"):
        m(x, positions(1, 2, torch.float32), mask[:, 0])
    with pytest.raises(ValueError, match="position"):
        m(x, positions(1, 3, torch.float32), mask)
    with pytest.raises(ValueError, match="raw past"):
        m(x, positions(1, 4, torch.float32), torch.ones(1, 1, 2, 4, dtype=torch.bool), raw_past=torch.zeros(1, 2, 5))
    with pytest.raises(ValueError, match="budget"):
        FlashNextQSAIndexer(m.geometry, max_mask_elements=3)(x, positions(1, 2, torch.float32), mask)
    raw = dict(model_type="qwen4_exp", text_config=dict(model_type="qwen4_exp_text", hidden_size=8,
        indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=4, indexer_budget=4, indexer_compress_ratio=2))
    assert QSAIndexerGeometry.from_dict(raw) == m.geometry
    raw["text_config"]["indexer_kv_heads"] = 2
    with pytest.raises(ValueError, match="single-KV"):
        QSAIndexerGeometry.from_dict(raw)


def test_small_indexer_checkpoint_binding_and_budget(tmp_path):
    m = make_module(torch.bfloat16)
    prefix = "model.language_model.layers.3.self_attn.indexer"
    save_file({prefix + "." + k: v.detach().clone() for k, v in m.state_dict().items()},
              tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "fixture"}))
    checkpoint = inspect_checkpoint(tmp_path)
    loaded = FlashNextQSAIndexer.from_checkpoint(checkpoint, m.geometry, prefix)
    for name, value in loaded.state_dict().items():
        assert torch.equal(value.view(torch.int16), m.state_dict()[name].view(torch.int16))
    with pytest.raises(ValueError, match="budget"):
        FlashNextQSAIndexer.from_checkpoint(checkpoint, m.geometry, prefix, max_weight_bytes=1)


def test_scatter_index_abi_is_portable_to_torch25(monkeypatch):
    original, seen = torch.Tensor.scatter, []

    def require_int64(self, dim, index, value):
        assert index.dtype == torch.int64
        seen.append(True)
        return original(self, dim, index, value)

    monkeypatch.setattr(torch.Tensor, "scatter", require_int64)
    result, _ = make_module()(torch.ones(1, 1, 8), positions(1, 1, torch.float32),
                             torch.ones(1, 1, 1, 1, dtype=torch.bool))
    assert seen == [True] and result.all()
