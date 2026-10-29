"""Explicit SDPA mask/empty-row/cache semantics, not a speed/default switch."""

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from test_flash_next_attention import inputs, module
from venturi.models.flash_next_attention import FlashNextQSAAttention


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_explicit_sdpa_bool_additive_empty_rows_are_finite_and_default_eager_unchanged(dtype):
    eager = module(dtype)
    sdpa = FlashNextQSAAttention(eager.geometry, attention_backend="sdpa").to(dtype).eval()
    sdpa.load_state_dict(eager.state_dict())
    x, pos, additive = inputs(2, 7, dtype)
    additive[0, :, -1] = torch.finfo(dtype).min
    with sdpa_kernel(SDPBackend.MATH), torch.inference_mode():
        a, state, probabilities = sdpa(x, pos, additive)
        b, bool_state, _ = sdpa(x, pos, additive == 0)
        old, _, _ = eager(x, pos, additive)
    assert probabilities is None and torch.isfinite(a).all() and torch.isfinite(b).all()
    assert not a[0, -1].any() and not b[0, -1].any()
    assert torch.isnan(old[0, -1]).all() and eager.attention_backend == "eager"
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert torch.equal(state.keys, bool_state.keys)


def test_sdpa_cached_chunk_uses_explicit_mask_not_upper_left_causal_triangle():
    sdpa = module()
    sdpa.attention_backend = "sdpa"
    x, pos, mask = inputs(2, 9)
    with sdpa_kernel(SDPBackend.MATH), torch.inference_mode():
        whole, _, _ = sdpa(x, pos, mask == 0)
        _, past, _ = sdpa(x[:, :4], tuple(p[:, :4] for p in pos), mask[:, :, :4, :4] == 0)
        saved = past.truncate(4)
        output, proposed, _ = sdpa(x[:, 4:], pos, mask[:, :, 4:] == 0, past=past)
    torch.testing.assert_close(output, whole[:, 4:], atol=2e-6, rtol=2e-6)
    assert torch.equal(past.keys, saved.keys) and proposed.keys.shape[2] == 9


def test_unknown_backend_rejected_not_automatically_selected():
    with pytest.raises(ValueError, match="backend"):
        FlashNextQSAAttention(module().geometry, attention_backend="auto")
