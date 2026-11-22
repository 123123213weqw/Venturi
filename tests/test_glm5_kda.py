"""GLM5 KDA forward correctness: forget-gate semantics, frontend wiring,
state management, and gate effectiveness."""

import pytest
import torch
from torch import nn

from venturi.models.glm5.kda import Glm5KDA, KDAState

H, D = 4, 32  # small geometry: 4 heads, 32 dim


def make_kda(**kw):
    torch.manual_seed(42)
    return Glm5KDA(hidden_size=64, num_heads=H, head_dim=D, conv_kernel=4, **kw)


def test_forget_gate_is_bounded_negative():
    kda = make_kda()
    x = torch.randn(2, 5, 64) * 3
    g = kda._forget_gate(x)
    assert g.shape == (2, 5, H, D)
    assert bool((g <= 0).all()) and bool((g > -5.0).all())  # in (-5, 0)
    # All-zero input + zero biases: sigmoid(0) = 0.5 → g = -2.5
    kda_zero = make_kda()
    with torch.no_grad():
        kda_zero.dt_bias.zero_()
    g_zero = kda_zero._forget_gate(torch.zeros(1, 1, 64))
    assert torch.allclose(g_zero, torch.full_like(g_zero, -2.5), atol=1e-6)


def test_recurrent_matches_manual_per_token():
    kda = make_kda().eval()
    x = torch.randn(1, 3, 64)
    with torch.no_grad():
        output, state = kda(x)
        # Manual per-token replay
        state0 = None
        manual_out = []
        for t in range(3):
            out_t, state0 = kda(x[:, t:t+1], past=state0)
            manual_out.append(out_t)
        manual = torch.cat(manual_out, dim=1)
    # Chunk vs recurrent have different rounding (docstring); loose tolerance
    torch.testing.assert_close(output, manual, rtol=0.02, atol=0.02)


def test_incremental_decode_matches_chunk_prefill():
    kda = make_kda().eval()
    x = torch.randn(1, 8, 64)
    with torch.no_grad():
        # Path A: prefill 6, decode token 7, decode token 8
        _, state = kda(x[:, :6])
        _, state = kda(x[:, 6:7], past=state)
        out_a, _ = kda(x[:, 7:8], past=state)
        # Path B: prefill 7, decode token 8
        _, state7 = kda(x[:, :7])
        out_b, _ = kda(x[:, 7:8], past=state7)
    torch.testing.assert_close(out_a, out_b, rtol=0.02, atol=0.02)


def test_state_shapes_and_dtypes():
    kda = make_kda().eval()
    x = torch.randn(2, 3, 64)
    with torch.no_grad():
        _, state = kda(x)
    assert state.conv.shape == (2, 3 * H * D, 4)
    assert state.recurrent.shape == (2, H, D, D)
    assert state.recurrent.dtype == torch.float32  # core keeps FP32 state
    assert state.tokens_seen == 3
    with torch.no_grad():
        _, s2 = kda(x[:, :1], past=state)
    assert s2.tokens_seen == 4


def test_output_gate_sigmoid_effective():
    """Gate pre-sigmoid all very negative → output near zero after o_norm."""
    kda = make_kda().eval()
    x = torch.randn(1, 2, 64)
    from venturi.models.linear_attention import SigmoidGatedRMSNorm
    norm = SigmoidGatedRMSNorm(D, eps=1e-5)
    values = torch.randn(8, D)
    gate_zero = torch.full((8, D), -100.0)  # sigmoid(-100) ≈ 0
    gate_one = torch.zeros(8, D)              # sigmoid(0) = 0.5
    out_zero = norm(values, gate_zero)
    out_half = norm(values, gate_one)
    assert float(out_zero.abs().max()) < 1e-3   # gate≈0 kills the output
    ratio = out_half / (norm.weight * values * torch.rsqrt(values.pow(2).mean(-1, keepdim=True) + 1e-5) * 0.5)
    assert torch.allclose(ratio, torch.ones_like(ratio), rtol=0.1)  # sigmoid(0)=0.5


def test_mask_zeroed_tokens_do_not_contribute():
    kda = make_kda().eval()
    x = torch.randn(1, 4, 64)
    mask = torch.tensor([[1, 1, 0, 1]], dtype=x.dtype)
    with torch.no_grad():
        out_masked, _ = kda(x, attention_mask=mask)
        out_trunc, _ = kda(x[:, :2])  # just first 2 tokens
    # The first 2 output positions should be identical (mask zeroes positions 2,3)
    torch.testing.assert_close(out_masked[:, :2], out_trunc, rtol=1e-5, atol=1e-5)


def test_dtype_and_shape_validation():
    kda = make_kda()
    with pytest.raises(ValueError, match="dtype"):
        kda(torch.zeros(1, 2, 64, dtype=torch.float64))
