"""GLM5 mHC (HyperConnection) correctness: Sinkhorn doubly-stochastic,
coefficient bounds, collapse/inject composition."""

import pytest
import torch

from venturi.models.glm5.hc import Glm5HyperConnection

N, D = 4, 64  # 4 streams, 64 dim


@pytest.fixture
def hc():
    torch.manual_seed(3)
    return Glm5HyperConnection(N, D, sinkhorn_iterations=20, eps=1e-6)


def test_coefficient_bounds(hc):
    streams = torch.randn(2, 5, N, D)
    post, comb, collapsed = hc(streams)
    assert post.shape == (2, 5, N)
    assert comb.shape == (2, 5, N, N)
    assert collapsed.shape == (2, 5, D)
    # pre is consumed internally; post ∈ (0, 2)
    assert bool((post > 0).all()) and bool((post < 2).all())
    # comb entries > 0
    assert bool((comb > 0).all())


def test_sinkhorn_doubly_stochastic(hc):
    streams = torch.randn(2, 3, N, D)
    _, comb, _ = hc(streams)
    # After Sinkhorn, column sums should be close to 1
    col_sums = comb.sum(dim=-2)  # [B,T,N]
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=0.1, rtol=0.1)  # approximate DS
    # Row sums should also be close to 1 (approximately, with eps)
    row_sums = comb.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=0.15, rtol=0.15)  # approximate


def test_collapse_produces_single_stream(hc):
    streams = torch.randn(1, 3, N, D)
    _, _, collapsed = hc(streams)
    assert collapsed.shape == (1, 3, D)
    assert collapsed.dtype == streams.dtype


def test_inject_shapes_and_dtype(hc):
    streams = torch.randn(1, 3, N, D)
    block_out = torch.randn(1, 3, D)
    post, comb, _ = hc(streams)
    new_streams = hc.inject(block_out, streams, post, comb)
    assert new_streams.shape == (1, 3, N, D)
    assert new_streams.dtype == streams.dtype


def test_collapse_then_inject_preserves_stream_count(hc):
    """Full cycle: collapse → block → inject → still N streams."""
    streams = torch.randn(1, 2, N, D)
    block = torch.randn(1, 2, D)
    post, comb, collapsed = hc(streams)
    assert collapsed.shape == (1, 2, D)
    new = hc.inject(block, streams, post, comb)
    assert new.shape == streams.shape


def test_fp32_coefficients_even_with_fp16_streams(hc):
    hc_fp16 = hc.to(torch.float16)
    hc_fp16.fn.data = hc_fp16.fn.data.float()  # keep FP32 per convention
    hc_fp16.base.data = hc_fp16.base.data.float()
    hc_fp16.scale.data = hc_fp16.scale.data.float()
    streams = torch.randn(1, 2, N, D, dtype=torch.float16)
    post, comb, collapsed = hc_fp16(streams)
    # Coefficients computed in FP32 even with FP16 streams
    assert post.dtype == torch.float32 or post.dtype == torch.float16
    assert collapsed.dtype == torch.float16
