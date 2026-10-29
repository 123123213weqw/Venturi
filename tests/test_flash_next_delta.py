"""Delta state/layout/normalization/gate contracts; not a GDN frontend."""

import pytest
import torch

from venturi.models.flash_next_delta import SigmoidGatedRMSNorm, gated_delta_rule


def fixture(length=7, dtype=torch.float32):
    rng = torch.Generator().manual_seed(541)
    q, k = (torch.randn(2, length, 6, 4, generator=rng).to(dtype) for _ in range(2))
    v = torch.randn(2, length, 6, 3, generator=rng).to(dtype)
    g = -torch.rand(2, length, 6, generator=rng)
    beta = torch.rand(2, length, 6, generator=rng).to(dtype)
    return q, k, v, g, beta


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_recurrent_chunk_agreement_and_owned_initial_state(dtype):
    values = fixture(67, dtype)
    state = torch.randn(2, 6, 4, 3)
    saved = state.clone()
    sequential, final = gated_delta_rule(*values, initial_state=state)
    chunked, other = gated_delta_rule(*values, initial_state=state, mode="chunk")
    tolerance = 3e-6 if dtype == torch.float32 else 8e-3
    torch.testing.assert_close(chunked, sequential, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(other, final, atol=3e-6, rtol=3e-6)
    assert final.dtype == torch.float32 and final.shape == (2, 6, 4, 3)
    assert torch.equal(saved, state) and final.data_ptr() != state.data_ptr()
    assert other.data_ptr() != state.data_ptr()


def test_exact_recurrent_partition_rollback_and_request_independence():
    values = fixture()
    output, whole = gated_delta_rule(*values)
    first, past = gated_delta_rule(*(t[:, :3] for t in values))
    original = past.clone()
    tail, proposed = gated_delta_rule(*(t[:, 3:] for t in values), initial_state=past)
    assert torch.equal(torch.cat((first, tail), 1), output) and torch.equal(proposed, whole)
    assert torch.equal(original, past)
    # Rejected recurrent suffix cannot be recovered by slicing the final matrix.
    # Restart from the saved prefix, replay only the accepted one-token prefix.
    accepted_output, accepted = gated_delta_rule(*(t[:, 3:4] for t in values), initial_state=past)
    assert torch.equal(accepted_output, output[:, 3:4]) and not torch.equal(accepted, proposed)
    isolated, isolated_state = gated_delta_rule(*(t[:1] for t in values))
    assert torch.equal(isolated, output[:1]) and torch.equal(isolated_state, whole[:1])


def test_zero_beta_decay_and_zero_query_analytic_result():
    q, k, v, g, beta = fixture(1)
    state = torch.randn(2, 6, 4, 3)
    g.zero_()
    beta.zero_()
    q.zero_()
    for mode in ("chunk", "recurrent"):
        output, final = gated_delta_rule(q, k, v, g, beta, initial_state=state, mode=mode)
        assert torch.equal(final, state) and not output.any()


def test_l2_norm_uses_sum_plus_epsilon_not_clamped_normalize():
    q = torch.full((1, 1, 1, 2), 1e-5)
    k, v = torch.ones_like(q), torch.ones(1, 1, 1, 1)
    g, b = torch.zeros(1, 1, 1), torch.ones(1, 1, 1)
    output, _ = gated_delta_rule(q, k, v, g, b)
    qn = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    kn = k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    expected = (qn * kn).sum(-1, keepdim=True) / (2**0.5)
    torch.testing.assert_close(output, expected)
    assert output.item() < 0.02  # clamp-based unit L2 would be near1 here


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_direct_weight_sigmoid_gate_and_cast_boundaries(dtype):
    rms = SigmoidGatedRMSNorm(4).to(dtype)
    x = torch.tensor([[1., -2., 3., -4.]], dtype=dtype)
    gate = torch.tensor([[-20., 0., 1., 20.]], dtype=dtype)
    with torch.no_grad():
        rms.weight.copy_(torch.tensor([0., 1., 2., -1.], dtype=dtype))
    y = rms(x, gate)
    norm = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)).to(dtype)
    expected = ((rms.weight * norm) * gate.float().sigmoid()).to(dtype)
    assert torch.equal(y, expected) and y[0, 0] == 0 and y[0, 1] < 0 and y[0, -1] > 0


def test_geometry_dtype_and_bounded_work_rejections():
    q, k, v, g, b = fixture()
    with pytest.raises(ValueError, match="shapes"):
        gated_delta_rule(q[:, :, :2], k[:, :, :2], v, g, b)
    with pytest.raises(ValueError, match="FP32"):
        gated_delta_rule(q, k, v, g, b, initial_state=torch.zeros(2, 6, 3, 4))
    with pytest.raises(ValueError, match="budget"):
        gated_delta_rule(q, k, v, g, b, max_work_elements=1)
    with pytest.raises(ValueError, match="bounded"):
        gated_delta_rule(q, k, v, g, b, chunk_size=True)
    with pytest.raises(ValueError, match="mode"):
        gated_delta_rule(q, k, v, g, b, mode="auto")
