import math

import pytest
import torch

from venturi.models.hyperconnections import (
    HyperConnectionSpec, reference_hc_coefficients, reference_hc_mixes,
    reference_hc_post, reference_hc_pre, reference_identity_pre,
)


@pytest.mark.parametrize("streams", [1, 2, 4])
@pytest.mark.parametrize("iterations", [1, 20])
def test_sinkhorn_against_independent_scalar_float64_equations(streams, iterations):
    spec = HyperConnectionSpec(streams, iterations)
    rng = torch.Generator().manual_seed(1017)
    x = torch.randn(2, 3, spec.coefficient_count, generator=rng) * 3
    scales = torch.tensor([.5, 2., .25])
    bases = torch.randn(spec.coefficient_count, generator=rng)
    before = x.clone()
    pre, post, comb = reference_hc_coefficients(x, scales, bases, spec)
    for b in range(2):
        for t in range(3):
            def z(i, part):
                return float(x[b, t, i]) * float(scales[part]) + float(bases[i])
            expected_pre = [1/(1+math.exp(-z(i, 0)))+spec.epsilon for i in range(streams)]
            expected_post = [2/(1+math.exp(-z(i+streams, 1))) for i in range(streams)]
            a = [[z(2*streams+i*streams+j, 2) for j in range(streams)] for i in range(streams)]
            for i in range(streams):
                row = [math.exp(v-max(a[i])) for v in a[i]]
                a[i] = [v/sum(row)+spec.epsilon for v in row]
            for step in range(iterations):
                if step:
                    for i in range(streams):
                        den = sum(a[i])+spec.epsilon
                        a[i] = [v/den for v in a[i]]
                col = [sum(a[i][j] for i in range(streams))+spec.epsilon for j in range(streams)]
                a = [[v/col[j] for j, v in enumerate(row)] for row in a]
            assert torch.allclose(pre[b, t], torch.tensor(expected_pre), atol=2e-7, rtol=2e-6)
            assert torch.allclose(post[b, t], torch.tensor(expected_post), atol=2e-7, rtol=2e-6)
            assert torch.allclose(comb[b, t], torch.tensor(a), atol=2e-7, rtol=2e-6)
    assert torch.equal(x, before)


def test_residual_matrix_orientation_is_not_conventional_left_multiply():
    residual = torch.tensor([[[[1.], [10.]]]])
    comb = torch.tensor([[[[.2, .8], [.3, .7]]]])
    output = torch.tensor([[[2.]]])
    post = torch.tensor([[[1., 3.]]])
    result = reference_hc_post(output, residual, post, comb)
    assert torch.allclose(result, torch.tensor([[[[5.2], [13.8]]]]))
    assert not torch.allclose(result, comb @ residual + output.unsqueeze(2)*post.unsqueeze(-1))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_collapse_fp32_accumulation_and_output_dtype(dtype):
    x = torch.arange(2*3*4*8).reshape(2, 3, 4, 8).to(dtype)
    pre = reference_identity_pre(x)
    assert torch.equal(reference_hc_pre(x, pre), x[:, :, 0])
    assert pre.dtype == torch.float32
    assert torch.equal(pre.sum(-1), torch.ones(2, 3))
    comb = torch.eye(4).expand(2, 3, 4, 4)
    out = reference_hc_post(torch.zeros(2, 3, 8, dtype=dtype), x, torch.zeros(2, 3, 4), comb)
    assert out.dtype == dtype and torch.equal(out, x)


def test_projection_normalizes_flattened_stream_not_each_copy():
    spec = HyperConnectionSpec(streams=2)
    stream = torch.tensor([[[[1., 2.], [30., 40.]]]])
    weight = torch.arange(32, dtype=torch.float32).reshape(8, 4)/32
    scale, base = torch.ones(3), torch.zeros(8)
    observed = reference_hc_mixes(stream, weight, scale, base, spec)
    flat = stream.flatten(2)
    projected = (flat @ weight.T) * torch.rsqrt(flat.square().mean(-1, keepdim=True)+spec.norm_epsilon)
    expected = reference_hc_coefficients(projected, scale, base, spec)
    assert all(torch.equal(a, b) for a, b in zip(observed, expected))


@pytest.mark.parametrize("kwargs", [dict(streams=True), dict(streams=0),
    dict(sinkhorn_iterations=0), dict(epsilon=0), dict(norm_epsilon=float('nan'))])
def test_bad_spec(kwargs):
    with pytest.raises(ValueError):
        HyperConnectionSpec(**kwargs)


def test_invalid_shapes_dtype_and_values_fail_closed():
    spec = HyperConnectionSpec()
    args = [torch.zeros(2, 3, 24), torch.ones(3), torch.zeros(24)]
    for index, value in [(0, torch.zeros(2, 3, 23)), (1, torch.ones(3, dtype=torch.float64)),
                         (2, torch.full((24,), float('nan')))]:
        changed = list(args); changed[index] = value
        with pytest.raises(ValueError):
            reference_hc_coefficients(*changed, spec)
    with pytest.raises(ValueError):
        reference_hc_pre(torch.zeros(2, 4, 8), torch.ones(2, 4))
    with pytest.raises(ValueError):
        reference_hc_post(torch.zeros(2, 3, 8), torch.zeros(2, 3, 4, 8),
                          torch.zeros(2, 3, 5), torch.zeros(2, 3, 4, 4))



def test_checkpoint_norm_epsilon_is_explicit_not_generic_default():
    x=torch.ones(1,1,4,2)*1e-10
    projection=torch.full((24,8),.1)
    scales=torch.ones(3);bases=torch.zeros(24)
    checkpoint=HyperConnectionSpec(norm_epsilon=1e-20)
    default=HyperConnectionSpec()
    exact=reference_hc_mixes(x,projection,scales,bases,checkpoint)
    wrong=reference_hc_mixes(x,projection,scales,bases,default)
    assert not torch.allclose(exact[0],wrong[0])
    assert all(torch.isfinite(v).all() for v in exact)
