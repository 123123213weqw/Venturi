"""Explicit SM70 warp-decode qualification, including partial final CTAs."""

import ctypes as C
import os

import pytest
import torch

from test_gdn import gdn_inputs, gdn_oracle
from test_gdn_compact import equal_bytes
from venturi.ops import CudaBackend
from venturi.ops.gdn_compact import prepare_gdn_compact


@pytest.fixture(scope="module")
def engines():
    candidate = os.environ.get("VENTURI_GDN_WARP_LIBRARY")
    baseline = os.environ.get("VENTURI_GDN_CONTROL_LIBRARY")
    if not candidate or not baseline:
        pytest.skip("explicit warp-decode and pinned control libraries required")
    result = CudaBackend(baseline), CudaBackend(candidate)
    enabled = result[1]._native.venturi_gdn_warp_decode_enabled
    enabled.restype = C.c_int
    assert enabled() == 1
    return result


@pytest.mark.parametrize("dv", [1, 3, 7, 13, 127, 128, 129])
@pytest.mark.parametrize("hv", [1, 3])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("packed", [False, True])
def test_single_token_bytes_tail_guards_and_fp64(engines, dv, hv, normalize, packed):
    values = gdn_inputs(tokens=1, hq=1, hv=hv, dk=128, dv=dv,
                        dtype=torch.float16, device="cuda:0")
    q, k, v, g, beta, state = values
    if not normalize:
        k.mul_(0.1)
    expected, expected_state = gdn_oracle(*values, normalize=normalize)
    qkv = torch.cat([q.flatten(1), k.flatten(1), v.flatten(1)], dim=1)
    actual = []
    for backend in engines:
        plan = prepare_gdn_compact(
            backend, tokens=1, qk_heads=q.shape[1], value_heads=v.shape[1],
            key_dim=128, value_dim=dv, normalize_qk=normalize,
            dtype=q.dtype, device=q.device,
        )
        og = torch.full((v.numel()+2,), 43, device=v.device, dtype=v.dtype)
        sg = torch.full((state.numel()+2,), 43, device=state.device, dtype=state.dtype)
        out, final = og[1:-1].view_as(v), sg[1:-1].view_as(state)
        if packed:
            plan.run_packed(qkv, g, beta, state=state, out=out, state_out=final)
        else:
            plan.run(q, k, v, g, beta, state=state, out=out, state_out=final)
        assert og[0] == og[-1] == sg[0] == sg[-1] == 43
        actual.append((out.cpu(), final.cpu()))
    assert all(equal_bytes(a, b) for a, b in zip(*actual))
    torch.testing.assert_close(actual[1][0].double(), expected, atol=2e-5, rtol=0.002)
    torch.testing.assert_close(actual[1][1].double(), expected_state, atol=2e-6, rtol=2e-4)


@pytest.mark.parametrize("field", range(6))
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -0.0])
def test_fast_path_nonfinite_and_signed_zero(engines, field, value):
    values = list(gdn_inputs(tokens=1, hq=1, hv=3, dk=128, dv=7,
                             dtype=torch.float16, device="cuda:0"))
    values[field].flatten()[0] = value
    actual = []
    for backend in engines:
        plan = prepare_gdn_compact(
            backend, tokens=1, qk_heads=1, value_heads=3, key_dim=128,
            value_dim=7, dtype=torch.float16, device="cuda:0",
        )
        out, state = torch.empty_like(values[2]), torch.empty_like(values[5])
        plan.run(*values[:5], state=values[5], out=out, state_out=state)
        actual.append((out, state))
    for old, new in zip(*actual):
        assert torch.equal(torch.isnan(old), torch.isnan(new))
        assert torch.equal(torch.isinf(old), torch.isinf(new))
        finite = torch.isfinite(old)
        assert equal_bytes(old[finite], new[finite])
