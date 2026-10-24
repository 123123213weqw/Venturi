"""A new integer-dot token tile must preserve old quantization and output bytes."""

import os

import pytest
import torch

from q8_dp4a_reference import projection_reference
from test_q8_dp4a import setup
from venturi.ops import CudaBackend, Q8RowSplitWeight, prepare_q8_dp4a


@pytest.mark.parametrize("passes", [1, 2])
@pytest.mark.parametrize("tokens", [1, 2, 8, 23, 32])
@pytest.mark.parametrize("shape", [(512, 5120), (1025, 6145)])
def test_old_new_cache_corner_wide_tail_and_original_profile_reference(
    passes, tokens, shape
):
    old = os.environ.get("VENTURI_DP4A_CONTROL_LIBRARY")
    new = os.environ.get("VENTURI_DP4A_LIBRARY")
    if not old or not new:
        pytest.skip("explicit pinned old/new DP4A libraries required")
    codes, scales, x_cpu, resident = setup(CudaBackend(new), *shape, tokens)
    control = CudaBackend(old).upload_linear(
        Q8RowSplitWeight(codes, scales, shape[1]), dtype=torch.float16, device="cuda:0"
    )
    plans = [
        prepare_q8_dp4a(r, tokens=tokens, residual_compensation=passes == 2)
        for r in (control, resident)
    ]
    x = x_cpu.cuda()
    outputs = [
        torch.empty(tokens, shape[0], dtype=torch.float16, device="cuda:0")
        for _ in plans
    ]
    scratch = [
        torch.empty(p.workspace_bytes, dtype=torch.uint8, device="cuda:0")
        for p in plans
    ]
    for p, out, work in zip(plans, outputs, scratch):
        p.run(x, out=out, workspace=work)
    assert torch.equal(outputs[0].view(torch.uint8), outputs[1].view(torch.uint8))
    assert torch.equal(scratch[0], scratch[1])
    ideal = projection_reference(x_cpu, codes, scales, passes)
    actual = outputs[1].cpu().double()
    assert torch.isfinite(actual).all()
    assert (actual - ideal).norm() / (ideal.norm() + 1e-12) < 0.001
