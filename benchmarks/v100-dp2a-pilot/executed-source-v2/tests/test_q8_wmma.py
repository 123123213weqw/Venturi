"""FP16 inputs/codes retain precision; new WMMA reduction has independent gates."""

import os

import pytest
import torch

from test_q8_dp4a import setup
from venturi.ops import CudaBackend, Q8RowSplitWeight, prepare_q8_wmma


def test_wmma_requires_resident_planes():
    with pytest.raises(ValueError, match="resident native Q8"):
        prepare_q8_wmma(object(), tokens=1)


@pytest.fixture
def backend():
    library = os.environ.get("VENTURI_WMMA_LIBRARY")
    if not library:
        pytest.skip("explicit SM70 WMMA research library required")
    return CudaBackend(library)


@pytest.mark.parametrize("warps", [1, 2, 4, 8])
@pytest.mark.parametrize("tokens", [1, 2, 3, 8, 16, 23, 32])
@pytest.mark.parametrize("shape", [(7, 1), (33, 67), (17, 128), (9, 129), (257, 5120)])
def test_group32_original_fp64_reference_and_tail_sentinels(
    backend, warps, tokens, shape
):
    codes, scales, x, resident = setup(backend, *shape, tokens)
    plan = prepare_q8_wmma(resident, tokens=tokens, split_k_warps=warps)
    guard = torch.full(
        (tokens * shape[0] + 2,), 47, dtype=torch.float16, device="cuda:0"
    )
    out = guard[1:-1].view(tokens, shape[0])
    plan.run(x.cuda(), out=out)
    actual = out.cpu().double()
    assert guard[0] == guard[-1] == 47 and torch.isfinite(actual).all()
    dense = codes.double() * scales.double().repeat_interleave(32, -1)
    ideal = x.double() @ dense[:, : shape[1]].T
    rms = ideal.square().mean().sqrt().clamp_min(1e-8)
    assert (actual - ideal).norm() / (ideal.norm() + 1e-12) < 0.001
    assert (
        (actual - ideal.half().double()).abs()
        <= 2 * torch.finfo(torch.float16).eps * torch.maximum(ideal.abs(), rms)
    ).all()
    assert plan.workspace_bytes == 0 and plan.precision_profile.endswith(
        f"splitk{warps}"
    )


@pytest.mark.parametrize("warps", [1, 2, 4, 8])
def test_subnormal_inputs_weight_scales_and_cancellation(backend, warps):
    codes = torch.ones(17, 256, dtype=torch.int8)
    scales = torch.ones(17, 8, dtype=torch.float16)
    x = torch.full((3, 256), 2**-24, dtype=torch.float16)
    x[1] = 1
    scales[0] = 2**-24
    x[2, ::2], x[2, 1::2] = 65504, -65504
    resident = backend.upload_linear(
        Q8RowSplitWeight(codes, scales, 256), dtype=torch.float16, device="cuda:0"
    )
    out = torch.empty(3, 17, dtype=torch.float16, device="cuda:0")
    prepare_q8_wmma(resident, tokens=3, split_k_warps=warps).run(x.cuda(), out=out)
    ideal = x.double() @ (codes.double() * scales.double().repeat_interleave(32, -1)).T
    assert torch.equal(out.cpu(), ideal.half())


def test_wmma_explicit_profile_aliases_native_graph_and_nonfinite(backend, monkeypatch):
    _, _, x_cpu, resident = setup(backend, 128, 128, 2)
    plan = prepare_q8_wmma(resident, tokens=2, split_k_warps=4)
    x = x_cpu.cuda()
    out = torch.empty(2, 128, dtype=torch.float16, device="cuda:0")
    for bad in (0, 3, 16, True, "4"):
        with pytest.raises(ValueError, match="split_k_warps"):
            prepare_q8_wmma(resident, tokens=2, split_k_warps=bad)
    with pytest.raises(ValueError, match="workspace"):
        plan.run(x, out=out, workspace=x)
    with pytest.raises(ValueError, match="alias"):
        plan.run(x, out=x)
    with pytest.raises(ValueError, match="alias"):
        plan.run(
            x,
            out=resident._payload._codes.flatten()[:512]
            .view(torch.float16)
            .view(2, 128),
        )
    for count in (0, 262141, 2**30):
        with pytest.raises(ValueError):
            prepare_q8_wmma(resident, tokens=count)
    for count in (0, 262141):
        assert (
            plan._run(
                x.data_ptr(),
                resident._payload._codes.data_ptr(),
                resident._payload._scales.data_ptr(),
                out.data_ptr(),
                count,
                128,
                128,
                128,
                4,
                0,
            )
            < 0
        )
    plan.run(x, out=out)
    expected = out.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        plan.run(x, out=out)
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    assert torch.equal(out, expected)

    def forbidden(*args, **kwargs):
        raise AssertionError("WMMA execution called Torch math/allocation")

    with monkeypatch.context() as m:
        for name in ("empty", "zeros", "mm", "matmul", "bmm"):
            m.setattr(torch, name, forbidden)
        plan.run(x, out=out)
    assert torch.equal(out, expected)
    x[0, 0] = float("nan")
    plan.run(x, out=out)
    assert not torch.isfinite(out[0]).any() and torch.isfinite(out[1]).all()


@pytest.mark.parametrize("warps", [1, 2, 4, 8])
def test_token_rows_scalar_batch_and_permutation_bytes(backend, warps):
    _, _, x_cpu, resident = setup(backend, 33, 513, 23)
    x = x_cpu.cuda()
    out = torch.empty(23, 33, dtype=torch.float16, device="cuda:0")
    prepare_q8_wmma(resident, tokens=23, split_k_warps=warps).run(x, out=out)
    expected = out.clone()
    scalar = prepare_q8_wmma(resident, tokens=1, split_k_warps=warps)
    for i in range(23):
        scalar.run(x[i : i + 1], out=out[i : i + 1])
    assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8))
    perm = torch.randperm(23, generator=torch.Generator().manual_seed(893)).cuda()
    shuffled = x[perm].contiguous()
    prepare_q8_wmma(resident, tokens=23, split_k_warps=warps).run(shuffled, out=out)
    assert torch.equal(out.view(torch.uint8), expected[perm].view(torch.uint8))
