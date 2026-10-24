"""Separate arithmetic qualification; this is not complete-model quality proof."""

import ctypes as C
import os
from types import SimpleNamespace

import pytest
import torch

from q8_dp2a_reference import projection_reference, quantize_reference
from venturi.ops import CudaBackend, Q8RowSplitWeight
from venturi.ops.q8_dp2a import DP2APilot, workspace_bytes
from venturi.ops.q8_fused import OutputDescriptor


@pytest.mark.parametrize("t,k", [(1, 1), (3, 257), (8, 8704), (23, 5120)])
def test_workspace_and_reference_bound(t, k):
    assert workspace_bytes(t, k) == t * ((k + 31) // 32) * 68
    x = torch.randn(t, k, generator=torch.Generator().manual_seed(712)).half()
    q, s, represented = quantize_reference(x)
    assert q.dtype == torch.int16 and q.abs().max() <= 32767
    error = (represented - x.double()).abs()
    bound = s.double().repeat_interleave(32, -1)[:, :k] * 0.5001 + 3e-7
    assert (error <= bound).all()
    assert not q.reshape(t, -1)[:, k:].any()


@pytest.mark.parametrize(
    "t,k", [(0, 32), (True, 32), (1, 3.5), (262141, 32), (1, -1), (262140, 32768)]
)
def test_workspace_invalid(t, k):
    with pytest.raises(ValueError):
        workspace_bytes(t, k)


def test_prepare_rejects_nonresident_and_dtype_before_gpu():
    pilot = DP2APilot.__new__(DP2APilot)
    for values in [[], object(), [object()] * 4]:
        with pytest.raises(ValueError):
            pilot.prepare(values, tokens=1)
    with pytest.raises(ValueError, match="FP16 or FP32"):
        pilot.prepare([object()], tokens=1, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="resident native Q8"):
        pilot.prepare([SimpleNamespace()], tokens=1)


def weight(n, k, seed=882):
    g = torch.Generator().manual_seed(seed)
    p = (k + 127) // 128 * 128
    return Q8RowSplitWeight(
        torch.randint(-127, 128, (n, p), dtype=torch.int8, generator=g),
        (torch.rand(n, p // 32, generator=g) * 0.01).half(),
        k,
    )


@pytest.fixture
def native():
    path, base = (
        os.environ.get("VENTURI_DP2A_PILOT_LIBRARY"),
        os.environ.get("VENTURI_DP2A_CONTROL_LIBRARY"),
    )
    if not path or not base:
        pytest.skip("Explicit remotely compiled SM70 pilot and control required")
    return DP2APilot(path), CudaBackend(base)


def test_signed_integer_lane_mapping_and_extremes(native):
    pilot, _ = native
    gen = torch.Generator().manual_seed(537)
    a = torch.randint(-32768, 32768, (512, 32), dtype=torch.int16, generator=gen)
    w = torch.randint(-128, 128, (512, 32), dtype=torch.int8, generator=gen)
    for i, (av, wv) in enumerate(
        [(-32768, -128), (32767, 127), (-32768, 127), (32767, -128)]
    ):
        a[i].fill_(av)
        w[i].fill_(wv)
    a[4] = torch.arange(32, dtype=torch.int16) * 1531 - 23412
    w[4] = torch.arange(32, dtype=torch.int8) * 7 - 113
    actual = torch.empty(512, dtype=torch.int32, device="cuda:0")
    ag, wg = a.cuda(), w.cuda()
    pilot.check(
        pilot.integer(
            ag.data_ptr(),
            wg.data_ptr(),
            actual.data_ptr(),
            512,
            torch.cuda.current_stream().cuda_stream,
        )
    )
    assert torch.equal(actual.cpu().long(), (a.long() * w.long()).sum(-1))


@pytest.mark.parametrize("tokens", [1, 2, 3, 8, 23])
@pytest.mark.parametrize("n,k", [(7, 1), (17, 257), (35, 4096), (1025, 8704)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_quantization_projection_padding_stride_and_fp64(native, tokens, n, k, dtype):
    pilot, backend = native
    w = weight(n, k)
    r = backend.upload_linear(w, dtype=torch.float16, device="cuda:0")
    p = pilot.prepare([r], tokens=tokens, dtype=dtype, output_strides=[n + 3])
    x = torch.randn(tokens, k, generator=torch.Generator().manual_seed(529)).half()
    xg = x.cuda()
    guard = torch.full(
        (p.workspace_bytes + 32,), 73, dtype=torch.uint8, device="cuda:0"
    )
    workspace = guard[16:-16]
    out = torch.full((tokens, n + 3), -9.25, dtype=dtype, device="cuda:0")
    p.run(xg, outputs=[out], workspace=workspace)
    q, s, _ = quantize_reference(x)
    size = q.numel() * 2
    packed = workspace.cpu()
    assert torch.equal(packed[:size].view(torch.int16).reshape(q.shape), q)
    assert torch.equal(packed[size:].view(torch.float32).reshape(s.shape), s)
    assert (guard[:16] == 73).all() and (guard[-16:] == 73).all()
    assert (out[:, n:] == -9.25).all()
    ref = projection_reference(x, w.codes, w.scales)
    actual = out[:, :n].cpu().double()
    if dtype == torch.float32:
        rms = ref.square().mean().sqrt().clamp_min(1e-8)
        assert (actual - ref).norm() / (ref.norm() + 1e-20) < 2e-6
        assert ((actual - ref).abs() <= 8e-6 * torch.maximum(ref.abs(), rms)).all()
    else:
        assert (actual - ref).norm() / (ref.norm() + 1e-20) < 0.001


@pytest.mark.parametrize("tokens", [1, 3, 8])
@pytest.mark.parametrize("count", [2, 3])
def test_fused_all_rows_replay_and_no_torch_math(native, monkeypatch, tokens, count):
    pilot, backend = native
    weights = [weight(n, 4097, 181 + n) for n in [17, 1025, 7][:count]]
    residents = [
        backend.upload_linear(w, dtype=torch.float16, device="cuda:0") for w in weights
    ]
    plan = pilot.prepare(residents, tokens=tokens)
    xcpu = torch.randn(
        tokens, 4097, generator=torch.Generator().manual_seed(338)
    ).half()
    x = xcpu.cuda()
    outs = [
        torch.empty(tokens, w.out_features, device="cuda:0", dtype=torch.float16)
        for w in weights
    ]
    workspace = torch.empty(plan.workspace_bytes, device="cuda:0", dtype=torch.uint8)
    plan.run(x, outputs=outs, workspace=workspace)
    for w, got in zip(weights, outs):
        ref = projection_reference(xcpu, w.codes, w.scales)
        assert (got.cpu().double() - ref).norm() / (ref.norm() + 1e-20) < 0.001
    expected = [out.cpu() for out in outs]
    graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(graph, stream=stream):
        plan.run(x, outputs=outs, workspace=workspace)
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    assert all(torch.equal(o.cpu(), e) for o, e in zip(outs, expected))

    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected Torch allocation/math in native pilot")

    with monkeypatch.context() as patch:
        for name in ["empty", "zeros", "mm", "matmul", "round"]:
            patch.setattr(torch, name, forbidden)
        plan.run(x, outputs=outs, workspace=workspace)
    x.neg_()
    graph.replay()
    assert all(torch.equal(o.cpu(), -e) for o, e in zip(outs, expected))


def test_nonfinite_zero_small_large_and_raw_contract(native):
    pilot, backend = native
    w = weight(7, 161)
    r = backend.upload_linear(w, dtype=torch.float16, device="cuda:0")
    p = pilot.prepare([r], tokens=1, dtype=torch.float32)
    x = torch.zeros(1, 161, dtype=torch.float16)
    x[0, 0] = float("inf")
    x[0, 32] = float("nan")
    x[0, 64:96] = torch.finfo(torch.float16).smallest_normal / 1024
    x[0, 96:128] = 65504
    x[0, 128:160] = -65504
    q, s, _ = quantize_reference(x)
    workspace = torch.empty(p.workspace_bytes, device="cuda:0", dtype=torch.uint8)
    out = torch.empty(1, 7, device="cuda:0", dtype=torch.float32)
    xg = x.cuda()
    p.run(xg, outputs=[out], workspace=workspace)
    got = workspace.cpu()
    assert torch.equal(got[: q.numel() * 2].view(torch.int16).reshape(q.shape), q)
    torch.testing.assert_close(
        got[q.numel() * 2 :].view(torch.float32).reshape(s.shape),
        s,
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    assert torch.isnan(out).all()
    assert (
        pilot.quantize(
            xg.data_ptr(), workspace.data_ptr(), p.workspace_bytes - 1, 1, 161, 0
        )
        == -1
    )
    for t, k in [(0, 1), (1, 0), (262141, 32), (262140, 32768)]:
        assert pilot.bytes(t, k) == 0
    for values in [
        dict(dtype=torch.bfloat16),
        dict(output_strides=[6]),
        dict(output_strides=[2**31]),
    ]:
        with pytest.raises(ValueError):
            pilot.prepare([r], tokens=1, **values)
    with pytest.raises(ValueError):
        p.run(xg, outputs=[out.half()], workspace=workspace)
    with pytest.raises(ValueError, match="alignment"):
        p.run(
            xg,
            outputs=[out],
            workspace=torch.empty(
                p.workspace_bytes + 1, dtype=torch.uint8, device="cuda:0"
            )[1:],
        )
    d = (OutputDescriptor * 1)(
        OutputDescriptor(
            r._payload._codes.data_ptr(),
            r._payload._scales.data_ptr(),
            out.data_ptr(),
            7,
            7,
        )
    )
    assert (
        pilot.project(workspace.data_ptr(), p.workspace_bytes, d, 1, 1, 161, 256, 3, 0)
        == -1
    )
    assert C.sizeof(OutputDescriptor) == 32
