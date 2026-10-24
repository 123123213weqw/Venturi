"""FP32 partials are not FP16 partials or full-K byte-equivalent arithmetic."""

import os
from types import SimpleNamespace

import pytest
import torch

from q8_dp4a_reference import projection_reference, quantize_reference
from venturi.ops import CudaBackend, Q8RowSplitWeight
from venturi.ops.q8_column_tp import (
    _bind,
    prepare_q8_column_partial,
    slice_q8_columns,
)


def weight(n, k, seed=71):
    gen = torch.Generator().manual_seed(seed)
    padded = (k + 127) // 128 * 128
    codes = torch.randint(-127, 128, (n, padded), dtype=torch.int8, generator=gen)
    scales = (torch.rand(n, padded // 32, generator=gen) * 0.01).half()
    return Q8RowSplitWeight(codes, scales, k)


@pytest.mark.parametrize("k,split", [(64, 32), (257, 128), (6144, 3072), (17408, 8704)])
def test_encoded_column_slicing_preserves_logical_codes_scales_and_owns_storage(
    k, split
):
    w = weight(3, k)
    a, b = slice_q8_columns(w, 0, split), slice_q8_columns(w, split, k)
    assert torch.equal(
        torch.cat([a.codes[:, :split], b.codes[:, : k - split]], 1), w.codes[:, :k]
    )
    assert torch.equal(a.scales[:, : split // 32], w.scales[:, : split // 32])
    assert torch.equal(
        b.scales[:, : (k - split + 31) // 32], w.scales[:, split // 32 : (k + 31) // 32]
    )
    assert not a.codes[:, split:].any() and not b.codes[:, k - split :].any()
    assert (
        a.codes.data_ptr() != w.codes.data_ptr()
        and b.scales.data_ptr() != w.scales.data_ptr()
    )
    saved = w.codes.clone()
    a.codes.zero_()
    assert torch.equal(w.codes, saved)


def test_column_slice_rejects_midgroup_and_non_q8_and_old_library():
    w = weight(7, 257)
    for a, b in [
        (1, 128),
        (0, 17),
        (128, 260),
        (128, 128),
        (-32, 32),
        (True, 32),
        (0, 32.0),
    ]:
        with pytest.raises(ValueError, match="scale-group"):
            slice_q8_columns(w, a, b)
    with pytest.raises(ValueError, match="Q8RowSplitWeight"):
        slice_q8_columns(object(), 0, 32)
    with pytest.raises(ValueError, match="resident native Q8"):
        prepare_q8_column_partial(object(), tokens=1)
    with pytest.raises(RuntimeError, match="exports absent"):
        _bind(SimpleNamespace(_native=object()))


@pytest.fixture
def backend():
    path = os.environ.get("VENTURI_COLUMN_TP_LIBRARY")
    if not path:
        pytest.skip("explicit remote SM70 FP32 column-TP library required")
    return CudaBackend(path)


@pytest.mark.parametrize("tokens", [1, 2, 3, 8, 23])
@pytest.mark.parametrize(
    "n,k,split",
    [(17, 257, 128), (33, 4096, 2048), (5120, 6144, 3072), (5120, 17408, 8704)],
)
def test_fp32_partial_quantizer_guards_and_two_rank_arithmetic(
    backend, tokens, n, k, split
):
    w = weight(n, k)
    x = torch.randn(tokens, k, generator=torch.Generator().manual_seed(798)).half()
    refs = []
    actual = []
    plans = []
    for start, stop in [(0, split), (split, k)]:
        piece = slice_q8_columns(w, start, stop)
        resident = backend.upload_linear(piece, dtype=torch.float16, device="cuda:0")
        plan = prepare_q8_column_partial(resident, tokens=tokens)
        workspace = torch.empty(
            plan.workspace_bytes, dtype=torch.uint8, device="cuda:0"
        )
        guarded = torch.full(
            (tokens * n + 2,), -93.25, dtype=torch.float32, device="cuda:0"
        )
        out = guarded[1:-1].view(tokens, n)
        local = x[:, start:stop].contiguous()
        plan.run(local.cuda(), out=out, workspace=workspace)
        got = out.cpu()
        actual.append(out)
        plans.append(plan)
        assert guarded[0] == guarded[-1] == -93.25
        q, d, _ = quantize_reference(local, 2)
        size = q.numel()
        assert torch.equal(workspace[:size].cpu().view(torch.int8).reshape(q.shape), q)
        assert torch.equal(
            workspace[size:].cpu().view(torch.float32).reshape(d.shape), d
        )
        ref = projection_reference(local, piece.codes, piece.scales, 2)
        refs.append(ref)
        rms = ref.square().mean().sqrt().clamp_min(1e-8)
        assert (got.double() - ref).norm() / (ref.norm() + 1e-12) < 2e-6
        assert (
            (got.double() - ref).abs() <= 8e-6 * torch.maximum(ref.abs(), rms)
        ).all()
    out = torch.empty((tokens, n), dtype=torch.float16, device="cuda:0")
    plans[0].reduce(*actual, out=out)
    ideal = sum(refs)
    assert (out.cpu().double() - ideal).norm() / (ideal.norm() + 1e-12) < 0.001
    assert torch.equal(out.cpu(), (actual[0].cpu() + actual[1].cpu()).half())


def test_sum_cast_once_alias_nonfinite_graph_and_no_torch_execution(
    backend, monkeypatch
):
    w = weight(7, 128)
    r = backend.upload_linear(w, dtype=torch.float16, device="cuda:0")
    plan = prepare_q8_column_partial(r, tokens=1)
    a = torch.tensor(
        [[4096.25, 65520.0, 1.0, float("inf"), 0.0, -0.0, -8192.5]], device="cuda:0"
    )
    b = torch.tensor(
        [[0.25, -65504.0, -1.0, float("-inf"), 0.0, -0.0, 0.5]], device="cuda:0"
    )
    out = torch.empty(1, 7, dtype=torch.float16, device="cuda:0")
    plan.reduce(a, b, out=out)
    assert torch.equal(
        out[:, :3].cpu(), torch.tensor([[4096.0, 16.0, 0.0]], dtype=torch.float16)
    )
    assert torch.isnan(out[0, 3])
    assert not torch.isfinite((a.half() + b.half())[0, 1])
    with pytest.raises(ValueError, match="alias"):
        plan.reduce(a, b, out=a.view(torch.float16)[:, :7])
    with pytest.raises(ValueError):
        plan.reduce(a.half(), b, out=out)
    x = torch.randn(1, 128, device="cuda:0", dtype=torch.float16)
    workspace = torch.empty(plan.workspace_bytes, dtype=torch.uint8, device="cuda:0")
    partial = torch.empty(1, 7, dtype=torch.float32, device="cuda:0")
    plan.run(x, out=partial, workspace=workspace)
    expected = partial.cpu().clone()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(graph, stream=stream):
        plan.run(x, out=partial, workspace=workspace)
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    assert torch.equal(partial.cpu(), expected)

    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected Torch allocation/math in native execution")

    with monkeypatch.context() as p:
        for name in ["empty", "zeros", "mm", "matmul", "round"]:
            p.setattr(torch, name, forbidden)
        plan.run(x, out=partial, workspace=workspace)
        plan.reduce(partial, partial, out=out)
    x.fill_(float("inf"))
    plan.run(x, out=partial, workspace=workspace)
    assert not torch.isfinite(partial).any()
    x.zero_()
    plan.run(x, out=partial, workspace=workspace)
    assert not partial.any() and not workspace.any()
    with pytest.raises(ValueError, match="alias"):
        plan.run(x, out=x[:, :14].view(torch.float32), workspace=workspace)
    with pytest.raises(ValueError):
        plan.run(x, out=partial.half(), workspace=workspace)
