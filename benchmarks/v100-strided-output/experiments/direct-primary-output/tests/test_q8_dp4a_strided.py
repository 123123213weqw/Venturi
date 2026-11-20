"""Pitched TP projection stores must not touch peer columns or change dot bytes."""

import ctypes as C
import os

import pytest
import torch

from q8_dp4a_reference import projection_reference
from test_q8_dp4a import setup
from venturi.ops import CudaBackend, Q8RowSplitWeight, prepare_q8_dp4a


def bind(backend):
    fn = backend._native.venturi_q8_dp4a_run_strided
    fn.argtypes = (
        [C.c_void_p, C.c_uint64] + [C.c_void_p] * 3 + [C.c_int] * 6 + [C.c_void_p]
    )
    fn.restype = C.c_int
    return fn


@pytest.fixture
def libraries():
    new = os.environ.get("VENTURI_DP4A_LIBRARY")
    old = os.environ.get("VENTURI_DP4A_CONTROL_LIBRARY")
    if not new or not old:
        pytest.skip("explicit pinned strided/control DP4A libraries required")
    return CudaBackend(new), CudaBackend(old)


def arguments(plan, workspace, out, stride):
    r = plan.resident
    return [
        workspace.data_ptr(),
        workspace.numel(),
        r._codes.data_ptr(),
        r._scales.data_ptr(),
        out.data_ptr(),
        plan.tokens,
        r.vocab_size,
        r.hidden_size,
        r._codes.shape[1],
        plan.passes,
        stride,
        torch.cuda.current_stream(r.device).cuda_stream,
    ]


@pytest.mark.parametrize("passes", [1, 2])
@pytest.mark.parametrize("tokens", [1, 2, 3, 8, 23, 32])
@pytest.mark.parametrize("shape", [(7, 67), (512, 5120), (1025, 6145)])
@pytest.mark.parametrize("padding", [0, 9])
def test_pitched_bytes_peer_columns_row_tails_and_independent_reference(
    libraries, passes, tokens, shape, padding
):
    backend, control = libraries
    codes, scales, x_cpu, resident = setup(backend, *shape, tokens)
    old = control.upload_linear(
        Q8RowSplitWeight(codes, scales, shape[1]), dtype=torch.float16, device="cuda:0"
    )
    plans = [
        prepare_q8_dp4a(r, tokens=tokens, residual_compensation=passes == 2)
        for r in (old, resident)
    ]
    scratch = [
        torch.empty(p.workspace_bytes, dtype=torch.uint8, device="cuda:0")
        for p in plans
    ]
    outputs = [
        torch.empty(tokens, shape[0], dtype=torch.float16, device="cuda:0")
        for _ in plans
    ]
    x = x_cpu.cuda()
    for p, work, out in zip(plans, scratch, outputs):
        p.run(x, workspace=work, out=out)
    assert torch.equal(outputs[0].view(torch.uint8), outputs[1].view(torch.uint8))
    assert torch.equal(scratch[0], scratch[1])
    stride = shape[0] + padding
    guard = torch.full((tokens * stride + 2,), 47, dtype=torch.float16, device="cuda:0")
    pitched = guard[1:-1].view(tokens, stride)
    args = arguments(plans[1], scratch[1], pitched, stride)
    backend._check(bind(backend)(*args))
    torch.cuda.synchronize()
    assert torch.equal(
        pitched[:, : shape[0]].contiguous().view(torch.uint8),
        outputs[0].view(torch.uint8),
    )
    assert guard[0] == guard[-1] == 47
    assert bool((pitched[:, shape[0] :] == 47).all())
    rows = list(dict.fromkeys([0, shape[0] // 2, shape[0] - 1]))
    ideal = projection_reference(x_cpu, codes[rows], scales[rows], passes)
    actual = pitched[:, rows].cpu().double()
    assert torch.isfinite(actual).all()
    assert (actual - ideal).norm() / (ideal.norm() + 1e-12) < 0.001


def test_raw_strided_geometry_rejected_before_launch(libraries):
    backend, _ = libraries
    _, _, x, resident = setup(backend, 7, 67, 2)
    plan = prepare_q8_dp4a(resident, tokens=2, residual_compensation=True)
    work = torch.empty(plan.workspace_bytes, dtype=torch.uint8, device="cuda:0")
    out = torch.empty(2, 7, dtype=torch.float16, device="cuda:0")
    plan.run(x.cuda(), out=out, workspace=work)
    fn = bind(backend)
    valid = arguments(plan, work, out, 7)
    invalid = [
        (0, 0),
        (1, work.numel() - 1),
        (2, 0),
        (3, 0),
        (4, 0),
        (5, 0),
        (5, -1),
        (6, 0),
        (7, 0),
        (8, 32),
        (9, 3),
        (10, 6),
        (10, -1),
        (10, 2**31 - 1),
        (0, work.data_ptr() + 1),
        (2, plan.resident._codes.data_ptr() + 1),
    ]
    for index, value in invalid:
        args = valid.copy()
        args[index] = value
        assert fn(*args) == -1, (index, value)
    torch.cuda.synchronize()


def test_strided_graph_replay_and_workspace_reuse_without_torch_math(
    libraries, monkeypatch
):
    backend, _ = libraries
    _, _, x, resident = setup(backend, 33, 67, 8)
    plan = prepare_q8_dp4a(resident, tokens=8, residual_compensation=True)
    work = torch.empty(plan.workspace_bytes, dtype=torch.uint8, device="cuda:0")
    x = x.cuda()
    compact = torch.empty(8, 33, dtype=torch.float16, device="cuda:0")
    pitched = torch.full((8, 71), 47, dtype=torch.float16, device="cuda:0")
    fn = bind(backend)
    args = arguments(plan, work, pitched, 71)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        plan.run(x, out=compact, workspace=work)
        args[-1] = stream.cuda_stream
        backend._check(fn(*args))
    torch.cuda.current_stream().wait_stream(stream)
    for value in (1.0, -0.5, 0.0):
        x.fill_(value)
        graph.replay()
        assert torch.equal(
            pitched[:, :33].contiguous().view(torch.uint8), compact.view(torch.uint8)
        )
        assert bool((pitched[:, 33:] == 47).all())
    args[-1] = torch.cuda.current_stream().cuda_stream

    def forbidden(*args, **kwargs):
        raise AssertionError("Pitched native store called Torch math/allocation")

    with monkeypatch.context() as patch:
        for name in ("empty", "zeros", "mm", "matmul"):
            patch.setattr(torch, name, forbidden)
        backend._check(fn(*args))
    torch.cuda.synchronize()
    assert bool((pitched[:, 33:] == 47).all())
