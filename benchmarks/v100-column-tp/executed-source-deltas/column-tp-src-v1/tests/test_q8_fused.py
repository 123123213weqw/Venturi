"""Complete-row fusion must preserve every output bit and all pitched sentinels."""

import os
import pytest
import torch
from test_q8_dp4a import setup
from q8_dp4a_reference import projection_reference
from venturi.ops import CudaBackend, prepare_q8_dp4a
from venturi.ops.q8_fused import prepare_q8_fused


@pytest.fixture
def backend():
    path = os.environ.get("VENTURI_Q8_FUSED_LIBRARY")
    if not path:
        pytest.skip("explicit fused research library required")
    return CudaBackend(path)


@pytest.mark.parametrize("passes", [1, 2])
@pytest.mark.parametrize("tokens", [1, 2, 3, 8, 23, 32])
@pytest.mark.parametrize("k", [67, 5120, 17408])
@pytest.mark.parametrize("ns", [(7, 33), (31, 17, 9)])
def test_complete_row_reference_bytes_and_pitched_guards(
    backend, passes, tokens, k, ns
):
    data = [setup(backend, n, k, tokens) for n in ns]
    x_cpu = data[0][2]
    x = x_cpu.cuda()
    res = [d[3] for d in data]
    strides = [n + 5 for n in ns]
    plan = prepare_q8_fused(
        res, tokens=tokens, residual_compensation=passes == 2, output_strides=strides
    )
    work = torch.empty(plan.workspace_bytes, dtype=torch.uint8, device="cuda:0")
    guards = [
        torch.full((tokens * s + 2,), 47, dtype=torch.float16, device="cuda:0")
        for s in strides
    ]
    outputs = [g[1:-1].view(tokens, s) for g, s in zip(guards, strides)]
    plan.run(x, outputs=outputs, workspace=work)
    scratch = work.clone()
    for (codes, scales, _, r), n, out, g in zip(data, ns, outputs, guards):
        old = prepare_q8_dp4a(r, tokens=tokens, residual_compensation=passes == 2)
        actual = torch.empty(tokens, n, dtype=torch.float16, device="cuda:0")
        old.run(x, out=actual, workspace=work)
        assert torch.equal(
            out[:, :n].contiguous().view(torch.uint8), actual.view(torch.uint8)
        )
        assert g[0] == g[-1] == 47 and bool((out[:, n:] == 47).all())
        assert torch.equal(scratch, work)
        ideal = projection_reference(x_cpu, codes, scales, passes)
        assert (actual.cpu().double() - ideal).norm() / (ideal.norm() + 1e-12) < 0.001


@pytest.mark.parametrize("tokens", [8, 23, 32])
def test_large_and_tiny_mixed_descriptor_tile_tails(backend, tokens):
    data = [setup(backend, n, 6145, tokens) for n in (1025, 17, 33)]
    x = data[0][2].cuda()
    res = [d[3] for d in data]
    p = prepare_q8_fused(res, tokens=tokens, residual_compensation=True)
    work = torch.empty(p.workspace_bytes, dtype=torch.uint8, device="cuda:0")
    outs = [
        torch.empty(tokens, n, dtype=torch.float16, device="cuda:0")
        for n in (1025, 17, 33)
    ]
    p.run(x, outputs=outs, workspace=work)
    for r, out in zip(res, outs):
        old = prepare_q8_dp4a(r, tokens=tokens, residual_compensation=True)
        expected = torch.empty_like(out)
        old.run(x, out=expected, workspace=work)
        assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8))


def test_alias_graph_nonfinite_and_no_torch_math(backend, monkeypatch):
    res = [setup(backend, n, 128, 3)[3] for n in (17, 33)]
    p = prepare_q8_fused(res, tokens=3, residual_compensation=True)
    x = torch.ones(3, 128, dtype=torch.float16, device="cuda:0")
    work = torch.empty(p.workspace_bytes, dtype=torch.uint8, device="cuda:0")
    outs = [torch.empty(3, n, dtype=torch.float16, device="cuda:0") for n in (17, 33)]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        p.run(x, outputs=outs, workspace=work)
    torch.cuda.current_stream().wait_stream(stream)
    x.fill_(float("inf"))
    g.replay()
    assert all(torch.isnan(o).all() for o in outs)
    x.zero_()
    g.replay()
    assert all(not o.any() for o in outs)
    with pytest.raises(ValueError, match="alias"):
        p.run(
            x,
            outputs=[work[:102].view(torch.float16).view(3, 17), outs[1]],
            workspace=work,
        )

    def forbidden(*a, **kw):
        raise AssertionError("Fused execution used Torch math/allocation")

    with monkeypatch.context() as m:
        for name in ("empty", "zeros", "matmul", "mm"):
            m.setattr(torch, name, forbidden)
        p.run(x, outputs=outs, workspace=work)
    torch.cuda.synchronize()


def test_cpu_count_preflight():
    for values in ([], [object()], [object()] * 4):
        with pytest.raises(ValueError, match="two or three"):
            prepare_q8_fused(values, tokens=1, residual_compensation=True)


def test_raw_invalid_descriptor_geometry(backend):
    from venturi.ops.q8_fused import OutputDescriptor

    values = [setup(backend, n, 67, 2)[3] for n in (7, 33)]
    p = prepare_q8_fused(values, tokens=2, residual_compensation=True)
    work = torch.empty(p.workspace_bytes, dtype=torch.uint8, device="cuda:0")
    x = torch.ones(2, 67, dtype=torch.float16, device="cuda:0")
    outs = [torch.empty(2, n, dtype=torch.float16, device="cuda:0") for n in (7, 33)]
    p.run(x, outputs=outs, workspace=work)
    desc = p._arguments(x, outs, work)
    args = [
        work.data_ptr(),
        work.numel(),
        desc,
        2,
        2,
        67,
        128,
        2,
        torch.cuda.current_stream().cuda_stream,
    ]
    for idx, value in [
        (0, 0),
        (1, work.numel() - 1),
        (2, None),
        (3, 1),
        (4, 0),
        (5, 0),
        (6, 67),
        (7, 3),
        (0, work.data_ptr() + 1),
    ]:
        bad = args.copy()
        bad[idx] = value
        assert p._run(*bad) == -1
    original = bytes(desc[0])
    for name, value in [
        ("codes", None),
        ("scales", None),
        ("out", None),
        ("rows", 0),
        ("stride", 6),
        ("stride", 2**31 - 1),
        ("codes", p.residents[0]._codes.data_ptr() + 1),
    ]:
        desc[0] = OutputDescriptor.from_buffer_copy(original)
        setattr(desc[0], name, value)
        assert p._run(*args) == -1
    torch.cuda.synchronize()


def test_descriptor_abi():
    import ctypes
    from venturi.ops.q8_fused import OutputDescriptor

    assert ctypes.sizeof(OutputDescriptor) == 32
