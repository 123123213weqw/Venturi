"""T16 reuse preserves exact dots, tails, pitches, graphs and descriptor guards."""

import os
import pytest
import torch
from test_q8_dp4a import setup
from q8_dp4a_reference import projection_reference
from venturi.ops import CudaBackend, prepare_q8_dp4a
from venturi.ops.q8_shared import prepare_q8_token16
from venturi.ops.q8_fused import prepare_q8_fused

TOKENS = [8, 16, 17, 23, 31, 32, 33, 46, 92, 184]


@pytest.fixture
def backend():
    path = os.environ.get("VENTURI_T16_LIBRARY")
    if not path:
        pytest.skip("explicit T16 candidate library required")
    return CudaBackend(path)


@pytest.mark.parametrize("tokens", TOKENS)
@pytest.mark.parametrize("passes", [1, 2])
@pytest.mark.parametrize("shape", [(7, 67), (1023, 6145), (1025, 6145)])
def test_plain_exact_bytes_workspace_and_guard(backend, tokens, passes, shape):
    n, k = shape
    codes, scales, x, r = setup(backend, n, k, tokens)
    plans = [
        prepare_q8_dp4a(r, tokens=tokens, residual_compensation=passes == 2),
        prepare_q8_token16(
            r,
            tokens=tokens,
            rows=8,
            residual_compensation=passes == 2,
        ),
    ]
    work = [
        torch.empty(p.workspace_bytes, dtype=torch.uint8, device="cuda:0")
        for p in plans
    ]
    guard = [
        torch.full((tokens * n + 2,), 47, dtype=torch.float16, device="cuda:0")
        for _ in plans
    ]
    xg = x.cuda()
    for p, w, g in zip(plans, work, guard):
        p.run(xg, out=g[1:-1].view(tokens, n), workspace=w)
    assert torch.equal(guard[0].view(torch.uint8), guard[1].view(torch.uint8))
    assert guard[1][0] == guard[1][-1] == 47 and torch.equal(*work)
    ts = sorted({0, tokens // 2, tokens - 1})
    rows = sorted({0, n // 2, n - 1})
    ref = projection_reference(x[ts], codes[rows], scales[rows], passes)
    actual = guard[1][1:-1].view(tokens, n).cpu()[ts][:, rows].double()
    assert (actual - ref).norm() / (ref.norm() + 1e-12) < 0.001


@pytest.mark.parametrize("tokens", TOKENS)
@pytest.mark.parametrize("passes", [1, 2])
@pytest.mark.parametrize("ns", [(1024, 17), (1025, 17, 33), (33, 17)])
def test_fused_cross_descriptor_boundary_and_pitched_tails(backend, tokens, passes, ns):
    data = [setup(backend, n, 6145, tokens) for n in ns]
    x = data[0][2].cuda()
    strides = [n + 5 for n in ns]
    p = prepare_q8_fused(
        [d[3] for d in data],
        tokens=tokens,
        residual_compensation=passes == 2,
        output_strides=strides,
    )
    work = torch.empty(p.workspace_bytes, dtype=torch.uint8, device="cuda:0")
    guards = [
        torch.full((tokens * s + 2,), 47, dtype=torch.float16, device="cuda:0")
        for s in strides
    ]
    ys = [g[1:-1].view(tokens, s) for g, s in zip(guards, strides)]
    p.run(x, outputs=ys, workspace=work)
    packed = work.clone()
    for n, d, y, g in zip(ns, data, ys, guards):
        old = prepare_q8_dp4a(d[3], tokens=tokens, residual_compensation=passes == 2)
        out = torch.empty(tokens, n, dtype=torch.float16, device="cuda:0")
        old.run(x, out=out, workspace=work)
        assert torch.equal(
            y[:, :n].contiguous().view(torch.uint8), out.view(torch.uint8)
        )
        assert g[0] == g[-1] == 47 and bool((y[:, n:] == 47).all())
        assert torch.equal(packed, work)


@pytest.mark.parametrize("group", [False, True])
def test_graph_dynamic_nonfinite_and_no_torch_math(backend, group, monkeypatch):
    tokens = 17
    data = [setup(backend, n, 6145, tokens) for n in ([1025, 17] if group else [1025])]
    x = data[0][2].cuda()
    if group:
        p = prepare_q8_fused(
            [d[3] for d in data], tokens=tokens, residual_compensation=True
        )
    else:
        p = prepare_q8_token16(
            data[0][3], tokens=tokens, rows=8, residual_compensation=True
        )
    work = torch.empty(p.workspace_bytes, dtype=torch.uint8, device="cuda:0")
    ys = [
        torch.empty(tokens, d[0].shape[0], dtype=torch.float16, device="cuda:0")
        for d in data
    ]

    def call():
        return p.run(
            x, workspace=work, **({"outputs": ys} if group else {"out": ys[0]})
        )

    call()
    expected = [y.clone() for y in ys]
    if not group:
        r = p.base.resident
        args = [
            work.data_ptr(),
            work.numel(),
            r._codes.data_ptr(),
            r._scales.data_ptr(),
            ys[0].data_ptr(),
            tokens,
            r.vocab_size,
            r.hidden_size,
            r._codes.shape[1],
            2,
            8,
            torch.cuda.current_stream().cuda_stream,
        ]
        for index, value in [
            (0, 0),
            (1, work.numel() - 1),
            (2, 0),
            (3, 0),
            (4, 0),
            (5, 0),
            (6, 0),
            (7, 0),
            (8, 1),
            (9, 3),
            (10, 16),
            (0, work.data_ptr() + 1),
            (2, r._codes.data_ptr() + 1),
        ]:
            bad = args.copy()
            bad[index] = value
            assert p._run(*bad) == -1
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        call()
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    assert all(
        torch.equal(a.view(torch.uint8), b.view(torch.uint8))
        for a, b in zip(expected, ys)
    )
    x.fill_(float("inf"))
    graph.replay()
    assert all(bool(torch.isnan(y).all()) for y in ys)
    x.zero_()
    graph.replay()
    assert all(not y.any() for y in ys)

    def forbidden(*a, **kw):
        raise AssertionError("Projection dispatch used Torch math/allocation")

    with monkeypatch.context() as m:
        for name in ["empty", "zeros", "matmul", "mm"]:
            m.setattr(torch, name, forbidden)
        call()
    torch.cuda.synchronize()


def test_t16_cpu_geometry_rejected_before_backend():
    for rows in [4, 16, True, 0]:
        with pytest.raises(ValueError, match="T16"):
            prepare_q8_token16(
                object(),
                tokens=16,
                rows=rows,
                residual_compensation=True,
            )


def test_t16_absent_export_is_explicit(monkeypatch):
    from types import SimpleNamespace
    import venturi.ops.q8_shared as shared

    native = SimpleNamespace()
    fake = SimpleNamespace(
        resident=SimpleNamespace(backend=SimpleNamespace(_native=native))
    )
    monkeypatch.setattr(shared, "prepare_q8_dp4a", lambda *a, **kw: fake)
    with pytest.raises(RuntimeError, match="T16 research export absent"):
        shared.prepare_q8_token16(object(), tokens=16, residual_compensation=True)
