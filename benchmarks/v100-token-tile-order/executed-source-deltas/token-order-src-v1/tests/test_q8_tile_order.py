"""Adjacent token CTAs preserve exact dots, tails, pitches and graph semantics."""

import ctypes as C
import os
import pytest
import torch
from test_q8_dp4a import setup
from q8_dp4a_reference import projection_reference
from venturi.ops import CudaBackend, prepare_q8_dp4a
from venturi.ops.q8_shared import prepare_q8_tiled_shared
from venturi.ops.q8_fused import prepare_q8_fused

TOKENS = [8, 9, 15, 16, 17, 23, 32, 46, 92, 184]


@pytest.fixture
def backend():
    path = os.environ.get("VENTURI_TILE_ORDER_LIBRARY")
    if not path:
        pytest.skip("explicit tile-order candidate library required")
    return CudaBackend(path)


@pytest.mark.parametrize("tokens", TOKENS)
@pytest.mark.parametrize("passes", [1, 2])
@pytest.mark.parametrize("shape", [(7, 67), (33, 5120), (1025, 6145)])
def test_plain_exact_bytes_workspace_and_guard(backend, tokens, passes, shape):
    n, k = shape
    codes, scales, x, r = setup(backend, n, k, tokens)
    plans = [
        prepare_q8_dp4a(r, tokens=tokens, residual_compensation=passes == 2),
        prepare_q8_tiled_shared(
            r,
            tokens=tokens,
            rows=16,
            token_tile=8,
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
@pytest.mark.parametrize("ns", [(1025, 17), (1025, 17, 33)])
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
    data = [setup(backend, n, 128, 17) for n in ([1025, 17, 33] if group else [1025])]
    x = data[0][2].cuda()
    if group:
        p = prepare_q8_fused(
            [d[3] for d in data], tokens=17, residual_compensation=True
        )
    else:
        p = prepare_q8_tiled_shared(
            data[0][3], tokens=17, rows=16, token_tile=8, residual_compensation=True
        )
    work = torch.empty(p.workspace_bytes, dtype=torch.uint8, device="cuda:0")
    ys = [
        torch.empty(17, d[3].vocab_size, dtype=torch.float16, device="cuda:0")
        for d in data
    ]

    def call():
        return p.run(
            x, workspace=work, **({"outputs": ys} if group else {"out": ys[0]})
        )

    call()
    expected = [y.clone() for y in ys]
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


def geometry(backend):
    fn = backend._native.venturi_q8_tile_geometry
    fn.argtypes = [C.c_uint64, C.c_int, C.c_int, C.POINTER(C.c_uint)]
    fn.restype = C.c_int
    return fn


def test_host_geometry_small_bijection_and_int32_grid_fallback(backend):
    fn = geometry(backend)
    for blocks in [1, 7, 2**31 - 1]:
        for tokens in [1, 8, 9, 16, 23, 262140]:
            for tile in [4, 8]:
                nt = (tokens + tile - 1) // tile
                expected = (
                    (blocks * nt, 1)
                    if tile == 8 and nt > 1 and blocks * nt <= 2**31 - 1
                    else (blocks, nt)
                )
                out = (C.c_uint * 2)()
                assert fn(blocks, tokens, tile, out) == 0 and tuple(out) == expected
                if blocks < 10 and tokens < 30:
                    pairs = {
                        (x // nt, x % nt)
                        if tile == 8 and out[1] == 1 and nt > 1
                        else (x, y)
                        for y in range(out[1])
                        for x in range(out[0])
                    }
                    assert pairs == {(b, t) for b in range(blocks) for t in range(nt)}


def test_host_geometry_invalid_preflight(backend):
    fn = geometry(backend)
    out = (C.c_uint * 2)()
    for b, t, tile, p in [
        (0, 16, 8, out),
        (2**31, 16, 8, out),
        (1, 0, 8, out),
        (1, 262141, 8, out),
        (1, 16, 3, out),
        (1, 16, 8, None),
    ]:
        assert fn(b, t, tile, p) == -1
