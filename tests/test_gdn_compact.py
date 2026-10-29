"""Compact launch keeps the eight-warp reduction, output and FP32 state bytes."""

import ctypes as C
import os
from types import SimpleNamespace

import pytest
import torch

from test_gdn import gdn_inputs, gdn_oracle
from venturi.ops import CudaBackend, TorchBackend
from venturi.ops.gdn_compact import prepare_gdn_compact


@pytest.fixture
def backends():
    library = os.environ.get("VENTURI_GDN_COMPACT_LIBRARY")
    control = os.environ.get("VENTURI_GDN_CONTROL_LIBRARY")
    if not library or not control:
        pytest.skip("explicit compact and pinned control libraries required")
    return CudaBackend(control), CudaBackend(library)


def plans(backends, t, dk, normalize=True):
    cfg = dict(
        tokens=t,
        qk_heads=2,
        value_heads=4,
        key_dim=dk,
        value_dim=13,
        normalize_qk=normalize,
        dtype=torch.float16,
        device="cuda:0",
    )
    return [
        backends[0].prepare_gdn(**cfg),
        backends[1].prepare_gdn(**cfg),
        prepare_gdn_compact(backends[1], **cfg),
    ]


def equal_bytes(a, b):
    return torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


@pytest.mark.parametrize("tokens", [1, 2, 4, 8, 23, 32])
@pytest.mark.parametrize("dk", [64, 128, 256])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("packed", [False, True])
def test_output_state_bytes_fp64_and_guarded_storage(
    backends, tokens, dk, normalize, packed
):
    values = gdn_inputs(tokens=tokens, dk=dk, dtype=torch.float16, device="cuda:0")
    q, k, v, g, beta, state = values
    if not normalize:
        k.mul_(0.1)
    ideal, ideal_state = gdn_oracle(*values, normalize=normalize)
    qkv = torch.cat([q.flatten(1), k.flatten(1), v.flatten(1)], 1)
    results = []
    for plan in plans(backends, tokens, dk, normalize):
        og = torch.full((v.numel() + 2,), 47, dtype=v.dtype, device=v.device)
        sg = torch.full(
            (state.numel() + 2,), 47, dtype=state.dtype, device=state.device
        )
        out, final = og[1:-1].view_as(v), sg[1:-1].view_as(state)
        if packed:
            plan.run_packed(qkv, g, beta, state=state, out=out, state_out=final)
        else:
            plan.run(q, k, v, g, beta, state=state, out=out, state_out=final)
        assert og[0] == og[-1] == sg[0] == sg[-1] == 47
        results.append((out.cpu(), final.cpu()))
    assert all(
        equal_bytes(a, b) for pair in results[1:] for a, b in zip(results[0], pair)
    )
    out, final = results[-1]
    error = (out.double() - ideal.half().double()).abs()
    assert (error <= 2 * torch.finfo(torch.float16).eps * ideal.abs() + 2e-5).all()
    torch.testing.assert_close(final.double(), ideal_state, atol=2e-6, rtol=2e-4)


@pytest.mark.parametrize("dk", [64, 128, 256])
def test_chunks_inplace_graph_alias_and_no_fallback(backends, dk, monkeypatch):
    q, k, v, g, beta, state = gdn_inputs(
        tokens=8, dk=dk, dtype=torch.float16, device="cuda:0"
    )
    plan = plans(backends, 8, dk)[-1]
    out, final = torch.empty_like(v), torch.empty_like(state)
    plan.run(q, k, v, g, beta, state=state, out=out, state_out=final)
    work = state.clone()
    parts = []
    start = 0
    for count in [1, 2, 5]:
        chunk = plans(backends, count, dk)[-1]
        part = torch.empty_like(v[start : start + count])
        chunk.run(
            *(x[start : start + count] for x in [q, k, v, g, beta]),
            state=work,
            out=part,
            state_out=work,
        )
        parts.append(part)
        start += count
    assert equal_bytes(torch.cat(parts), out) and equal_bytes(work, final)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()

    def forbidden(*a, **kw):
        raise AssertionError("Compact GDN used numerical fallback/allocation")

    with monkeypatch.context() as m:
        for name in ["empty", "empty_like", "matmul", "mm", "cat"]:
            m.setattr(torch, name, forbidden)
        with torch.cuda.graph(graph, stream=stream):
            plan.run(q, k, v, g, beta, state=state, out=out, state_out=final)
        graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    assert equal_bytes(torch.cat(parts), out) and equal_bytes(work, final)
    with pytest.raises(ValueError, match="alias"):
        plan.run(q, k, v, g, beta, state=state, out=v, state_out=final)
    with pytest.raises(ValueError, match="dtype"):
        plan.run(q, k, v, g, beta, state=state, out=out, state_out=final.half())


@pytest.mark.parametrize("dk", [64, 128])
@pytest.mark.parametrize("field", range(6))
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), 0.0])
def test_nonfinite_virtual_lanes_and_zero_propagation(backends, dk, field, value):
    values = list(gdn_inputs(tokens=2, dk=dk, dtype=torch.float16, device="cuda:0"))
    values[field].flatten()[0] = value
    actual = []
    for plan in [plans(backends, 2, dk)[0], plans(backends, 2, dk)[-1]]:
        out, final = torch.empty_like(values[2]), torch.empty_like(values[5])
        plan.run(*values[:5], state=values[5], out=out, state_out=final)
        actual.append((out, final))
    for old, new in zip(*actual):
        assert torch.equal(torch.isnan(old), torch.isnan(new))
        assert torch.equal(torch.isinf(old), torch.isinf(new))
        finite = torch.isfinite(old)
        assert equal_bytes(old[finite], new[finite])


def test_raw_geometry_and_preserved_sm70_bf16_rejection(backends):
    assert plans(backends, 1, 128)[-1].key_dim == 128
    q, k, v, g, beta, state = gdn_inputs(
        tokens=1, dk=128, dtype=torch.float16, device="cuda:0"
    )
    out, final = torch.empty_like(v), torch.empty_like(state)
    fn = backends[1]._native.venturi_gdn_compact_run
    assert len(fn.argtypes) == 18
    args = [x.data_ptr() for x in [q, k, v, g, beta, state, out, final]] + [
        1,
        2,
        4,
        128,
        13,
        1,
        0,
        0,
        128**-0.5,
        torch.cuda.current_stream().cuda_stream,
    ]
    for i, value in [
        (0, 0),
        (8, 0),
        (9, 3),
        (11, 32),
        (13, 2),
        (14, 2),
        (15, 1),
        (16, float("nan")),
    ]:
        bad = args.copy()
        bad[i] = value
        assert fn(*bad) == -1
    assert fn(*args) == 0
    assert C.sizeof(C.c_void_p) == 8


def test_explicit_backend_and_missing_export_preflight():
    with pytest.raises(ValueError, match="explicit native"):
        prepare_gdn_compact(TorchBackend(), tokens=1, qk_heads=2, value_heads=4)
    b = SimpleNamespace(
        execution_kind="native-cuda",
        validate_device=lambda *a: torch.device("cuda:0"),
        _native=SimpleNamespace(),
    )
    with pytest.raises(RuntimeError, match="export absent"):
        prepare_gdn_compact(b, tokens=1, qk_heads=2, value_heads=4)


@pytest.mark.parametrize("dk", [64, 128])
def test_zero_signed_zero_and_dynamic_graph_inputs(backends, dk):
    values = list(gdn_inputs(tokens=3, dk=dk, dtype=torch.float16, device="cuda:0"))
    for x in values:
        x.fill_(-0.0)
    candidate = plans(backends, 3, dk)[-1]
    old = plans(backends, 3, dk)[0]
    outputs = [
        (torch.empty_like(values[2]), torch.empty_like(values[5])) for _ in range(2)
    ]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        candidate.run(
            *values[:5], state=values[5], out=outputs[1][0], state_out=outputs[1][1]
        )
    torch.cuda.current_stream().wait_stream(stream)
    for replacement in [-0.0, 0.25, float("nan"), 0.0]:
        values[2].fill_(replacement)
        values[4].fill_(0.5)
        graph.replay()
        old.run(
            *values[:5], state=values[5], out=outputs[0][0], state_out=outputs[0][1]
        )
        for a, b in zip(*outputs):
            assert torch.equal(torch.isnan(a), torch.isnan(b))
            finite = torch.isfinite(a)
            assert equal_bytes(a[finite], b[finite])
