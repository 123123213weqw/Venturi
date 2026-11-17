"""SM70 Q5 public contract, tail/rounding regressions and both execution lanes."""

import os

import pytest
import torch

from venturi.ops import CudaBackend, Q5RowSplitWeight, TorchBackend
from venturi.ops.q5_linear import (
    BLAS_BYTES,
    Q5Dispatcher,
    prepare_q5,
    tune_q5,
    workspace_geometry,
    export_q5_calibration,
    load_q5_calibration,
)


def weight(n=19, k=131):
    g = torch.Generator().manual_seed(n + k)
    return Q5RowSplitWeight.from_dense(torch.randn(n, k, generator=g) * 0.025)


def expected(w, x):
    return (x.cpu().double() @ w.dequantize_reference().half().double().T).half()


def test_public_q5_reference_and_owned_storage():
    w = weight()
    b = TorchBackend()
    assert b.capabilities.supports(
        "linear",
        dtype=torch.float16,
        device="cpu",
        weight_format=w.format,
        weight_layout=w.layout,
    )
    r = b.upload_linear(w, dtype=torch.float16, device="cpu")
    x = torch.randn(4, w.in_features).half()
    out = torch.empty(4, w.out_features).half()
    oracle = expected(w, x)
    w.codes.zero_()
    r.prepare(tokens=4).run(x, out=out)
    torch.testing.assert_close(out, oracle, rtol=0.002, atol=0.002)


def test_reference_rounding_cancellation_regression():
    w = Q5RowSplitWeight.from_codes(
        torch.tensor([[15, -14]]), torch.tensor([[0.1]], dtype=torch.float16)
    )
    r = TorchBackend().upload_linear(w, dtype=torch.float16, device="cpu")
    out = torch.empty(1, 1).half()
    r.prepare(tokens=1).run(torch.ones(1, 2).half(), out=out)
    assert out.item() == 0.1005859375  # not the old FP32-scale result .0999756


def test_workspace_budget_is_bounded_and_aligned():
    for k in (1, 131, 5120):
        for n in (1, 129, 17408):
            budget = BLAS_BYTES + 1024 * 1024
            chunk, size = workspace_geometry(n, k, budget)
            assert 1 <= chunk <= n and size <= budget and size % 256 == 0
    with pytest.raises(ValueError):
        workspace_geometry(1, 131, BLAS_BYTES + 261)


def test_native_q5_capability_is_sm70_f16_and_extension_gated():
    b = object.__new__(CudaBackend)
    b.q4_backend = None
    for arch in (70, 89):
        for present in (False, True):
            b.architecture, b._q5_available = arch, present
            for dtype in (torch.float16, torch.bfloat16):
                assert b.capabilities.supports(
                    "linear",
                    dtype=dtype,
                    device="cuda:0",
                    weight_format="q5_g64_fp16",
                    weight_layout="row_split_k128_v1",
                ) == (arch == 70 and present and dtype == torch.float16)


@pytest.fixture
def native():
    lib = os.environ.get("VENTURI_OPS_LIBRARY")
    if (
        not lib
        or not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (7, 0)
    ):
        pytest.skip("Explicit SM70 VENTURI_OPS_LIBRARY required")
    return CudaBackend(lib)


@pytest.mark.parametrize(
    "n,k", [(1, 1), (17, 63), (19, 64), (129, 131), (33, 255), (128, 320)]
)
@pytest.mark.parametrize("tokens", [1, 3, 4, 17])
def test_all_routes_against_cpu_oracle(native, n, k, tokens):
    w = weight(n, k)
    r = native.upload_linear(w, dtype=torch.float16, device="cuda:0")
    x = torch.randn(tokens, k).half().cuda()
    oracle = expected(w, x)
    for route in ("scalar", "cublas", "wmma"):
        if route == "wmma" and k % 64:
            with pytest.raises(ValueError, match="multiple of 64"):
                prepare_q5(r, tokens=tokens, route=route)
            continue
        p = prepare_q5(r, tokens=tokens, route=route)
        ws = (
            torch.empty(p.workspace_bytes, dtype=torch.uint8, device="cuda")
            if p.workspace_bytes
            else None
        )
        out = torch.empty(tokens, n, device="cuda", dtype=torch.float16)
        p.run(x, out=out, workspace=ws)
        torch.testing.assert_close(out.cpu(), oracle, rtol=0.002, atol=0.002)
        p.close()


def test_odd_k_lone_tail_and_misaligned_token_rows(native):
    codes = torch.zeros(19, 131, dtype=torch.int16)
    codes[:, -1] = 1
    w = Q5RowSplitWeight.from_codes(codes, torch.ones(19, 3).half())
    r = native.upload_linear(w, dtype=torch.float16, device="cuda:0")
    x = torch.zeros(3, 131).half().cuda()
    x[:, -1] = 1
    p = prepare_q5(r, tokens=3, route="scalar")
    out = torch.empty(3, 19, device="cuda", dtype=torch.float16)
    p.run(x, out=out)
    assert torch.equal(out.cpu(), torch.ones(3, 19).half())
    p.close()


def test_cublas_many_chunks_and_stream_isolation(native):
    w = weight(33, 131)
    r = native.upload_linear(w, dtype=torch.float16, device="cuda:0")
    p = prepare_q5(r, tokens=3, route="cublas", workspace_limit_bytes=BLAS_BYTES + 1024)
    assert p.chunk_rows < 33
    x = torch.randn(3, 131).half().cuda()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    outputs = [torch.empty(3, 33, device="cuda", dtype=torch.float16) for _ in streams]
    workspaces = [
        torch.empty(p.workspace_bytes, device="cuda", dtype=torch.uint8)
        for _ in streams
    ]
    torch.cuda.synchronize()
    for stream, out, ws in zip(streams, outputs, workspaces):
        with torch.cuda.stream(stream):
            p.run(x, out=out, workspace=ws)
    for stream in streams:
        stream.synchronize()
    for out in outputs:
        torch.testing.assert_close(out.cpu(), expected(w, x), rtol=0.002, atol=0.002)
    p.close()


def test_gpu_rounding_boundary_all_routes(native):
    codes = torch.zeros(16, 64, dtype=torch.int16)
    codes[:, :2] = torch.tensor([15, -14])
    w = Q5RowSplitWeight.from_codes(codes, torch.full((16, 1), 0.1).half())
    r = native.upload_linear(w, dtype=torch.float16, device="cuda:0")
    x = torch.zeros(4, 64).half().cuda()
    x[:, :2] = 1
    for route in ("scalar", "wmma", "cublas"):
        p = prepare_q5(r, tokens=4, route=route)
        out = torch.empty(4, 16, device="cuda", dtype=torch.float16)
        ws = (
            torch.empty(p.workspace_bytes, dtype=torch.uint8, device="cuda")
            if p.workspace_bytes
            else None
        )
        p.run(x, out=out, workspace=ws)
        assert torch.equal(out.cpu(), torch.full((4, 16), 0.1005859375).half())
        p.close()


def test_q5_public_mlp_composition(native):
    from venturi.models.gated_mlp import GatedMLPWeights

    w = GatedMLPWeights(
        torch.ones(64), weight(128, 64), weight(128, 64), weight(64, 128)
    )
    x = torch.randn(4, 64).half()
    outputs = []
    for backend, device in ((TorchBackend(), "cpu"), (native, "cuda:0")):
        p = w.upload(backend, dtype=torch.float16, device=device).prepare(tokens=4)
        out = torch.empty(4, 64, device=device, dtype=torch.float16)
        p.run(x.to(device), out=out, workspace=p.empty_workspace())
        outputs.append(out.cpu())
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0.003, atol=0.003)


def test_calibration_and_dynamic_dispatch(native):
    w = weight(129, 320)
    r = native.upload_linear(w, dtype=torch.float16, device="cuda:0")
    xs = {m: torch.randn(m, 320).half().cuda() for m in (1, 4, 17)}
    before = prepare_q5(r, tokens=4)
    assert before.kernel == "cublas" and before.reason == "uncalibrated-safe-baseline"
    for x in xs.values():
        rec = tune_q5(r, x, warmup=1, iterations=2, repeats=2)
        assert rec["candidates"][rec["selected"]]["correct"]
    records = export_q5_calibration(r)
    r._payload.selections.clear()
    load_q5_calibration(r, records)
    records[0]["identity"]["device_uuid"] = "wrong-device"
    with pytest.raises(ValueError, match="identity mismatch"):
        load_q5_calibration(r, records)
    assert before.kernel == "cublas"  # existing plans/graphs are immutable
    d = Q5Dispatcher(r, xs)
    ws = (
        torch.empty(d.workspace_bytes, device="cuda", dtype=torch.uint8)
        if d.workspace_bytes
        else None
    )
    for m in (1, 17, 4, 1, 4):
        out = torch.empty(m, 129, device="cuda", dtype=torch.float16)
        d.run(xs[m], out=out, workspace=ws)
        torch.testing.assert_close(
            out.cpu(), expected(w, xs[m]), rtol=0.002, atol=0.002
        )
    with pytest.raises(ValueError, match="not explicitly prepared"):
        d.run(torch.zeros(2, 320).half().cuda(), out=out, workspace=ws)
    d.close()
    before.close()


@pytest.mark.parametrize("route", ["scalar", "wmma", "cublas"])
def test_graph_alias_and_closed_plan(native, route):
    w = weight(128, 320)
    r = native.upload_linear(w, dtype=torch.float16, device="cuda:0")
    p = prepare_q5(r, tokens=4, route=route)
    x = torch.randn(4, 320).half().cuda()
    out = torch.empty(4, 128, device="cuda", dtype=torch.float16)
    ws = (
        torch.empty(p.workspace_bytes, dtype=torch.uint8, device="cuda")
        if p.workspace_bytes
        else None
    )
    for _ in range(3):
        p.run(x, out=out, workspace=ws)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        p.run(x, out=out, workspace=ws)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out.cpu(), expected(w, x), rtol=0.002, atol=0.002)
    with pytest.raises(ValueError, match="alias"):
        p.run(x, out=x.flatten()[:512].view(4, 128), workspace=ws)
    p.close()
    with pytest.raises(RuntimeError, match="closed"):
        p.run(x, out=out, workspace=ws)
