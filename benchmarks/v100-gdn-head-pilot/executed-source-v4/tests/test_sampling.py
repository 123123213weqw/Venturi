"""Exact tie/nonfinite semantics and graph-safe native greedy selection."""

import os

import pytest
import torch

from venturi.ops import CudaBackend, TorchBackend, prepare_greedy


@pytest.fixture(
    params=[
        "cpu-fp16",
        "cpu-bf16",
        "cpu-fp32",
        "native-fp16",
        "native-bf16",
        "native-fp32",
    ]
)
def selector_engine(request):
    mode, dtype = request.param.split("-")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[
        dtype
    ]
    if mode == "cpu":
        return TorchBackend(), dtype, "cpu"
    library = os.environ.get("VENTURI_SM70_LIBRARY") or os.environ.get(
        "VENTURI_BLOCKS_LIBRARY"
    )
    if not library:
        pytest.skip("Explicit native SM70/SM89 library required")
    if dtype == torch.bfloat16 and os.environ.get("VENTURI_SM70_LIBRARY"):
        pytest.skip("SM70 rejects native BF16")
    return CudaBackend(library), dtype, "cuda:0"


def buffers(engine, rows, vocab):
    backend, dtype, device = engine
    p = prepare_greedy(backend, rows=rows, vocab_size=vocab, dtype=dtype, device=device)
    out = torch.empty(rows, 2, dtype=torch.int64, device=device)
    scratch = torch.empty(p.workspace_bytes, dtype=torch.uint8, device=device)
    return p, out, scratch


@pytest.mark.parametrize("vocab", [1, 31, 257, 4096, 4097, 248320, 524289])
def test_exact_independent_rows_and_cross_tile_ties(selector_engine, vocab):
    p, out, w = buffers(selector_engine, 4, vocab)
    g = torch.Generator().manual_seed(5001)
    x = torch.randn(4, vocab, generator=g).to(dtype=p.dtype)
    x[0, 0] = x[0, -1] = 10
    x[1].fill_(-torch.finfo(p.dtype).max)
    x[2].fill_(-0.0)
    x[2, -1] = +0.0
    x[3, -1] = 11
    # CPU float64 argmax preserves all source values exactly, including FP32.
    expected = torch.stack((x.double().argmax(1), torch.zeros(4, dtype=torch.int64)), 1)
    original = x.clone()
    device_x = x.to(p.device)
    assert p.run(device_x, out=out, workspace=w) is out
    assert torch.equal(out.cpu(), expected)
    assert torch.equal(device_x.cpu(), original)


def test_nonfinite_rows_never_emit_tokens_and_next_call_recovers(selector_engine):
    p, out, w = buffers(selector_engine, 6, 13017)
    x = torch.zeros(6, 13017, dtype=p.dtype, device=p.device)
    x[0, -1] = float("nan")
    x[1, 256] = float("inf")
    x[2, 8193] = -float("inf")
    x[3].fill_(float("nan"))
    x[4, -1] = 0.5
    p.run(x, out=out, workspace=w)
    assert out.cpu().tolist() == [[-1, 1]] * 4 + [[13016, 0], [0, 0]]
    x.fill_(-3)
    x[:, 4096] = 1
    p.run(x, out=out, workspace=w)
    assert out.cpu().tolist() == [[4096, 0]] * 6


def test_subnormal_signed_zero_and_one_ulp_ordering(selector_engine):
    p, out, w = buffers(selector_engine, 3, 8193)
    info = torch.finfo(p.dtype)
    tiny = info.tiny * info.eps
    x = torch.zeros(3, 8193, dtype=p.dtype)
    x[0, 6000] = tiny
    x[1].fill_(-tiny)
    x[1, 4096] = -0.0
    x[2].fill_(1)
    x[2, 8192] = 1 + info.eps
    assert float(x[0, 6000]) > 0
    expected = torch.stack((x.double().argmax(1), torch.zeros(3, dtype=torch.int64)), 1)
    p.run(x.to(p.device), out=out, workspace=w)
    assert torch.equal(out.cpu(), expected)
    assert out.cpu().tolist() == [[6000, 0], [4096, 0], [8192, 0]]


def test_metadata_alignment_alias_and_no_implicit_conversion():
    b = TorchBackend()
    for rows, vocab in ((0, 4), (1, 0), (True, 4), (65536, 4), (1, 2**31)):
        with pytest.raises(ValueError):
            prepare_greedy(b, rows=rows, vocab_size=vocab, device="cpu")
    with pytest.raises(ValueError, match="FP16/BF16/FP32"):
        prepare_greedy(b, vocab_size=5, dtype=torch.float64, device="cpu")
    p, out, w = buffers((b, torch.float32, "cpu"), 1, 64)
    x = torch.ones(1, 64)
    for bad in (x.double(), x.T, x[:, ::2], x.requires_grad_()):
        with pytest.raises(ValueError):
            p.run(bad, out=out, workspace=w)
    x = x.detach()
    for bad in (out.int(), torch.empty(2, 1, dtype=torch.int64)):
        with pytest.raises(ValueError):
            p.run(x, out=bad, workspace=w)
    with pytest.raises(ValueError):
        p.run(x, out=out, workspace=w[:-1])
    unaligned = torch.empty(p.workspace_bytes + 1, dtype=torch.uint8)[1:]
    with pytest.raises(ValueError, match="alignment"):
        p.run(x, out=out, workspace=unaligned)
    shared = torch.empty(256, dtype=torch.uint8)
    with pytest.raises(ValueError, match="alias"):
        p.run(shared.view(torch.float32).view(1, 64), out=out, workspace=shared[:16])
    with pytest.raises(ValueError, match="alias"):
        p.run(
            shared.view(torch.float32).view(1, 64),
            out=shared[:16].view(torch.int64).view(1, 2),
            workspace=w,
        )


def test_native_capture_no_allocations_fallback_sync_or_scalar_read(
    selector_engine, monkeypatch
):
    if selector_engine[2] == "cpu":
        pytest.skip("Native capture/no-fallback checks")
    p, out, w = buffers(selector_engine, 8, 248320)
    x = torch.randn(8, 248320, dtype=p.dtype, device=p.device)
    expected = torch.stack(
        (x.cpu().double().argmax(1), torch.zeros(8, dtype=torch.int64)), 1
    )
    p.run(x, out=out, workspace=w)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()

    def forbidden(*a, **kw):
        raise AssertionError("No allocation/numerical Torch fallback/host sync/read")

    def prohibit(patch):
        for name in (
            "empty",
            "empty_like",
            "zeros",
            "zeros_like",
            "argmax",
            "isfinite",
        ):
            patch.setattr(torch, name, forbidden)
        for name in ("argmax", "isfinite", "item", "tolist", "cpu", "to"):
            patch.setattr(torch.Tensor, name, forbidden)
        patch.setattr(torch.cuda, "synchronize", forbidden)

    # PyTorch synchronizes BEFORE starting capture. That is graph setup, not
    # run(); prohibit synchronization/allocation/fallback in the operator body.
    with torch.cuda.graph(graph):
        with monkeypatch.context() as patch:
            prohibit(patch)
            p.run(x, out=out, workspace=w)
    with monkeypatch.context() as patch:
        prohibit(patch)
        graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out.cpu(), expected)
    x.fill_(0)
    x[0, -1] = float("nan")
    graph.replay()
    torch.cuda.synchronize()
    assert out.cpu().tolist() == [[-1, 1]] + [[0, 0]] * 7


def test_native_reentrant_streams_with_independent_buffers(selector_engine):
    if selector_engine[2] == "cpu":
        pytest.skip("Native stream test")
    p, out, w = buffers(selector_engine, 1, 4097)
    other_out, other_w = torch.empty_like(out), torch.empty_like(w)
    a = torch.zeros(1, 4097, dtype=p.dtype, device=p.device)
    b = a.clone()
    a[0, 29], b[0, 4096] = 1, 1
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        p.run(a, out=out, workspace=w)
    p.run(b, out=other_out, workspace=other_w)
    torch.cuda.synchronize()
    assert out.cpu().tolist() == [[29, 0]]
    assert other_out.cpu().tolist() == [[4096, 0]]
    with pytest.raises(RuntimeError, match="greedy selection"):
        p.backend._check(
            p.backend._native.venturi_greedy_run(
                a.data_ptr(),
                out.data_ptr(),
                w.data_ptr(),
                p.workspace_bytes - 1,
                p.rows,
                p.vocab_size,
                0,
                0,
            )
        )
