"""Explicit FP16 port: independent numeric checks, architecture and ABI guards.

SM70 tests opt in with VENTURI_SM70_LIBRARY. SM89 uses VENTURI_BLOCKS_LIBRARY.
Neither variable starts compilation or changes a GPU service.
"""

from contextlib import nullcontext
import os

import pytest
import torch

from test_gdn import gdn_inputs, gdn_oracle
from test_qwen_model import (
    test_hybrid_manual_composition_state_and_native_no_fallback as hybrid_check,
    test_session_failure_atomic_position_poison_reset_and_invalid_ids as failure_check,
    test_failure_after_committed_prefix_of_one_forward_poison_entire_append as chunk_failure_check,
    test_session_owner_stream_and_two_request_continuations as stream_check,
)
from test_qwen_gdn_layer import fixture_weights as gdn_fixture
from qwen_gdn_reference import qualify_layer_operations
from venturi.models import ResidentQwenGDNLayer
from venturi.ops import CudaBackend, Q8RowSplitWeight, TorchBackend


@pytest.fixture(params=["cpu-fp16", "native-fp16"])
def port(request):
    if request.param == "cpu-fp16":
        return TorchBackend(), torch.float16, "cpu"
    library = os.environ.get("VENTURI_SM70_LIBRARY") or os.environ.get(
        "VENTURI_BLOCKS_LIBRARY"
    )
    if not library:
        pytest.skip("explicit SM70 or SM89 native library required")
    backend = CudaBackend(library)
    expected = 70 if os.environ.get("VENTURI_SM70_LIBRARY") else 89
    assert backend.architecture == expected
    return backend, torch.float16, "cuda:0"


def test_architecture_guard_without_gpu(monkeypatch):
    backend = object.__new__(CudaBackend)
    backend.architecture = 70
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (7, 0))
    assert backend._device(torch.float16, "cuda:0") == torch.device("cuda:0")
    with pytest.raises(ValueError, match="not native BF16"):
        backend._device(torch.bfloat16, "cuda:0")
    backend.architecture = 89
    with pytest.raises(ValueError, match="requires SM89"):
        backend._device(torch.float16, "cuda:0")


@pytest.mark.parametrize("k", [65, 129, 5120])
@pytest.mark.parametrize("tokens", [1, 9])
def test_q8_fp16_embedding_projection_and_encoded_storage(port, k, tokens):
    backend, dtype, device = port
    source = (
        torch.randn(31, k, generator=torch.Generator().manual_seed(1042)).bfloat16()
        * 0.03
    )
    w = Q8RowSplitWeight.from_dense(source)
    resident = backend.upload_linear(w, dtype=dtype, device=device)
    if device != "cpu":
        assert torch.equal(resident.storage_tensors[0].cpu(), w.codes)
        assert torch.equal(resident.storage_tensors[1].cpu(), w.scales)
    exact = (w.codes.double() * w.scales.double().repeat_interleave(32, -1))[:, :k]
    lookup = resident.prepare_embedding([30, 1, 7, 7])
    embeddings = torch.empty(4, k, dtype=dtype, device=device)
    lookup.run(out=embeddings)
    assert torch.equal(embeddings.cpu(), exact[[30, 1, 7, 7]].to(dtype))
    x = (
        torch.randn(tokens, k, generator=torch.Generator().manual_seed(1019)) * 0.2
    ).to(device=device, dtype=dtype)
    out = torch.empty(tokens, 31, dtype=dtype, device=device)
    resident.prepare(tokens=tokens).run(x, out=out)
    ideal = x.cpu().double() @ exact.T
    error = (out.cpu().double() - ideal.to(dtype).double()).abs()
    assert bool((error <= 2 * torch.finfo(dtype).eps * ideal.abs() + 2e-4).all())


@pytest.mark.parametrize("dk", [64, 128, 256])
@pytest.mark.parametrize("normalize", [False, True])
def test_fp16_gdn_packed_unpacked_oracle_and_chunks(port, dk, normalize):
    backend, dtype, device = port
    q, k, v, g, beta, s = gdn_inputs(dk=dk, dtype=dtype, device=device)
    if not normalize:
        k.mul_(0.1)
    ideal, ideal_s = gdn_oracle(q, k, v, g, beta, s, normalize=normalize)

    def prepare(t):
        return backend.prepare_gdn(
            tokens=t,
            qk_heads=2,
            value_heads=4,
            key_dim=dk,
            value_dim=13,
            dtype=dtype,
            device=device,
            normalize_qk=normalize,
        )

    plan = prepare(9)
    packed = torch.cat((q.flatten(1), k.flatten(1), v.flatten(1)), dim=-1)
    out, final = torch.empty_like(v), torch.empty_like(s)
    a, b = torch.empty_like(v), torch.empty_like(s)
    plan.run(q, k, v, g, beta, state=s, out=out, state_out=final)
    plan.run_packed(packed, g, beta, state=s, out=a, state_out=b)
    assert torch.equal(a, out) and torch.equal(b, final)
    error = (out.cpu().double() - ideal.to(dtype).double()).abs()
    assert bool((error <= 2 * torch.finfo(dtype).eps * ideal.abs() + 2e-5).all())
    torch.testing.assert_close(final.cpu().double(), ideal_s, atol=2e-6, rtol=2e-4)
    state = s.clone()
    parts = []
    pos = 0
    for t in (3, 1, 5):
        p = prepare(t)
        value = torch.empty_like(v[pos : pos + t])
        p.run_packed(
            packed[pos : pos + t],
            g[pos : pos + t],
            beta[pos : pos + t],
            state=state,
            out=value,
            state_out=state,
        )
        parts.append(value)
        pos += t
    assert torch.equal(torch.cat(parts), out) and torch.equal(state, final)


def test_fp16_gdn_layer_actual_input_op_oracles(tmp_path, port):
    backend, dtype, device = port
    weights, raw, _ = gdn_fixture(tmp_path)
    r = ResidentQwenGDNLayer.from_q8(
        weights, backend, layer=0, dtype=dtype, device=device
    )
    p = r.prepare(tokens=9)
    state = r.zero_state()
    w = p.empty_workspace()
    x = (
        torch.randn(9, r.hidden_size, generator=torch.Generator().manual_seed(1452))
        * 0.2
    ).to(device=device, dtype=dtype)
    expected, h, s, checks = qualify_layer_operations(
        p, raw, x, state.history, state.recurrent, workspace=w
    )
    out = torch.empty_like(x)
    p.run(
        x,
        out=out,
        history=state.history,
        state=state.recurrent,
        history_out=state.history,
        state_out=state.recurrent,
        workspace=w,
    )
    assert (
        torch.equal(out, expected)
        and torch.equal(state.history, h)
        and torch.equal(state.recurrent, s)
    )
    assert all(item["passed"] for item in checks)
    p.close()


def test_fp16_hybrid_integration(tmp_path, port, monkeypatch):
    hybrid_check(tmp_path, port, monkeypatch)


def test_fp16_hybrid_failure_reset(tmp_path, port, monkeypatch):
    failure_check(tmp_path, port, monkeypatch)


def test_fp16_hybrid_second_chunk_failure(tmp_path, port, monkeypatch):
    chunk_failure_check(tmp_path, port, monkeypatch)


def test_fp16_hybrid_stream_ownership(tmp_path, port):
    stream_check(tmp_path, port)


def test_fp16_q8_gdn_graph_no_allocation_fallback(port, monkeypatch):
    backend, dtype, device = port
    if device == "cpu":
        pytest.skip("CUDA capture requires native port")
    q, k, v, g, beta, state = gdn_inputs(dtype=dtype, device=device)
    packed = torch.cat((q.flatten(1), k.flatten(1), v.flatten(1)), -1)
    p = backend.prepare_gdn(
        tokens=9,
        qk_heads=2,
        value_heads=4,
        key_dim=64,
        value_dim=13,
        dtype=dtype,
        device=device,
    )
    out, final = torch.empty_like(v), torch.empty_like(state)
    weight = Q8RowSplitWeight.from_dense(torch.randn(31, 65) * 0.02)
    r = backend.upload_linear(weight, dtype=dtype, device=device)
    embedding = r.prepare_embedding([1] * 9)
    head = r.prepare(tokens=9)
    x = torch.empty(9, 65, dtype=dtype, device=device)
    logits = torch.empty(9, 31, dtype=dtype, device=device)

    def run():
        p.run_packed(packed, g, beta, state=state, out=out, state_out=final)
        embedding.run(out=x)
        head.run(x, out=logits)

    run()
    torch.cuda.synchronize()
    expected = tuple(t.clone() for t in (out, final, x, logits))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()

    def forbidden(*a, **k):
        raise AssertionError(
            "native FP16 capture used allocation/view/numerical fallback"
        )

    with torch.cuda.stream(stream), monkeypatch.context() as m:
        for name in ("empty", "empty_like", "mm", "matmul", "cat"):
            m.setattr(torch, name, forbidden)
        for name in (
            "to",
            "clone",
            "copy_",
            "float",
            "view",
            "reshape",
            "repeat_interleave",
        ):
            m.setattr(torch.Tensor, name, forbidden)
        with torch.cuda.graph(graph, stream=stream):
            run()
        graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    assert all(
        torch.equal(actual, ideal)
        for actual, ideal in zip((out, final, x, logits), expected)
    )


def test_sm70_cabi_rejects_bf16_before_launch(port):
    backend, _, device = port
    if device == "cpu" or backend.architecture != 70:
        pytest.skip("SM70-only direct C ABI guard")
    assert not backend._native.venturi_dense_create(1, 1, 1, 1)
    assert b"SM70" in backend._native.venturi_blocks_error()
    # Rejection occurs on the host before any pointer is dereferenced/launched.
    assert backend._native.venturi_add_run(None, None, None, 1, 1, None) == -1
    assert b"SM70" in backend._native.venturi_blocks_error()
