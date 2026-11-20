"""Explicit-library decoder qualification; no local compilation or model download."""

import os

import pytest
import torch

from decoder_checks import qualified_decoder_composition
from test_decoder_layer import layer_weights, rope_tables
from venturi.models import AttentionDimensions
from venturi.ops import AdaBackend, CudaBackend

pytestmark = pytest.mark.cuda


@pytest.fixture(params=[False, True], ids=["dense", "q4"])
def backend(request):
    blocks, ada = (
        os.environ.get("VENTURI_BLOCKS_LIBRARY"),
        os.environ.get("VENTURI_ADA_LIBRARY"),
    )
    if not blocks or (request.param and not ada):
        pytest.skip("explicit native libraries required")
    return CudaBackend(
        blocks, q4_backend=AdaBackend(ada) if request.param else None
    ), request.param


@pytest.mark.parametrize("layout", ["split_half", "interleaved"])
def test_decoder_per_op_fp64_prefill_decode_kv_and_arena(backend, layout):
    engine, q4 = backend
    dims = AttentionDimensions(4, 2, 64, rotary_dim=32, rope_layout=layout)
    weights = layer_weights(32, 80, dims, torch.bfloat16, q4)
    resident = weights.upload(engine, dimensions=dims)
    pointers = tuple(t.data_ptr() for t in resident.storage_tensors)
    cache = resident.empty_cache(capacity=16)
    arena = resident.prepare(tokens=7).empty_workspace()
    x = torch.randn(
        12, 32, generator=torch.Generator().manual_seed(194), dtype=torch.bfloat16
    ).cuda()
    cos, sin = rope_tables(12, dims, "cuda")
    # Uninitialized/stale tail must never be read or overwritten beyond the append.
    cache._keys.fill_(47)
    cache._values.fill_(47)
    position = 0
    for count in (3, 7, 1, 1):
        plan = resident.prepare(tokens=count)
        assert tuple(t.data_ptr() for t in plan.storage_tensors) == pointers
        step = plan.at(position=position)
        a, b, c = (
            x[position : position + count],
            cos[position : position + count],
            sin[position : position + count],
        )
        prefix = (cache.keys.clone(), cache.values.clone()) if position else None
        expected, kv, checks = qualified_decoder_composition(
            step, weights, a, b, c, prefix=prefix
        )
        guard = torch.full((a.numel() + 2,), 47.0, dtype=a.dtype, device="cuda")
        out = guard[1:-1].view_as(a)
        step.run(a, b, c, out=out, workspace=arena, cache=cache)
        torch.cuda.synchronize()
        assert torch.equal(out, expected)
        assert torch.equal(cache.keys, kv[0]) and torch.equal(cache.values, kv[1])
        assert cache.length == position + count
        assert all(check["passed"] for check in checks.values())
        assert guard[0].item() == guard[-1].item() == 47
        assert bool((cache._keys[cache.length :] == 47).all())
        assert bool((cache._values[cache.length :] == 47).all())
        position += count
        plan.close()


def test_decoder_capture_commits_only_replay_and_native_has_no_fallback(
    backend, monkeypatch
):
    engine, q4 = backend
    dims = AttentionDimensions(2, 1, 64)
    resident = layer_weights(32, 80, dims, torch.bfloat16, q4).upload(
        engine, dimensions=dims
    )
    plan = resident.prepare(tokens=3)
    cache, arena = resident.empty_cache(capacity=8), plan.empty_workspace(max_tokens=7)
    x = torch.randn(3, 32, dtype=torch.bfloat16, device="cuda")
    cos, sin = rope_tables(3, dims, "cuda")
    out = torch.empty_like(x)
    step = plan.at(position=0)
    step.run(x, cos, sin, out=out, workspace=arena, cache=cache)
    expected = out.clone()
    cache.reset()
    allocated = torch.cuda.memory_allocated()

    def forbidden(*args, **kwargs):
        raise AssertionError("native execution used Tensor allocation/math fallback")

    with monkeypatch.context() as m:
        for name in ("empty", "empty_like", "mm", "matmul", "softmax", "cat"):
            m.setattr(torch, name, forbidden)
        for name in ("to", "clone", "copy_", "softmax", "__matmul__"):
            m.setattr(torch.Tensor, name, forbidden)
        graph = step.capture(x, cos, sin, out=out, workspace=arena, cache=cache)
        assert cache.length == 0
        graph.run()
        assert cache.length == 3
        with pytest.raises(ValueError, match="visible length"):
            graph.run()
        cache.truncate(0)
        graph.run()
    torch.cuda.synchronize()
    assert torch.equal(out, expected)
    # Capture may establish a graph pool, but repeated native execution must not
    # allocate additional tensor storage, decode weights, or append by copying.
    cache.reset()
    live = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        graph.run()
        cache.reset()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == torch.cuda.max_memory_allocated() == live
    assert live >= allocated
    cache.reset()
    # Pointer-changing user mutation must not replay a graph against stale storage.
    x.set_(torch.randn_like(x))
    with pytest.raises(ValueError, match="storage changed"):
        graph.run()
    plan.close()


def test_decoder_stream_ownership_and_two_request_isolation(backend):
    engine, q4 = backend
    dims = AttentionDimensions(2, 1, 64)
    resident = layer_weights(32, 80, dims, torch.bfloat16, q4).upload(
        engine, dimensions=dims
    )
    plan = resident.prepare(tokens=5)
    step = plan.at(position=0)
    inputs = [torch.randn(5, 32, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    outputs = [torch.empty_like(x) for x in inputs]
    cos, sin = rope_tables(5, dims, "cuda")
    streams = [torch.cuda.Stream() for _ in inputs]
    lanes = []
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            lanes.append((resident.empty_cache(capacity=8), plan.empty_workspace()))
    expected = []
    for x in inputs:
        cache, arena = resident.empty_cache(capacity=8), plan.empty_workspace()
        out = torch.empty_like(x)
        step.run(x, cos, sin, out=out, workspace=arena, cache=cache)
        expected.append(out)
    for stream, x, out, (cache, arena) in zip(streams, inputs, outputs, lanes):
        with torch.cuda.stream(stream):
            step.run(x, cos, sin, out=out, workspace=arena, cache=cache)
    for stream in streams:
        stream.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(outputs, expected))
    with pytest.raises(RuntimeError, match="owning CUDA stream"):
        lanes[0][0].reset()
    # One lane's workspace cannot silently be used by another stream.
    with torch.cuda.stream(streams[1]):
        lanes[1][0].reset()
        with pytest.raises(RuntimeError, match="owning CUDA stream"):
            step.run(
                inputs[1],
                cos,
                sin,
                out=outputs[1],
                workspace=lanes[0][1],
                cache=lanes[1][0],
            )
    torch.cuda.synchronize()
    plan.close()


def test_native_workspace_alias_rejected_before_any_kv_write(backend):
    engine, q4 = backend
    dims = AttentionDimensions(2, 1, 64)
    resident = layer_weights(32, 80, dims, torch.bfloat16, q4).upload(
        engine, dimensions=dims
    )
    plan = resident.prepare(tokens=1)
    step = plan.at(position=0)
    arena, cache = plan.empty_workspace(), resident.empty_cache(capacity=8)
    cache._keys.fill_(47)
    x = torch.ones(1, 32, device="cuda", dtype=torch.bfloat16)
    cos, sin = rope_tables(1, dims, "cuda")
    out = arena.activation[:32].view_as(x)
    with pytest.raises(ValueError, match="alias"):
        step.run(x, cos, sin, out=out, workspace=arena, cache=cache)
    assert cache.length == 0 and not cache.failed and bool((cache._keys == 47).all())
    # A raw torch.cuda.graph context cannot accidentally advance host KV state.
    from unittest.mock import patch

    with patch("torch.cuda.is_current_stream_capturing", return_value=True):
        with pytest.raises(RuntimeError, match="explicit capture"):
            step.run(x, cos, sin, out=torch.empty_like(x), workspace=arena, cache=cache)
    plan.close()


def test_native_prefill_chunk_and_single_token_equivalence(backend):
    engine, q4 = backend
    dims = AttentionDimensions(4, 2, 64)
    weights = layer_weights(32, 80, dims, torch.bfloat16, q4)
    resident = weights.upload(engine, dimensions=dims)
    plans = {t: resident.prepare(tokens=t) for t in (1, 3, 7, 17)}
    arena = plans[17].empty_workspace()
    x = torch.randn(
        17, 32, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(51)
    ).cuda()
    cos, sin = rope_tables(17, dims, "cuda")
    baseline = None
    for chunks in ([17], [3, 7, 7], [1] * 17):
        cache = resident.empty_cache(capacity=17)
        out, position = torch.empty_like(x), 0
        for count in chunks:
            plans[count].at(position=position).run(
                x[position : position + count],
                cos[position : position + count],
                sin[position : position + count],
                out=out[position : position + count],
                workspace=arena,
                cache=cache,
            )
            position += count
        if baseline is None:
            baseline = out.clone()
        scale = baseline.double().square().mean().sqrt()
        error = (out.double() - baseline.double()).abs()
        # Fixed BF16 cross-mode diagnostic budget on this synthetic fixture,
        # in addition to independent per-Op qualification in the preceding test.
        assert error.square().mean().sqrt() <= torch.finfo(x.dtype).eps / 2 * scale
        assert error.max() <= 2 * torch.finfo(x.dtype).eps * scale
        assert cache.length == 17
    torch.cuda.synchronize()
    for plan in plans.values():
        plan.close()


def test_mixed_dense_q4_layer_uses_correct_scratch_lifetimes(backend):
    from dataclasses import replace

    engine, q4_attention = backend
    ada = os.environ.get("VENTURI_ADA_LIBRARY")
    if not ada:
        pytest.skip("explicit Ada library required for mixed formats")
    engine = CudaBackend(engine.library, q4_backend=AdaBackend(ada))
    dims = AttentionDimensions(2, 1, 64)
    attention_weights = layer_weights(32, 80, dims, torch.bfloat16, q4_attention)
    other = layer_weights(32, 80, dims, torch.bfloat16, not q4_attention)
    weights = replace(attention_weights, mlp=other.mlp)
    resident = weights.upload(engine, dimensions=dims)
    plan = resident.prepare(tokens=3)
    arena, cache = plan.empty_workspace(), resident.empty_cache(capacity=3)
    assert arena.linear is not None
    if not q4_attention:
        assert plan.mlp.linear_workspace_bytes == 0
    x = torch.randn(3, 32, device="cuda", dtype=torch.bfloat16)
    cos, sin = rope_tables(3, dims, "cuda")
    step = plan.at(position=0)
    expected, _, _ = qualified_decoder_composition(step, weights, x, cos, sin)
    out = torch.empty_like(x)
    step.run(x, cos, sin, out=out, workspace=arena, cache=cache)
    assert torch.equal(out, expected)
    # A Q4-only sibling can reuse the same lane even though it needs no scratch.
    q4resident = layer_weights(32, 80, dims, torch.bfloat16, True).upload(
        engine, dimensions=dims
    )
    q4plan = q4resident.prepare(tokens=3)
    q4plan.at(position=0).run(
        x, cos, sin, out=out, workspace=arena, cache=q4resident.empty_cache(capacity=3)
    )
    torch.cuda.synchronize()
    plan.close()
    q4plan.close()
