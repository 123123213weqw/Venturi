"""RoPE conventions, causal alignment, GQA sharing, graph and buffer contracts."""

import math
import os

import pytest
import torch

from attention_reference import attention_reference, rope_reference
from venturi.ops import CudaBackend, TorchBackend


@pytest.fixture(params=["reference", "cuda"])
def backend(request):
    if request.param == "reference":
        return TorchBackend(), "cpu"
    library = os.environ.get("VENTURI_BLOCKS_LIBRARY")
    if not library:
        pytest.skip("explicit native library required")
    return CudaBackend(library), "cuda:0"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("layout", ["split_half", "interleaved"])
@pytest.mark.parametrize("tokens,heads,dim,rotary", [(1, 1, 8, 8), (7, 3, 65, 32)])
def test_rope_oracle_tail_guards_inplace(
    backend, dtype, layout, tokens, heads, dim, rotary
):
    engine, device = backend
    gen = torch.Generator().manual_seed(847)
    x = torch.randn(tokens, heads, dim, generator=gen, dtype=dtype).to(device)
    angles = torch.randn(tokens, rotary // 2, generator=gen, dtype=torch.float64) * 100
    cos, sin = angles.cos().float().to(device), angles.sin().float().to(device)
    expected = rope_reference(x, cos, sin, rotary, layout)
    plan = engine.prepare_rope(
        tokens=tokens,
        heads=heads,
        head_dim=dim,
        rotary_dim=rotary,
        layout=layout,
        dtype=dtype,
        device=device,
    )
    guard = torch.full((x.numel() + 2,), 47, dtype=dtype, device=device)
    out = guard[1:-1].view_as(x)
    assert plan.run(x, cos, sin, out=out) is out
    tolerance = 2 * torch.finfo(dtype).eps
    torch.testing.assert_close(out.cpu(), expected, rtol=tolerance, atol=2e-6)
    assert guard[0].item() == guard[-1].item() == 47
    assert torch.equal(out[..., rotary:], x[..., rotary:])
    plan.run(x, cos, sin, out=x)
    torch.testing.assert_close(x.cpu(), expected, rtol=tolerance, atol=2e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "tokens,kv_tokens,hq,hkv,dim",
    [
        (1, 1, 1, 1, 64),
        (1, 129, 6, 2, 256),
        (7, 7, 4, 4, 128),
        (7, 19, 6, 2, 64),
        (17, 129, 4, 1, 128),
    ],
)
def test_causal_gqa_independent_fp64(backend, dtype, tokens, kv_tokens, hq, hkv, dim):
    engine, device = backend
    gen = torch.Generator().manual_seed(723)
    q = torch.randn(tokens, hq, dim, generator=gen, dtype=dtype).to(device)
    k = torch.randn(kv_tokens, hkv, dim, generator=gen, dtype=dtype).to(device)
    v = torch.randn(kv_tokens, hkv, dim, generator=gen, dtype=dtype).to(device)
    plan = engine.prepare_attention(
        tokens=tokens,
        kv_tokens=kv_tokens,
        query_heads=hq,
        kv_heads=hkv,
        head_dim=dim,
        dtype=dtype,
        device=device,
    )
    expected = attention_reference(q, k, v, 1 / math.sqrt(dim))
    guard = torch.full((q.numel() + 2,), 47, dtype=dtype, device=device)
    out = guard[1:-1].view_as(q)
    plan.run(q, k, v, out=out)
    torch.testing.assert_close(
        out.cpu(), expected, rtol=2 * torch.finfo(dtype).eps, atol=2e-5
    )
    assert guard[0].item() == guard[-1].item() == 47
    # Mutating keys/values invisible to row zero must not affect that row.
    first = out[0].clone()
    if tokens > 1:
        k[kv_tokens - tokens + 1 :].fill_(20)
        v[kv_tokens - tokens + 1 :].fill_(-30)
        plan.run(q, k, v, out=out)
        assert torch.equal(first, out[0])
    # Uniform logits must average visible V, including the prefix.
    q.zero_()
    expected = attention_reference(q, k, v, plan.scale)
    plan.run(q, k, v, out=out)
    torch.testing.assert_close(
        out.cpu(), expected, rtol=2 * torch.finfo(dtype).eps, atol=2e-5
    )


def test_attention_large_logits_are_stable_and_alias_rejected(backend):
    engine, device = backend
    q = torch.full((3, 4, 64), 90.0, device=device)
    k = torch.full((9, 2, 64), -90.0, device=device)
    k[2] = 90.0
    v = torch.randn(9, 2, 64, device=device)
    plan = engine.prepare_attention(
        tokens=3,
        kv_tokens=9,
        query_heads=4,
        kv_heads=2,
        head_dim=64,
        dtype=torch.float32,
        device=device,
    )
    out = torch.empty_like(q)
    expected = attention_reference(q, k, v, plan.scale)
    plan.run(q, k, v, out=out)
    torch.testing.assert_close(out.cpu(), expected, rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError, match="alias"):
        plan.run(q, k, v, out=q)
    with pytest.raises(ValueError, match="shape"):
        plan.run(q, k.half(), v, out=out)
    with pytest.raises(ValueError, match="inference"):
        plan.run(q.requires_grad_(), k, v, out=out)


@pytest.mark.parametrize(
    "changes",
    [
        {"tokens": 0},
        {"kv_tokens": 1},
        {"query_heads": 3},
        {"head_dim": 65},
        {"scale": 0},
        {"scale": float("inf")},
        {"scale": 1e-50},
    ],
)
def test_attention_invalid_configuration(changes):
    config = dict(
        tokens=2,
        kv_tokens=2,
        query_heads=4,
        kv_heads=2,
        head_dim=64,
        dtype=torch.float32,
        device="cpu",
    )
    config.update(changes)
    with pytest.raises(ValueError):
        TorchBackend().prepare_attention(**config)


@pytest.mark.parametrize(
    "changes",
    [
        {"rotary_dim": 3},
        {"rotary_dim": 18},
        {"layout": "unknown"},
        {"heads": True},
    ],
)
def test_rope_invalid_configuration(changes):
    config = dict(tokens=2, heads=4, head_dim=16, dtype=torch.float32, device="cpu")
    config.update(changes)
    with pytest.raises(ValueError):
        TorchBackend().prepare_rope(**config)


def test_rope_wrong_tables_and_partial_alias_rejected():
    plan = TorchBackend().prepare_rope(
        tokens=2, heads=3, head_dim=16, dtype=torch.float32, device="cpu"
    )
    raw = torch.empty(97)
    x = raw[:-1].view(2, 3, 16)
    cos, sin = torch.ones(2, 8), torch.zeros(2, 8)
    with pytest.raises(ValueError, match="alias"):
        plan.run(x, cos, sin, out=raw[1:].view_as(x))
    with pytest.raises(ValueError, match="shape"):
        plan.run(x, cos.double(), sin, out=torch.empty_like(x))


@pytest.mark.cuda
def test_attention_rope_native_graph_stream_without_tensor_fallback(monkeypatch):
    library = os.environ.get("VENTURI_BLOCKS_LIBRARY")
    if not library:
        pytest.skip("explicit native library required")
    engine, device = CudaBackend(library), "cuda:0"
    rope = engine.prepare_rope(
        tokens=7, heads=6, head_dim=64, dtype=torch.bfloat16, device=device
    )
    attn = engine.prepare_attention(
        tokens=7,
        kv_tokens=19,
        query_heads=6,
        kv_heads=2,
        head_dim=64,
        dtype=torch.bfloat16,
        device=device,
    )
    x = torch.randn(7, 6, 64, device=device, dtype=torch.bfloat16)
    q = torch.empty_like(x)
    k = torch.randn(19, 2, 64, device=device, dtype=x.dtype)
    v = torch.randn_like(k)
    out = torch.empty_like(x)
    angles = torch.randn(7, 32, device=device)
    cos, sin = angles.cos(), angles.sin()
    qref = rope_reference(x, cos, sin, 64, "split_half")
    expected = attention_reference(qref, k, v, attn.scale)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def run():
        rope.run(x, cos, sin, out=q)
        attn.run(q, k, v, out=out)

    with torch.cuda.stream(stream):
        run()
    stream.synchronize()

    def forbidden(*args, **kwargs):
        raise AssertionError("native run used Tensor math or allocation")

    with monkeypatch.context() as m:
        for name in ("empty", "empty_like", "matmul", "mm", "softmax"):
            m.setattr(torch, name, forbidden)
        for name in ("to", "clone", "copy_", "softmax", "__matmul__"):
            m.setattr(torch.Tensor, name, forbidden)
        with torch.cuda.stream(stream):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                run()
            graph.replay()
        stream.synchronize()
    torch.testing.assert_close(out.cpu(), expected, rtol=0.02, atol=0.002)


def test_causal_chunk_and_token_at_a_time_equivalence(backend):
    engine, device = backend
    dtype = torch.float32
    q = torch.randn(7, 6, 64, device=device, dtype=dtype)
    k = torch.randn(19, 2, 64, device=device, dtype=dtype)
    v = torch.randn_like(k)
    whole = engine.prepare_attention(
        tokens=7,
        kv_tokens=19,
        query_heads=6,
        kv_heads=2,
        head_dim=64,
        dtype=dtype,
        device=device,
    )
    out = torch.empty_like(q)
    whole.run(q, k, v, out=out)
    for token in range(7):
        single = engine.prepare_attention(
            tokens=1,
            kv_tokens=13 + token,
            query_heads=6,
            kv_heads=2,
            head_dim=64,
            dtype=dtype,
            device=device,
        )
        selected = torch.empty_like(q[token : token + 1])
        single.run(q[token : token + 1], k[: 13 + token], v[: 13 + token], out=selected)
        torch.testing.assert_close(
            selected, out[token : token + 1], rtol=2e-6, atol=2e-6
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_attention_long_prefix(backend, dtype):
    engine, device = backend
    q = torch.randn(1, 2, 64, device=device, dtype=dtype)
    k = torch.randn(4096, 1, 64, device=device, dtype=dtype)
    v = torch.randn_like(k)
    plan = engine.prepare_attention(
        tokens=1,
        kv_tokens=4096,
        query_heads=2,
        kv_heads=1,
        head_dim=64,
        dtype=dtype,
        device=device,
    )
    out = torch.empty_like(q)
    expected = attention_reference(q, k, v, plan.scale)
    plan.run(q, k, v, out=out)
    torch.testing.assert_close(
        out.cpu(), expected, rtol=2 * torch.finfo(dtype).eps, atol=2e-5
    )
