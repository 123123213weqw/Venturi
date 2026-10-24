"""Full Qwen gated-attention layer with explicit cache lifecycle and state tests."""

from dataclasses import replace
import os

import pytest
import torch

from attention_reference import attention_reference
from qwen_gdn_reference import q8_source_oracle
from test_qwen_source import fixture_tensors, source_config, write_source
from test_qwen_attention_ops import oracle as qk_oracle
from venturi.models import QwenQ8Weights, ResidentQwenAttentionLayer
from venturi.ninfer import NInferArtifact
from venturi.ops import CudaBackend, TorchBackend
from venturi.qwen_export import export_qwen_q8


def fixture_weights(root):
    cfg, raw = source_config(), fixture_tensors()
    c = cfg["text_config"]
    c["head_dim"] = 64
    c["rope_parameters"].update(mrope_interleaved=True, mrope_section=[3, 3, 2])
    p = "model.language_model.layers.1.self_attn."
    for key, shape in {
        "q_proj.weight": (256, 32),
        "k_proj.weight": (64, 32),
        "v_proj.weight": (64, 32),
        "o_proj.weight": (32, 128),
        "q_norm.weight": (64,),
        "k_norm.weight": (64,),
    }.items():
        raw[p + key] = torch.empty(shape)
    gen = torch.Generator().manual_seed(912)
    for name, tensor in list(raw.items()):
        if not name.startswith("model.language_model.layers.1."):
            continue
        scale = 0.15 if tensor.ndim == 2 else 0.05
        raw[name] = (
            torch.randn(tensor.shape, generator=gen, dtype=torch.float64) * scale
        ).bfloat16()
    source = write_source(root / "source", raw=cfg, tensors=raw)
    artifact = export_qwen_q8(source, root / "q8.ninfer")
    return QwenQ8Weights.from_ninfer(NInferArtifact(artifact["output"])), raw, c


@pytest.fixture(params=["fp64", "cpu-bf16", "cuda"])
def engine(request):
    if request.param != "cuda":
        return (
            TorchBackend(),
            torch.float64 if request.param == "fp64" else torch.bfloat16,
            "cpu",
        )
    lib = os.environ.get("VENTURI_BLOCKS_LIBRARY")
    if not lib:
        pytest.skip("explicit native library required")
    return CudaBackend(lib), torch.bfloat16, "cuda:0"


def input_tensor(r, t):
    return (
        torch.randn(
            t, 32, generator=torch.Generator().manual_seed(807), dtype=torch.float64
        )
        * 0.4
    ).to(dtype=r.dtype, device=r.device)


def check(a, b, *, floor=2e-5):
    value, ideal = a.cpu().double(), b.cpu().double()
    err = (value - ideal.to(a.dtype).double()).abs()
    if a.dtype == torch.float64:
        torch.testing.assert_close(value, ideal, atol=2e-12, rtol=2e-11)
    else:
        assert bool(
            (err <= 2 * torch.finfo(a.dtype).eps * ideal.abs() + floor).all()
        ), dict(
            error=err.max().item(),
            indices=(err > 2 * torch.finfo(a.dtype).eps * ideal.abs() + floor)
            .nonzero()[:4]
            .tolist(),
        )


def manual_qualified(step, raw, x, cos, sin, w):
    """Independent FP64 each-Op oracle on actual inputs, then native composition."""
    p, r, b = step.layer, step.layer.resident, w.buffers
    dtype = r.dtype
    prefix = "model.language_model.layers.1."

    def norm(value, key):
        v = value.cpu().double()
        return (
            v
            * torch.rsqrt(v.square().mean(-1, keepdim=True) + 1e-6)
            * (1 + raw[prefix + key].double())
        )

    def linear(op, source, target, key, scratch):
        op.run(source, out=target, workspace=scratch)
        check(
            target,
            source.cpu().double() @ q8_source_oracle(raw[prefix + key]).T,
            floor=2e-4,
        )

    p.norm.run(x, out=b["normalized"])
    check(b["normalized"], norm(x, "input_layernorm.weight"))
    for op, target, name, scratch in zip(
        p.projections[:3],
        (b["qg"], b["key"], step.value_tail),
        ("q", "k", "v"),
        w.projection_scratch[:3],
    ):
        linear(
            op, b["normalized"], target, "self_attn." + name + "_proj.weight", scratch
        )
    p.qk_norm.run(
        b["qg"], w.key_heads, cos, sin, query_out=w.query_heads, key_out=step.key_tail
    )
    d = r.dimensions
    q = (
        b["qg"]
        .cpu()
        .reshape(p.tokens, d.query_heads, 2 * d.head_dim)[..., : d.head_dim]
    )
    check(
        w.query_heads,
        qk_oracle(q, r.qk_norm.query_gamma, cos, sin, d.rotary_dim, dtype),
    )
    check(
        step.key_tail,
        qk_oracle(w.key_heads, r.qk_norm.key_gamma, cos, sin, d.rotary_dim, dtype),
    )
    step.attention.run(
        w.query_heads, step.key_prefix, step.value_prefix, out=w.attended_heads
    )
    check(
        w.attended_heads,
        attention_reference(
            w.query_heads.cpu(),
            step.key_prefix.cpu(),
            step.value_prefix.cpu(),
            d.head_dim**-0.5,
        ),
    )
    attended = w.attended_heads.cpu().double().clone()
    gates = (
        b["qg"]
        .cpu()
        .reshape(p.tokens, d.query_heads, 2 * d.head_dim)[..., d.head_dim :]
        .double()
        .sigmoid()
        .to(dtype)
        .double()
    )
    p.gate.run(w.attended_heads, b["qg"], out=w.attended_heads)
    check(w.attended_heads, attended * gates)
    linear(
        p.output,
        b["attended"],
        b["projected"],
        "self_attn.o_proj.weight",
        w.projection_scratch[3],
    )
    p.add.run(x, b["projected"], out=b["residual"])
    check(b["residual"], x.cpu().double() + b["projected"].cpu().double())
    p.mlp.norm.run(b["residual"], out=w.mlp.hidden)
    check(w.mlp.hidden, norm(b["residual"], "post_attention_layernorm.weight"))
    for op, target, name in (
        (p.mlp.gate, w.mlp.gate, "gate"),
        (p.mlp.up, w.mlp.up, "up"),
    ):
        linear(
            op,
            w.mlp.hidden,
            target,
            "mlp." + name + "_proj.weight",
            w.mlp.linear if op.workspace_bytes else None,
        )
    gate, up = w.mlp.gate.cpu().double().clone(), w.mlp.up.cpu().double().clone()
    p.mlp.swiglu.run(w.mlp.gate, w.mlp.up, out=w.mlp.gate)
    check(w.mlp.gate, gate * gate.sigmoid() * up)
    linear(
        p.mlp.down,
        w.mlp.gate,
        w.mlp.hidden,
        "mlp.down_proj.weight",
        w.mlp.linear if p.mlp.down.workspace_bytes else None,
    )
    out = torch.empty_like(x)
    p.mlp.add.run(b["residual"], w.mlp.hidden, out=out)
    check(out, b["residual"].cpu().double() + w.mlp.hidden.cpu().double())
    return out


@pytest.mark.parametrize("tokens", [1, 7])
def test_layer_op_math_chunking_and_dense_cache(tmp_path, engine, tokens):
    backend, dtype, device = engine
    weights, raw, _ = fixture_weights(tmp_path)
    r = ResidentQwenAttentionLayer.from_q8(
        weights, backend, layer=1, dtype=dtype, device=device
    )
    cache = r.empty_cache(capacity=16)
    plan = r.prepare(tokens=tokens)
    w = plan.empty_workspace()
    x = input_tensor(r, tokens)
    co, si = r.rope.tables(
        backend, position=0, tokens=tokens, dtype=dtype, device=device
    )
    step = plan.at(position=0, cache=cache)
    expected = manual_qualified(step, raw, x, co, si, w)
    out = torch.empty_like(x)
    step.run(x, co, si, out=out, workspace=w)
    assert torch.equal(out, expected) and cache.length == tokens and not cache.failed
    keys, values = cache.keys.clone(), cache.values.clone()
    with pytest.raises(ValueError, match="visible length"):
        step.run(x, co, si, out=out, workspace=w)
    assert cache.length == tokens
    cache.reset()
    parts = []
    position = 0
    keep = []
    for count in (1,) if tokens == 1 else (2, 1, 4):
        p = r.prepare(tokens=count)
        ws = p.empty_workspace()
        s = p.at(position=position, cache=cache)
        c, t = r.rope.tables(
            backend, position=position, tokens=count, dtype=dtype, device=device
        )
        v = x[position : position + count]
        target = torch.empty_like(v)
        expected = manual_qualified(s, raw, v, c, t, ws)
        s.run(v, c, t, out=target, workspace=ws)
        assert torch.equal(target, expected)
        parts.append(target)
        keep.append((p, ws))
        position += count
    if dtype == torch.float64 or r.device.type == "cuda":
        tol = 2e-12 if dtype == torch.float64 else 0
        torch.testing.assert_close(torch.cat(parts), out, atol=tol, rtol=tol)
        torch.testing.assert_close(cache.keys, keys, atol=tol, rtol=tol)
        torch.testing.assert_close(cache.values, values, atol=tol, rtol=tol)
    assert cache.length == tokens


def test_preflight_owner_alias_view_and_closed_plan_before_cache_mutation(
    tmp_path, engine
):
    backend, dtype, device = engine
    weights, _, _ = fixture_weights(tmp_path)
    r = ResidentQwenAttentionLayer.from_q8(
        weights, backend, layer=1, dtype=dtype, device=device
    )
    p = r.prepare(tokens=3)
    cache = r.empty_cache(capacity=8)
    w = p.empty_workspace()
    x = input_tensor(r, 3)
    out = torch.full_like(x, 17)
    cache._keys.fill_(19)
    cache._values.fill_(23)
    w.activation.fill_(29)
    before = tuple(t.clone() for t in (*cache.storage_tensors, w.activation, out))
    c, s = r.rope.tables(backend, position=0, tokens=3, dtype=dtype, device=device)
    step = p.at(position=0, cache=cache)
    for bad in (
        replace(w, query_heads=w.query_heads.clone()),
        replace(w, mlp=replace(w.mlp, gate=w.mlp.up)),
        replace(w, buffers={**w.buffers, "key": w.buffers["key"].clone()}),
    ):
        with pytest.raises(ValueError):
            step.run(x, c, s, out=out, workspace=bad)
        assert cache.length == 0 and not cache.failed
        assert all(
            torch.equal(a, b)
            for a, b in zip((*cache.storage_tensors, w.activation, out), before)
        )
    with pytest.raises(ValueError, match="alias"):
        step.run(x, c, s, out=x, workspace=w)
    other = ResidentQwenAttentionLayer.from_q8(
        weights, backend, layer=1, dtype=dtype, device=device
    )
    with pytest.raises(ValueError, match="different"):
        p.at(position=0, cache=other.empty_cache(capacity=8))
    p.close()
    with pytest.raises(RuntimeError, match="closed"):
        step.run(x, c, s, out=out, workspace=w)
    assert cache.length == 0 and all(
        torch.equal(a, b)
        for a, b in zip((*cache.storage_tensors, w.activation, out), before)
    )


@pytest.mark.cuda
def test_layer_capture_no_allocations_views_fallback_and_commit(tmp_path, monkeypatch):
    lib = os.environ.get("VENTURI_BLOCKS_LIBRARY")
    if not lib:
        pytest.skip("explicit native library required")
    weights, _, _ = fixture_weights(tmp_path)
    backend = CudaBackend(lib)
    r = ResidentQwenAttentionLayer.from_q8(weights, backend, layer=1)
    p = r.prepare(tokens=3)
    cache = r.empty_cache(capacity=8)
    w = p.empty_workspace()
    x = input_tensor(r, 3)
    out = torch.empty_like(x)
    co, si = r.rope.tables(
        backend, position=0, tokens=3, dtype=r.dtype, device=r.device
    )
    step = p.at(position=0, cache=cache)
    step.run(x, co, si, out=out, workspace=w)
    expected = out.clone()
    cache.reset()

    def forbidden(*a, **k):
        raise AssertionError(
            "Attention execution used Tensor allocation/view/numerical fallback"
        )

    with monkeypatch.context() as m:
        for name in ("empty", "empty_like", "cat", "matmul", "sigmoid"):
            m.setattr(torch, name, forbidden)
        for name in (
            "to",
            "clone",
            "copy_",
            "view",
            "reshape",
            "contiguous",
            "exp",
            "sigmoid",
        ):
            m.setattr(torch.Tensor, name, forbidden)
        captured = step.capture(x, co, si, out=out, workspace=w, warmup=1)
        assert cache.length == 0
        captured.run()
    assert cache.length == 3 and torch.equal(out, expected)
    with pytest.raises(ValueError, match="visible length"):
        captured.run()
    cache.reset()
    captured.run()
    assert torch.equal(out, expected)


def test_submission_failure_poisons_cache_without_advancing_visibility(
    tmp_path, engine, monkeypatch
):
    backend, dtype, device = engine
    weights, _, _ = fixture_weights(tmp_path)
    r = ResidentQwenAttentionLayer.from_q8(
        weights, backend, layer=1, dtype=dtype, device=device
    )
    p = r.prepare(tokens=3)
    cache = r.empty_cache(capacity=8)
    w = p.empty_workspace()
    x = input_tensor(r, 3)
    out = torch.empty_like(x)
    c, s = r.rope.tables(backend, position=0, tokens=3, dtype=dtype, device=device)
    step = p.at(position=0, cache=cache)
    original = type(p.gate).run

    def fail(self, *args, **kwargs):
        raise RuntimeError("Injected failure after KV writes")

    monkeypatch.setattr(type(p.gate), "run", fail)
    with pytest.raises(RuntimeError, match="Injected"):
        step.run(x, c, s, out=out, workspace=w)
    assert cache.failed and cache.length == 0
    with pytest.raises(RuntimeError, match="failed"):
        step.run(x, c, s, out=out, workspace=w)
    with pytest.raises(RuntimeError, match="failed"):
        cache.truncate(0)
    monkeypatch.setattr(type(p.gate), "run", original)
    cache.reset()
    step.run(x, c, s, out=out, workspace=w)
    assert cache.length == 3 and not cache.failed


@pytest.mark.cuda
def test_independent_request_caches_stream_ownership_and_graph_views(tmp_path):
    lib = os.environ.get("VENTURI_BLOCKS_LIBRARY")
    if not lib:
        pytest.skip("explicit native library required")
    weights, _, _ = fixture_weights(tmp_path)
    backend = CudaBackend(lib)
    r = ResidentQwenAttentionLayer.from_q8(weights, backend, layer=1)
    p = r.prepare(tokens=3)
    x = input_tensor(r, 3)
    requests = []
    expected = []
    streams = [torch.cuda.Stream() for _ in range(2)]
    for index, stream in enumerate(streams):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            cache = r.empty_cache(capacity=8)
            w = p.empty_workspace()
            step = p.at(position=0, cache=cache)
            value = x + index * 0.25
            target = torch.empty_like(x)
            co, si = r.rope.tables(
                backend, position=0, tokens=3, dtype=r.dtype, device=r.device
            )
            step.run(value, co, si, out=target, workspace=w)
            expected.append(target.clone())
            cache.reset()
            requests.append((step, cache, w, value, target, co, si))
    for stream, (step, cache, w, value, target, co, si) in zip(streams, requests):
        with torch.cuda.stream(stream):
            step.run(value, co, si, out=target, workspace=w)
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    assert all(
        torch.equal(req[4], reference) for req, reference in zip(requests, expected)
    )
    assert all(req[1].length == 3 for req in requests)
    step, cache, w, value, target, co, si = requests[0]
    with pytest.raises(RuntimeError, match="owning"):
        step.run(value, co, si, out=target, workspace=w)
