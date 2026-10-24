"""Independent staged QK/partial-RoPE/gate tests, including packed head order."""

import os

import pytest
import torch

from venturi.models.qwen_rope import QwenTextRoPE
from venturi.ops import CudaBackend, TorchBackend


@pytest.fixture(params=["fp64", "cpu-bf16", "float16", "bfloat16", "float32"])
def engine(request):
    if request.param in ("fp64", "cpu-bf16"):
        return (
            TorchBackend(),
            torch.float64 if request.param == "fp64" else torch.bfloat16,
            "cpu",
        )
    lib = os.environ.get("VENTURI_BLOCKS_LIBRARY")
    if not lib:
        pytest.skip("explicit native library required")
    return CudaBackend(lib), getattr(torch, request.param), "cuda:0"


def sample(shape, dtype, device, seed):
    return (
        torch.randn(
            shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64
        )
        * 0.4
    ).to(dtype=dtype, device=device)


def oracle(raw, gamma, cos, sin, r, dtype):
    x = raw.cpu().double()
    h = r // 2
    norm = (
        (
            x
            * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
            * gamma.cpu().double()
        )
        .to(dtype)
        .double()
    )
    c, s = (t.cpu().to(dtype).double()[:, None, :] for t in (cos, sin))
    a, b = norm[..., :h], norm[..., h:r]
    out = norm.clone()
    out[..., :h] = (
        ((a * c).to(dtype).double() - (b * s).to(dtype).double()).to(dtype).double()
    )
    out[..., h:r] = (
        ((b * c).to(dtype).double() + (a * s).to(dtype).double()).to(dtype).double()
    )
    return out


def check(actual, ideal):
    d = actual.dtype
    err = (actual.cpu().double() - ideal.to(d).double()).abs()
    if d == torch.float64:
        torch.testing.assert_close(actual.cpu().double(), ideal, atol=2e-12, rtol=2e-11)
    else:
        assert bool((err <= 2 * torch.finfo(d).eps * ideal.abs() + 2e-5).all()), (
            err.max().item()
        )


@pytest.mark.parametrize("d", [64, 128, 256])
@pytest.mark.parametrize("fraction", [0.25, 1.0])
def test_qk_partial_rope_head_packing_and_activation_boundaries(engine, d, fraction):
    backend, dtype, device = engine
    tokens, hq, hkv = 7, 6, 2
    r = int(d * fraction)
    qg = sample((tokens, 2 * hq * d), dtype, device, 2)
    key = sample((tokens, hkv, d), dtype, device, 3)
    qraw, kraw = (
        sample((d,), torch.float64, "cpu", 4),
        sample((d,), torch.float64, "cpu", 5),
    )
    resident = backend.upload_qwen_qk_norm(qraw, kraw, dtype=dtype, device=device)
    plan = resident.prepare(tokens=tokens, query_heads=hq, kv_heads=hkv, rotary_dim=r)
    cos, sin = (
        sample((tokens, r // 2), torch.float32, device, seed) for seed in (6, 7)
    )
    qo, ko = (
        torch.empty(tokens, hq, d, dtype=dtype, device=device),
        torch.empty_like(key),
    )
    plan.run(qg, key, cos, sin, query_out=qo, key_out=ko)
    # Query values are the first D channels of EACH 2D head, not a flat half.
    qn, kn = torch.empty_like(qo), torch.empty_like(ko)
    plan.run_normalized(qg, key, query_out=qn, key_out=kn)
    for raw, gamma, actual in (
        (qg.cpu().reshape(tokens, hq, 2 * d)[..., :d], resident.query_gamma, qn),
        (key, resident.key_gamma, kn),
    ):
        value = raw.cpu().double()
        check(
            actual,
            value
            * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
            * gamma.cpu().double(),
        )
    # A whole FP64 norm+rounded-RoPE cascade can amplify a one-ULP norm
    # difference through cancellation. Qualify each stage on actual inputs,
    # and require exact parity between fused and staged rounded rotations.
    half = r // 2
    for norm, actual in ((qn, qo), (kn, ko)):
        value = norm.cpu().double()
        c, s = (t.cpu().to(dtype).double()[:, None, :] for t in (cos, sin))
        expected = value.clone()
        expected[..., :half] = (
            (
                (value[..., :half] * c).to(dtype).double()
                - (value[..., half:r] * s).to(dtype).double()
            )
            .to(dtype)
            .double()
        )
        expected[..., half:r] = (
            (
                (value[..., half:r] * c).to(dtype).double()
                + (value[..., :half] * s).to(dtype).double()
            )
            .to(dtype)
            .double()
        )
        check(actual, expected)
        if dtype in (torch.float16, torch.bfloat16):
            assert torch.equal(actual.cpu(), expected.to(dtype))
    gamma_before = [t.clone() for t in resident.storage_tensors]
    gate = backend.prepare_qwen_output_gate(
        tokens=tokens, heads=hq, head_dim=d, dtype=dtype, device=device
    )
    expected_gate = (
        qg.cpu()
        .reshape(tokens, hq, 2 * d)[..., d:]
        .double()
        .sigmoid()
        .to(dtype)
        .double()
    )
    actual_input = qo.cpu().double().clone()
    gate.run(qo, qg, out=qo)
    check(qo, actual_input * expected_gate)
    assert all(
        torch.equal(a, b) for a, b in zip(gamma_before, resident.storage_tensors)
    )


def test_qwen_ops_reject_aliases_and_invalid_geometry(engine):
    backend, dtype, device = engine
    d = 64
    resident = backend.upload_qwen_qk_norm(
        torch.zeros(d), torch.zeros(d), dtype=dtype, device=device
    )
    p = resident.prepare(tokens=1, query_heads=2, kv_heads=1, rotary_dim=16)
    qg = sample((1, 256), dtype, device, 1)
    key = sample((1, 1, 64), dtype, device, 2)
    co = torch.ones(1, 8, dtype=torch.float32, device=device)
    si = torch.zeros_like(co)
    qo = torch.empty(1, 2, 64, dtype=dtype, device=device)
    for qout, kout in ((qg[:, :128].reshape_as(qo), torch.empty_like(key)), (qo, key)):
        with pytest.raises(ValueError, match="alias"):
            p.run(qg, key, co, si, query_out=qout, key_out=kout)
    for config in (
        dict(query_heads=3, kv_heads=2, rotary_dim=16),
        dict(query_heads=2, kv_heads=1, rotary_dim=65),
        dict(query_heads=True, kv_heads=1, rotary_dim=16),
    ):
        with pytest.raises(ValueError):
            resident.prepare(tokens=1, **config)
    gate = backend.prepare_qwen_output_gate(
        tokens=1, heads=2, head_dim=64, dtype=dtype, device=device
    )
    with pytest.raises(ValueError, match="alias"):
        gate.run(qo, qg, out=qg[:, 128:].reshape_as(qo))


@pytest.mark.cuda
def test_qwen_ops_graph_has_no_numerical_fallback_or_tensor_views(monkeypatch):
    lib = os.environ.get("VENTURI_BLOCKS_LIBRARY")
    if not lib:
        pytest.skip("explicit native library required")
    backend, dtype, device = CudaBackend(lib), torch.bfloat16, "cuda:0"
    qg, key = (
        sample((9, 6 * 256 * 2), dtype, device, 1),
        sample((9, 2, 256), dtype, device, 2),
    )
    res = backend.upload_qwen_qk_norm(
        torch.zeros(256), torch.ones(256) * 0.1, dtype=dtype, device=device
    )
    p = res.prepare(tokens=9, query_heads=6, kv_heads=2, rotary_dim=64)
    gate = backend.prepare_qwen_output_gate(
        tokens=9, heads=6, head_dim=256, dtype=dtype, device=device
    )
    qo = torch.empty(9, 6, 256, dtype=dtype, device=device)
    ko = torch.empty_like(key)
    cos, sin = (sample((9, 32), torch.float32, device, seed) for seed in (3, 4))

    def run():
        p.run(qg, key, cos, sin, query_out=qo, key_out=ko)
        gate.run(qo, qg, out=qo)

    run()
    expected = (qo.clone(), ko.clone())
    stream, graph = torch.cuda.Stream(), torch.cuda.CUDAGraph()
    stream.wait_stream(torch.cuda.current_stream())

    def forbidden(*a, **k):
        raise AssertionError(
            "Native Qwen attention used a Tensor fallback/allocation/view"
        )

    with torch.cuda.stream(stream):
        with monkeypatch.context() as m:
            for name in ("empty", "empty_like", "cat", "matmul", "sigmoid"):
                m.setattr(torch, name, forbidden)
            for name in (
                "to",
                "clone",
                "copy_",
                "reshape",
                "view",
                "contiguous",
                "sigmoid",
                "rsqrt",
            ):
                m.setattr(torch.Tensor, name, forbidden)
            with torch.cuda.graph(graph, stream=stream):
                run()
            graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(qo, expected[0]) and torch.equal(ko, expected[1])


def rope_config():
    return dict(
        head_dim=256,
        max_position_embeddings=262144,
        rope_parameters=dict(
            rope_type="default",
            rope_theta=10000000,
            partial_rotary_factor=0.25,
            mrope_interleaved=True,
            mrope_section=[11, 11, 10],
        ),
    )


@pytest.mark.parametrize("start", [0, 7, 262140])
def test_text_equal_axis_rope_matches_independent_fp32_tables(start):
    cfg = rope_config()
    r = QwenTextRoPE.from_config(cfg)
    cos, sin = r.tables(
        TorchBackend(), position=start, tokens=4, dtype=torch.bfloat16, device="cpu"
    )
    inv = 1.0 / (10000000 ** (torch.arange(0, 64, 2).float() / 64))
    phase = torch.arange(start, start + 4).float()[:, None] * inv
    assert torch.equal(cos, phase.cos().bfloat16().float()) and torch.equal(
        sin, phase.sin().bfloat16().float()
    )
    with pytest.raises(ValueError):
        r.tables(
            TorchBackend(),
            position=262143,
            tokens=2,
            dtype=torch.bfloat16,
            device="cpu",
        )


def test_rope_rejects_nondefault_or_nontext_conventions():
    for key, value in (
        ("rope_type", "dynamic"),
        ("partial_rotary_factor", 0.01),
        ("mrope_section", [1, 2, 3]),
    ):
        cfg = rope_config()
        cfg["rope_parameters"][key] = value
        with pytest.raises(ValueError):
            QwenTextRoPE.from_config(cfg)


def test_qwen_eager_source_norm_rotary_and_gate_conventions():
    pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5RMSNorm,
        Qwen3_5TextRotaryEmbedding,
        apply_rotary_pos_emb,
    )

    cfg = rope_config()
    cfg.update(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=6,
        num_key_value_heads=2,
        num_hidden_layers=1,
        layer_types=["full_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=4,
    )
    hf = Qwen3_5TextConfig(**cfg)
    qg, key = (
        sample((7, 6 * 512), torch.bfloat16, "cpu", 19),
        sample((7, 2, 256), torch.bfloat16, "cpu", 20),
    )
    rawq, rawk = (
        sample((256,), torch.bfloat16, "cpu", 21),
        sample((256,), torch.bfloat16, "cpu", 22),
    )
    backend = TorchBackend()
    resident = backend.upload_qwen_qk_norm(
        rawq, rawk, dtype=torch.bfloat16, device="cpu"
    )
    tables = QwenTextRoPE.from_config(cfg)
    co, si = tables.tables(
        backend, position=262130, tokens=7, dtype=torch.bfloat16, device="cpu"
    )
    module = Qwen3_5TextRotaryEmbedding(hf)
    positions = torch.arange(262130, 262137)[None]
    hfco, hfsi = module(torch.empty(1, 7, 32, dtype=torch.bfloat16), positions)
    assert torch.equal(co, hfco[0, :, :32].float()) and torch.equal(
        si, hfsi[0, :, :32].float()
    )
    qnorm, knorm = Qwen3_5RMSNorm(256).bfloat16(), Qwen3_5RMSNorm(256).bfloat16()
    with torch.no_grad():
        qnorm.weight.copy_(rawq)
        knorm.weight.copy_(rawk)
    q = qnorm(qg.reshape(7, 6, 512)[..., :256])[None].transpose(1, 2)
    k = knorm(key)[None].transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, hfco, hfsi)
    qo, ko = torch.empty(7, 6, 256, dtype=torch.bfloat16), torch.empty_like(key)
    resident.prepare(tokens=7, query_heads=6, kv_heads=2, rotary_dim=64).run(
        qg, key, co, si, query_out=qo, key_out=ko
    )
    assert torch.equal(qo, q[0].transpose(0, 1)) and torch.equal(
        ko, k[0].transpose(0, 1)
    )
    expected = qo * qg.reshape(7, 6, 512)[..., 256:].sigmoid()
    backend.prepare_qwen_output_gate(
        tokens=7, heads=6, head_dim=256, dtype=torch.bfloat16, device="cpu"
    ).run(qo, qg, out=qo)
    assert torch.equal(qo, expected)
