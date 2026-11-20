"""Hybrid integration, independent source oracle and per-request state ownership."""

import copy
import os

import pytest
import torch

from attention_reference import attention_reference
from qwen_gdn_reference import gdn_layer_oracle, q8_source_oracle
from test_qwen_attention_layer import fixture_weights
from venturi.models import (
    QwenQ8Weights,
    ResidentQwenModel,
    qwen_resident_bytes,
)
from venturi.ninfer import NInferArtifact
from venturi.ops import CudaBackend, TorchBackend
from venturi.qwen_export import export_qwen_q8
from test_qwen_source import write_source


def hybrid_fixture(root, *, tied=False):
    weights, raw, _ = fixture_weights(root)
    cfg = dict(model_type="qwen3_5", text_config=copy.deepcopy(weights.config))
    cfg["text_config"]["num_hidden_layers"] = 4
    cfg["text_config"]["layer_types"] *= 2
    cfg["text_config"]["tie_word_embeddings"] = tied
    cfg["tie_word_embeddings"] = tied
    if tied:
        raw.pop("lm_head.weight")
    gen = torch.Generator().manual_seed(1201)
    for name, value in list(raw.items()):
        if name.startswith("model.language_model.layers.0."):
            scale = 0.08 if value.ndim > 1 else 0.03
            raw[name] = (torch.randn(value.shape, generator=gen) * scale).to(
                value.dtype
            )
    raw["model.language_model.layers.0.linear_attn.norm.weight"] += 0.9
    for name, value in list(raw.items()):
        if ".layers.0." in name or ".layers.1." in name:
            i = 2 if ".layers.0." in name else 3
            dest = name.replace(".layers.0.", f".layers.{i}.").replace(
                ".layers.1.", f".layers.{i}."
            )
            raw[dest] = (value * 0.875).to(value.dtype)
    source = write_source(root / "hybrid-source", raw=cfg, tensors=raw)
    artifact = export_qwen_q8(source, root / "hybrid-q8.ninfer")
    return QwenQ8Weights.from_ninfer(NInferArtifact(artifact["output"])), raw


@pytest.fixture(params=["fp64", "cpu-bf16", "cuda"])
def engine(request):
    if request.param != "cuda":
        return (
            TorchBackend(),
            torch.float64 if request.param == "fp64" else torch.bfloat16,
            "cpu",
        )
    library = os.environ.get("VENTURI_BLOCKS_LIBRARY")
    if not library:
        pytest.skip("explicit native library required")
    return CudaBackend(library), torch.bfloat16, "cuda:0"


def load(weights, engine):
    backend, dtype, device = engine
    return ResidentQwenModel.from_q8(
        weights, backend, dtype=dtype, device=device, reserve_bytes=0
    )


def zero_norm(x, raw, eps):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * (1 + raw.double())


def independent_forward(raw, cfg, ids, *, tied):
    """Complete FP64 original-source oracle, never using runtime bindings/ops."""
    x = q8_source_oracle(raw["model.language_model.embed_tokens.weight"])[ids]
    eps, hq, hk, d = (
        cfg[k]
        for k in (
            "rms_norm_eps",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
        )
    )
    for i, kind in enumerate(cfg["layer_types"]):
        p = f"model.language_model.layers.{i}."
        if kind == "linear_attention":
            channels = (
                2 * cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"]
                + cfg["linear_num_value_heads"] * cfg["linear_value_head_dim"]
            )
            history = torch.zeros(
                cfg["linear_conv_kernel_dim"] - 1, channels, dtype=torch.float64
            )
            state = torch.zeros(
                cfg["linear_num_value_heads"],
                cfg["linear_value_head_dim"],
                cfg["linear_key_head_dim"],
                dtype=torch.float64,
            )
            x, _, _, _ = gdn_layer_oracle(raw, cfg, x, history, state, layer=i)
            continue

        def norm(v, name):
            return zero_norm(v, raw[p + name], eps)

        def linear(v, name):
            return v @ q8_source_oracle(raw[p + name]).T

        normalized = norm(x, "input_layernorm.weight")
        qg = linear(normalized, "self_attn.q_proj.weight").view(len(ids), hq, 2 * d)
        q = norm(qg[..., :d], "self_attn.q_norm.weight")
        k = norm(
            linear(normalized, "self_attn.k_proj.weight").view(len(ids), hk, d),
            "self_attn.k_norm.weight",
        )
        v = linear(normalized, "self_attn.v_proj.weight").view(len(ids), hk, d)
        r = int(d * cfg["rope_parameters"]["partial_rotary_factor"])
        # Original inverse frequency and text phase are FP32, including FP64
        # diagnostic activations; trig then activation cast follows the adapter.
        inv = 1.0 / cfg["rope_parameters"]["rope_theta"] ** (
            torch.arange(0, r, 2, dtype=torch.float32) / r
        )
        phase = torch.arange(len(ids), dtype=torch.float32)[:, None] * inv[None, :]
        co, si = phase.cos().double()[:, None], phase.sin().double()[:, None]

        def rotate(z):
            a, b = z[..., : r // 2], z[..., r // 2 : r]
            return torch.cat((a * co - b * si, b * co + a * si, z[..., r:]), -1)

        mixed = attention_reference(rotate(q), rotate(k), v, d**-0.5)
        gated = mixed * qg[..., d:].sigmoid()
        residual = x + linear(gated.flatten(1), "self_attn.o_proj.weight")
        normalized = norm(residual, "post_attention_layernorm.weight")
        gate = linear(normalized, "mlp.gate_proj.weight")
        up = linear(normalized, "mlp.up_proj.weight")
        x = residual + linear(gate * gate.sigmoid() * up, "mlp.down_proj.weight")
    x = zero_norm(x, raw["model.language_model.norm.weight"], eps)
    head = "model.language_model.embed_tokens.weight" if tied else "lm_head.weight"
    return x @ q8_source_oracle(raw[head]).T


@pytest.mark.parametrize("tied", [False, True])
def test_hybrid_fp64_full_source_oracle_chunking_and_tied_storage(tmp_path, tied):
    weights, raw = hybrid_fixture(tmp_path, tied=tied)
    model = load(weights, (TorchBackend(), torch.float64, "cpu"))
    assert (model.embedding is model.head) == tied
    assert (
        qwen_resident_bytes(weights, reference=True, dtype=torch.float64)
        == model.storage_nbytes
    )
    ids = [2, 4, 1, 8, 3, 9, 2]
    ideal = independent_forward(raw, model.config, ids, tied=tied)
    session = model.create_session(16, max_tokens=3, reserve_bytes=0)
    actual = session.forward(ids, all_logits=True)
    torch.testing.assert_close(actual, ideal, atol=3e-11, rtol=3e-10)
    assert session.position == len(ids)
    session.reset()
    result = torch.cat(
        [
            session.forward(ids[:2], all_logits=True),
            session.forward(ids[2:5], all_logits=True),
            session.forward(ids[5:], all_logits=True),
        ]
    )
    torch.testing.assert_close(result, ideal, atol=3e-11, rtol=3e-10)
    session.close()


def manual_composition(model, ids):
    """Separately owned per-layer workspaces/states, no session orchestration."""
    t = len(ids)
    states, caches, plans, keep = [], [], [], []
    x = torch.empty(
        t, model.config["hidden_size"], dtype=model.dtype, device=model.device
    )
    embedding = model.embedding.prepare_embedding(ids)
    embedding.run(out=x)
    co, si = model.rope.tables(
        model.backend, position=0, tokens=t, dtype=model.dtype, device=model.device
    )
    for layer in model.layers:
        p = layer.prepare(tokens=t)
        plans.append(p)
        ws = p.empty_workspace()
        keep.append(ws)
        out = torch.empty_like(x)
        if hasattr(layer, "zero_state"):
            state = layer.zero_state()
            states.append(state)
            p.run(
                x,
                out=out,
                history=state.history,
                state=state.recurrent,
                history_out=state.history,
                state_out=state.recurrent,
                workspace=ws,
            )
        else:
            cache = layer.empty_cache(capacity=16)
            caches.append(cache)
            p.at(position=0, cache=cache).run(x, co, si, out=out, workspace=ws)
        x = out
    normalized = torch.empty_like(x)
    model.norm.prepare(tokens=t).run(x, out=normalized)
    head = model.head.prepare(tokens=t)
    out = torch.empty(
        t, model.config["vocab_size"], dtype=model.dtype, device=model.device
    )
    scratch = (
        torch.empty(head.workspace_bytes, dtype=torch.uint8, device=model.device)
        if head.workspace_bytes
        else None
    )
    head.run(normalized, out=out, workspace=scratch)
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    for p in (*plans, head):
        if hasattr(p, "close"):
            p.close()
    return out, states, caches


def test_hybrid_manual_composition_state_and_native_no_fallback(
    tmp_path, engine, monkeypatch
):
    weights, _ = hybrid_fixture(tmp_path)
    model = load(weights, engine)
    assert (
        qwen_resident_bytes(
            weights,
            reference=model.backend.name == "torch-reference",
            dtype=model.dtype,
        )
        == model.storage_nbytes
    )
    ids = [2, 4, 1, 8, 3, 9, 2]
    expected, states, caches = manual_composition(model, ids)
    session = model.create_session(16, max_tokens=7, reserve_bytes=0)
    if model.device.type == "cuda":

        def forbidden(*a, **k):
            raise AssertionError("native model used a numerical Torch fallback")

        for name in ("matmul", "mm", "bmm", "index_select", "softmax"):
            monkeypatch.setattr(torch, name, forbidden)
        monkeypatch.setattr(
            torch.nn.functional, "scaled_dot_product_attention", forbidden
        )
    result = session.forward(ids, all_logits=True)
    assert torch.equal(result, expected.to(result.dtype))
    assert session.position == 7 and not session.failed
    assert set(session._lanes) == {7}
    lane = session._lanes[7]
    assert lane.workspaces[0] is lane.workspaces[2]
    assert lane.workspaces[1] is lane.workspaces[3]
    assert lane.workspaces[0] is not lane.workspaces[1]
    for a, b in zip((s for s in session._states if s is not None), states):
        assert torch.equal(a.history, b.history) and torch.equal(
            a.recurrent, b.recurrent
        )
    for a, b in zip((c for c in session._caches if c is not None), caches):
        assert (
            a.length == b.length == 7
            and torch.equal(a.keys, b.keys)
            and torch.equal(a.values, b.values)
        )
    if model.device.type == "cuda" or model.dtype == torch.float64:
        # Native Q8 row kernels are shape-independent; CPU BF16 GEMM is not.
        session.reset()
        split = torch.cat(
            (
                session.forward(ids[:3], all_logits=True),
                session.forward(ids[3:], all_logits=True),
            )
        )
        torch.testing.assert_close(
            split,
            result,
            atol=2e-12 if model.dtype == torch.float64 else 0,
            rtol=2e-11 if model.dtype == torch.float64 else 0,
        )
    session.close()


def test_model_budget_and_geometry_preflight_never_read_payload(tmp_path, monkeypatch):
    weights, _ = hybrid_fixture(tmp_path)

    def forbidden(*a, **k):
        raise AssertionError("payload read before preflight")

    monkeypatch.setattr(type(weights), "read_linear", forbidden)
    monkeypatch.setattr(type(weights), "read_float", forbidden)
    with pytest.raises(MemoryError, match="No implicit offload"):
        ResidentQwenModel.from_q8(
            weights,
            TorchBackend(),
            device="cpu",
            reserve_bytes=0,
            memory_budget_bytes=1,
        )
    weights.config["head_dim"] = 16
    with pytest.raises(ValueError, match="head_dim"):
        ResidentQwenModel.from_q8(
            weights, TorchBackend(), device="cpu", reserve_bytes=0
        )


def test_session_failure_atomic_position_poison_reset_and_invalid_ids(
    tmp_path, engine, monkeypatch
):
    weights, _ = hybrid_fixture(tmp_path)
    model = load(weights, engine)
    session = model.create_session(12, max_tokens=3, reserve_bytes=0)
    other = model.create_session(12, max_tokens=3, reserve_bytes=0)
    with pytest.raises(ValueError, match="Vocabulary"):
        session.forward([True])
    with pytest.raises(ValueError, match="capacity"):
        session.forward([1] * 13)
    assert not session.failed and session.position == 0 and not session._lanes
    clean = other.forward([1, 2, 3], all_logits=True)
    lane = session._lane(3)
    original = lane.layers[3].norm.run

    def fail(op, *a, **k):
        if op is lane.layers[3].norm:
            raise RuntimeError("injected last-layer submission failure")
        return original.__func__(op, *a, **k)

    monkeypatch.setattr(type(lane.layers[3].norm), "run", fail)
    with pytest.raises(RuntimeError, match="injected"):
        session.forward([1, 2, 3])
    assert session.failed and session.position == 0
    assert all(c.failed for c in session._caches if c is not None)
    with pytest.raises(RuntimeError, match="reset required"):
        session.forward([1])
    monkeypatch.setattr(type(lane.layers[3].norm), "run", original.__func__)
    session.reset()
    assert all(
        not bool(s.recurrent.any()) and not bool(s.history.any())
        for s in session._states
        if s is not None
    )
    result = session.forward([1, 2, 3], all_logits=True)
    assert torch.equal(result, clean) and session.position == other.position == 3
    assert all(
        s1.history.data_ptr() != s2.history.data_ptr()
        for s1, s2 in zip(session._states, other._states)
        if s1 is not None
    )
    session.close()
    other.close()
    with pytest.raises(RuntimeError, match="closed"):
        session.forward([1])


def test_closed_later_plan_preflight_before_any_state_change(tmp_path, engine):
    weights, _ = hybrid_fixture(tmp_path)
    model = load(weights, engine)
    session = model.create_session(12, max_tokens=3, reserve_bytes=0)
    lane = session._lane(3)
    lane.layers[-1].close()
    with pytest.raises(RuntimeError, match="closed"):
        session.forward([1, 2, 3])
    assert session.position == 0 and not session.failed
    assert all(
        not bool(s.recurrent.any()) and not bool(s.history.any())
        for s in session._states
        if s is not None
    )
    assert all(c.length == 0 and not c.failed for c in session._caches if c is not None)
    session.close()


def test_shape_lanes_bounded_and_returned_logits_owned(tmp_path, engine):
    weights, _ = hybrid_fixture(tmp_path)
    model = load(weights, engine)
    s = model.create_session(32, max_tokens=5, reserve_bytes=0)
    a = s.forward([1, 2], all_logits=True)
    save = a.clone()
    assert not s._lanes
    for count in (3, 4, 5, 1, 2):
        s.forward([1] * count)
    assert set(s._lanes) == {1, 5} and s.workspace_bytes > 0 and s.cache_bytes > 0
    assert torch.equal(a, save)
    s.close()


def test_session_owner_stream_and_two_request_continuations(tmp_path, engine):
    if engine[2] == "cpu":
        pytest.skip("native multi-stream integration")
    weights, _ = hybrid_fixture(tmp_path)
    model = load(weights, engine)
    a, b = (
        torch.cuda.Stream(device=model.device),
        torch.cuda.Stream(device=model.device),
    )
    with torch.cuda.stream(a):
        sa = model.create_session(16, max_tokens=3, reserve_bytes=0)
        first = sa.forward([1, 2, 3], all_logits=True)
    with torch.cuda.stream(b):
        sb = model.create_session(16, max_tokens=3, reserve_bytes=0)
        sb.forward([6, 5])
        second = sb.forward([4, 3])
        with pytest.raises(RuntimeError, match="owning CUDA stream"):
            sa.forward([1])
    assert sa.position == 3 and not sa.failed
    with torch.cuda.stream(a):
        continuation = sa.forward([4, 5])
        sa.reset()
        total = sa.forward([1, 2, 3, 4, 5], all_logits=True)
        assert torch.equal(total[:3], first)
        assert torch.equal(total[-1], continuation)
        assert not torch.equal(second, continuation)
        sa.close()
    with torch.cuda.stream(b):
        sb.close()


def test_failure_after_committed_prefix_of_one_forward_poison_entire_append(
    tmp_path, engine, monkeypatch
):
    weights, _ = hybrid_fixture(tmp_path)
    model = load(weights, engine)
    session = model.create_session(16, max_tokens=2, reserve_bytes=0)
    session.forward([1])
    lane = session._lane(2)
    original = lane.layers[3].norm.run
    count = 0

    def fail(op, *args, **kwargs):
        nonlocal count
        if op is lane.layers[3].norm:
            count += 1
            if count == 2:
                raise RuntimeError("injected second-chunk failure")
        return original.__func__(op, *args, **kwargs)

    monkeypatch.setattr(type(lane.layers[3].norm), "run", fail)
    with pytest.raises(RuntimeError, match="second-chunk"):
        session.forward([2, 3, 4, 5])
    # Some private caches may have enqueued a longer prefix. The public whole
    # request never commits it or permits continuation using partially advanced
    # GDN state. Only resetting to an empty request makes it usable again.
    assert session.position == 1 and session.failed and count == 2
    assert all(c.failed for c in session._caches if c is not None)
    monkeypatch.setattr(type(lane.layers[3].norm), "run", original.__func__)
    session.reset()
    assert session.position == 0 and not session.failed
    session.close()
