"""Separate MTP source bindings, independent block reference and native rollback."""

import copy
import os
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from test_qwen_model import hybrid_fixture, zero_norm
from test_qwen_source import write_source
from test_qwen_tensor_parallel import same_state
from qwen_gdn_reference import q8_source_oracle
from attention_reference import attention_reference
from test_fp32_parameters import mixed_port as base_port
from venturi.models import (
    NativeQwenModel,
    NativeQwenPipelineModel,
    NativeQwenMTPModel,
    QwenMTPPlan,
    QwenMTPWeights,
    QwenQ8Weights,
)
from venturi.ninfer import NInferArtifact
from venturi.qwen_export import export_qwen_q8, export_qwen_mtp_q8


def mtp_fixture(root, *, tied=False):
    weights, raw = hybrid_fixture(root, tied=tied)
    cfg = copy.deepcopy(weights.config)
    cfg.update(mtp_num_hidden_layers=1, mtp_use_dedicated_embeddings=False)
    raw = {k: v for k, v in raw.items() if not k.startswith("mtp.")}
    for k, v in list(raw.items()):
        if k.startswith("model.language_model.layers.3."):
            raw[k.replace("model.language_model.layers.3.", "mtp.layers.0.")] = (
                v * 0.73
            ).bfloat16()
    h = cfg["hidden_size"]
    gen = torch.Generator().manual_seed(7493)
    raw["mtp.fc.weight"] = (torch.randn(h, h * 2, generator=gen) * 0.05).bfloat16()
    for s in ("pre_fc_norm_embedding", "pre_fc_norm_hidden", "norm"):
        raw[f"mtp.{s}.weight"] = (torch.randn(h, generator=gen) * 0.05).bfloat16()
    source = write_source(
        root / "mtp-source",
        raw=dict(model_type="qwen3_5", tie_word_embeddings=tied, text_config=cfg),
        tensors=raw,
    )
    target = QwenQ8Weights.from_ninfer(
        NInferArtifact(export_qwen_q8(source, root / "mtp-target.ninfer")["output"])
    )
    draft = QwenMTPWeights.from_ninfer(
        NInferArtifact(export_qwen_mtp_q8(source, root / "mtp-draft.ninfer")["output"])
    )
    return target, draft, raw, source


def test_mtp_separate_schema_and_original_quantization(tmp_path):
    target, draft, raw, source = mtp_fixture(tmp_path)
    draft.validate_target(target)
    assert len(draft.plan.bindings) == 15
    assert len([s for s in draft.plan.bindings if s.format == "q8_g32_fp16"]) == 8
    assert target.artifact.artifact_id != draft.artifact.artifact_id
    for s in draft.plan.bindings:
        value = raw[s.source]
        if s.format == "q8_g32_fp16":
            encoded = draft.read_linear(s.binding)
            padded = encoded.codes.shape[1]
            actual = (
                encoded.codes.double().view(value.shape[0], -1, 32)
                * encoded.scales.double()[:, :, None]
            ).reshape(value.shape[0], padded)[:, : value.shape[1]]
            assert torch.equal(actual, q8_source_oracle(value))
        else:
            assert torch.equal(draft.read_float(s.binding), value)
    with pytest.raises(ValueError, match="complete unfused"):
        QwenQ8Weights.from_ninfer(draft.artifact)
    cfg = copy.deepcopy(target.config)
    cfg["mtp_use_dedicated_embeddings"] = True
    with pytest.raises(ValueError, match="shared"):
        QwenMTPPlan.from_config(cfg)
    changed = copy.deepcopy(target)
    changed.plan.config["rms_norm_eps"] *= 2
    with pytest.raises(ValueError, match="configuration"):
        draft.validate_target(changed)
    raw.pop("mtp.fc.weight")
    incomplete = write_source(
        tmp_path / "incomplete",
        raw=dict(model_type="qwen3_5", text_config=target.config),
        tensors=raw,
    )
    with pytest.raises(Exception, match="15 original"):
        export_qwen_mtp_q8(incomplete, tmp_path / "bad.ninfer")
    assert not (tmp_path / "bad.ninfer").exists()


def mtp_oracle(
    raw,
    cfg,
    ids,
    hidden,
    cache=None,
    *,
    embedding=None,
    quantizer=q8_source_oracle,
    compute_logits=True,
):
    """FP64 equations from source tensor names, independent of production bindings."""
    eps, hq, hk, d = (
        cfg[k]
        for k in (
            "rms_norm_eps",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
        )
    )
    dtype = hidden.dtype

    def cast(x):
        return x.to(dtype).double()

    def norm(x, name):
        return cast(zero_norm(x.double(), raw[name], eps))

    def linear(x, name):
        return cast(x.double() @ quantizer(raw[name]).T)

    emb = cast(
        quantizer(raw["model.language_model.embed_tokens.weight"])[ids]
        if embedding is None
        else embedding
    )
    x = linear(
        torch.cat(
            (
                norm(emb, "mtp.pre_fc_norm_embedding.weight"),
                norm(hidden, "mtp.pre_fc_norm_hidden.weight"),
            ),
            -1,
        ),
        "mtp.fc.weight",
    )
    p = "mtp.layers.0."
    v = norm(x, p + "input_layernorm.weight")
    qg = linear(v, p + "self_attn.q_proj.weight").reshape(len(ids), hq, 2 * d)
    key = linear(v, p + "self_attn.k_proj.weight").reshape(len(ids), hk, d)
    value = linear(v, p + "self_attn.v_proj.weight").reshape(len(ids), hk, d)
    query = zero_norm(qg[..., :d], raw[p + "self_attn.q_norm.weight"], eps)
    key = zero_norm(key, raw[p + "self_attn.k_norm.weight"], eps)
    pos = 0 if cache is None else len(cache[0])
    r = int(d * cfg["rope_parameters"]["partial_rotary_factor"])
    inv = 1.0 / cfg["rope_parameters"]["rope_theta"] ** (
        torch.arange(0, r, 2, dtype=torch.float32) / r
    )
    phase = (
        torch.arange(pos, pos + len(ids), dtype=torch.float32)[:, None] * inv[None, :]
    )
    co, si = (
        phase.cos().to(dtype).double()[:, None],
        phase.sin().to(dtype).double()[:, None],
    )

    def rotate(z):
        a, b = z[..., : r // 2], z[..., r // 2 : r]
        return cast(torch.cat((a * co - b * si, b * co + a * si, z[..., r:]), -1))

    query, key = rotate(query), rotate(key)
    keys, values = (
        (key, value)
        if cache is None
        else (torch.cat((cache[0], key)), torch.cat((cache[1], value)))
    )
    attended = cast(attention_reference(query, keys, values, d**-0.5))
    attended = cast(attended * qg[..., d:].sigmoid())
    residual = cast(x + linear(attended.flatten(1), p + "self_attn.o_proj.weight"))
    normalized = norm(residual, p + "post_attention_layernorm.weight")
    gate, up = (linear(normalized, p + f"mlp.{s}_proj.weight") for s in ("gate", "up"))
    out = cast(
        residual + linear(cast(gate * gate.sigmoid() * up), p + "mlp.down_proj.weight")
    )
    out = norm(out, "mtp.norm.weight")
    head = (
        "model.language_model.embed_tokens.weight"
        if cfg["tie_word_embeddings"]
        else "lm_head.weight"
    )
    return out, linear(out[-1:], head)[0] if compute_logits else None, (keys, values)


@pytest.fixture(params=["native-fp16", "native-bf16"])
def mtp_engine(request):
    if os.environ.get("VENTURI_MTP_TEST") != "1":
        pytest.skip("explicit native MTP opt-in required")
    return base_port.__wrapped__(request)


def load_models(target_weights, mtp_weights, engine, mode="single"):
    backend, dtype, device = engine
    kw = dict(
        weight_mode="resident",
        dtype=dtype,
        device=device,
        max_tokens=3,
        reserve_bytes=0,
    )
    if mode == "pp":
        target = NativeQwenPipelineModel.from_q8(
            target_weights, backend, pipeline_device="cuda:1", split_layer=2, **kw
        )
    else:
        if mode == "tp":
            kw.update(tensor_parallel_device="cuda:1", tensor_parallel_transport="host")
        target = NativeQwenModel.from_q8(target_weights, backend, **kw)
    try:
        return target, NativeQwenMTPModel.from_q8(target, mtp_weights, reserve_bytes=0)
    except BaseException:
        target.close()
        raise


@pytest.mark.parametrize("mode", ["single", "tp", "pp"])
def test_mtp_block_source_oracle_and_incremental_kv(tmp_path, mtp_engine, mode):
    a, b, raw, _ = mtp_fixture(tmp_path)
    target, model = load_models(a, b, mtp_engine, mode)
    s = model.create_session(24)
    cache = None
    try:
        for index, ids in enumerate(([1, 3, 2], [4], [5, 6])):
            h = (
                torch.randn(
                    len(ids),
                    a.config["hidden_size"],
                    generator=torch.Generator().manual_seed(190 + index),
                )
                * 0.31
            ).to(model.dtype)
            actual = s.draft_forward(ids, h)
            expected, logits, cache = mtp_oracle(raw, a.config, ids, h, cache)
            torch.testing.assert_close(
                actual["hidden"].double(), expected, atol=0.008, rtol=0.02
            )
            torch.testing.assert_close(
                actual["logits"].double(), logits, atol=0.008, rtol=0.02
            )
            state = s.draft_snapshot()
            torch.testing.assert_close(
                state["keys"].double(), cache[0], atol=0.008, rtol=0.02
            )
            torch.testing.assert_close(
                state["values"].double(), cache[1], atol=0.008, rtol=0.02
            )
    finally:
        s.close()
        model.close()
        target.close()


@pytest.mark.parametrize("mode", ["single", "tp", "pp"])
@pytest.mark.parametrize("forced", ["native", "accept", "reject"])
def test_mtp_exact_target_output_state_eos_and_limits(
    tmp_path, mtp_engine, mode, forced
):
    a, b, _, _ = mtp_fixture(tmp_path)
    target, model = load_models(a, b, mtp_engine, mode)
    ref, s = target.create_session(32), model.create_session(32)
    prompt = [1, 2, 3, 4, 5]
    try:
        for maximum, stop in (
            (1, None),
            (2, None),
            (3, None),
            (8, None),
            (9, 0),
            (9, 1),
            (9, 3),
        ):
            ref.reset()
            s.reset()
            expected = ref.generate(prompt, max_new_tokens=maximum)
            eos = [] if stop is None else [expected[stop]]
            if eos:
                ref.reset()
                expected = ref.generate(
                    prompt, max_new_tokens=maximum, eos_token_ids=eos
                )
            if forced != "native":
                s._force_proposals(
                    expected
                    if forced == "accept"
                    else [(v + 1) % a.config["vocab_size"] for v in expected]
                )
            actual = s.generate(prompt, max_new_tokens=maximum, eos_token_ids=eos)
            timings = s.generation_timings()
            assert (
                0
                < timings["native_time_to_first_token_seconds"]
                <= timings["native_generation_seconds"]
            )
            assert actual == expected
            assert s.position == len(prompt) + len(expected) - 1
            assert s.stats()["draft_position"] == s.position
            same_state(s.snapshot(), ref.snapshot())
            stats = s.stats()
            if forced == "reject":
                assert stats["accepted_tokens"] == 0
            if forced == "accept":
                assert stats["accepted_tokens"] == stats["proposed_tokens"]
            with pytest.raises(RuntimeError, match="fresh/reset"):
                s.generate([actual[-1]], max_new_tokens=1)
    finally:
        ref.close()
        s.close()
        model.close()
        target.close()


@pytest.mark.parametrize("mode", ["single", "tp", "pp"])
def test_mtp_failure_reset_lifetime_and_native_generation(
    tmp_path, mtp_engine, mode, monkeypatch
):
    a, b, _, _ = mtp_fixture(tmp_path, tied=True)
    target, model = load_models(a, b, mtp_engine, mode)
    ref, s = target.create_session(32), model.create_session(32)
    expected = ref.generate([1, 2, 3, 4, 5], max_new_tokens=7)
    try:
        s._inject_failure(layer=3, position=5)
        with pytest.raises(RuntimeError, match="injected"):
            s.generate([1, 2, 3, 4, 5], max_new_tokens=7)
        assert s.failed
        s.reset()
        target.close()
        model.close()

        def forbidden(*args, **kwargs):
            raise AssertionError("Torch math/alloc called during native MTP generation")

        with monkeypatch.context() as context:
            for name in (
                "empty",
                "zeros",
                "matmul",
                "mm",
                "softmax",
                "sigmoid",
                "cat",
                "tensor",
            ):
                context.setattr(torch, name, forbidden)
            assert s.generate([1, 2, 3, 4, 5], max_new_tokens=7) == expected
        with pytest.raises(RuntimeError, match="closed"):
            model.create_session(32)
    finally:
        ref.close()
        s.close()
        model.close()
        target.close()


@pytest.mark.parametrize("count", [1, 2, 4, 8])
def test_mtp_pipeline_concurrent_request_isolation(tmp_path, mtp_engine, count):
    a, b, _, _ = mtp_fixture(tmp_path)
    target, model = load_models(a, b, mtp_engine, "pp")
    sessions = [model.create_session(24) for _ in range(count)]
    refs = [target.create_session(24) for _ in range(count)]
    prompts = [[1 + i, 2, 3] for i in range(count)]
    try:
        expected = [s.generate(p, max_new_tokens=7) for s, p in zip(refs, prompts)]
        with ThreadPoolExecutor(max_workers=count) as pool:
            jobs = [
                pool.submit(s.generate, p, max_new_tokens=7)
                for s, p in zip(sessions, prompts)
            ]
            assert [j.result(timeout=90) for j in jobs] == expected
        for s, ref in zip(sessions, refs):
            same_state(s.snapshot(), ref.snapshot())
    finally:
        for s in sessions + refs:
            s.close()
        model.close()
        target.close()
