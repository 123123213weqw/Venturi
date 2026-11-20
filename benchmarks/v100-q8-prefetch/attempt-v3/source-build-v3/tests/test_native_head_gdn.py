"""Distributed GDN caches are request-owned, including rollback and diagnostics."""

import copy
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from test_qwen_model import hybrid_fixture, independent_forward
from test_qwen_mtp import mtp_fixture, load_models
from test_qwen_source import write_source
from test_qwen_tensor_parallel import same_state
from venturi.models import NativeQwenModel, QwenQ8Weights, QwenMTPWeights
from venturi.models.qwen_native import _head_gdn_workspace
from venturi.ninfer import NInferArtifact
from venturi.ops import CudaBackend
from venturi.qwen_export import export_qwen_q8, export_qwen_mtp_q8

PROFILE = "q8-residual-column-mlp-head-gdn"


def eligible_mtp(root, *, tied=False):
    root.mkdir(parents=True, exist_ok=True)
    (root / "original").mkdir()
    original, _, raw, _ = mtp_fixture(root / "original", tied=tied)
    cfg = copy.deepcopy(original.config)
    cfg.update(linear_num_value_heads=6, linear_value_head_dim=64)
    generator = torch.Generator().manual_seed(1907)
    shapes = {
        "in_proj_qkv.weight": (640, 32),
        "in_proj_z.weight": (384, 32),
        "in_proj_a.weight": (6, 32),
        "in_proj_b.weight": (6, 32),
        "out_proj.weight": (32, 384),
        "conv1d.weight": (640, 1, 4),
        "A_log": (6,),
        "dt_bias": (6,),
        "norm.weight": (64,),
    }
    for layer, kind in enumerate(cfg["layer_types"]):
        if kind != "linear_attention":
            continue
        prefix = f"model.language_model.layers.{layer}.linear_attn."
        for name, shape in shapes.items():
            value = torch.randn(shape, generator=generator) * (
                0.06 if len(shape) > 1 else 0.03
            )
            if name == "norm.weight":
                value += 0.9
            raw[prefix + name] = value.to(
                torch.float32 if name in ("A_log", "norm.weight") else torch.bfloat16
            )
    source = write_source(
        root / "eligible",
        raw=dict(model_type="qwen3_5", text_config=cfg, tie_word_embeddings=tied),
        tensors=raw,
    )
    target = QwenQ8Weights.from_ninfer(
        NInferArtifact(export_qwen_q8(source, root / "target.ninfer")["output"])
    )
    draft = QwenMTPWeights.from_ninfer(
        NInferArtifact(export_qwen_mtp_q8(source, root / "draft.ninfer")["output"])
    )
    return target, draft, raw, source


@pytest.fixture
def engine():
    path = os.environ.get("VENTURI_NATIVE_HEAD_GDN_LIBRARY")
    if not path:
        pytest.skip("Explicit native head-GDN SM70 library and two V100s required")
    return CudaBackend(path), torch.float16, "cuda:0"


def load(weights, engine, rows=8, profile=PROFILE):
    backend, dtype, device = engine
    return NativeQwenModel.from_q8(
        weights,
        backend,
        weight_mode="resident",
        dtype=dtype,
        device=device,
        max_tokens=rows,
        reserve_bytes=0,
        tensor_parallel_device="cuda:1",
        tensor_parallel_transport="host",
        projection_profile=profile,
    )


def test_head_profile_fails_closed_before_cuda_access():
    weights, backend = object.__new__(QwenQ8Weights), object.__new__(CudaBackend)
    backend._native = SimpleNamespace()
    kw = dict(weight_mode="resident", dtype=torch.float16, projection_profile=PROFILE)
    with pytest.raises(ValueError, match="two-device host TP"):
        NativeQwenModel.from_q8(weights, backend, **kw)
    with pytest.raises(RuntimeError, match="exports absent"):
        NativeQwenModel.from_q8(
            weights,
            backend,
            tensor_parallel_device="cuda:1",
            tensor_parallel_transport="host",
            **kw,
        )
    with pytest.raises(ValueError, match="complete resident FP16"):
        NativeQwenModel.from_q8(
            weights,
            backend,
            tensor_parallel_device="cuda:1",
            tensor_parallel_transport="host",
            **{**kw, "weight_mode": "streaming"},
        )


def test_actual_qwen_head_workspace_bound():
    cfg = dict(
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    assert _head_gdn_workspace(cfg, 32) == 1_254_400


def test_eligible_fixture_preserves_original_odd_fixture(tmp_path):
    (tmp_path / "odd").mkdir()
    original, _ = hybrid_fixture(tmp_path / "odd")
    target, draft, _, _ = eligible_mtp(tmp_path / "aligned")
    assert original.config["linear_value_head_dim"] == 13
    assert target.config["linear_value_head_dim"] == 64
    assert target.config["linear_num_value_heads"] == 6
    draft.validate_target(target)


@pytest.mark.parametrize("rows", [1, 2, 3, 8])
def test_full_native_teacher_chunks_reset_and_canonical_state(tmp_path, engine, rows):
    weights, _, raw, _ = eligible_mtp(tmp_path)
    model = load(weights, engine, rows=rows)
    request = model.create_session(32)
    try:
        ids = [1, 2, 3, 4, 5, 6, 7, 8]
        expected = independent_forward(raw, weights.config, ids, tied=False)
        actual = request.forward(ids, all_logits=True)
        rel = float((actual.double() - expected).norm() / expected.norm())
        assert rel < 0.02
        assert torch.equal(actual.argmax(-1), expected.argmax(-1))
        first = request.snapshot()
        assert first["gdn"][0][0].shape == (3, 640)
        assert first["gdn"][0][1].shape == (6, 64, 64)
        memory = request.info()
        assert memory["head_gdn_enabled"] and memory["peer_request_device_bytes"] > 0
        assert memory["head_gdn_primary_workspace_bytes"] == _head_gdn_workspace(
            weights.config, rows
        )
        request.reset()
        second = request.forward(ids, all_logits=True)
        assert torch.equal(actual, second)
        same_state(first, request.snapshot())
    finally:
        request.close()
        model.close()


def test_eight_private_ragged_sequences_selective_reset_failure_native_only(
    tmp_path, engine
):
    weights, _, _, _ = eligible_mtp(tmp_path)
    model = load(weights, engine, rows=8)
    batch = model.create_batch([40] * 8)
    references = [model.create_session(40) for _ in range(8)]
    prompts = [[1, 2, 3, 4][: 1 + i % 4] for i in range(8)]
    try:
        expected = [
            r.generate(p, max_new_tokens=5) for r, p in zip(references, prompts)
        ]
        with (
            patch.object(
                torch, "matmul", side_effect=AssertionError("Python fallback")
            ),
            patch.object(torch, "empty", side_effect=AssertionError("Python fallback")),
        ):
            actual = batch.generate(prompts, max_new_tokens=5)
        assert actual == expected
        for i, ref in enumerate(references):
            same_state(batch.snapshot(i), ref.snapshot())
        assert (
            batch.info()["all_peer_request_device_bytes"]
            == 8 * references[0].info()["peer_request_device_bytes"]
        )
        untouched = batch.snapshot(4)
        batch.reset(slots=[7, 0])
        same_state(untouched, batch.snapshot(4))
        for i in (7, 0):
            references[i].reset()
        actual = batch.generate(
            [prompts[7], prompts[0]],
            slots=[7, 0],
            max_new_tokens=3,
            eos_token_ids=[[expected[7][0]], []],
        )
        assert actual == [
            references[i].generate(
                prompts[i],
                max_new_tokens=3,
                eos_token_ids=[expected[i][0]] if i == 7 else [],
            )
            for i in (7, 0)
        ]
        batch.reset(slots=[7, 0])
        batch._inject_failure(slot=7, layer=2, position=0)
        with pytest.raises(RuntimeError, match="injected"):
            batch.generate([prompts[7], prompts[0]], slots=[7, 0], max_new_tokens=3)
        batch.reset(slots=[7, 0])
        same_state(untouched, batch.snapshot(4))
    finally:
        batch.close()
        for ref in references:
            ref.close()
        model.close()


@pytest.mark.parametrize(
    "forced", ["native", "accept", "reject-first", "reject-second"]
)
def test_mtp2_peer_prefixes_correction_eos_and_rollback(
    tmp_path, engine, monkeypatch, forced
):
    import test_qwen_mtp2 as checks

    base_loader = load_models

    def loader(a, b, e, mode, **kw):
        return base_loader(a, b, e, "tp", profile=PROFILE, draft_profile="q8-residual")

    monkeypatch.setattr(checks, "mtp_fixture", eligible_mtp)
    monkeypatch.setattr(checks, "load_models", loader)
    from test_mtp_prefix import prefix as prefix_fixture

    prefix_fixture.__wrapped__(monkeypatch)
    checks.test_depth2_output_target_and_corrected_draft_caches(
        tmp_path, engine, "tp", PROFILE, forced
    )


@pytest.fixture
def head_batch_checks(monkeypatch):
    import test_qwen_mtp_batch as checks

    original = checks.models
    monkeypatch.setattr(checks, "mtp_fixture", eligible_mtp)

    def models(
        root, backend, tp, profile=PROFILE, tied=False, draft_profile="q8-residual"
    ):
        return original(
            root, backend, True, profile=PROFILE, tied=tied, draft_profile="q8-residual"
        )

    monkeypatch.setattr(checks, "models", models)
    return checks


@pytest.mark.parametrize("count", [1, 2, 4, 8])
@pytest.mark.parametrize("forced", ["native", "mixed"])
def test_mtp1_private_target_and_corrected_draft_cohorts(
    tmp_path, engine, head_batch_checks, count, forced
):
    head_batch_checks.test_target_and_corrected_draft_bytes(
        tmp_path, engine[0], True, PROFILE, count, forced
    )


@pytest.mark.parametrize("maximum", [1, 4, 11])
def test_mtp1_peer_eos_ragged_and_selective_reset(
    tmp_path, engine, head_batch_checks, maximum
):
    head_batch_checks.test_ragged_budgets_eos_tails_and_selective_reset(
        tmp_path, engine[0], True, maximum
    )


@pytest.mark.parametrize("position", [0, 3])
def test_mtp1_peer_failure_reset_close_and_native_only(
    tmp_path, engine, head_batch_checks, position, monkeypatch
):
    head_batch_checks.test_failure_poison_reset_lifetime_no_python_math(
        tmp_path, engine[0], position, monkeypatch
    )


def test_mtp2_peer_failure_reset_close_and_native_only(tmp_path, engine, monkeypatch):
    import test_qwen_mtp2 as checks

    original = load_models
    monkeypatch.setattr(checks, "mtp_fixture", eligible_mtp)

    def loader(a, b, e, mode, **kwargs):
        return original(a, b, e, "tp", profile=PROFILE, draft_profile="q8-residual")

    monkeypatch.setattr(checks, "load_models", loader)
    from test_mtp_prefix import prefix as prefix_fixture

    prefix_fixture.__wrapped__(monkeypatch)
    checks.test_depth2_failure_reset_lifetime_and_native_only(
        tmp_path, engine, "tp", monkeypatch
    )
