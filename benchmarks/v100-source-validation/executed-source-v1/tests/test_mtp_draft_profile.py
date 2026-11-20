"""Explicit changed draft arithmetic; target parity alone is insufficient."""

import inspect
import os
from types import SimpleNamespace

import pytest
import torch

import test_qwen_mtp2 as chain
import test_qwen_mtp_batch as packed
from test_mtp_prefix import prefix as prefix_fixture  # noqa: F401
from test_qwen_mtp import mtp_fixture, load_models, mtp_oracle
from test_qwen_mtp2 import same_draft
from venturi.models import (
    NativeQwenModel,
    NativeQwenPipelineModel,
    NativeQwenMTPModel,
    QwenMTPWeights,
)
from venturi.models.qwen_mtp_native import _bind
from venturi.ops import CudaBackend


def test_explicit_draft_profile_cpu_preflight_and_default():
    assert (
        inspect.signature(NativeQwenMTPModel.from_q8)
        .parameters["draft_projection_profile"]
        .default
        == "fp16"
    )
    weights = object.__new__(QwenMTPWeights)
    target = object.__new__(NativeQwenModel)
    target._dtype = torch.float16
    target._backend = SimpleNamespace(_native=SimpleNamespace())
    with pytest.raises(ValueError, match="explicit fp16/q8-residual"):
        NativeQwenMTPModel.from_q8(target, weights, draft_projection_profile="auto")
    with pytest.raises(RuntimeError, match="exports absent"):
        NativeQwenMTPModel.from_q8(
            target, weights, draft_projection_profile="q8-residual"
        )
    target._dtype = torch.bfloat16
    with pytest.raises(ValueError, match="FP16 non-PP"):
        NativeQwenMTPModel.from_q8(
            target, weights, draft_projection_profile="q8-residual"
        )
    pp = object.__new__(NativeQwenPipelineModel)
    pp._dtype = torch.float16
    with pytest.raises(ValueError, match="FP16 non-PP"):
        NativeQwenMTPModel.from_q8(pp, weights, draft_projection_profile="q8-residual")


def test_additive_profile_exports_do_not_break_old_binding():
    class Legacy:
        def __getattr__(self, name):
            if name in [
                "venturi_qwen_mtp_create_profiled",
                "venturi_qwen_mtp_projection_info",
            ]:
                raise AttributeError(name)
            return SimpleNamespace()

    native = _bind(SimpleNamespace(_native=Legacy()))
    assert native.venturi_qwen_create_profiled is None
    assert native.venturi_qwen_projection_info is None
    assert native.venturi_qwen_create is not None


def test_cli_draft_profile_requires_artifact_before_backend(
    tmp_path, monkeypatch, capsys
):
    from venturi import qwen_native_cli
    from test_qwen_model import hybrid_fixture

    w, _ = hybrid_fixture(tmp_path)
    monkeypatch.setattr(
        qwen_native_cli,
        "CudaBackend",
        lambda *a: pytest.fail("Backend before preflight"),
    )
    with pytest.raises(SystemExit) as error:
        qwen_native_cli.main(
            [
                "generate",
                str(w.artifact.path),
                "--token-ids",
                "1,2",
                "--mtp-draft-projection-profile",
                "q8-residual",
            ]
        )
    assert error.value.code == 1 and "explicit artifact" in capsys.readouterr().err


@pytest.fixture
def engine():
    path = os.environ.get("VENTURI_MTP_DRAFT_PROFILE_LIBRARY")
    if not path:
        pytest.skip("Explicit SM70 residual-draft library required")
    return CudaBackend(path), torch.float16, "cuda:0"


@pytest.mark.parametrize("mode", ["single", "tp"])
def test_independent_draft_hidden_logits_and_incremental_kv(tmp_path, engine, mode):
    a, b, raw, _ = mtp_fixture(tmp_path)
    target, model = load_models(
        a,
        b,
        engine,
        mode,
        profile="q8-residual-column-mlp" if mode == "tp" else "q8-residual",
        draft_profile="q8-residual",
    )
    session = model.create_session(32)
    cache = None
    try:
        info = model.projection_info
        assert info["mtp_draft_projection_profile"] == "q8-residual"
        assert info["mtp_draft_activation_quantization_passes"] == 2
        maximum = max(s.shape[1] for s in b.plan.bindings if s.format == "q8_g32_fp16")
        assert (
            info["mtp_draft_primary_projection_workspace_bytes"]
            == 3 * ((maximum + 31) // 32) * 72
        )
        assert info["mtp_draft_secondary_projection_workspace_bytes"] == 0
        for i, ids in enumerate([[1, 3, 2], [4], [5, 6]]):
            hidden = (
                torch.randn(
                    len(ids),
                    a.config["hidden_size"],
                    generator=torch.Generator().manual_seed(397 + i),
                )
                * 0.31
            ).half()
            actual = session.draft_forward(ids, hidden)
            out, logits, cache = mtp_oracle(raw, a.config, ids, hidden, cache)
            for got, ref in [
                (actual["hidden"], out),
                (actual["logits"], logits),
                (session.draft_snapshot()["keys"], cache[0]),
                (session.draft_snapshot()["values"], cache[1]),
            ]:
                torch.testing.assert_close(got.double(), ref, atol=0.008, rtol=0.02)
    finally:
        session.close()
        model.close()
        target.close()


@pytest.mark.parametrize("mode", ["single", "tp"])
def test_default_draft_bytes_equal_qualified_old_library(tmp_path, engine, mode):
    control = os.environ.get("VENTURI_MTP_DRAFT_CONTROL_LIBRARY")
    if not control:
        pytest.skip("Explicit qualified original FP16 draft control required")
    a, b, _, _ = mtp_fixture(tmp_path)
    old_engine = CudaBackend(control), torch.float16, "cuda:0"
    pairs = [
        load_models(
            a,
            b,
            e,
            mode,
            profile="q8-residual-column-mlp" if mode == "tp" else "q8-residual",
        )
        for e in [old_engine, engine]
    ]
    sessions = [model.create_session(32) for target, model in pairs]
    try:
        for i, ids in enumerate([[1, 3, 2], [4], [5, 6]]):
            h = (
                torch.randn(
                    len(ids),
                    a.config["hidden_size"],
                    generator=torch.Generator().manual_seed(191 + i),
                )
                * 0.31
            ).half()
            results = [s.draft_forward(ids, h) for s in sessions]
            for k in ["hidden", "logits"]:
                assert torch.equal(
                    results[0][k].view(torch.uint8), results[1][k].view(torch.uint8)
                )
            same_draft(*(s.draft_snapshot() for s in sessions))
        assert pairs[1][1].projection_info["mtp_draft_projection_profile"] == "fp16"
        assert (
            pairs[1][1].projection_info["mtp_draft_primary_projection_workspace_bytes"]
            == 0
        )
    finally:
        for s in sessions:
            s.close()
        for target, model in pairs:
            model.close()
            target.close()


@pytest.fixture
def residual_chain(monkeypatch):
    def load(*args, **kwargs):
        kwargs["draft_profile"] = "q8-residual"
        return load_models(*args, **kwargs)

    monkeypatch.setattr(chain, "load_models", load)


@pytest.mark.parametrize("mode", ["single", "tp"])
@pytest.mark.parametrize(
    "forced", ["native", "accept", "reject-first", "reject-second"]
)
@pytest.mark.usefixtures("prefix_fixture")
def test_prefix_transactions_same_profile_corrected_draft(
    tmp_path, engine, residual_chain, mode, forced
):
    chain.test_depth2_output_target_and_corrected_draft_caches(
        tmp_path,
        engine,
        mode,
        "q8-residual-column-mlp" if mode == "tp" else "q8-residual",
        forced,
    )


@pytest.mark.parametrize("mode", ["single", "tp"])
@pytest.mark.usefixtures("prefix_fixture")
def test_prefix_failure_reset_lifetime_native_only(
    tmp_path, engine, residual_chain, mode, monkeypatch
):
    chain.test_depth2_failure_reset_lifetime_and_native_only(
        tmp_path, engine, mode, monkeypatch
    )


@pytest.mark.parametrize("count", [2, 8])
@pytest.mark.usefixtures("prefix_fixture")
def test_concurrent_private_requests(tmp_path, engine, residual_chain, count):
    chain.test_depth2_concurrent_private_requests(tmp_path, engine, "tp", count)


@pytest.fixture
def residual_batch(monkeypatch):
    original = packed.models

    def load(*args, **kwargs):
        kwargs.update(draft_profile="q8-residual", profile="q8-residual-column-mlp")
        # Some inherited checks pass target profile positionally.
        if len(args) > 3:
            args = args[:3] + ("q8-residual-column-mlp",) + args[4:]
            kwargs.pop("profile")
        return original(*args, **kwargs)

    monkeypatch.setattr(packed, "models", load)


@pytest.mark.parametrize("count", [2, 8])
@pytest.mark.parametrize("forced", ["native", "accept", "reject", "mixed"])
def test_mtp_cohort_target_and_draft_bytes(
    tmp_path, engine, residual_batch, count, forced
):
    packed.test_target_and_corrected_draft_bytes(
        tmp_path, engine[0], True, "q8-residual-column-mlp", count, forced
    )


@pytest.mark.parametrize("maximum", [1, 7])
def test_cohort_ragged_eos_selective_reset(tmp_path, engine, residual_batch, maximum):
    packed.test_ragged_budgets_eos_tails_and_selective_reset(
        tmp_path, engine[0], True, maximum
    )


@pytest.mark.parametrize("position", [0, 3])
def test_cohort_failure_reset_and_native_only(
    tmp_path, engine, residual_batch, position, monkeypatch
):
    packed.test_failure_poison_reset_lifetime_no_python_math(
        tmp_path, engine[0], position, monkeypatch
    )


def test_overlapping_cohort_scalar_target_owners(tmp_path, engine, residual_batch):
    packed.test_overlapping_batches_scalar_and_original_target(tmp_path, engine[0])
