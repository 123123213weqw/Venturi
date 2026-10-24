"""New arithmetic stays explicit; private state and old numerical gates remain."""

import os
from types import SimpleNamespace

import pytest
import torch

import test_qwen_batch as batch_checks
import test_qwen_mtp as mtp_checks
from test_completion_benchmark import load as load_tool
from test_qwen_model import hybrid_fixture, independent_forward
from venturi.models import NativeQwenModel
from venturi.ops import CudaBackend


def test_profile_preflight_before_device_payload_and_missing_export(tmp_path):
    weights, _ = hybrid_fixture(tmp_path)
    backend = object.__new__(CudaBackend)

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid profile touched device")

    backend._device = forbidden
    for profile in (None, True, 1, "auto", "q8"):
        with pytest.raises(ValueError, match="Projection profile"):
            NativeQwenModel.from_q8(
                weights, backend, weight_mode="resident", projection_profile=profile
            )
    for options in (
        dict(weight_mode="streaming", dtype=torch.float16),
        dict(weight_mode="resident", dtype=torch.bfloat16),
        dict(weight_mode="resident", dtype=torch.float16, _stage_range=(0, 2)),
    ):
        with pytest.raises(ValueError, match="complete resident FP16"):
            NativeQwenModel.from_q8(
                weights, backend, projection_profile="q8-residual", **options
            )
    backend._native = SimpleNamespace()
    with pytest.raises(RuntimeError, match="exports absent"):
        NativeQwenModel.from_q8(
            weights,
            backend,
            weight_mode="resident",
            dtype=torch.float16,
            projection_profile="q8-residual",
        )


@pytest.fixture
def engine(monkeypatch):
    library = os.environ.get("VENTURI_PROFILE_LIBRARY")
    if not library:
        pytest.skip("explicit new model-profile library required")
    backend = CudaBackend(library)
    original = batch_checks.load

    def profiled(*args, **kwargs):
        model = original(*args, profile="q8-residual", **kwargs)
        info = model.projection_info
        assert model.projection_profile == "q8-residual"
        assert info["activation_quantization_passes"] == 2
        assert info["primary_projection_workspace_bytes"] > 0
        return model

    monkeypatch.setattr(batch_checks, "load", profiled)
    return backend, torch.float16, "cuda:0"


@pytest.mark.parametrize("tp", [False, True])
@pytest.mark.parametrize("count", [1, 2, 4, 8])
def test_profile_same_arithmetic_ragged_logits_private_state(
    tmp_path, engine, tp, count
):
    batch_checks.test_ragged_forward_exact_logits_and_state_with_inactive_slots(
        tmp_path, engine, tp, count
    )


@pytest.mark.parametrize("tp", [False, True])
def test_profile_budgets_eos_reset_lifetime_and_native_only(
    tmp_path, engine, tp, monkeypatch
):
    batch_checks.test_generate_eos_unequal_budgets_active_compaction_lifetime_no_torch(
        tmp_path, engine, tp, monkeypatch
    )


@pytest.mark.parametrize("tp", [False, True])
def test_profile_failure_poison_preflight_raw_abi_recovery(tmp_path, engine, tp):
    batch_checks.test_failure_poison_preflight_reset_and_raw_c_abi(tmp_path, engine, tp)


def test_profile_two_batch_owners_and_source_guard(tmp_path, engine):
    batch_checks.test_two_batches_share_only_immutable_model_and_source_guard(
        tmp_path, engine
    )


@pytest.mark.parametrize("tp", [False, True])
def test_profile_independent_fp64_full_fixture_unchanged_logit_gates(
    tmp_path, engine, tp
):
    backend, dtype, device = engine
    weights, raw = hybrid_fixture(tmp_path)
    ids = [1, 7, 3]
    reference = independent_forward(raw, weights.config, ids, tied=False)
    check = load_tool("qualify_qwen_streaming").logits_comparison
    for profile in ("fp16", "q8-residual"):
        model = NativeQwenModel.from_q8(
            weights,
            backend,
            weight_mode="resident",
            dtype=dtype,
            device=device,
            max_tokens=8,
            reserve_bytes=0,
            projection_profile=profile,
            **(
                dict(tensor_parallel_device="cuda:1", tensor_parallel_transport="host")
                if tp
                else {}
            ),
        )
        session = model.create_session(32)
        try:
            actual = session.forward(ids, all_logits=True)
            assert all(check(a, b)["passed"] for a, b in zip(actual, reference))
            info = session.info()
            assert info["projection_profile"] == profile
            assert bool(info["peer_projection_workspace_bytes"]) == (
                tp and profile != "fp16"
            )
        finally:
            session.close()
            model.close()


@pytest.fixture
def mtp_profile_engine(engine, monkeypatch):
    original = mtp_checks.load_models

    def profiled(*args, **kwargs):
        return original(*args, profile="q8-residual", **kwargs)

    monkeypatch.setattr(mtp_checks, "load_models", profiled)
    return engine


@pytest.mark.parametrize("mode", ["single", "tp"])
@pytest.mark.parametrize("forced", ["native", "accept", "reject"])
def test_profile_mtp_accept_reject_eos_target_cache_bytes(
    tmp_path, mtp_profile_engine, mode, forced
):
    mtp_checks.test_mtp_exact_target_output_state_eos_and_limits(
        tmp_path, mtp_profile_engine, mode, forced
    )


@pytest.mark.parametrize("mode", ["single", "tp"])
def test_profile_mtp_failure_reset_lifetime(
    tmp_path, mtp_profile_engine, mode, monkeypatch
):
    mtp_checks.test_mtp_failure_reset_lifetime_and_native_generation(
        tmp_path, mtp_profile_engine, mode, monkeypatch
    )
