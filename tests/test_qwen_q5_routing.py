"""Native Q5 model-loop routing, fallback, state and lifecycle regression tests."""

import os

import pytest
import torch

from venturi.models.qwen import qwen_q5


def test_q5_control_rejects_invalid_options_before_native_access():
    model = object()
    for route in ("auto", None, 1):
        with pytest.raises(ValueError, match="route"):
            qwen_q5.configure(model, route)
    for budget in (True, 0, 2**31):
        with pytest.raises(ValueError, match="budget"):
            qwen_q5.configure(model, "cublas", budget)


def fixture(tmp_path):
    path = os.environ.get("VENTURI_QWEN_LIBRARY")
    ops = os.environ.get("VENTURI_OPS_LIBRARY")
    if not path or not ops or not torch.cuda.is_available():
        pytest.skip("Explicit SM70 ops/Qwen pair required")
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("Native Q5 adaptive service qualification is SM70 only")
    from test_qwen_model import hybrid_fixture
    from test_row_codec_execution import _codec_artifact_from_q8
    from venturi.models import QwenWeights, NativeQwenModel
    from venturi.models.qwen.native_runtime import QwenRuntime
    from venturi.ops import CudaBackend

    source, _ = hybrid_fixture(tmp_path)
    output = _codec_artifact_from_q8(
        source.source, tmp_path / "q5.safetensors", "q5_g64_fp16"
    )
    return NativeQwenModel.from_q8(
        QwenWeights.open(output),
        QwenRuntime(path, CudaBackend(ops)),
        weight_mode="resident",
        dtype=torch.float16,
        device="cuda:0",
        max_tokens=16,
        reserve_bytes=0,
    )


def test_native_q5_route_calibration_state_batch_and_fallback(tmp_path):
    model = fixture(tmp_path)
    try:
        info = model.configure_q5_linear("cublas", workspace_limit_bytes=8 * 1024**2)
        assert info["version"] == 2 and info["mode"] == "cublas"
        for rows in ([], [True], [1, 1], [17]):
            with pytest.raises(ValueError):
                model.calibrate_q5_linear(rows)
        info = model.calibrate_q5_linear([1, 2, 8, 16], repeats=3, iterations=2)
        assert info["calibration"]
        assert all(
            r["routes"]["cublas"]["eligible_and_correct"] for r in info["calibration"]
        )
        assert set(r["tokens"] for r in info["calibration"]) == {1, 2, 8, 16}
        assert all(
            r["selected"] == "cublas" for r in info["calibration"] if r["tokens"] > 8
        )
        with pytest.raises(RuntimeError, match="immutable"):
            model.configure_q5_linear("cublas", workspace_limit_bytes=16 * 1024**2)
        session = model.create_session(24)
        try:
            with pytest.raises(RuntimeError, match="before requests"):
                model.calibrate_q5_linear([1])
            baseline = session.forward([1, 2, 3])
            state = session.snapshot()
            session.reset()
            model.configure_q5_linear("adaptive", workspace_limit_bytes=8 * 1024**2)
            actual = session.forward([1, 2, 3])
            assert model.q5_linear_info()["guarded_cublas_calls"] > 0
            torch.testing.assert_close(actual, baseline, rtol=0.01, atol=0.01)
            snapshot = session.snapshot()
            for kind in ("gdn", "kv"):
                for a, b in zip(snapshot[kind], state[kind]):
                    if a is not None:
                        for x, y in zip(a, b):
                            assert torch.equal(x, y)
            generated = []
            for route in ("cublas", "adaptive"):
                session.reset()
                model.configure_q5_linear(route, workspace_limit_bytes=8 * 1024**2)
                generated.append(session.generate([1, 2], max_new_tokens=4))
            assert generated[0] == generated[1]
        finally:
            session.close()
        batch = model.create_batch([16] * 3)
        try:
            logits = []
            for route in ("cublas", "adaptive"):
                batch.reset()
                model.configure_q5_linear(route, workspace_limit_bytes=8 * 1024**2)
                logits.append(batch.forward([[1, 2], [3, 4]]))
            torch.testing.assert_close(logits[0], logits[1], rtol=0.01, atol=0.01)
            batch.reset()
            batch.forward([[1], [2], [3]])
            assert model.q5_linear_info()["uncalibrated_fallback_calls"] > 0
        finally:
            batch.close()
    finally:
        model.close()
    with pytest.raises(RuntimeError, match="closed"):
        model.q5_linear_info()


def test_q5_projection_diagnostics_do_not_claim_q8_storage():
    from types import SimpleNamespace as NS
    from venturi.models.qwen_load.verify import projection_info

    spec = NS(
        dtype=torch.float16,
        projection_profile="fp16",
        tensor_parallel_device=None,
        column_profile=False,
        dequant_resident=False,
        weights=NS(plan=NS(bindings=[NS(format="q5_g64_fp16"), NS(format="fp32")])),
    )
    info = projection_info(spec, {}, 0, None)
    assert "Q5 G64" in info["projection_precision_scope"]
    assert "Q8" not in info["projection_precision_scope"]
