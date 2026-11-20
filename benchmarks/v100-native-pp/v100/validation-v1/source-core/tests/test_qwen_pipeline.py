"""Native stage-local ownership, exact state parity and concurrent requests."""

import os
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from test_qwen_model import hybrid_fixture
from test_fp32_parameters import mixed_port as base_port
from test_qwen_tensor_parallel import same_state
from venturi.models import NativeQwenModel, NativeQwenPipelineModel
from venturi.ops import TorchBackend


def test_pipeline_requires_explicit_native_backend():
    with pytest.raises(TypeError, match="explicit CudaBackend"):
        NativeQwenPipelineModel.from_q8(
            None, TorchBackend(), weight_mode="resident", pipeline_device="cuda:1"
        )


@pytest.fixture(params=["native-fp16", "native-bf16"])
def pp_engine(request):
    if os.environ.get("VENTURI_PP_TEST") != "1":
        pytest.skip("explicit two-device PP opt-in required")
    b, dtype, device = base_port.__wrapped__(request)
    if torch.cuda.device_count() < 2:
        pytest.skip("two GPUs required")
    return b, dtype, device


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("split", [1, 2, 3])
def test_pipeline_exact_logits_trace_state_and_continuation(
    tmp_path, pp_engine, tied, split
):
    b, dtype, device = pp_engine
    weights, _ = hybrid_fixture(tmp_path, tied=tied)
    kw = dict(
        weight_mode="resident",
        dtype=dtype,
        device=device,
        max_tokens=3,
        reserve_bytes=0,
    )
    reference = NativeQwenModel.from_q8(weights, b, **kw)
    pipeline = NativeQwenPipelineModel.from_q8(
        weights, b, pipeline_device="cuda:1", split_layer=split, **kw
    )
    a, c = reference.create_session(24), pipeline.create_session(24)
    try:
        same_state(a.snapshot(), c.snapshot())
        for ids in ([1, 2, 3, 4, 5], [6], [7, 8]):
            x, y = [], []
            u, v = (
                a.forward(ids, all_logits=True, trace=x),
                c.forward(ids, all_logits=True, trace=y),
            )
            assert torch.equal(u.view(torch.uint8), v.view(torch.uint8))
            assert len(x) == len(y)
            assert all(
                torch.equal(
                    p["hidden"].view(torch.uint8), q["hidden"].view(torch.uint8)
                )
                for p, q in zip(x, y)
            )
            same_state(a.snapshot(), c.snapshot())
        info = c.info()
        assert info["pipeline_split_layer"] == split and info["pipeline_handoffs"] == 4
        assert info["peer_request_device_bytes"] > 0
        a.reset()
        c.reset()
        assert a.generate([1, 2, 3], max_new_tokens=6) == c.generate(
            [1, 2, 3], max_new_tokens=6
        )
        same_state(a.snapshot(), c.snapshot())
        assert torch.cuda.current_device() == 0
    finally:
        a.close()
        c.close()
        reference.close()
        pipeline.close()


@pytest.mark.parametrize("count", [1, 2, 4, 8])
def test_pipeline_concurrent_request_isolation(tmp_path, pp_engine, count):
    b, dtype, device = pp_engine
    weights, _ = hybrid_fixture(tmp_path)
    pipeline = NativeQwenPipelineModel.from_q8(
        weights,
        b,
        pipeline_device="cuda:1",
        split_layer=2,
        weight_mode="resident",
        dtype=dtype,
        device=device,
        max_tokens=3,
        reserve_bytes=0,
    )
    sessions = [pipeline.create_session(24) for _ in range(count)]
    prompts = [[1 + i, 2, 3] for i in range(count)]
    try:
        expected = [
            s.generate(ids, max_new_tokens=5) for s, ids in zip(sessions, prompts)
        ]
        states = [s.snapshot() for s in sessions]
        for s in sessions:
            s.reset()
        with ThreadPoolExecutor(max_workers=count) as pool:
            jobs = [
                pool.submit(s.generate, ids, max_new_tokens=5)
                for s, ids in zip(sessions, prompts)
            ]
            assert [j.result(timeout=60) for j in jobs] == expected
        for s, state in zip(sessions, states):
            same_state(s.snapshot(), state)
    finally:
        for s in sessions:
            s.close()
        pipeline.close()


@pytest.mark.parametrize("layer", [0, 3])
def test_pipeline_failure_reset_lifetime_and_no_python_math(
    tmp_path, pp_engine, monkeypatch, layer
):
    b, dtype, device = pp_engine
    weights, _ = hybrid_fixture(tmp_path)
    kw = dict(
        weight_mode="resident",
        dtype=dtype,
        device=device,
        max_tokens=3,
        reserve_bytes=0,
    )
    reference = NativeQwenModel.from_q8(weights, b, **kw)
    pipeline = NativeQwenPipelineModel.from_q8(
        weights, b, pipeline_device="cuda:1", **kw
    )
    a, c = reference.create_session(24), pipeline.create_session(24)
    expected = a.generate([1, 2, 3], max_new_tokens=5)
    try:
        c._inject_failure(layer=layer, position=0)
        with pytest.raises(RuntimeError, match="injected"):
            c.forward([1, 2, 3])
        assert c.failed and c.position == 0
        with pytest.raises(RuntimeError, match="reset required"):
            c.generate([1], max_new_tokens=1)
        c.reset()
        pipeline.close()
        with pytest.raises(RuntimeError, match="closed"):
            pipeline.create_session(24)

        def forbidden(*a, **k):
            raise AssertionError("Native pipeline called Torch math")

        with monkeypatch.context() as patch:
            for name in (
                "matmul",
                "mm",
                "bmm",
                "argmax",
                "cos",
                "sin",
                "empty",
                "zeros",
            ):
                patch.setattr(torch, name, forbidden)
            for name in ("matmul", "__matmul__", "argmax", "float", "to", "cpu"):
                patch.setattr(torch.Tensor, name, forbidden)
            assert c.generate([1, 2, 3], max_new_tokens=5) == expected
        same_state(a.snapshot(), c.snapshot())
    finally:
        a.close()
        c.close()
        reference.close()
        pipeline.close()
