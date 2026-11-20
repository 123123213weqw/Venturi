"""Two-device row shards preserve complete dot products and request state."""

import os
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from test_fp32_parameters import mixed_port as base_port
from test_qwen_model import hybrid_fixture
from venturi.models import NativeQwenModel


@pytest.fixture(params=["native-fp16", "native-bf16"])
def parallel_engine(request):
    if os.environ.get("VENTURI_TP_TEST") != "1":
        pytest.skip("explicit two-device tensor-parallel test opt-in required")
    b, dtype, dev = base_port.__wrapped__(request)
    if torch.cuda.device_count() < 2:
        pytest.skip("two CUDA devices required")
    return b, dtype, dev


def same_state(a, b):
    for kind in ("gdn", "kv"):
        for u, v in zip(a[kind], b[kind]):
            assert (u is None) == (v is None)
            if u is not None:
                for x, y in zip(u, v):
                    assert torch.equal(x.view(torch.uint8), y.view(torch.uint8))


@pytest.mark.parametrize("tied", [False, True])
def test_parallel_logits_tails_chunks_and_exact_state(tmp_path, parallel_engine, tied):
    b, dtype, dev = parallel_engine
    weights, _ = hybrid_fixture(tmp_path, tied=tied)
    kwargs = dict(
        weight_mode="resident", dtype=dtype, device=dev, max_tokens=3, reserve_bytes=0
    )
    first = NativeQwenModel.from_q8(weights, b, **kwargs)
    second = NativeQwenModel.from_q8(
        weights,
        b,
        tensor_parallel_device="cuda:1",
        tensor_parallel_transport="host",
        **kwargs,
    )
    a, c = first.create_session(24), second.create_session(24)
    try:
        initial = c.info()
        assert initial["peer_device_index"] == 1 and initial["peer_device_bytes"] > 0
        for ids in ([1, 2, 3, 4, 5], [6], [7, 8]):
            ta, tb = [], []
            x, y = (
                a.forward(ids, all_logits=True, trace=ta),
                c.forward(ids, all_logits=True, trace=tb),
            )
            assert torch.equal(x.view(torch.uint8), y.view(torch.uint8))
            assert len(ta) == len(tb)
            assert all(
                torch.equal(
                    u["hidden"].view(torch.uint8), v["hidden"].view(torch.uint8)
                )
                for u, v in zip(ta, tb)
            )
            same_state(a.snapshot(), c.snapshot())
        assert (
            c.info()["model_weight_upload_bytes"]
            == initial["model_weight_upload_bytes"]
        )
        assert (
            c.info()["peer_input_copy_bytes"] > 0
            and c.info()["peer_output_copy_bytes"] > 0
        )
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
        first.close()
        second.close()


def test_parallel_failure_reset_lifetime_and_no_torch(
    tmp_path, parallel_engine, monkeypatch
):
    b, dtype, dev = parallel_engine
    weights, _ = hybrid_fixture(tmp_path)
    kwargs = dict(
        weight_mode="resident", dtype=dtype, device=dev, max_tokens=3, reserve_bytes=0
    )
    reference = NativeQwenModel.from_q8(weights, b, **kwargs)
    model = NativeQwenModel.from_q8(
        weights,
        b,
        tensor_parallel_device="cuda:1",
        tensor_parallel_transport="host",
        **kwargs,
    )
    baseline, s = reference.create_session(24), model.create_session(24)
    expected = baseline.generate([1, 2, 3], max_new_tokens=5)
    try:
        s._inject_failure(layer=2, position=0)
        with pytest.raises(RuntimeError, match="injected"):
            s.forward([1, 2, 3])
        assert s.failed and s.position == 0
        s.reset()
        model.close()

        def forbidden(*args, **kwargs):
            raise AssertionError("Native TP generation called Torch math")

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
            assert s.generate([1, 2, 3], max_new_tokens=5) == expected
        same_state(s.snapshot(), baseline.snapshot())
    finally:
        baseline.close()
        s.close()
        reference.close()
        model.close()


def test_parallel_shared_scratch_threaded_requests_and_invalid_peer(
    tmp_path, parallel_engine
):
    b, dtype, dev = parallel_engine
    weights, _ = hybrid_fixture(tmp_path)
    kwargs = dict(
        weight_mode="resident", dtype=dtype, device=dev, max_tokens=3, reserve_bytes=0
    )
    with pytest.raises(ValueError, match="different"):
        NativeQwenModel.from_q8(weights, b, tensor_parallel_device="cuda:0", **kwargs)
    with pytest.raises(ValueError, match="resident"):
        NativeQwenModel.from_q8(
            weights,
            b,
            tensor_parallel_device="cuda:1",
            **{**kwargs, "weight_mode": "streaming"},
        )
    model = NativeQwenModel.from_q8(
        weights,
        b,
        tensor_parallel_device="cuda:1",
        tensor_parallel_transport="host",
        **kwargs,
    )
    sessions = [model.create_session(24) for _ in range(4)]
    prompts = [[1, 2, 3], [4, 5], [6], [7, 8, 9, 1]]
    try:
        expected = [
            s.generate(ids, max_new_tokens=5) for s, ids in zip(sessions, prompts)
        ]
        for s in sessions:
            s.reset()
        with ThreadPoolExecutor(max_workers=4) as pool:
            jobs = [
                pool.submit(s.generate, ids, max_new_tokens=5)
                for s, ids in zip(sessions, prompts)
            ]
            assert [j.result() for j in jobs] == expected
    finally:
        for s in sessions:
            s.close()
        model.close()


def test_parallel_peer_fails_fast_when_copy_probe_is_known_broken(
    tmp_path, parallel_engine
):
    if os.environ.get("VENTURI_EXPECT_P2P_FAILURE") != "1":
        pytest.skip("explicit known-broken byte-probe environment required")
    b, dtype, dev = parallel_engine
    weights, _ = hybrid_fixture(tmp_path)
    with pytest.raises(RuntimeError, match="byte probe failed"):
        NativeQwenModel.from_q8(
            weights,
            b,
            weight_mode="resident",
            dtype=dtype,
            device=dev,
            tensor_parallel_device="cuda:1",
            tensor_parallel_transport="peer",
            max_tokens=3,
            reserve_bytes=0,
        )
