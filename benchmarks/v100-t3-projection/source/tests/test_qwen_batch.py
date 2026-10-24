"""Grouped projections must never merge independent recurrent/attention sequences."""

import ctypes as C
import os
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from test_fp32_parameters import mixed_port as base_port
from test_qwen_model import hybrid_fixture
from test_qwen_tensor_parallel import same_state
from venturi.models import NativeQwenBatchSession, NativeQwenModel


def test_batch_python_slot_prompt_and_capacity_preflight():
    batch = NativeQwenBatchSession()
    batch.capacities = (12, 16, 20)
    batch.model = SimpleNamespace(_cfg=dict(vocab_size=8))
    for slots in ([], [0, 0], [-1], [3], [True], "0"):
        with pytest.raises(ValueError):
            batch._slots(slots)
    selected, tokens, offsets = batch._inputs([[1, 2], [3]], [2, 0])
    assert (
        list(selected) == [2, 0]
        and list(tokens) == [1, 2, 3]
        and list(offsets) == [0, 2, 3]
    )
    for prompts, slots in (([], None), ([[]], None), ([[9]], None), ([[1]], [0, 1])):
        with pytest.raises(ValueError):
            batch._inputs(prompts, slots)
    model = NativeQwenModel()
    model._mode, model._max_tokens, model._cfg = (
        "resident",
        4,
        dict(max_position_embeddings=16),
    )
    for capacities in ([], [12] * 5, [0], [17], [True]):
        with pytest.raises(ValueError):
            NativeQwenBatchSession.create(model, capacities)
    with pytest.raises(ValueError):
        NativeQwenBatchSession.create(object(), [12])


@pytest.fixture(params=["native-fp16", "native-bf16"])
def batch_engine(request):
    if os.environ.get("VENTURI_BATCH_TEST") != "1":
        pytest.skip("explicit native batching test opt-in required")
    return base_port.__wrapped__(request)


def load(tmp_path, engine, tp, *, tied=False, profile="fp16"):
    backend, dtype, device = engine
    if tp and torch.cuda.device_count() < 2:
        pytest.skip("two devices required")
    weights, _ = hybrid_fixture(tmp_path, tied=tied)
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
    return model


def prompts(model, count):
    return [
        [(7 * i + j + 1) % model.config["vocab_size"] for j in range(1 + 2 * (i % 6))]
        for i in range(count)
    ]


@pytest.mark.parametrize("tp", [False, True])
@pytest.mark.parametrize("count", [1, 2, 4, 8])
def test_ragged_forward_exact_logits_and_state_with_inactive_slots(
    tmp_path, batch_engine, tp, count
):
    model = load(tmp_path, batch_engine, tp, tied=count == 4)
    batch, scalar = (
        model.create_batch([64] * count),
        [model.create_session(64) for _ in range(count)],
    )
    try:
        assert all(kv is None or kv[0].numel() == 0 for kv in batch.snapshot(0)["kv"])
        groups = prompts(model, count)
        actual = batch.forward(groups)
        expected = torch.stack([s.forward(ids) for s, ids in zip(scalar, groups)])
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        for i, s in enumerate(scalar):
            same_state(batch.snapshot(i), s.snapshot())
            assert batch.info()["requests"][i]["position"] == len(groups[i])
        slots = list(dict.fromkeys([count - 1, 0]))
        groups = [
            [(i + j + 3) % model.config["vocab_size"] for j in range(2 + i)]
            for i in range(len(slots))
        ]
        actual = batch.forward(groups, slots=slots)
        expected = torch.stack(
            [scalar[i].forward(ids) for i, ids in zip(slots, groups)]
        )
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        for i, s in enumerate(scalar):
            same_state(batch.snapshot(i), s.snapshot())
        owned = actual.clone()
        batch.forward([[1]], slots=[0])
        assert torch.equal(owned, actual)
        info = batch.info()
        assert info["shared_batch_workspace_device_bytes"] > 0
        assert info["all_request_device_bytes"] == sum(
            x["request_device_bytes"] for x in info["requests"]
        )
    finally:
        batch.close()
        for s in scalar:
            s.close()
        model.close()


@pytest.mark.parametrize("tp", [False, True])
def test_generate_eos_unequal_budgets_active_compaction_lifetime_no_torch(
    tmp_path, batch_engine, tp, monkeypatch
):
    model = load(tmp_path, batch_engine, tp)
    batch, scalar = (
        model.create_batch([64] * 8),
        [model.create_session(64) for _ in range(8)],
    )
    groups = prompts(model, 8)
    maximum = list(range(1, 9))
    try:
        raw = [
            s.generate(ids, max_new_tokens=n, eos_token_ids=[])
            for s, ids, n in zip(scalar, groups, maximum)
        ]
        eos = [
            [] if i % 3 == 0 else [row[min(len(row) - 1, i % 3)]]
            for i, row in enumerate(raw)
        ]
        for s in scalar:
            s.reset()
        expected = [
            s.generate(ids, max_new_tokens=n, eos_token_ids=e)
            for s, ids, n, e in zip(scalar, groups, maximum, eos)
        ]
        model.close()
        with pytest.raises(RuntimeError, match="closed"):
            model.create_batch([12])

        def forbidden(*args, **kwargs):
            raise AssertionError("Native batch generation called Torch math")

        with monkeypatch.context() as patch:
            for name in (
                "empty",
                "zeros",
                "matmul",
                "mm",
                "bmm",
                "argmax",
                "cos",
                "sin",
            ):
                patch.setattr(torch, name, forbidden)
            for name in ("to", "cpu", "float", "matmul", "__matmul__", "argmax"):
                patch.setattr(torch.Tensor, name, forbidden)
            actual = batch.generate(groups, max_new_tokens=maximum, eos_token_ids=eos)
        assert actual == expected
        for i, s in enumerate(scalar):
            same_state(batch.snapshot(i), s.snapshot())
            row = batch.info()["requests"][i]
            assert (
                row["position"] == len(groups[i]) + len(actual[i]) - 1
                and not row["failed"]
            )
            assert (
                row["native_generation_seconds"]
                >= row["native_time_to_first_token_seconds"]
                > 0
            )
        for i in [7, 0, 3]:
            scalar[i].reset()
        batch.reset(slots=[7, 0, 3])
        expected = [scalar[i].generate([i, 1], max_new_tokens=3) for i in [7, 0, 3]]
        assert (
            batch.generate(
                [[i, 1] for i in [7, 0, 3]], max_new_tokens=3, slots=[7, 0, 3]
            )
            == expected
        )
        for i, s in enumerate(scalar):
            same_state(batch.snapshot(i), s.snapshot())
    finally:
        batch.close()
        for s in scalar:
            s.close()
        model.close()


@pytest.mark.parametrize("tp", [False, True])
def test_failure_poison_preflight_reset_and_raw_c_abi(tmp_path, batch_engine, tp):
    model = load(tmp_path, batch_engine, tp)
    batch = model.create_batch([24] * 3)
    try:
        before = batch.info()
        with pytest.raises(ValueError, match="capacity"):
            batch.generate([[1], [2], [3]], max_new_tokens=[2, 100, 2])
        assert batch.info()["requests"] == before["requests"]
        raw_output = torch.empty(2, model.config["vocab_size"])
        status = batch._native.forward(
            batch._handle,
            (C.c_int * 2)(0, 0),
            2,
            (C.c_int64 * 2)(1, 2),
            (C.c_int * 3)(0, 1, 2),
            raw_output.data_ptr(),
            2 * model.config["vocab_size"],
        )
        assert status == -1 and not any(r["failed"] for r in batch.info()["requests"])
        batch._inject_failure(slot=1, layer=2, position=0)
        with pytest.raises(RuntimeError, match="injected"):
            batch.forward([[1, 2], [3], [4, 5]])
        assert all(r["failed"] for r in batch.info()["requests"])
        with pytest.raises(RuntimeError, match="failed"):
            batch.generate([[1]], max_new_tokens=2)
        batch.reset(slots=[1])
        assert batch.generate([[3]], max_new_tokens=2, slots=[1])
        assert batch.info()["requests"][0]["failed"]
        batch.reset()
        assert not any(r["failed"] or r["position"] for r in batch.info()["requests"])
        reference = [model.create_session(24) for _ in range(3)]
        try:
            expected = [
                s.generate([i + 1], max_new_tokens=4) for i, s in enumerate(reference)
            ]
            assert (
                batch.generate([[i + 1] for i in range(3)], max_new_tokens=4)
                == expected
            )
            for i, s in enumerate(reference):
                same_state(batch.snapshot(i), s.snapshot())
        finally:
            for s in reference:
                s.close()
    finally:
        batch.close()
        model.close()


def test_two_batches_share_only_immutable_model_and_source_guard(
    tmp_path, batch_engine
):
    model = load(tmp_path, batch_engine, False)
    a, b = model.create_batch([24] * 2), model.create_batch([24] * 2)
    try:
        expected = a.generate([[1, 2], [3]], max_new_tokens=4)
        a.reset()
        with ThreadPoolExecutor(max_workers=2) as pool:
            x = pool.submit(a.generate, [[1, 2], [3]], max_new_tokens=4)
            y = pool.submit(b.generate, [[1, 2], [3]], max_new_tokens=4)
            assert x.result() == y.result() == expected
        for i in range(2):
            same_state(a.snapshot(i), b.snapshot(i))
        a.reset()
        with model._weights.artifact.files[0].path.open("ab") as f:
            f.write(b"changed")
        with pytest.raises(RuntimeError, match="source changed"):
            a.generate([[1]], max_new_tokens=2)
        assert all(not r["failed"] and r["position"] == 0 for r in a.info()["requests"])
    finally:
        a.close()
        b.close()
        model.close()
