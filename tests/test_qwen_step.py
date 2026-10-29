"""One native mixed iteration, separate request state and no logits round-trip."""
import threading
from types import SimpleNamespace

import pytest
import torch

from test_qwen_batch import batch_engine, load
from test_qwen_tensor_parallel import same_state
from venturi.models.qwen_batch import NativeQwenBatchSession


def metadata_batch():
    batch = NativeQwenBatchSession()
    batch.capacities = (16, 16, 16)
    batch.model = SimpleNamespace(_cfg=dict(vocab_size=8), _max_tokens=4)
    return batch


@pytest.mark.parametrize("options", [
    dict(prompts=[]), dict(prompts=[[]]), dict(prompts=[[8]]),
    dict(prompts=[[1] * 5]), dict(prompts=[[1]], slots=[True]),
    dict(prompts=[[1]], sample_slots=[1]),
    dict(prompts=[[1]], sample_slots=[0, 0]),
    dict(prompts=[[1]], sample_slots="0"),
    dict(prompts=[[1]], decode_slots=[1]),
    dict(prompts=[[1, 2]], decode_slots=[0]),
    dict(prompts=[[1]], decode_slots=[True]),
    dict(prompts=[[1]], decode_slots=[0, 0]),
])
def test_step_invalid_metadata_never_reaches_native(options):
    batch = metadata_batch()
    with pytest.raises(ValueError):
        batch.step(**options)


def test_step_rejects_mtp_handle_before_any_access():
    class NotAnOrdinaryBatch(NativeQwenBatchSession):
        pass

    with pytest.raises(ValueError, match="not MTP"):
        NotAnOrdinaryBatch().step([[1]])


def test_step_missing_optional_export_is_explicit_not_fallback():
    batch = metadata_batch()
    batch._lock, batch._check = threading.RLock(), lambda: None
    batch.model._backend = SimpleNamespace(_native=SimpleNamespace())
    with pytest.raises(RuntimeError, match="rebuild explicitly"):
        batch.step([[1]])


def test_step_sample_order_partial_prefill_and_metadata_only_submission():
    batch = metadata_batch()
    batch._handle, batch._lock, batch._check = 123, threading.RLock(), lambda: None
    calls = []

    class Function:
        def __call__(self, owner, slots, count, tokens, offsets, samples, ns, decodes, nd, output, elements):
            calls.append((owner, list(slots), list(tokens), list(offsets), list(samples), list(decodes), elements))
            for i, slot in enumerate(samples):
                output[i] = (slot + 5) % 8
            return 0

    def check(code):
        assert code == 0

    native = SimpleNamespace(venturi_qwen_batch_step=Function())
    batch.model._backend = SimpleNamespace(_native=native, _check=check)
    assert batch.step([[1], [2, 3]], slots=[2, 0], sample_slots=[0, 2], decode_slots=[2]) == [5, 7]
    assert calls[-1] == (123, [2, 0], [1, 2, 3], [0, 1, 3], [0, 2], [2], 2)
    assert batch.step([[4]], slots=[1], sample_slots=[]) == []
    assert calls[-1][-3:] == ([], [], 0)


@pytest.mark.cuda
@pytest.mark.parametrize("tp", [False, True])
def test_native_mixed_steps_exact_scalar_ids_private_states_and_slot_reuse(tmp_path, batch_engine, tp, monkeypatch):
    model = load(tmp_path, batch_engine, tp)
    batch = model.create_batch([48] * 3)
    scalar = [model.create_session(48) for _ in range(3)]
    try:
        # A partial prefill is intentionally not sampled. Slots have different
        # positions; the next call combines continuing prefill and decode rows.
        operations = [
            ([2, 0], [[1, 2, 3], [4, 5]], [2], []),
            ([0, 1, 2], [[6, 7], [2], [3]], [2, 0], [2]),
            ([1, 0], [[3, 4, 5], [6]], [0, 1], [0]),
            ([2], [[7]], [], [2]),
        ]
        for slots, inputs, samples, decodes in operations:
            expected = {}
            for slot, ids in zip(slots, inputs):
                expected[slot] = int(torch.argmax(scalar[slot].forward(ids)))

            def forbidden(*args, **kwargs):
                raise AssertionError("Native step used Torch math/allocation fallback")

            with monkeypatch.context() as m:
                for name in ("empty", "zeros", "mm", "matmul", "argmax"):
                    m.setattr(torch, name, forbidden)
                actual = batch.step(inputs, slots=slots, sample_slots=samples, decode_slots=decodes)
            assert actual == [expected[s] for s in samples]
            for slot in range(3):
                same_state(batch.snapshot(slot), scalar[slot].snapshot())
        batch.reset(slots=[0])
        scalar[0].reset()
        actual = batch.step([[7, 6], [5]], slots=[0, 2], decode_slots=[2])
        expected = [int(torch.argmax(scalar[s].forward(ids))) for s, ids in zip([0, 2], [[7, 6], [5]])]
        assert actual == expected
        for slot in range(3):
            same_state(batch.snapshot(slot), scalar[slot].snapshot())
    finally:
        batch.close()
        for session in scalar:
            session.close()
        model.close()


@pytest.mark.cuda
def test_native_step_invalid_preflight_and_execution_poison_are_distinct(tmp_path, batch_engine):
    model = load(tmp_path, batch_engine, False)
    batch = model.create_batch([32] * 3)
    try:
        batch.step([[1], [2], [3]])
        before = [batch.snapshot(i) for i in range(3)]
        for options in (dict(prompts=[[1] * 9]), dict(prompts=[[1]], sample_slots=[2])):
            with pytest.raises(ValueError):
                batch.step(**options)
            for i in range(3):
                same_state(before[i], batch.snapshot(i))
        batch._inject_failure(slot=0, layer=0, position=1)
        with pytest.raises(RuntimeError, match="injected"):
            batch.step([[4], [5]], slots=[0, 1], decode_slots=[0, 1])
        info = batch.info()
        assert [r["failed"] for r in info["requests"]] == [True, True, False]
        same_state(before[2], batch.snapshot(2))
        with pytest.raises(RuntimeError):
            batch.step([[4]], slots=[0])
        batch.reset(slots=[0, 1])
        assert len(batch.step([[1], [2]], slots=[0, 1])) == 2
    finally:
        batch.close()
        model.close()
