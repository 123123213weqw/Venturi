"""Opt-in tiny native Q8 continuous scheduling versus scalar trajectories/state."""
import pytest
import torch

from test_qwen_batch import batch_engine, load
from test_qwen_tensor_parallel import same_state
from venturi.runtime import ContinuousScheduler, ContinuousWorker


@pytest.mark.cuda
@pytest.mark.parametrize("metadata_mode", ["checked-json", "sealed"])
@pytest.mark.parametrize("prefill_policy", ["round-robin", "balanced", "balanced-short"])
@pytest.mark.parametrize("tp,profile", [
    (False, "fp16"), (True, "fp16"),
    (False, "q8-residual"), (True, "q8-residual"),
    (True, "q8-residual-column-mlp"),
    (True, "q8-residual-column-mlp-head-gdn"),
])
def test_native_continuous_completion_refill_and_long_prefill_private_states(tmp_path, batch_engine, tp, profile, metadata_mode, prefill_policy, monkeypatch):
    if profile.endswith("head-gdn"):
        from test_native_head_gdn import eligible_mtp, load as head_load
        weights, _, _, _ = eligible_mtp(tmp_path / "eligible-head")
        model = head_load(weights, batch_engine, metadata_mode=metadata_mode)
    else:
        model = load(tmp_path, batch_engine, tp, profile=profile, metadata_mode=metadata_mode)
    batch = model.create_batch([64] * 2)
    scheduler = ContinuousScheduler(batch, token_budget=8, prefill_chunk_size=3, prefill_policy=prefill_policy)
    cases = [([1, 2], 2), ([3, 4, 5, 6, 7, 1, 2], 9), ([6, 5, 4, 3, 2], 3)]
    refs = [model.create_session(64) for _ in cases]
    try:
        expected = [s.generate(ids, max_new_tokens=n, eos_token_ids=[])
                    for s, (ids, n) in zip(refs, cases)]
        a = scheduler.submit(cases[0][0], max_new_tokens=cases[0][1])
        b = scheduler.submit(cases[1][0], max_new_tokens=cases[1][1])
        events = []
        for _ in range(10):
            events.extend(scheduler.tick())
            if a.future.done():
                break
        assert a.future.result().output_ids == tuple(expected[0])
        assert not b.future.done()
        same_state(batch.snapshot(0), refs[0].snapshot())
        c = scheduler.submit(cases[2][0], max_new_tokens=cases[2][1])

        def forbidden(*args, **kwargs):
            raise AssertionError("Continuous native scheduling used Torch allocation/math")

        for _ in range(30):
            with monkeypatch.context() as m:
                for name in ("empty", "zeros", "mm", "matmul", "argmax"):
                    m.setattr(torch, name, forbidden)
                current = scheduler.tick()
            events.extend(current)
            if any(e.request_id == c.request_id and e.finish_reason for e in current):
                same_state(batch.snapshot(0), refs[2].snapshot())
            if any(e.request_id == b.request_id and e.finish_reason for e in current):
                same_state(batch.snapshot(1), refs[1].snapshot())
            if b.future.done() and c.future.done():
                break
        assert b.future.result().output_ids == tuple(expected[1])
        assert c.future.result().output_ids == tuple(expected[2])
        assert scheduler.info()["mixed_iterations"] > 0
        assert scheduler.info()["completed_requests"] == 3
        assert scheduler.info()["emitted_tokens"] == sum(n for _, n in cases)
        for ticket, output in zip((a, b, c), expected):
            assert [e.token_id for e in events if e.request_id == ticket.request_id] == output
    finally:
        scheduler.close()
        batch.close()
        for s in refs:
            s.close()
        model.close()


@pytest.mark.cuda
@pytest.mark.parametrize("metadata_mode", ["checked-json", "sealed"])
@pytest.mark.parametrize("prefill_policy", ["round-robin", "balanced", "balanced-short"])
@pytest.mark.parametrize("tp,profile", [
    (False, "fp16"), (True, "fp16"),
    (False, "q8-residual"), (True, "q8-residual"),
    (True, "q8-residual-column-mlp"),
    (True, "q8-residual-column-mlp-head-gdn"),
])
def test_native_worker_thread_owns_device_and_private_state(tmp_path, batch_engine, tp, profile, metadata_mode, prefill_policy, monkeypatch):
    import threading

    if profile.endswith("head-gdn"):
        from test_native_head_gdn import eligible_mtp, load as head_load
        weights, _, _, _ = eligible_mtp(tmp_path / "eligible-head")
        model = head_load(weights, batch_engine, metadata_mode=metadata_mode)
    else:
        model = load(tmp_path, batch_engine, tp, profile=profile, metadata_mode=metadata_mode)
    batch = model.create_batch([64] * 2)
    refs = [model.create_session(64) for _ in range(2)]
    scheduler = ContinuousScheduler(batch, token_budget=8, prefill_chunk_size=3, prefill_policy=prefill_policy)
    worker = ContinuousWorker(scheduler)
    prompts = [[1, 2], [3, 4, 5, 6, 7, 1, 2]]
    owner_ids = set()
    step = batch.step

    def owned(*args, **kwargs):
        owner_ids.add(threading.get_ident())
        return step(*args, **kwargs)

    monkeypatch.setattr(batch, "step", owned)
    try:
        expected = [s.generate(p, max_new_tokens=5, eos_token_ids=[]) for s, p in zip(refs, prompts)]
        # Queue before explicit start to fix slot assignment for private-state
        # checks; the worker still exclusively owns all resets and steps.
        tickets = [scheduler.submit(p, max_new_tokens=5) for p in prompts]

        def forbidden(*args, **kwargs):
            raise AssertionError("Continuous worker used Torch allocation/math")

        with monkeypatch.context() as m:
            for name in ("empty", "zeros", "mm", "matmul", "argmax"):
                m.setattr(torch, name, forbidden)
            worker.start()
            actual = [t.future.result(timeout=60).output_ids for t in tickets]
        assert actual == [tuple(x) for x in expected]
        assert len(owner_ids) == 1 and threading.get_ident() not in owner_ids
        for slot, ref in enumerate(refs):
            same_state(batch.snapshot(slot), ref.snapshot())
    finally:
        worker.close(timeout=60)
        batch.close()
        for ref in refs:
            ref.close()
        model.close()


@pytest.mark.cuda
@pytest.mark.parametrize("head", [False, True])
def test_sealed_target_preserves_mtp_binding_and_private_state(tmp_path, batch_engine, head):
    from test_native_head_gdn import eligible_mtp, load as head_load
    from test_qwen_mtp import mtp_fixture
    from venturi.models import NativeQwenMTPModel

    if head:
        weights, draft, _, _ = eligible_mtp(tmp_path / "eligible-head")
        target = head_load(weights, batch_engine, metadata_mode="sealed")
    else:
        weights, draft, _, _ = mtp_fixture(tmp_path)
        from venturi.models import NativeQwenModel
        backend, dtype, device = batch_engine
        target = NativeQwenModel.from_q8(weights, backend, weight_mode="resident",
                    dtype=dtype, device=device, max_tokens=8, reserve_bytes=0,
                    metadata_mode="sealed")
    model = request = reference = None
    try:
        model = NativeQwenMTPModel.from_q8(target, draft, reserve_bytes=0)
        request, reference = model.create_session(64), target.create_session(64)
        expected = reference.generate([1, 2, 3], max_new_tokens=6, eos_token_ids=[])
        actual = request.generate([1, 2, 3], max_new_tokens=6, eos_token_ids=[])
        assert actual == expected
        same_state(request.snapshot(), reference.snapshot())
        target._weights.artifact.path.touch()
        with pytest.raises(RuntimeError, match="source changed"):
            request.generate([1], max_new_tokens=1)
    finally:
        for item in (request, reference, model, target):
            if item is not None:
                item.close()
