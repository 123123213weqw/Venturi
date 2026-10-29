"""One inference owner, concurrent admission, real idle waiting and safe close."""
from concurrent.futures import ThreadPoolExecutor
import threading
import time

import pytest

from test_continuous_scheduler import Batch, reference
from venturi.runtime import ContinuousScheduler, ContinuousWorker


def test_no_thread_until_start_and_idle_wait_does_not_tick():
    batch = Batch()
    scheduler = ContinuousScheduler(batch)
    worker = ContinuousWorker(scheduler)
    assert not worker.info()["started"]
    with pytest.raises(RuntimeError, match="explicitly"):
        worker.submit([1], max_new_tokens=2)
    worker.start()
    worker.start()
    try:
        time.sleep(.02)
        assert not batch.calls and not batch.resets
        tickets = [worker.submit([i], max_new_tokens=3) for i in range(1, 8)]
        for i, ticket in enumerate(tickets, 1):
            assert ticket.future.result(timeout=3).output_ids == reference([i], 3)
        count = len(batch.calls)
        time.sleep(.02)
        assert len(batch.calls) == count
    finally:
        worker.close(timeout=3)
    assert not worker.info()["alive"] and worker.info()["closed"]
    worker.close()
    with pytest.raises(RuntimeError, match="closed"):
        worker.start()


def test_concurrent_submitters_keep_private_outputs_and_do_not_run_model():
    batch = Batch(capacities=(64,) * 4)
    owner_ids = set()
    original = batch.step

    def step(*args, **kwargs):
        owner_ids.add(threading.get_ident())
        return original(*args, **kwargs)

    batch.step = step
    with ContinuousWorker(ContinuousScheduler(batch, token_budget=8, prefill_chunk_size=3)) as worker:
        def submit(i):
            prompt = [i % 31, (i + 1) % 31, (i + 2) % 31]
            ticket = worker.submit(prompt, max_new_tokens=5)
            return ticket.future.result(timeout=3).output_ids, reference(prompt, 5)

        with ThreadPoolExecutor(max_workers=8) as pool:
            for actual, expected in pool.map(submit, range(32)):
                assert actual == expected
    assert len(owner_ids) == 1 and threading.get_ident() not in owner_ids


def test_close_timeout_keeps_live_owner_and_buffers_until_boundary():
    entered, release = threading.Event(), threading.Event()
    batch = Batch()
    batch.hook = lambda: (entered.set(), release.wait(timeout=3))
    worker = ContinuousWorker(ContinuousScheduler(batch)).start()
    ticket = worker.submit([1], max_new_tokens=8)
    assert entered.wait(timeout=3)
    with pytest.raises(TimeoutError, match="still in flight"):
        worker.close(timeout=0)
    assert worker.info()["alive"] and not worker.info()["closed"]
    # Do not destroy the model/batch on an observation timeout.
    release.set()
    worker.close(timeout=3)
    assert ticket.future.cancelled()
    assert batch.states == [[], []]


def test_failed_owner_propagates_and_never_restarts_or_changes_strategy():
    batch = Batch()
    batch.fail_step = True
    worker = ContinuousWorker(ContinuousScheduler(batch)).start()
    ticket = worker.submit([1], max_new_tokens=8)
    with pytest.raises(RuntimeError, match="step fault"):
        ticket.future.result(timeout=3)
    with pytest.raises(RuntimeError, match="failure"):
        worker.close(timeout=3)
    with pytest.raises(RuntimeError, match="failed"):
        worker.submit([2], max_new_tokens=1)
    assert len(batch.calls) == 1 and not worker.info()["alive"]


def test_close_without_start_and_foreign_cancellation():
    a = ContinuousWorker(ContinuousScheduler(Batch()))
    b = ContinuousScheduler(Batch())
    foreign = b.submit([1], max_new_tokens=1)
    with pytest.raises(ValueError, match="different"):
        a.cancel(foreign)
    a.close()
    assert a.info()["closed"] and not a.info()["started"]


def test_thread_start_failure_requires_cleanup_and_does_not_pretend_to_run(monkeypatch):
    from venturi.runtime import worker as module
    scheduler = ContinuousScheduler(Batch())
    pending = scheduler.submit([1], max_new_tokens=2)
    worker = ContinuousWorker(scheduler)

    class BrokenThread:
        def __init__(self, **kwargs): pass
        def start(self): raise RuntimeError("thread admission fault")

    monkeypatch.setattr(module.threading, "Thread", BrokenThread)
    with pytest.raises(RuntimeError, match="thread admission fault"):
        worker.start()
    assert not worker.info()["started"]
    with pytest.raises(RuntimeError, match="failed"):
        worker.close()
    assert pending.future.cancelled() and scheduler.info()["closed"]


@pytest.mark.parametrize("timeout", [-1, True, float("nan"), float("inf")])
def test_join_deadline_is_explicit_finite_nonnegative(timeout):
    worker = ContinuousWorker(ContinuousScheduler(Batch()))
    with pytest.raises(ValueError):
        worker.close(timeout=timeout)
    worker.close()
