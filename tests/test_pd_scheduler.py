"""Framework-free PD ownership, bounded admission and first-token contracts."""
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from venturi.runtime import PDScheduler, PDWorkerPool
from venturi.runtime.state import RequestState, StateComponent, StateIdentity, TransferReceipt


IDENTITY = StateIdentity("fixture-checkpoint", "recurrence-and-prefix-v1", "int64", "exact")


class Target:
    def __init__(self, capacities=(64, 64), rows=8):
        self.capacities, self.max_token_rows, self.vocab_size = capacities, rows, 97
        self.state_identity = IDENTITY
        self.prefix = [[] for _ in capacities]
        self.value = [0] * len(capacities)
        self.epochs = [0] * len(capacities)
        self.calls, self.resets, self.owners = [], [], set()
        self.fail_step, self.fail_reset = False, False
        self.entered = self.release = None

    def reset(self, *, slots):
        if self.fail_reset:
            raise RuntimeError("injected reset failure")
        for s in slots:
            self.prefix[s], self.value[s] = [], 0
            self.epochs[s] += 1
        self.resets.append(tuple(slots))

    def step(self, prompts, *, slots, sample_slots, decode_slots=()):
        self.owners.add(threading.get_ident())
        if self.entered is not None:
            self.entered.set()
            assert self.release.wait(3)
        if self.fail_step:
            raise RuntimeError("injected step failure")
        assert sum(map(len, prompts)) <= self.max_token_rows
        self.calls.append((tuple(tuple(p) for p in prompts), tuple(slots), tuple(sample_slots), tuple(decode_slots)))
        for ids, s in zip(prompts, slots):
            assert len(self.prefix[s]) + len(ids) <= self.capacities[s]
            for token in ids:
                self.value[s] = (self.value[s] * 3 + token + 1) % self.vocab_size
                self.prefix[s].append(token)
            self.epochs[s] += 1
        return [(self.value[s] + len(self.prefix[s])) % self.vocab_size for s in sample_slots]

    def export_state(self, slot, *, position):
        assert position == len(self.prefix[slot])
        components = (StateComponent("prefix", "int64", (position,), position * 8),
                      StateComponent("recurrence", "int64", (1,), 8))
        return RequestState(self.state_identity, position, components,
                            (self, slot, self.epochs[slot], tuple(self.prefix[slot]), self.value[slot]))


class Transport:
    def __init__(self):
        self.calls = []
        self.fail = False
        self.entered = self.release = None
        self.bad_receipt = False

    def transfer(self, state, target, *, slot):
        if self.entered is not None:
            self.entered.set()
            assert self.release.wait(3)
        if self.fail:
            raise RuntimeError("injected transfer failure")
        source, source_slot, epoch, prefix, value = state.payload
        assert source.epochs[source_slot] == epoch
        assert state.identity == target.state_identity and not target.prefix[slot]
        target.prefix[slot], target.value[slot] = list(prefix), value
        target.epochs[slot] += 1
        self.calls.append((source, source_slot, target, slot, tuple(prefix)))
        return TransferReceipt(state.position, state.nbytes + int(self.bad_receipt), 0.01, "fixture")


def reference(prompt, maximum, eos=()):
    model = Target((128,), 128)
    output = model.step([prompt], slots=[0], sample_slots=[0])
    while len(output) < maximum and output[-1] not in eos:
        output += model.step([[output[-1]]], slots=[0], sample_slots=[0], decode_slots=[0])
    return output, model.prefix[0], model.value[0]


def make(*, p=1, d=1, capacities=(64, 64), rows=8, **kwargs):
    producers = [Target(capacities, rows) for _ in range(p)]
    consumers = [Target(capacities, rows) for _ in range(d)]
    transport = Transport()
    return PDScheduler(producers, consumers, transport, prefill_chunk_size=3, **kwargs), producers, consumers, transport


def drain(scheduler, tickets, limit=200):
    events = []
    for _ in range(limit):
        for i in range(len(scheduler.prefill_targets)):
            events.extend(scheduler.tick_prefill(i))
        for i in range(len(scheduler.decode_targets)):
            events.extend(scheduler.tick_decode(i))
        if all(t.future.done() for t in tickets):
            return events
    pytest.fail("PD fixture failed to finish within bounded ticks")


@pytest.mark.parametrize("p,d", [(1, 1), (2, 1), (1, 2), (4, 4)])
def test_pd_unequal_arrivals_private_state_exact_first_token_and_refill(p, d):
    scheduler, producers, consumers, transport = make(p=p, d=d)
    cases = [([1, 2], 7), ([4] * 11, 4), ([3, 7, 2], 9), ([8], 2), ([7] * 13, 3)]
    try:
        tickets = [scheduler.submit(prompt, max_new_tokens=n) for prompt, n in cases[:3]]
        events = []
        for i in range(p):
            events.extend(scheduler.tick_prefill(i))
        tickets += [scheduler.submit(prompt, max_new_tokens=n) for prompt, n in cases[3:]]
        events += drain(scheduler, tickets)
        for ticket, (prompt, n) in zip(tickets, cases):
            expected, prefix, value = reference(prompt, n)
            actual = ticket.future.result()
            assert actual.output_ids == tuple(expected)
            assert [e.token_id for e in events if e.request_id == ticket.request_id] == expected
            assert actual.finish_reason == "length"
            assert 0 <= actual.queue_seconds <= actual.time_to_first_token_seconds <= actual.wall_seconds
        assert len(transport.calls) == len(cases)
        assert all(len(prefix) in [len(prompt) for prompt, _ in cases]
                   for _, _, _, _, prefix in transport.calls)
        assert scheduler.info()["completed_requests"] == len(cases)
        assert scheduler.info()["emitted_tokens"] == sum(n for _, n in cases)
        assert not scheduler.has_work
    finally:
        scheduler.close()
    assert scheduler.info()["prefill_free_slots"] == [2] * p
    assert scheduler.info()["decode_free_slots"] == [2] * d


def test_pd_exact_state_at_handoff_and_end_no_replayed_prompt_or_pending_token():
    scheduler, p, d, transport = make(capacities=(64,))
    try:
        prompt = [3, 1, 8, 4, 7]
        ticket = scheduler.submit(prompt, max_new_tokens=6)
        assert not scheduler.tick_prefill()
        first = scheduler.tick_prefill()
        assert len(first) == 1
        assert d[0].prefix[0] == p[0].prefix[0] == prompt
        assert d[0].value[0] == p[0].value[0]
        events = list(first) + drain(scheduler, [ticket])
        expected, prefix, value = reference(prompt, 6)
        assert d[0].prefix[0] == prefix and d[0].value[0] == value
        assert p[0].prefix[0] == prompt  # Source never consumed a decode token.
        assert len(d[0].prefix[0]) == len(prompt) + 6 - 1
        assert [e.token_id for e in events] == expected
        assert all(len(ids) == 1 for prompts, _, _, _ in d[0].calls for ids in prompts)
    finally:
        scheduler.close()


@pytest.mark.parametrize("stop", ["length-one", "eos-first", "eos-later"])
def test_pd_early_completion_and_eos(stop):
    scheduler, p, d, transport = make()
    prompt, n = [3, 1], 5
    expected, _, _ = reference(prompt, n)
    eos = ()
    if stop == "length-one":
        n = 1
    elif stop == "eos-first":
        eos = [expected[0]]
    else:
        eos = [expected[2]]
    try:
        ticket = scheduler.submit(prompt, max_new_tokens=n, eos_token_ids=eos)
        drain(scheduler, [ticket])
        assert ticket.future.result().output_ids == tuple(reference(prompt, n, eos)[0])
        assert len(transport.calls) == int(stop == "eos-later")
    finally:
        scheduler.close()


def test_pd_reserves_destination_before_prefill_and_applies_backpressure():
    scheduler, p, d, transport = make(capacities=(32,), max_pending=1)
    try:
        a = scheduler.submit([1], max_new_tokens=5)
        with pytest.raises(RuntimeError, match="queue is full"):
            scheduler.submit([2], max_new_tokens=2)
        scheduler.tick_prefill()
        b = scheduler.submit([2], max_new_tokens=2)
        before = len(p[0].calls)
        scheduler.tick_prefill()
        assert len(p[0].calls) == before  # Free P slot, but D slot still occupied.
        assert scheduler.info()["pending_requests"] == 1
        drain(scheduler, [a, b])
        assert a.future.result().output_ids == tuple(reference([1], 5)[0])
        assert b.future.result().output_ids == tuple(reference([2], 2)[0])
    finally:
        scheduler.close()


@pytest.mark.parametrize("phase", ["pending", "partial-prefill", "decode"])
def test_pd_safe_boundary_cancellation_and_slot_reuse(phase):
    scheduler, p, d, transport = make(capacities=(32,))
    try:
        a = scheduler.submit([1] * 7, max_new_tokens=5)
        if phase != "pending":
            scheduler.tick_prefill()
        if phase == "decode":
            scheduler.tick_prefill()
            scheduler.tick_prefill()
        assert scheduler.cancel(a)
        scheduler.tick_prefill()
        scheduler.tick_decode()
        b = scheduler.submit([2, 5], max_new_tokens=4)
        drain(scheduler, [b])
        assert a.future.cancelled()
        assert b.future.result().output_ids == tuple(reference([2, 5], 4)[0])
        assert d[0].prefix[0] == reference([2, 5], 4)[1]
        assert scheduler.info()["cancelled_requests"] == 1
    finally:
        scheduler.close()


@pytest.mark.parametrize("stage", ["prefill", "transfer", "decode"])
def test_pd_cancel_during_inflight_keeps_leases_and_close_never_frees_work(stage):
    scheduler, p, d, transport = make(capacities=(32,))
    owner = p[0] if stage == "prefill" else d[0] if stage == "decode" else transport
    owner.entered, owner.release = threading.Event(), threading.Event()
    ticket = scheduler.submit([1, 2], max_new_tokens=5)
    if stage == "decode":
        scheduler.tick_prefill()
    errors, events = [], []

    def run():
        try:
            events.extend(scheduler.tick_decode() if stage == "decode" else scheduler.tick_prefill())
        except BaseException as e:
            errors.append(e)
    thread = threading.Thread(target=run)
    thread.start()
    assert owner.entered.wait(2)
    try:
        assert scheduler.cancel(ticket)
        with pytest.raises(RuntimeError, match="in flight"):
            scheduler.close()
    finally:
        owner.release.set()
        thread.join(3)
    assert not thread.is_alive() and not errors and not events
    assert ticket.future.cancelled() and not scheduler.has_work
    scheduler.close()


@pytest.mark.parametrize("stage", ["prefill", "transfer", "decode", "receipt", "reset"])
def test_pd_failure_is_surfaced_without_hidden_retry_or_fallback(stage):
    scheduler, p, d, transport = make(capacities=(32,))
    ticket = scheduler.submit([1], max_new_tokens=5)
    if stage == "prefill":
        p[0].fail_step = True
    elif stage == "transfer":
        transport.fail = True
    elif stage == "receipt":
        transport.bad_receipt = True
    elif stage == "reset":
        p[0].fail_reset = True
    if stage == "decode":
        scheduler.tick_prefill()
        d[0].fail_step = True
    with pytest.raises(RuntimeError):
        scheduler.tick_decode() if stage == "decode" else scheduler.tick_prefill()
    with pytest.raises(RuntimeError):
        ticket.future.result()
    assert scheduler.info()["failed"]
    with pytest.raises(RuntimeError, match="failed"):
        scheduler.submit([2], max_new_tokens=2)
    p[0].fail_reset = False
    scheduler.close()


def test_pd_request_timeout_is_boundary_safe_and_cleanup_can_retry():
    clock = [10.0]
    scheduler, p, d, transport = make(clock=lambda: clock[0])
    ticket = scheduler.submit([1, 2], max_new_tokens=2, timeout_seconds=1)
    clock[0] = 12.0
    scheduler.tick_prefill()
    with pytest.raises(TimeoutError):
        ticket.future.result()
    assert not p[0].calls
    p[0].fail_reset = True
    with pytest.raises(RuntimeError, match="reset failure"):
        scheduler.close()
    assert not scheduler.info()["closed"]
    p[0].fail_reset = False
    scheduler.close()
    assert scheduler.info()["closed"]


@pytest.mark.parametrize("p,d", [(1, 1), (4, 4)])
def test_pd_worker_pool_real_threads_finish_refill_and_sleep_when_idle(p, d):
    scheduler, producers, consumers, transport = make(p=p, d=d)
    sink = []
    pool = PDWorkerPool(scheduler, event_sink=lambda events: sink.extend(events))
    pool.start()
    try:
        cases = [([1, i + 2, 3], i + 2) for i in range(10)]
        tickets = [pool.submit(prompt, max_new_tokens=n) for prompt, n in cases]
        for ticket, (prompt, n) in zip(tickets, cases):
            assert ticket.future.result(timeout=4).output_ids == tuple(reference(prompt, n)[0])
        before = scheduler.info()
        time.sleep(0.02)
        after = scheduler.info()
        assert before["prefill_steps"] == after["prefill_steps"]
        assert before["decode_steps"] == after["decode_steps"]
        assert all(threading.get_ident() not in target.owners for target in producers + consumers)
        assert all(len(target.owners) <= 1 for target in producers + consumers)
    finally:
        pool.close(timeout=4)
    assert not any(t.is_alive() for t in pool._threads)


def test_pd_pool_timeout_retains_targets_and_sink_failure_fails_pending():
    scheduler, p, d, transport = make(capacities=(32,))
    p[0].entered, p[0].release = threading.Event(), threading.Event()
    pool = PDWorkerPool(scheduler)
    pool.start()
    ticket = pool.submit([1], max_new_tokens=4)
    assert p[0].entered.wait(2)
    try:
        with pytest.raises(TimeoutError, match="in flight"):
            pool.close(timeout=0)
        assert not scheduler.info()["closed"]
    finally:
        p[0].release.set()
        pool.close(timeout=4)
    assert ticket.future.done()
    scheduler, p, d, _ = make(capacities=(32,))
    def bad_sink(events):
        raise RuntimeError("sink failed")
    pool = PDWorkerPool(scheduler, event_sink=bad_sink).start()
    ticket = pool.submit([1], max_new_tokens=8)
    with pytest.raises(RuntimeError, match="sink failed"):
        ticket.future.result(timeout=3)
    with pytest.raises(RuntimeError, match="pool failed"):
        pool.close(timeout=4)


@pytest.mark.parametrize("bad", [[], [True], [-1], [97], "abc"])
def test_pd_invalid_request_fails_before_target_calls(bad):
    scheduler, p, d, _ = make()
    with pytest.raises(ValueError):
        scheduler.submit(bad, max_new_tokens=2)
    assert not p[0].calls and not d[0].calls
    scheduler.close()


def test_pd_identity_alias_capacity_ticket_and_schema_guards():
    a, b, transport = Target(), Target(), Transport()
    with pytest.raises(ValueError, match="both roles"):
        PDScheduler([a], [a], transport)
    b.state_identity = replace(IDENTITY, checkpoint="other")
    with pytest.raises(ValueError, match="share checkpoint"):
        PDScheduler([a], [b], transport)
    scheduler, _, _, _ = make(capacities=(8,))
    for options in ({"max_new_tokens": True}, {"max_new_tokens": 8},
                    {"max_new_tokens": 2, "timeout_seconds": float("nan")},
                    {"max_new_tokens": 2, "eos_token_ids": [True]}):
        with pytest.raises(ValueError):
            scheduler.submit([1, 2], **options)
    other, _, _, _ = make()
    ticket = other.submit([1], max_new_tokens=2)
    with pytest.raises(ValueError, match="different"):
        scheduler.cancel(ticket)
    with pytest.raises(ValueError):
        RequestState(IDENTITY, 0, (), None)
    with pytest.raises(ValueError):
        StateComponent("x", "int64", (-1,), 2)
    with pytest.raises(ValueError):
        RequestState(IDENTITY, 1, (StateComponent("x", "int64", (1,), 8),) * 2, None)
    scheduler.close()
    other.close()


def test_pd_import_does_not_import_models_frameworks_or_start_threads():
    root = Path(__file__).resolve().parents[1]
    code = """import sys,threading
before={t.ident for t in threading.enumerate()}
from venturi.runtime import PDScheduler, PDWorkerPool
assert not any(n in sys.modules for n in ('torch','transformers','venturi.models','vllm','sglang'))
assert before=={t.ident for t in threading.enumerate()}
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=root,
                            env=dict(os.environ, PYTHONPATH=str(root / "src")), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_pd_prefill_and_decode_actually_overlap_and_other_lane_error_drains_safely():
    scheduler, p, d, _ = make(capacities=(32, 32))
    first = scheduler.submit([1, 2], max_new_tokens=6)
    scheduler.tick_prefill()
    second = scheduler.submit([4, 5], max_new_tokens=6)
    rendezvous = threading.Barrier(2)
    original_p, original_d = p[0].step, d[0].step

    def prefill(*args, **kwargs):
        rendezvous.wait(timeout=2)
        return original_p(*args, **kwargs)

    def decode(*args, **kwargs):
        rendezvous.wait(timeout=2)
        return original_d(*args, **kwargs)

    p[0].step, d[0].step = prefill, decode
    errors = []
    def tick(function):
        try:
            function()
        except BaseException as e:
            errors.append(e)
    threads = [threading.Thread(target=tick, args=(fn,))
               for fn in (scheduler.tick_prefill, scheduler.tick_decode)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(3)
    assert not errors and not any(t.is_alive() for t in threads)
    p[0].step, d[0].step = original_p, original_d
    drain(scheduler, [first, second])
    assert first.future.result().output_ids == tuple(reference([1, 2], 6)[0])
    assert second.future.result().output_ids == tuple(reference([4, 5], 6)[0])
    scheduler.close()


def test_pd_failure_waits_for_other_inflight_lane_before_releasing_its_lease():
    scheduler, p, d, transport = make(capacities=(32, 32))
    first = scheduler.submit([1], max_new_tokens=6)
    scheduler.tick_prefill()
    d[0].entered, d[0].release = threading.Event(), threading.Event()
    errors = []
    def decode():
        try:
            scheduler.tick_decode()
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=decode)
    thread.start()
    assert d[0].entered.wait(2)
    second = scheduler.submit([3], max_new_tokens=4)
    p[0].fail_step = True
    try:
        with pytest.raises(RuntimeError):
            scheduler.tick_prefill()
        assert not first.future.done()  # Its GPU/state owner is still in flight.
        with pytest.raises(RuntimeError):
            second.future.result()
    finally:
        d[0].release.set()
        thread.join(3)
    assert not thread.is_alive() and not errors
    with pytest.raises(RuntimeError):
        first.future.result()
    scheduler.close()


def test_pd_router_does_not_let_first_large_prefill_target_hoard_the_pool():
    scheduler, p, d, transport = make(p=4, d=4, capacities=(64,) * 8, rows=32)
    try:
        tickets = [scheduler.submit([i + 1], max_new_tokens=3) for i in range(8)]
        # All requests fit the first worker, but routing must still use the pool.
        scheduler.tick_prefill(0)
        assert len(transport.calls) == 1
        assert not scheduler.work_for("prefill", 0)
        assert scheduler.work_for("prefill", 1)
        drain(scheduler, tickets)
        assert {id(call[0]) for call in transport.calls} == {id(target) for target in p}
        assert {id(call[2]) for call in transport.calls} == {id(target) for target in d}
        assert all(t.future.result().output_ids == tuple(reference([i + 1], 3)[0])
                   for i, t in enumerate(tickets))
    finally:
        scheduler.close()


@pytest.mark.parametrize("options", [
    dict(position=True), dict(payload_bytes=True), dict(wall_seconds=float("inf")),
    dict(wall_seconds=-1), dict(staging_bytes=-1), dict(transport=""),
])
def test_pd_receipt_schema_rejects_invalid_types_and_nonfinite_values(options):
    values = dict(position=1, payload_bytes=8, wall_seconds=0.1, transport="fixture")
    values.update(options)
    with pytest.raises(ValueError):
        TransferReceipt(**values)
