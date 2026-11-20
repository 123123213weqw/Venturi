"""Scheduling contracts against an independent deterministic stateful backend."""
import random
from concurrent.futures import CancelledError

import pytest

from venturi.runtime import ContinuousScheduler


def greedy(history, vocab=31):
    return (sum((i + 1) * token for i, token in enumerate(history)) + len(history) * 3) % vocab


def reference(prompt, maximum, eos=()):
    history, result = list(prompt), []
    for _ in range(maximum):
        token = greedy(history)
        result.append(token)
        if token in eos:
            break
        history.append(token)
    return tuple(result)


class Clock:
    def __init__(self):
        self.now = 1.0

    def __call__(self):
        return self.now


class Batch:
    vocab_size = 31
    max_token_rows = 16

    def __init__(self, capacities=(64, 64), clock=None):
        self.capacities = capacities
        self.states = [[] for _ in capacities]
        self.calls, self.resets = [], []
        self.clock, self.hook = clock or Clock(), None
        self.fail_reset, self.fail_step = False, False

    def reset(self, *, slots):
        if self.fail_reset:
            raise RuntimeError("reset fault")
        self.resets.append(tuple(slots))
        for s in slots:
            self.states[s] = []

    def step(self, prompts, *, slots, sample_slots, decode_slots):
        assert sum(map(len, prompts)) <= self.max_token_rows
        assert len(set(slots)) == len(slots)
        assert set(sample_slots) <= set(slots) and set(decode_slots) <= set(slots)
        assert all(len(p) == 1 for s, p in zip(slots, prompts) if s in decode_slots)
        self.calls.append((tuple(slots), tuple(map(tuple, prompts)), tuple(sample_slots), tuple(decode_slots)))
        for s, p in zip(slots, prompts):
            self.states[s].extend(p)
        self.clock.now += 0.001
        if self.hook:
            self.hook()
        if self.fail_step:
            raise RuntimeError("step fault after cache append")
        return [greedy(self.states[s]) for s in sample_slots]


def run(scheduler, maximum=1000):
    events = []
    for _ in range(maximum):
        info = scheduler.info()
        if not info["pending_requests"] and not info["active_requests"]:
            return events
        events.extend(scheduler.tick())
    raise AssertionError("Scheduler failed to make bounded progress")


def test_finished_request_returns_and_new_prompt_interleaves_with_live_decode():
    batch = Batch()
    scheduler = ContinuousScheduler(batch, token_budget=4, prefill_chunk_size=2, clock=batch.clock)
    a = scheduler.submit([1], max_new_tokens=2)
    b = scheduler.submit([2], max_new_tokens=6)
    scheduler.tick()
    scheduler.tick()
    assert a.future.done() and not b.future.done()
    assert a.future.result().output_ids == reference([1], 2)
    c = scheduler.submit([3, 4, 5, 6, 7], max_new_tokens=3)
    events = scheduler.tick()
    slots, prompts, samples, decodes = batch.calls[-1]
    assert len(decodes) == 1 and len(prompts) == 2 and sorted(map(len, prompts)) == [1, 2]
    assert [e.request_id for e in events] == [b.request_id]
    assert not c.future.done() and batch.resets[-1] == (0,)
    run(scheduler)
    assert b.future.result().output_ids == reference([2], 6)
    assert c.future.result().output_ids == reference([3, 4, 5, 6, 7], 3)
    info = scheduler.info()
    assert info["mixed_iterations"] > 0 and info["completed_requests"] == 3
    assert info["emitted_tokens"] == 11 and info["free_slots"] == 2
    assert info["input_rows"] == info["decode_rows"] + info["prefill_rows"]


def test_partial_prefill_is_never_sampled_and_eos_is_per_request():
    batch = Batch()
    s = ContinuousScheduler(batch, token_budget=4, prefill_chunk_size=2, clock=batch.clock)
    prompt = [1, 2, 3, 4, 5]
    a = s.submit(prompt, max_new_tokens=9, eos_token_ids=[greedy(prompt)])
    b = s.submit([7], max_new_tokens=7)
    first = s.tick()
    assert [e.request_id for e in first] == [b.request_id]
    assert not a.future.done()
    run(s)
    assert a.future.result().output_ids == (greedy(prompt),)
    assert a.future.result().finish_reason == "eos"
    assert b.future.result().finish_reason == "length"
    assert b.future.result().output_ids == reference([7], 7)


def test_cancellation_before_and_during_native_work_never_leaks_reused_state():
    batch = Batch(capacities=(64,))
    s = ContinuousScheduler(batch, token_budget=4, prefill_chunk_size=2, clock=batch.clock)
    queued = s.submit([1], max_new_tokens=3)
    assert s.cancel(queued)
    assert s.tick() == () and not batch.calls
    active = s.submit([2, 3, 4], max_new_tokens=3)
    s.tick()
    batch.hook = lambda: s.cancel(active)
    assert s.tick() == ()
    assert active.future.cancelled() and s.info()["free_slots"] == 1
    batch.hook = None
    fresh = s.submit([5, 6], max_new_tokens=2)
    run(s)
    assert fresh.future.result().output_ids == reference([5, 6], 2)
    assert batch.states[0] == [5, 6, fresh.future.result().output_ids[0]]
    assert s.info()["cancelled_requests"] == 2
    with pytest.raises(CancelledError):
        active.future.result()


def test_submission_during_iteration_is_admitted_next_tick_and_not_reentrant():
    batch = Batch()
    s = ContinuousScheduler(batch, prefill_chunk_size=2, clock=batch.clock)
    a = s.submit([1], max_new_tokens=3)
    incoming = []

    def hook():
        batch.hook = None
        for call in (s.tick, s.reset, s.close):
            with pytest.raises(RuntimeError, match="iteration"):
                call()
        incoming.append(s.submit([2], max_new_tokens=1))

    batch.hook = hook
    s.tick()
    assert s.info()["pending_requests"] == 1 and not incoming[0].future.done()
    s.tick()
    assert incoming[0].future.done() and not a.future.done()
    run(s)
    assert a.future.result().output_ids == reference([1], 3)


@pytest.mark.parametrize("where", ["reset", "step"])
def test_fault_fails_outstanding_futures_and_needs_explicit_reset(where):
    batch = Batch()
    s = ContinuousScheduler(batch, clock=batch.clock)
    tickets = [s.submit([i], max_new_tokens=2) for i in (1, 2, 3)]
    setattr(batch, "fail_" + where, True)
    with pytest.raises(RuntimeError, match=where + " fault"):
        s.tick()
    assert s.info()["failed"] and s.info()["failed_requests"] == 3
    for t in tickets:
        with pytest.raises(RuntimeError, match=where + " fault"):
            t.future.result()
    with pytest.raises(RuntimeError, match="explicit reset"):
        s.submit([1], max_new_tokens=1)
    setattr(batch, "fail_" + where, False)
    s.reset()
    assert not s.info()["failed"] and batch.states == [[], []]
    good = s.submit([6], max_new_tokens=2)
    run(s)
    assert good.future.result().output_ids == reference([6], 2)


def test_capacities_best_fit_queue_bound_foreign_cancel_and_closed_owner():
    batch = Batch(capacities=(8, 32))
    s = ContinuousScheduler(batch, max_pending=2, clock=batch.clock)
    long = s.submit([1, 2, 3], max_new_tokens=10)
    short = s.submit([2], max_new_tokens=2)
    with pytest.raises(RuntimeError, match="queue is full"):
        s.submit([3], max_new_tokens=1)
    s.tick()
    assert batch.states[1] == [1, 2, 3] and batch.states[0] == [2]
    other = ContinuousScheduler(Batch())
    with pytest.raises(ValueError, match="different scheduler"):
        other.cancel(long)
    s.close()
    s.close()
    assert long.future.cancelled() and short.future.cancelled()
    assert batch.states == [[], []]
    for call in (s.tick, s.reset, lambda: s.submit([2], max_new_tokens=1)):
        with pytest.raises(RuntimeError, match="closed"):
            call()


@pytest.mark.parametrize("options", [
    dict(prompt=[]), dict(prompt="hello"), dict(prompt=[True]), dict(prompt=[31]),
    dict(prompt=[1], max_new_tokens=True), dict(prompt=[1], max_new_tokens=0),
    dict(prompt=[1] * 64), dict(prompt=[1], eos_token_ids=[False]),
])
def test_invalid_request_is_rejected_before_backend_access(options):
    batch = Batch()
    s = ContinuousScheduler(batch)
    args = dict(prompt=[1], max_new_tokens=2)
    args.update(options)
    with pytest.raises(ValueError):
        s.submit(**args)
    assert not batch.resets and not batch.calls and s.info()["pending_requests"] == 0


@pytest.mark.parametrize("options", [
    dict(token_budget=1), dict(token_budget=17), dict(token_budget=True),
    dict(prefill_chunk_size=0), dict(max_pending=0), dict(clock=None),
    dict(prefill_policy="automatic"), dict(prefill_policy=None),
])
def test_invalid_configuration_has_no_native_side_effects(options):
    batch = Batch()
    with pytest.raises((ValueError, TypeError)):
        ContinuousScheduler(batch, **options)
    assert not batch.resets and not batch.calls


@pytest.mark.parametrize("slots", [1, 2, 4, 8])
@pytest.mark.parametrize("policy", ["round-robin", "balanced", "balanced-short"])
def test_randomized_arrival_lengths_cancellation_and_budget_conservation(slots, policy):
    rng = random.Random(71 + slots)
    batch = Batch(capacities=(64,) * slots)
    budget = slots + 2
    s = ContinuousScheduler(batch, token_budget=budget, prefill_chunk_size=5,
                            prefill_policy=policy, clock=batch.clock)
    pending = []
    for iteration in range(120):
        if rng.random() < 0.65:
            prompt = [rng.randrange(31) for _ in range(rng.randint(1, 20))]
            maximum = rng.randint(1, 7)
            eos = [greedy(prompt)] if rng.random() < 0.2 else []
            ticket = s.submit(prompt, max_new_tokens=maximum, eos_token_ids=eos)
            pending.append((ticket, reference(prompt, maximum, eos)))
        if pending and rng.random() < 0.12:
            s.cancel(rng.choice(pending)[0])
        s.tick()
    run(s)
    for ticket, expected in pending:
        if not ticket.future.cancelled():
            result = ticket.future.result()
            assert result.output_ids == expected
            assert 0 <= result.queue_seconds <= result.time_to_first_token_seconds <= result.wall_seconds
    assert all(sum(map(len, prompts)) <= budget for _, prompts, _, _ in batch.calls)
    assert not s.info()["active_requests"] and not s.info()["pending_requests"]
    assert not s._requests  # The scheduler does not retain completed output history.


def test_balanced_water_filling_matches_independent_per_row_reference():
    from venturi.runtime.continuous import _balanced_rows

    rng = random.Random(723)
    for _ in range(1500):
        caps = [rng.randint(1, 32) for _ in range(rng.randint(1, 32))]
        budget = rng.randint(0, sum(caps) + 5)
        expected, left = [0] * len(caps), budget
        while left and any(a < b for a, b in zip(expected, caps)):
            for i, cap in enumerate(caps):
                if left and expected[i] < cap:
                    expected[i] += 1
                    left -= 1
        assert _balanced_rows(caps, budget) == expected


@pytest.mark.parametrize("count", [1, 2, 4, 8])
@pytest.mark.parametrize("policy", ["balanced", "balanced-short"])
def test_balanced_equal_prefills_fill_rows_without_early_decode_fragmentation(count, policy):
    batch = Batch(capacities=(64,) * count)
    batch.max_token_rows = 32
    scheduler = ContinuousScheduler(batch, token_budget=32, prefill_chunk_size=16,
                                    prefill_policy=policy, clock=batch.clock)
    tickets = [scheduler.submit([i] * 16, max_new_tokens=3) for i in range(count)]
    steps = max(1, count // 2)
    for index in range(steps):
        events = scheduler.tick()
        slots, prompts, samples, decodes = batch.calls[-1]
        assert len(prompts) == count and not decodes
        assert set(map(len, prompts)) == {min(16, 32 // count)}
        assert len(events) == (count if index == steps - 1 else 0)
        assert len(samples) == len(events)
    run(scheduler)
    assert all(t.future.result().output_ids == reference([i] * 16, 3)
               for i, t in enumerate(tickets))
    assert scheduler.info()["prefill_policy"] == policy


@pytest.mark.parametrize("prompts,chunk,decode_first", [
    ([4, 3, 4, 4], 16, False),     # Heterogeneous short remainders.
    ([24, 24, 24, 24], 16, False),  # Uniform but long prompts.
    ([4, 4, 4], 16, True),          # Preserve a live decoder's plan.
])
def test_balanced_short_gate_keeps_original_plan_elsewhere(prompts, chunk, decode_first):
    traces = []
    for policy in ("round-robin", "balanced-short"):
        batch = Batch(capacities=(64,) * 4)
        scheduler = ContinuousScheduler(batch, token_budget=8, prefill_chunk_size=chunk,
                                        prefill_policy=policy, clock=batch.clock)
        if decode_first:
            scheduler.submit([1], max_new_tokens=8)
            scheduler.tick()
        for n in prompts:
            scheduler.submit([2] * n, max_new_tokens=3)
        scheduler.tick()
        traces.append(batch.calls[-1])
    assert traces[0] == traces[1]


def test_balanced_short_prompt_shares_reused_and_decode_still_first():
    batch = Batch(capacities=(64,) * 4)
    scheduler = ContinuousScheduler(batch, token_budget=8, prefill_chunk_size=8,
                                    prefill_policy="balanced", clock=batch.clock)
    a = scheduler.submit([1], max_new_tokens=5)
    scheduler.tick()
    b = scheduler.submit([2], max_new_tokens=2)
    c = scheduler.submit([3] * 12, max_new_tokens=3)
    d = scheduler.submit([4] * 12, max_new_tokens=3)
    scheduler.tick()
    slots, prompts, samples, decodes = batch.calls[-1]
    assert slots == (0, 1, 2, 3) and decodes == (0,)
    assert tuple(map(len, prompts)) == (1, 1, 3, 3)
    assert samples == (0, 1)
    run(scheduler)
    for t, prompt, maximum in ((a, [1], 5), (b, [2], 2), (c, [3] * 12, 3), (d, [4] * 12, 3)):
        assert t.future.result().output_ids == reference(prompt, maximum)


def test_default_prefill_policy_preserves_front_loaded_round_robin():
    batch = Batch(capacities=(64,) * 4)
    scheduler = ContinuousScheduler(batch, token_budget=8, prefill_chunk_size=4)
    for _ in range(4):
        scheduler.submit([1] * 16, max_new_tokens=1)
    scheduler.tick()
    assert batch.calls[-1][0] == (0, 1)
    assert tuple(map(len, batch.calls[-1][1])) == (4, 4)
    assert scheduler.info()["prefill_policy"] == "round-robin"
