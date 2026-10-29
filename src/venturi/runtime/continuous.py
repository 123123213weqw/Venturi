"""Bounded decode-first iteration scheduling over an explicit prepared batch.

This owner controls ordinary greedy decoding only. It does not load weights,
choose precision/devices, share prefixes, preempt state, use MTP, or start HTTP
threads. The supplied batch is exclusively leased until close. Native model
execution stays in batch.step; this module handles token/slot metadata only.
"""

from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
import itertools
import threading
from time import perf_counter


@dataclass(frozen=True)
class Completion:
    request_id: int
    output_ids: tuple[int, ...]
    finish_reason: str
    queue_seconds: float
    time_to_first_token_seconds: float
    wall_seconds: float
    max_inter_token_seconds: float


@dataclass(frozen=True)
class RequestTicket:
    request_id: int
    future: Future
    _owner: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class TokenEvent:
    request_id: int
    token_id: int
    timestamp: float
    finish_reason: str | None


@dataclass
class _Request:
    ticket: RequestTicket
    prompt: tuple[int, ...]
    maximum: int
    eos: frozenset[int]
    submitted: float
    slot: int | None = None
    admitted: float | None = None
    prompt_offset: int = 0
    output: list[int] = field(default_factory=list)
    first: float | None = None
    last: float | None = None
    max_gap: float = 0.0


def _positive(value, name):
    if type(value) is not int or not 0 < value < 2**31:
        raise ValueError(name + " must be a positive int32")
    return value


def _balanced_rows(caps, budget):
    """Max-min row sharing; unused short-request shares are redistributed.

    caps are already in rotating request order. The final one-row remainders
    follow that order, not the sort order used to find the water level. This
    operates on request metadata, never on tensors or individual input tokens.
    """
    counts = [0] * len(caps)
    ranked = sorted(range(len(caps)), key=caps.__getitem__)
    for at, index in enumerate(ranked):
        share, extra = divmod(budget, len(caps) - at)
        if caps[index] <= share:
            counts[index] = caps[index]
            budget -= caps[index]
            continue
        rest = set(ranked[at:])
        for index in range(len(caps)):
            if index in rest:
                counts[index] = share + (extra > 0)
                extra = max(0, extra - 1)
        break
    return counts


class ContinuousScheduler:
    """Thread-safe submission/cancel; exactly one caller owns each tick.

    token_budget is a per-iteration row limit, not context capacity. It must be
    at least the prepared slot count so every live decode can progress. Prefills
    take bounded round-robin chunks from the remaining budget by default.
    Explicit prefill_policy="balanced" shares that budget across waiting
    prefills, redistributing unused shares without delaying live decodes.
    "balanced-short" limits sharing to equal short remainders with no live
    decodes; heterogeneous/long prompts otherwise keep the default plan.
    Returned events
    can feed streaming responses; futures complete when THEIR request finishes,
    never when an unrelated cohort finishes. Callbacks on futures must stay light.
    """

    def __init__(self, batch, *, token_budget=None, prefill_chunk_size=128,
                 max_pending=1024, prefill_policy="round-robin", clock=perf_counter):
        capacities = tuple(batch.capacities)
        if not capacities or any(type(c) is not int or not 0 < c < 2**31 for c in capacities):
            raise ValueError("Explicit prepared request capacities required")
        limit = _positive(batch.max_token_rows, "max_token_rows")
        budget = limit if token_budget is None else _positive(token_budget, "token_budget")
        if not len(capacities) <= budget <= limit:
            raise ValueError("Token budget must cover slots and fit prepared token rows")
        _positive(prefill_chunk_size, "prefill_chunk_size")
        _positive(max_pending, "max_pending")
        if prefill_policy not in ("round-robin", "balanced", "balanced-short"):
            raise ValueError("prefill_policy must be round-robin, balanced or balanced-short")
        if not callable(clock):
            raise TypeError("Explicit monotonic clock required")
        self.batch, self.capacities = batch, capacities
        self.vocab_size = _positive(batch.vocab_size, "vocab_size")
        self.token_budget, self.prefill_chunk_size = budget, min(prefill_chunk_size, budget)
        self.prefill_policy = prefill_policy
        self.max_pending, self._clock = max_pending, clock
        self._owner, self._ids = object(), itertools.count(1)
        self._lock, self._tick_lock = threading.RLock(), threading.Lock()
        self._pending, self._active = deque(), {}
        self._requests, self._free = {}, set(range(len(capacities)))
        self._closed, self._failed, self._prefill_cursor = False, False, 0
        self._totals = dict(iterations=0, input_rows=0, decode_rows=0,
                            prefill_rows=0, mixed_iterations=0, emitted_tokens=0,
                            completed_requests=0, cancelled_requests=0, failed_requests=0)

    def _check(self):
        if self._closed:
            raise RuntimeError("Continuous scheduler closed")
        if self._failed:
            raise RuntimeError("Continuous scheduler failed; explicit reset required")

    def _tokens(self, values, *, empty=False):
        if not isinstance(values, (list, tuple)) or (not values and not empty):
            raise ValueError("Explicit token ID list required")
        if any(type(v) is not int or not 0 <= v < self.vocab_size for v in values):
            raise ValueError("Token IDs must belong to the model vocabulary")
        return tuple(values)

    def submit(self, prompt, *, max_new_tokens, eos_token_ids=()):
        submitted = self._clock()
        ids = self._tokens(prompt)
        maximum = _positive(max_new_tokens, "max_new_tokens")
        eos = frozenset(self._tokens(eos_token_ids, empty=True))
        if len(ids) + maximum > max(self.capacities):
            raise ValueError("Request cannot fit any prepared slot capacity")
        with self._lock:
            self._check()
            if len(self._pending) >= self.max_pending:
                raise RuntimeError("Continuous pending queue is full")
            ticket = RequestTicket(next(self._ids), Future(), self._owner)
            request = _Request(ticket, ids, maximum, eos, submitted)
            self._pending.append(request)
            self._requests[ticket.request_id] = request
            return ticket

    def cancel(self, ticket):
        if not isinstance(ticket, RequestTicket) or ticket._owner is not self._owner:
            raise ValueError("Ticket belongs to a different scheduler")
        # Future cancellation is observed at the next safe iteration boundary,
        # including a cancellation that arrives while native work is in flight.
        return ticket.future.cancel()

    def _retire(self, request):
        self._requests.pop(request.ticket.request_id, None)
        if request.slot is not None:
            self._active.pop(request.slot)
            self._free.add(request.slot)

    def _remove_cancelled(self):
        for request in list(self._requests.values()):
            if request.ticket.future.cancelled():
                self._retire(request)
                self._totals["cancelled_requests"] += 1
        self._pending = deque(r for r in self._pending if not r.ticket.future.cancelled())

    def _admit(self):
        admissions, keep = [], deque()
        for request in self._pending:
            if request.ticket.future.cancelled():
                self._retire(request)
                self._totals["cancelled_requests"] += 1
                continue
            needed = len(request.prompt) + request.maximum
            fits = [s for s in self._free if needed <= self.capacities[s]]
            if not fits:
                keep.append(request)
                continue
            slot = min(fits, key=lambda s: (self.capacities[s], s))
            self._free.remove(slot)
            request.slot, request.admitted = slot, self._clock()
            self._active[slot] = request
            admissions.append(slot)
        self._pending = keep
        if admissions:
            # Reset BEFORE reuse, never copy the previous tenant's state. One
            # reset submission covers all newly admitted slots for this tick.
            self.batch.reset(slots=admissions)

    def _plan(self):
        decodes = [r for r in self._active.values() if r.prompt_offset == len(r.prompt)]
        prefills = [r for r in self._active.values() if r.prompt_offset < len(r.prompt)]
        plan, remaining = [], self.token_budget
        for request in sorted(decodes, key=lambda r: r.ticket.request_id):
            plan.append((request, [request.output[-1]], True))
            remaining -= 1
        if prefills:
            prefills.sort(key=lambda r: r.ticket.request_id)
            start = self._prefill_cursor % len(prefills)
            ordered = prefills[start:] + prefills[:start]
            lengths = [len(r.prompt) - r.prompt_offset for r in ordered]
            caps = [min(self.prefill_chunk_size, length) for length in lengths]
            # Only the no-decode, equal SHORT-remainder cohort gets row sharing.
            # Long/heterogeneous prefills and arrivals during live decoding keep
            # the original bounded round-robin plan. No timer or GPU heuristic.
            balanced = self.prefill_policy == "balanced" or (
                self.prefill_policy == "balanced-short" and not decodes
                and lengths[0] <= self.prefill_chunk_size
                and all(length == lengths[0] for length in lengths)
            )
            counts = _balanced_rows(caps, remaining) if balanced else caps
            served = 0
            for request, count in zip(ordered, counts):
                if not remaining:
                    break
                count = min(count, remaining)
                if not count:
                    continue
                ids = request.prompt[request.prompt_offset:request.prompt_offset + count]
                plan.append((request, list(ids), False))
                remaining -= count
                served += 1
            advance = 1 if balanced and served == len(prefills) else served
            self._prefill_cursor = (start + advance) % len(prefills)
        return plan

    def _fail(self, error):
        self._failed = True
        completions = []
        for request in list(self._requests.values()):
            future = request.ticket.future
            if not future.done():
                completions.append((future, error))
                self._totals["failed_requests"] += 1
        self._requests.clear()
        self._active.clear()
        self._pending.clear()
        self._free = set(range(len(self.capacities)))
        return completions

    def tick(self):
        """Run at most one mixed group; return this iteration's token events."""
        if not self._tick_lock.acquire(blocking=False):
            raise RuntimeError("A continuous iteration is already owned")
        publish, failures, events, error = [], [], [], None
        try:
            with self._lock:
                self._check()
                try:
                    self._remove_cancelled()
                    self._admit()
                    plan = self._plan()
                except BaseException as exc:
                    failures, error, plan = self._fail(exc), exc, []
            if not plan and error is None:
                return ()
            slots = [r.slot for r, _, _ in plan]
            decode_slots = [r.slot for r, _, decode in plan if decode]
            sample_slots = [r.slot for r, ids, decode in plan
                            if decode or r.prompt_offset + len(ids) == len(r.prompt)]
            if error is None:
                try:
                    output = self.batch.step([ids for _, ids, _ in plan], slots=slots,
                                             sample_slots=sample_slots, decode_slots=decode_slots)
                    if (len(output) != len(sample_slots) or
                        any(type(v) is not int or not 0 <= v < self.vocab_size for v in output)):
                        raise RuntimeError("Invalid native step sample result; state cannot be retried")
                except BaseException as exc:
                    with self._lock:
                        failures = self._fail(exc)
                    error = exc
            if error is None:
                now = self._clock()
                samples = dict(zip(sample_slots, output))
                with self._lock:
                    rows = sum(len(ids) for _, ids, _ in plan)
                    self._totals["iterations"] += 1
                    self._totals["input_rows"] += rows
                    self._totals["decode_rows"] += len(decode_slots)
                    self._totals["prefill_rows"] += rows - len(decode_slots)
                    self._totals["mixed_iterations"] += bool(decode_slots and rows > len(decode_slots))
                    for request, ids, decode in plan:
                        if not decode:
                            request.prompt_offset += len(ids)
                        if request.ticket.future.cancelled():
                            self._retire(request)
                            self._totals["cancelled_requests"] += 1
                            continue
                        if request.slot not in samples:
                            continue
                        token = samples[request.slot]
                        request.output.append(token)
                        self._totals["emitted_tokens"] += 1
                        if request.first is None:
                            request.first = now
                        if request.last is not None:
                            request.max_gap = max(request.max_gap, now - request.last)
                        request.last = now
                        reason = ("eos" if token in request.eos else
                                  "length" if len(request.output) >= request.maximum else None)
                        events.append(TokenEvent(request.ticket.request_id, token, now, reason))
                        if reason is not None:
                            completion = Completion(request.ticket.request_id, tuple(request.output), reason,
                                request.admitted - request.submitted, request.first - request.submitted,
                                now - request.submitted, request.max_gap)
                            publish.append((request.ticket.future, completion))
                            self._retire(request)
                            self._totals["completed_requests"] += 1
            # Do not run user Future callbacks while holding the state lock.
            for future, exc in failures:
                if future.set_running_or_notify_cancel():
                    future.set_exception(exc)
            for future, value in publish:
                if future.set_running_or_notify_cancel():
                    future.set_result(value)
                else:
                    # Cancellation won the final publish boundary, so don't
                    # leak a final token/completion event to that client.
                    events = [e for e in events if e.request_id != value.request_id]
                    with self._lock:
                        self._totals["completed_requests"] -= 1
                        self._totals["cancelled_requests"] += 1
                        self._totals["emitted_tokens"] -= 1
            if error is not None:
                raise error
            return tuple(events)
        finally:
            self._tick_lock.release()

    def info(self):
        with self._lock:
            return dict(self._totals, active_requests=len(self._active),
                        pending_requests=len(self._pending), free_slots=len(self._free),
                        token_budget=self.token_budget, prefill_chunk_size=self.prefill_chunk_size,
                        prefill_policy=self.prefill_policy,
                        closed=self._closed, failed=self._failed,
                        scope="Per scheduler instance; throughput needs a shared wall interval, not summed request durations")

    @property
    def has_work(self):
        """Cheap owner wake predicate; never probes a device or copies stats."""
        with self._lock:
            return bool(self._pending or self._active)

    def _reset(self, *, close):
        if not self._tick_lock.acquire(blocking=False):
            raise RuntimeError("Cannot reset an in-flight continuous iteration")
        failures, tickets, error = [], [], None
        try:
            with self._lock:
                if self._closed:
                    if close:
                        return
                    raise RuntimeError("Continuous scheduler closed")
                try:
                    self.batch.reset(slots=list(range(len(self.capacities))))
                except BaseException as exc:
                    failures, error = self._fail(exc), exc
                else:
                    tickets = [r.ticket for r in self._requests.values()]
                    self._requests.clear()
                    self._pending.clear()
                    self._active.clear()
                    self._free = set(range(len(self.capacities)))
                    self._failed = False
                self._closed = close
            for ticket in tickets:
                ticket.future.cancel()
            for future, exc in failures:
                if future.set_running_or_notify_cancel():
                    future.set_exception(exc)
            if error is not None:
                raise error
        finally:
            self._tick_lock.release()

    def reset(self):
        """Cancel outstanding requests and explicitly reset all leased slots."""
        self._reset(close=False)

    def close(self):
        """Cancel/reset the lease, but do not close the caller-owned batch/model."""
        self._reset(close=True)
