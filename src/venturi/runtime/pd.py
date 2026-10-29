"""Opt-in native-independent prefill/decode pools with bounded admission.

Each target is exclusively leased. One owner ticks each P or D lane; different
lanes may execute concurrently. Destination capacity is reserved BEFORE prefill.
The first sampled token travels as control metadata: it is not consumed twice.
No HTTP server, device discovery, precision change, MTP or state preemption.
"""
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
import itertools
import math
import threading
from time import perf_counter

from .continuous import Completion, RequestTicket, TokenEvent, _positive
from .state import RequestState, StateIdentity, TransferReceipt


@dataclass
class _PDRequest:
    ticket: RequestTicket
    prompt: tuple
    maximum: int
    eos: frozenset
    submitted: float
    deadline: float | None
    admitted: float | None = None
    producer: int | None = None
    source_slot: int | None = None
    consumer: int | None = None
    target_slot: int | None = None
    offset: int = 0
    phase: str = "pending"
    busy: bool = False
    output: list = field(default_factory=list)
    first: float | None = None
    last: float | None = None
    max_gap: float = 0.0


class PDScheduler:
    def __init__(self, prefill_targets, decode_targets, transport, *,
                 prefill_chunk_size=128, max_pending=128, clock=perf_counter):
        self.prefill_targets, self.decode_targets = tuple(prefill_targets), tuple(decode_targets)
        all_targets = self.prefill_targets + self.decode_targets
        if not self.prefill_targets or not self.decode_targets:
            raise ValueError("PD requires explicit nonempty prefill and decode pools")
        if len({id(t) for t in all_targets}) != len(all_targets):
            raise ValueError("A PD target cannot serve both roles or multiple lanes")
        identity = all_targets[0].state_identity
        if not isinstance(identity, StateIdentity):
            raise TypeError("PD targets require an explicit StateIdentity")
        self.vocab_size = _positive(all_targets[0].vocab_size, "vocab_size")
        for target in all_targets:
            if target.state_identity != identity or target.vocab_size != self.vocab_size:
                raise ValueError("PD targets must share checkpoint/layout/precision/arithmetic and vocabulary")
            if (not target.capacities or any(type(c) is not int or not 0 < c < 2**31 for c in target.capacities)
                    or len(target.capacities) > _positive(target.max_token_rows, "max_token_rows")):
                raise ValueError("PD target capacities must fit prepared token rows")
        if not callable(getattr(transport, "transfer", None)) or not callable(clock):
            raise TypeError("Explicit state transport and monotonic clock required")
        self.transport, self._clock = transport, clock
        self.prefill_chunk_size = _positive(prefill_chunk_size, "prefill_chunk_size")
        self.max_pending = _positive(max_pending, "max_pending")
        self._owner, self._ids = object(), itertools.count(1)
        self._lock = threading.RLock()
        self._p_locks = [threading.Lock() for _ in self.prefill_targets]
        self._d_locks = [threading.Lock() for _ in self.decode_targets]
        self._p_free = [set(range(len(t.capacities))) for t in self.prefill_targets]
        self._d_free = [set(range(len(t.capacities))) for t in self.decode_targets]
        self._pending, self._requests = deque(), {}
        self._cursor = [0] * len(self.prefill_targets)
        self._route_cursor = 0
        self._producer_cursor = 0
        self._closed, self._closing, self._error = False, False, None
        self._stats = dict(prefill_steps=0, decode_steps=0, prefill_rows=0,
                           decode_rows=0, handoffs=0, transferred_bytes=0,
                           transfer_wall_seconds=0.0, emitted_tokens=0,
                           completed_requests=0, cancelled_requests=0, failed_requests=0)

    def _check(self):
        if self._closed or self._closing:
            raise RuntimeError("PD scheduler closed")
        if self._error is not None:
            raise RuntimeError("PD scheduler failed; drain/close before reusing targets") from self._error

    def submit(self, prompt, *, max_new_tokens, eos_token_ids=(), timeout_seconds=None):
        if (not isinstance(prompt, (list, tuple)) or not prompt
                or any(type(i) is not int or not 0 <= i < self.vocab_size for i in prompt)
                or not isinstance(eos_token_ids, (list, tuple))
                or any(type(i) is not int or not 0 <= i < self.vocab_size for i in eos_token_ids)):
            raise ValueError("PD requires nonempty prompt and valid token ID sequences")
        maximum = _positive(max_new_tokens, "max_new_tokens")
        if timeout_seconds is not None and (type(timeout_seconds) not in (int, float)
                or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
            raise ValueError("Request timeout must be finite and positive")
        needed = len(prompt) + maximum
        if any(not any(needed <= c for t in pool for c in t.capacities)
               for pool in (self.prefill_targets, self.decode_targets)):
            raise ValueError("PD request must fit a prefill AND a decode slot")
        now = self._clock()
        with self._lock:
            self._check()
            if len(self._pending) >= self.max_pending:
                raise RuntimeError("PD pending queue is full")
            ticket = RequestTicket(next(self._ids), Future(), self._owner)
            request = _PDRequest(ticket, tuple(prompt), maximum, frozenset(eos_token_ids),
                                 now, None if timeout_seconds is None else now + timeout_seconds)
            self._pending.append(request)
            self._requests[ticket.request_id] = request
            return ticket

    def cancel(self, ticket):
        if not isinstance(ticket, RequestTicket) or ticket._owner is not self._owner:
            raise ValueError("Ticket belongs to a different PD scheduler")
        return ticket.future.cancel()

    def _retire(self, request):
        self._requests.pop(request.ticket.request_id, None)
        if request.source_slot is not None:
            self._p_free[request.producer].add(request.source_slot)
            request.source_slot = None
        if request.target_slot is not None:
            self._d_free[request.consumer].add(request.target_slot)
            request.target_slot = None

    def _sweep(self, publish):
        now = self._clock()
        for r in list(self._requests.values()):
            if r.busy:
                continue
            if r.ticket.future.cancelled():
                self._stats["cancelled_requests"] += 1
            elif self._error is not None or r.deadline is not None and now >= r.deadline:
                publish.append((r.ticket.future, self._error or TimeoutError("PD request deadline reached")))
                self._stats["failed_requests"] += 1
            else:
                continue
            self._retire(r)
        self._pending = deque(r for r in self._pending if r.ticket.request_id in self._requests)

    def _destination(self, needed):
        candidates = [(len(t.capacities) - len(free), t.capacities[s],
                       (i - self._route_cursor) % len(self.decode_targets), i, s)
                      for i, (t, free) in enumerate(zip(self.decode_targets, self._d_free))
                      for s in free if t.capacities[s] >= needed]
        return min(candidates)[-2:] if candidates else None

    def _producer(self, needed):
        candidates = [(len(t.capacities) - len(free), t.capacities[s],
                       (i - self._producer_cursor) % len(self.prefill_targets), i, s)
                      for i, (t, free) in enumerate(zip(self.prefill_targets, self._p_free))
                      for s in free if t.capacities[s] >= needed]
        return min(candidates)[-2:] if candidates else None

    def _admit(self, lane):
        target = self.prefill_targets[lane]
        for r in list(self._pending):
            needed = len(r.prompt) + r.maximum
            producer = self._producer(needed)
            destination = self._destination(needed)
            if producer is None or producer[0] != lane or destination is None:
                continue
            slot = producer[1]
            consumer, target_slot = destination
            self._p_free[lane].remove(slot)
            self._d_free[consumer].remove(target_slot)
            r.producer, r.source_slot = lane, slot
            r.consumer, r.target_slot = consumer, target_slot
            r.admitted, r.phase = self._clock(), "prefill"
            self._pending.remove(r)
            # Reserved slots cannot be observed by a decode tick yet. Reset is
            # serialized with native work on other slots, never with live state.
            target.reset(slots=[slot])
            self.decode_targets[consumer].reset(slots=[target_slot])
            self._route_cursor = (consumer + 1) % len(self.decode_targets)
            self._producer_cursor = (lane + 1) % len(self.prefill_targets)

    def _emit(self, r, token, events, publish):
        now = self._clock()
        if type(token) is not int or not 0 <= token < self.vocab_size:
            raise RuntimeError("Invalid PD token result")
        r.output.append(token)
        if r.first is None:
            r.first = now
        if r.last is not None:
            r.max_gap = max(r.max_gap, now - r.last)
        r.last = now
        reason = "eos" if token in r.eos else "length" if len(r.output) >= r.maximum else None
        events.append(TokenEvent(r.ticket.request_id, token, now, reason))
        self._stats["emitted_tokens"] += 1
        if reason:
            publish.append((r.ticket.future, Completion(r.ticket.request_id, tuple(r.output), reason,
                r.admitted - r.submitted, r.first - r.submitted, now - r.submitted, r.max_gap)))
            self._stats["completed_requests"] += 1
            self._retire(r)

    def _publish(self, publish, events):
        cancelled = set()
        for future, value in publish:
            if future.set_running_or_notify_cancel():
                if isinstance(value, BaseException):
                    future.set_exception(value)
                else:
                    future.set_result(value)
            elif isinstance(value, Completion):
                cancelled.add(value.request_id)
                with self._lock:
                    self._stats["completed_requests"] -= 1
                    self._stats["cancelled_requests"] += 1
                    self._stats["emitted_tokens"] -= 1
        return tuple(e for e in events if e.request_id not in cancelled)

    def _fatal(self, error, selected, publish):
        with self._lock:
            if self._error is None:
                self._error = error
            for r in selected:
                r.busy = False
            # Other in-flight lanes keep their leases until THEIR safe boundary.
            self._sweep(publish)

    def tick_prefill(self, lane=0, *, _wake=None):
        if type(lane) is not int or not 0 <= lane < len(self.prefill_targets):
            raise ValueError("Invalid prefill lane")
        lock, target = self._p_locks[lane], self.prefill_targets[lane]
        if not lock.acquire(blocking=False):
            raise RuntimeError("PD prefill lane already owned")
        selected, publish, events = [], [], []
        try:
            with self._lock:
                self._check()
                self._sweep(publish)
                self._admit(lane)
                jobs = [r for r in self._requests.values()
                        if r.producer == lane and r.phase == "prefill"]
                plan, budget = [], target.max_token_rows
                if jobs:
                    cursor = self._cursor[lane] % len(jobs)
                    jobs = jobs[cursor:] + jobs[:cursor]
                    for r in jobs:
                        count = min(budget, self.prefill_chunk_size, len(r.prompt) - r.offset)
                        if count:
                            plan.append((r, r.prompt[r.offset:r.offset + count]))
                            budget -= count
                    self._cursor[lane] = (cursor + len(plan)) % len(jobs)
                selected = [r for r, _ in plan]
                for r in selected:
                    r.busy = True
            if _wake is not None:
                _wake()  # Outside metadata locks; wake newly eligible P lanes.
            if plan:
                complete = [r.source_slot for r, ids in plan if r.offset + len(ids) == len(r.prompt)]
                tokens = target.step([list(ids) for _, ids in plan],
                    slots=[r.source_slot for r in selected], sample_slots=complete, decode_slots=())
                if len(tokens) != len(complete):
                    raise RuntimeError("Invalid PD prefill sample count")
                samples = dict(zip(complete, tokens))
                with self._lock:
                    self._stats["prefill_steps"] += 1
                    self._stats["prefill_rows"] += sum(len(ids) for _, ids in plan)
                for r, ids in plan:
                    with self._lock:
                        r.offset += len(ids)
                        r.busy = False
                        self._sweep(publish)
                        if r.ticket.request_id not in self._requests or r.source_slot not in samples:
                            continue
                        token = samples[r.source_slot]
                        if type(token) is not int or not 0 <= token < self.vocab_size:
                            raise RuntimeError("Invalid PD prefill token")
                        if r.maximum == 1 or token in r.eos:
                            self._emit(r, token, events, publish)
                            continue
                        r.phase, r.busy = "transfer", True
                    # Source stays leased and destination stays reserved until
                    # synchronous transfer returns a validated commit receipt.
                    state = target.export_state(r.source_slot, position=len(r.prompt))
                    if not isinstance(state, RequestState) or state.identity != target.state_identity:
                        raise RuntimeError("Invalid PD exported state identity")
                    receipt = self.transport.transfer(state, self.decode_targets[r.consumer], slot=r.target_slot)
                    if (not isinstance(receipt, TransferReceipt) or receipt.position != len(r.prompt)
                            or receipt.payload_bytes != state.nbytes or receipt.staging_bytes < 0
                            or not math.isfinite(receipt.wall_seconds) or receipt.wall_seconds < 0):
                        raise RuntimeError("Invalid PD transfer receipt")
                    with self._lock:
                        self._stats["handoffs"] += 1
                        self._stats["transferred_bytes"] += receipt.payload_bytes
                        self._stats["transfer_wall_seconds"] += receipt.wall_seconds
                        self._p_free[lane].add(r.source_slot)
                        r.source_slot = None
                        r.phase, r.busy = "decode", False
                        self._sweep(publish)
                        if r.ticket.request_id in self._requests:
                            self._emit(r, token, events, publish)
            return self._publish(publish, events)
        except BaseException as error:
            self._fatal(error, selected, publish)
            self._publish(publish, [])
            raise
        finally:
            lock.release()

    def tick_decode(self, lane=0):
        if type(lane) is not int or not 0 <= lane < len(self.decode_targets):
            raise ValueError("Invalid decode lane")
        lock, target = self._d_locks[lane], self.decode_targets[lane]
        if not lock.acquire(blocking=False):
            raise RuntimeError("PD decode lane already owned")
        selected, publish, events = [], [], []
        try:
            with self._lock:
                self._check()
                self._sweep(publish)
                selected = [r for r in self._requests.values()
                            if r.consumer == lane and r.phase == "decode"]
                for r in selected:
                    r.busy = True
            if selected:
                slots = [r.target_slot for r in selected]
                tokens = target.step([[r.output[-1]] for r in selected], slots=slots,
                                     sample_slots=slots, decode_slots=slots)
                if len(tokens) != len(selected):
                    raise RuntimeError("Invalid PD decode sample count")
                with self._lock:
                    self._stats["decode_steps"] += 1
                    self._stats["decode_rows"] += len(selected)
                    for r in selected:
                        r.busy = False
                    self._sweep(publish)
                    for r, token in zip(selected, tokens):
                        if r.ticket.request_id in self._requests:
                            self._emit(r, token, events, publish)
            return self._publish(publish, events)
        except BaseException as error:
            self._fatal(error, selected, publish)
            self._publish(publish, [])
            raise
        finally:
            lock.release()

    def work_for(self, role, lane):
        with self._lock:
            if self._closed or self._closing or self._error is not None:
                return False
            if role == "decode":
                return any(r.phase == "decode" and r.consumer == lane for r in self._requests.values())
            if any(r.phase == "prefill" and r.producer == lane for r in self._requests.values()):
                return True
            for r in self._pending:
                producer = self._producer(len(r.prompt) + r.maximum)
                if (producer is not None and producer[0] == lane
                        and self._destination(len(r.prompt) + r.maximum) is not None):
                    return True
            return False

    @property
    def has_work(self):
        with self._lock:
            return bool(self._requests)

    def info(self):
        with self._lock:
            return dict(self._stats, pending_requests=len(self._pending),
                        active_requests=len(self._requests) - len(self._pending),
                        prefill_workers=len(self.prefill_targets), decode_workers=len(self.decode_targets),
                        prefill_free_slots=[len(s) for s in self._p_free],
                        decode_free_slots=[len(s) for s in self._d_free],
                        closed=self._closed, failed=self._error is not None,
                        scope="State handoff and native token events, not HTTP streaming. Transfer wall times may overlap; do not sum them as global elapsed time.")

    def close(self):
        locks = []
        try:
            for lock in self._p_locks + self._d_locks:
                if not lock.acquire(blocking=False):
                    raise RuntimeError("PD lane still in flight; retain all targets and join owners")
                locks.append(lock)
            with self._lock:
                if self._closed:
                    return
                self._closing = True
                tickets = [r.ticket for r in self._requests.values()]
            # Callbacks are outside metadata locks. No worker remains in flight.
            for ticket in tickets:
                ticket.future.cancel()
            for target in self.prefill_targets + self.decode_targets:
                target.reset(slots=list(range(len(target.capacities))))
            with self._lock:
                self._requests.clear()
                self._pending.clear()
                self._p_free = [set(range(len(t.capacities))) for t in self.prefill_targets]
                self._d_free = [set(range(len(t.capacities))) for t in self.decode_targets]
                self._closed, self._closing = True, False
        finally:
            for lock in reversed(locks):
                lock.release()


class PDWorkerPool:
    """One inference thread per explicit P/D target; idle lanes sleep.

Optional event_sink runs on inference owners and MUST remain nonblocking. It is
not an HTTP/SSE implementation. close(timeout) never frees in-flight targets.
"""
    def __init__(self, scheduler, *, event_sink=None):
        if not isinstance(scheduler, PDScheduler) or event_sink is not None and not callable(event_sink):
            raise TypeError("Explicit PDScheduler and optional callable event sink required")
        self.scheduler, self.event_sink = scheduler, event_sink
        self._condition, self._threads = threading.Condition(), []
        self._stopping, self._closed, self._error = False, False, None

    def start(self):
        with self._condition:
            if self._closed or self._stopping:
                raise RuntimeError("PD worker pool closed")
            if self._threads:
                return self
            try:
                for role, targets in (("prefill", self.scheduler.prefill_targets),
                                      ("decode", self.scheduler.decode_targets)):
                    for lane in range(len(targets)):
                        thread = threading.Thread(target=self._run, args=(role, lane),
                                                  name=f"venturi-pd-{role}-{lane}", daemon=False)
                        thread.start()
                        self._threads.append(thread)
            except BaseException as error:
                self._stopping, self._error = True, error
                self._condition.notify_all()
                raise
            return self

    def _run(self, role, lane):
        try:
            while True:
                with self._condition:
                    while not self._stopping and not self.scheduler.work_for(role, lane):
                        self._condition.wait()
                    if self._stopping:
                        return
                events = (self.scheduler.tick_prefill(lane, _wake=self._wake) if role == "prefill"
                          else self.scheduler.tick_decode(lane))
                if events and self.event_sink is not None:
                    self.event_sink(events)
                with self._condition:
                    self._condition.notify_all()
        except BaseException as error:
            with self._condition:
                self._stopping, self._error = True, error
                self._condition.notify_all()
            publish = []
            self.scheduler._fatal(error, [], publish)
            self.scheduler._publish(publish, [])

    def _wake(self):
        with self._condition:
            self._condition.notify_all()

    def submit(self, prompt, **options):
        with self._condition:
            if not self._threads or self._stopping or self._closed:
                raise RuntimeError("Start a healthy PD worker pool before submission") from self._error
            ticket = self.scheduler.submit(prompt, **options)
            self._condition.notify_all()
            return ticket

    def cancel(self, ticket):
        result = self.scheduler.cancel(ticket)
        with self._condition:
            self._condition.notify_all()
        return result

    def info(self):
        with self._condition:
            return dict(started=bool(self._threads),
                        alive_workers=sum(t.is_alive() for t in self._threads),
                        stopping=self._stopping, closed=self._closed,
                        error=repr(self._error) if self._error is not None else None,
                        scheduling=self.scheduler.info())

    def close(self, *, timeout=180):
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout < 0:
            raise ValueError("Finite nonnegative PD join timeout required")
        if threading.current_thread() in self._threads:
            raise RuntimeError("Cannot close PD worker pool from its inference callback")
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        deadline = perf_counter() + timeout
        for thread in self._threads:
            thread.join(max(0, deadline - perf_counter()))
        if any(thread.is_alive() for thread in self._threads):
            raise TimeoutError("PD inference still in flight; do not free targets/models")
        if not self._closed:
            self.scheduler.close()
            self._closed = True
        if self._error is not None:
            raise RuntimeError("PD worker pool failed") from self._error
