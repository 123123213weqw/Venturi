"""Explicit homogeneous replicas over the common continuous scheduler.

One worker owns each batch; an entire request remains on that worker. This is
request-level replica routing, not TP/PP/PD, shared KV, MTP or a model loader.
No framework, GPU, HTTP server or threads are initialized at import/construction.
"""
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
import itertools
import math
import threading
from time import monotonic
import weakref

from .continuous import ContinuousScheduler
from .worker import ContinuousWorker


@dataclass(frozen=True)
class ContinuousReplica:
    name: str
    identity: str
    scheduler: ContinuousScheduler


@dataclass(frozen=True)
class ReplicaTicket:
    request_id: int
    replica: str
    future: Future
    _owner: object = field(repr=False, compare=False)


class ContinuousReplicaPool:
    """Bounded routing with private state and one inference owner per replica.

    identity is a caller-authenticated checkpoint/tokenizer/precision/arithmetic
    key, NOT a hash verification performed here. Do not mix different models in
    one pool or separately submit/tick/close its borrowed schedulers. Different
    batches must also be independently prepared on their declared resources.

    Default routing rotates healthy fitting replicas. Explicit least-work
    minimizes outstanding (prompt + output-budget) tokens per prepared slot.
    That is a CPU heuristic, not a measured cost model or a universal speed win.
    max_requests counts ALL unfinished admitted requests (active plus queued).
    """
    def __init__(self, replicas, *, max_requests=256, routing="round-robin"):
        if (type(replicas) is not tuple or not 1 <= len(replicas) <= 32 or
                any(type(r) is not ContinuousReplica for r in replicas) or
                type(max_requests) is not int or not 1 <= max_requests <= 65536 or
                routing not in ("least-work", "round-robin")):
            raise ValueError("Explicit replica tuple and bounded request capacity required")
        if any(not isinstance(r.name, str) or not 1 <= len(r.name) <= 128 or
               not isinstance(r.identity, str) or not 1 <= len(r.identity) <= 1024 or
               not isinstance(r.scheduler, ContinuousScheduler) for r in replicas):
            raise ValueError("Named replicas, explicit identity and prepared schedulers required")
        if (len({r.name for r in replicas}) != len(replicas) or
                len({r.identity for r in replicas}) != 1 or
                len({r.scheduler.vocab_size for r in replicas}) != 1 or
                len({id(r.scheduler) for r in replicas}) != len(replicas) or
                len({id(r.scheduler.batch) for r in replicas}) != len(replicas)):
            raise ValueError("Unique names/schedulers/batches and homogeneous identity/vocabulary required")
        if any(r.scheduler.has_work or r.scheduler.info()['closed'] or
               r.scheduler.info()['failed'] for r in replicas):
            raise ValueError("Fresh healthy idle schedulers must be exclusively supplied")
        self.replicas = replicas
        self.max_requests = max_requests
        self.routing = routing
        self._lock = threading.RLock()
        self._owner, self._ids = object(), itertools.count(1)
        self._pending, self._work = {}, [0] * len(replicas)
        self._deferred = set()
        self._cursor = 0
        self._started = self._stopping = self._closed = False
        self._start_error = None
        self._totals = dict(submitted=0, completed=0, cancelled=0, failed=0)
        owner = weakref.ref(self)
        def boundary(index):
            def notify():
                pool = owner()
                if pool is not None:
                    pool._sweep(index)
            return notify
        self._workers = tuple(ContinuousWorker(r.scheduler, on_boundary=boundary(i))
                              for i, r in enumerate(replicas))

    def start(self):
        try:
            with self._lock:
                if self._stopping or self._closed or self._start_error is not None:
                    raise RuntimeError("Replica pool closed or startup failed")
                if self._started:
                    return self
                for worker in self._workers:
                    worker.start()
                self._started = True
            return self
        except BaseException as error:
            with self._lock:
                self._start_error = error
                self._stopping = True
            # Signal EVERY owner before joining any; cleanup runs outside our
            # lock because resolving futures may invoke application callbacks.
            for worker in self._workers:
                worker.request_stop()
            for worker in self._workers:
                try:
                    worker.close(timeout=180)
                except BaseException:
                    pass  # Keep worker handles; caller can inspect/close again.
            with self._lock:
                self._closed = all(w.info()['closed'] and not w.info()['alive'] for w in self._workers)
            raise

    def submit(self, prompt, *, max_new_tokens, eos_token_ids=()):
        began = monotonic()
        # Freeze arbitrary input iterables ONCE before touching a replica.
        if (not isinstance(prompt, (tuple, list)) or not prompt or
                not isinstance(eos_token_ids, (tuple, list, set, frozenset)) or
                type(max_new_tokens) is not int or not 1 <= max_new_tokens < 2**31):
            raise ValueError("Explicit nonempty prompt, EOS sequence and positive output budget required")
        prompt, eos = tuple(prompt), tuple(eos_token_ids)
        vocab = self.replicas[0].scheduler.vocab_size
        if any(type(n) is not int or not 0 <= n < vocab for n in prompt + eos):
            raise ValueError("Token IDs must fit the homogeneous replica vocabulary")
        required = len(prompt) + max_new_tokens
        with self._lock:
            if not self._started or self._stopping or self._closed:
                raise RuntimeError("Start an open replica pool explicitly before submission")
            if len(self._pending) >= self.max_requests:
                raise RuntimeError("Replica pool request capacity exhausted")
            candidates = []
            fits = False
            for index, (replica, worker) in enumerate(zip(self.replicas, self._workers)):
                scheduler = replica.scheduler
                if required > max(scheduler.capacities):
                    continue
                fits = True
                info = worker.info()
                if (info['error'] is not None or info['stopping'] or info['closed'] or
                        not info['alive'] or info['scheduling']['failed'] or
                        info['scheduling']['pending_requests'] >= scheduler.max_pending):
                    continue
                score = ((self._work[index] + required) / len(scheduler.capacities)
                         if self.routing == "least-work" else 0)
                candidates.append((score, (index-self._cursor) % len(self.replicas), index))
            if not fits:
                raise ValueError("Request cannot fit any replica's declared slot capacity")
            if not candidates:
                raise RuntimeError("No healthy fitting replica has queue capacity")
            index = min(candidates)[2]
            # Submission validates/adopts metadata only; GPU execution belongs
            # to the existing worker. A failure is NEVER retried on another GPU.
            inner = self._workers[index].submit(prompt, max_new_tokens=max_new_tokens, eos_token_ids=eos)
            route_seconds = monotonic() - began
            request_id = next(self._ids)
            public = Future()
            ticket = ReplicaTicket(request_id, self.replicas[index].name, public, self._owner)
            self._pending[request_id] = (index, required, inner, public, route_seconds)
            self._work[index] += required
            self._cursor = (index+1) % len(self.replicas)
            self._totals['submitted'] += 1
        # Already-complete futures may run callbacks synchronously. Never add
        # callbacks while holding the routing lock or retain the pool via tickets.
        owner = weakref.ref(self)
        def finished(future):
            pool = owner()
            if pool is not None:
                pool._finish(request_id, future)
        def public_done(future):
            pool = owner()
            if pool is not None and future.cancelled():
                pool._cancel_id(request_id)
        public.add_done_callback(public_done)
        inner.future.add_done_callback(finished)
        return ticket

    def _finish(self, request_id, future):
        with self._lock:
            entry = self._pending.get(request_id)
            if entry is None:
                return
            if future.cancelled():
                # Register BEFORE querying retirement, so a racing last tick
                # cannot sweep an empty set and leave a permanent charge.
                self._deferred.add(request_id)
        if future.cancelled() and self.replicas[entry[0]].scheduler.contains(entry[2]):
            entry[3].cancel()
            return
        with self._lock:
            entry = self._pending.pop(request_id, None)
            if entry is None:
                return
            self._deferred.discard(request_id)
            index, cost, inner, public, route_seconds = entry
            self._work[index] -= cost
        action, result = None, None
        if future.cancelled():
            outcome, action = 'cancelled', 'cancel'
        elif not public.set_running_or_notify_cancel():
            outcome = 'cancelled'
        else:
            try:
                value = future.result()
                value = replace(value, request_id=request_id,
                                queue_seconds=value.queue_seconds+route_seconds,
                                time_to_first_token_seconds=value.time_to_first_token_seconds+route_seconds,
                                wall_seconds=value.wall_seconds+route_seconds)
            except BaseException as error:
                outcome, action, result = 'failed', 'exception', error
            else:
                outcome, action, result = 'completed', 'result', value
        with self._lock:
            self._totals[outcome] += 1
        # Counters and capacity commit before publishing; callbacks can inspect
        # or submit again, but never run under our routing lock.
        if action == 'cancel':
            public.cancel()
        elif action == 'exception':
            public.set_exception(result)
        elif action == 'result':
            public.set_result(result)

    def _sweep(self, index):
        with self._lock:
            deferred = [(n, self._pending[n][2].future) for n in self._deferred
                        if n in self._pending and self._pending[n][0] == index]
        for request_id, future in deferred:
            self._finish(request_id, future)

    def _cancel_id(self, request_id):
        with self._lock:
            entry = self._pending.get(request_id)
            if entry is None:
                return False
            worker, inner = self._workers[entry[0]], entry[2]
        return worker.cancel(inner)

    def cancel(self, ticket):
        if not isinstance(ticket, ReplicaTicket) or ticket._owner is not self._owner:
            raise ValueError("Ticket belongs to a different replica pool")
        return self._cancel_id(ticket.request_id)

    def info(self):
        with self._lock:
            return dict(self._totals, started=self._started, stopping=self._stopping,
                        closed=self._closed, unfinished=len(self._pending),
                        max_requests=self.max_requests, routing=self.routing,
                        replicas=[dict(name=r.name, outstanding_token_budget=self._work[i],
                                       worker=self._workers[i].info()) for i, r in enumerate(self.replicas)],
                        scope="Whole-request routing; each replica owns private continuous batching/state; no TP/PP/PD, prefix sharing or MTP")

    def close(self, *, timeout=180):
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout < 0:
            raise ValueError("Explicit finite nonnegative shared join timeout required")
        if any(w._thread is threading.current_thread() for w in self._workers):
            raise RuntimeError("Cannot close/join the pool from an inference-owner callback")
        deadline = monotonic()+timeout
        with self._lock:
            self._stopping = True
        for worker in self._workers:
            worker.request_stop()
        errors = []
        for worker in self._workers:
            try:
                worker.close(timeout=max(0, deadline-monotonic()))
            except BaseException as error:
                errors.append(error)
        with self._lock:
            self._closed = all(w.info()['closed'] and not w.info()['alive'] for w in self._workers)
        if any(isinstance(e, TimeoutError) for e in errors):
            raise TimeoutError("Replica owners still in flight; retain every borrowed batch/model")
        if errors:
            raise RuntimeError("Replica execution or cleanup failed; all owners were signalled") from errors[0]

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()
