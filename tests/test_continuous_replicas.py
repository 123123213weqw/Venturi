"""Replica routing, private state, overload, cancellation and safe owner exit."""
from concurrent.futures import ThreadPoolExecutor
import gc
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import weakref

import pytest

from test_continuous_scheduler import Batch, reference
from venturi.runtime import ContinuousReplica, ContinuousReplicaPool, ContinuousScheduler


def test_replica_import_and_benchmark_help_do_not_initialize_frameworks():
    code = """
import importlib.abc, runpy, sys
class Reject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch', 'transformers', 'tokenizers'):
            raise AssertionError('Unexpected framework import: '+fullname)
sys.meta_path.insert(0, Reject())
import venturi.runtime
assert 'venturi.models' not in sys.modules
sys.argv=['benchmark_continuous_replicas.py', '--help']
runpy.run_path('tools/benchmark_continuous_replicas.py', run_name='__main__')
"""
    root = Path(__file__).parents[1]
    result = subprocess.run([sys.executable, '-c', code], cwd=root,
        env=dict(os.environ, PYTHONPATH=str(root/'src')), capture_output=True,
        text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert '--checkpoint' in result.stdout


def make_pool(*, capacities=((128,)*4, (128,)*4), max_requests=64, hooks=None, pending=64):
    batches = [Batch(capacities=c) for c in capacities]
    for i, batch in enumerate(batches):
        batch.max_token_rows = max(16, len(batch.capacities))
        if hooks:
            batch.hook = hooks[i]
    specs = tuple(ContinuousReplica(str(i), 'same-authenticated-greedy-identity',
        ContinuousScheduler(b, token_budget=b.max_token_rows, prefill_chunk_size=4,
                            max_pending=pending, clock=b.clock)) for i, b in enumerate(batches))
    return ContinuousReplicaPool(specs, max_requests=max_requests), batches


@pytest.mark.parametrize('requests', [1, 4, 8, 16, 32])
def test_distinct_concurrent_requests_match_independent_reference(requests):
    pool, batches = make_pool()
    assert not pool.info()['started'] and not any(b.calls for b in batches)
    with pool:
        def send(i):
            ids = [i % 31, (i+3) % 31, (i+8) % 31]
            t = pool.submit(ids, max_new_tokens=5)
            r = t.future.result(timeout=5)
            assert r.request_id == t.request_id
            assert r.output_ids == reference(ids, 5)
            return t
        with ThreadPoolExecutor(max_workers=min(requests, 16)) as workers:
            tickets = list(workers.map(send, range(requests)))
        assert len({t.request_id for t in tickets}) == requests
        info = pool.info()
        assert info['completed'] == requests and info['unfinished'] == 0
        assert all(r['outstanding_token_budget'] == 0 for r in info['replicas'])
    assert all(not r['worker']['alive'] for r in pool.info()['replicas'])


def test_two_owners_really_overlap_and_weighted_routing_is_not_thread_fanout():
    barrier = threading.Barrier(2)
    ids = [set(), set()]
    # Each first tick runs on a different owner and meets at a real barrier.
    pool, batches = make_pool(hooks=[None, None])
    seen = [False, False]
    for i, batch in enumerate(batches):
        def h(i=i):
            ids[i].add(threading.get_ident())
            if not seen[i]:
                seen[i] = True
                barrier.wait(timeout=3)
        batch.hook = h
    with pool:
        a = pool.submit([1], max_new_tokens=1)
        b = pool.submit([2], max_new_tokens=1)
        assert a.replica != b.replica
        assert a.future.result(timeout=5).output_ids == reference([1], 1)
        assert b.future.result(timeout=5).output_ids == reference([2], 1)
    assert len(ids[0]) == len(ids[1]) == 1 and ids[0] != ids[1]


def test_capacity_fit_and_pending_budget_fail_before_extra_admission():
    entered, release = [threading.Event(), threading.Event()], threading.Event()
    pool, batches = make_pool(capacities=((8,), (64,)), max_requests=2,
        hooks=[lambda i=i: (entered[i].set(), release.wait(3)) for i in range(2)])
    pool.start()
    try:
        a = pool.submit([1], max_new_tokens=5)
        # Long request only fits replica1; replica0 remains blocked.
        b = pool.submit([2]*10, max_new_tokens=2)
        assert b.replica == '1'
        assert all(e.wait(3) for e in entered)
        with pytest.raises(RuntimeError, match='capacity'):
            pool.submit([4], max_new_tokens=4)
        with pytest.raises(ValueError, match='fit'):
            # Avoid exercising global overload first.
            other, _ = make_pool(capacities=((2,),))
            with other:
                other.submit([1,2], max_new_tokens=1)
        release.set()
        for t in (a,b):t.future.result(timeout=3)
    finally:
        release.set()
        pool.close(timeout=5)


def test_completion_callback_can_reenter_submission_and_sees_committed_counters():
    pool, _ = make_pool()
    done = threading.Event()
    second = []
    with pool:
        a = pool.submit([1], max_new_tokens=1)
        def cb(f):
            assert pool.info()['completed'] >= 1
            second.append(pool.submit([2], max_new_tokens=2))
            done.set()
        a.future.add_done_callback(cb)
        assert done.wait(3)
        assert second[0].future.result(timeout=3).output_ids == reference([2], 2)


@pytest.mark.parametrize('public_cancel', [False, True])
def test_cancellation_keeps_capacity_until_safe_boundary_and_resets_private_slots(public_cancel):
    entered, release = threading.Event(), threading.Event()
    pool, batches = make_pool(capacities=((64,),), max_requests=1,
        hooks=[lambda: (entered.set(), release.wait(3))])
    pool.start()
    try:
        t = pool.submit([1], max_new_tokens=12)
        assert entered.wait(3)
        assert t.future.cancel() if public_cancel else pool.cancel(t)
        assert pool.info()['unfinished'] == 1
        with pytest.raises(RuntimeError, match='capacity'):
            pool.submit([2], max_new_tokens=1)
        release.set()
        for i in range(300):
            if pool.info()['unfinished'] == 0:break
            time.sleep(.005)
        assert t.future.cancelled() and pool.info()['cancelled'] == 1
        assert pool.info()['unfinished'] == 0
        assert pool.submit([2], max_new_tokens=1).future.result(timeout=3).output_ids == reference([2], 1)
        assert batches[0].resets[-1] == (0,)
    finally:
        release.set();pool.close(timeout=5)


def test_close_signals_all_replicas_before_wait_and_timeout_retains_owners():
    entered = [threading.Event(), threading.Event()]
    release = threading.Event()
    pool, _ = make_pool(hooks=[lambda i=i: (entered[i].set(), release.wait(3)) for i in range(2)])
    pool.start()
    a = pool.submit([1], max_new_tokens=10)
    b = pool.submit([2], max_new_tokens=10)
    assert all(e.wait(3) for e in entered)
    try:
        with pytest.raises(TimeoutError, match='in flight'):
            pool.close(timeout=0)
        info = pool.info()
        assert not info['closed'] and all(r['worker']['stopping'] for r in info['replicas'])
        assert all(r['worker']['alive'] for r in info['replicas'])
    finally:
        release.set();pool.close(timeout=5)
    assert a.future.cancelled() and b.future.cancelled()
    pool.close()


def test_failed_replica_does_not_retry_or_poison_other_replica():
    pool, batches = make_pool()
    batches[0].fail_step = True
    pool.start()
    try:
        a = pool.submit([1], max_new_tokens=2)
        with pytest.raises(RuntimeError, match='step fault'):
            a.future.result(timeout=3)
        b = pool.submit([2], max_new_tokens=2)
        assert b.replica == '1' and b.future.result(timeout=3).output_ids == reference([2], 2)
        assert len(batches[0].calls) == 1
    finally:
        with pytest.raises(RuntimeError, match='cleanup failed'):
            pool.close(timeout=5)
    assert pool.info()['closed']


def test_cancelled_budget_retires_even_when_execution_and_cleanup_fail():
    entered, release = threading.Event(), threading.Event()
    pool, batches = make_pool(max_requests=1,
        hooks=[lambda: (entered.set(), release.wait(3)), None])
    pool.start()
    try:
        t = pool.submit([1], max_new_tokens=8)
        assert entered.wait(3)
        pool.cancel(t)
        batches[0].fail_step = batches[0].fail_reset = True
        release.set()
        for _ in range(300):
            if not pool.info()['unfinished']:
                break
            time.sleep(.005)
        assert pool.info()['unfinished'] == 0
        assert pool.info()['cancelled'] == 1
        next_ticket = pool.submit([2], max_new_tokens=1)
        assert next_ticket.replica == '1'
        assert next_ticket.future.result(timeout=3).output_ids == reference([2], 1)
    finally:
        release.set()
        with pytest.raises(RuntimeError, match='cleanup failed'):
            pool.close(timeout=5)


def test_ticket_does_not_retain_closed_pool_and_foreign_cancellation_rejected():
    pool, _ = make_pool()
    other, _ = make_pool()
    with pool:
        ticket = pool.submit([1], max_new_tokens=1)
        ticket.future.result(timeout=3)
        with pytest.raises(ValueError, match='different'):
            other.cancel(ticket)
    other.close()
    ref = weakref.ref(pool)
    del pool
    gc.collect()
    assert ref() is None


def test_constructor_rejects_mixed_or_duplicate_resources_and_pending_schedulers():
    pool, _ = make_pool()
    a,b = pool.replicas
    for replicas in ((a,a), (a,ContinuousReplica('other','different',b.scheduler)),
                     (a,ContinuousReplica('other',a.identity,a.scheduler))):
        with pytest.raises(ValueError):ContinuousReplicaPool(replicas)
    a.scheduler.submit([1], max_new_tokens=2)
    with pytest.raises(ValueError, match='idle'):
        ContinuousReplicaPool((a,))
    pool.close()


@pytest.mark.parametrize('prompt,maximum,eos', [([],1,()),([True],1,()),([31],1,()),([1],0,()),([1],True,()),([1],1,(False,))])
def test_invalid_inputs_never_admitted(prompt, maximum, eos):
    pool, batches = make_pool()
    with pool:
        with pytest.raises(ValueError):pool.submit(prompt,max_new_tokens=maximum,eos_token_ids=eos)
        assert not any(b.calls for b in batches) and pool.info()['submitted'] == 0


def test_unstarted_close_and_explicit_start():
    pool, _ = make_pool()
    with pytest.raises(RuntimeError, match='explicitly'):
        pool.submit([1],max_new_tokens=1)
    pool.close()
    assert pool.info()['closed']
    with pytest.raises(RuntimeError):pool.start()


@pytest.mark.parametrize('routing,expected', [('least-work','1'), ('round-robin','0')])
def test_cost_policy_and_round_robin_control_are_explicit(routing,expected):
    entered = [threading.Event(), threading.Event()]
    release = threading.Event()
    pool, _ = make_pool(hooks=[lambda i=i: (entered[i].set(),release.wait(3))for i in range(2)])
    pool.routing = routing
    pool.start()
    try:
        a = pool.submit([1],max_new_tokens=50)
        b = pool.submit([2],max_new_tokens=1)
        assert all(e.wait(3)for e in entered)
        c = pool.submit([3],max_new_tokens=1)
        assert a.replica=='0' and b.replica=='1' and c.replica==expected
    finally:
        release.set();pool.close(timeout=5)


def test_safe_boundary_metadata_is_not_future_completion():
    batch = Batch()
    scheduler = ContinuousScheduler(batch)
    ticket = scheduler.submit([1],max_new_tokens=2)
    assert scheduler.contains(ticket)
    ticket.future.cancel()
    assert ticket.future.done() and scheduler.contains(ticket)
    scheduler.tick()
    assert not scheduler.contains(ticket)
    scheduler.close()


def test_startup_failure_joins_started_owner_without_leaking_handles(monkeypatch):
    pool, _ = make_pool()
    def fail():raise RuntimeError('thread startup failed')
    monkeypatch.setattr(pool._workers[1],'start',fail)
    with pytest.raises(RuntimeError,match='startup failed'):pool.start()
    assert all(not r['worker']['alive']for r in pool.info()['replicas'])
    assert pool.info()['closed']


def test_callback_cannot_join_its_own_owner_or_stop_healthy_pool():
    entered,release=threading.Event(),threading.Event()
    pool,_=make_pool(hooks=[lambda:(entered.set(),release.wait(3)),None])
    pool.start()
    try:
        a=pool.submit([1],max_new_tokens=1)
        assert entered.wait(3)
        errors=[];called=threading.Event()
        def cb(f):
            try:pool.close(timeout=0)
            except RuntimeError as e:errors.append(str(e))
            called.set()
        a.future.add_done_callback(cb);release.set()
        assert called.wait(3)and 'inference-owner'in errors[0]
        assert not pool.info()['stopping']
        pool.submit([2],max_new_tokens=1).future.result(timeout=3)
    finally:
        release.set();pool.close(timeout=5)
