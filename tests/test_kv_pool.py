"""Unified KV pool mechanism tests (ops/csrc/kv), independent of any model.

Builds a tiny harness DSO from the kv headers and drives it through ctypes:
residency under pressure, spill/reload byte exactness, and read-only prefix
sharing. Gated to CUDA hosts exactly like the other native integration tests.
"""

import ctypes
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.cuda

FIELDS = (
    "enabled", "granule", "physical", "mapped", "virtual", "ram_backing", "valid",
    "evictions", "reloads", "d2h", "h2d", "ooms", "preflights", "peak_gpu",
    "peak_ram", "gpu_limit", "ram_limit", "entries",
)


@pytest.fixture(scope="module")
def harness(tmp_path_factory):
    if os.environ.get("VENTURI_TEST_CUDA") != "1":
        pytest.skip("Set VENTURI_TEST_CUDA=1 on a CUDA host")
    import torch

    assert torch.cuda.is_available(), "CUDA integration was requested but unavailable"
    compiler = shutil.which("nvcc")
    if not compiler:
        pytest.skip("nvcc is required to build the KV pool harness")
    from venturi.ops.build import _compile_library

    version = subprocess.run([compiler, "--version"], capture_output=True, text=True, check=True).stdout
    library = _compile_library(
        tmp_path_factory.mktemp("kv-harness"),
        "kvtest",
        ROOT / "tests" / "kv_harness.cu",
        ROOT,
        compiler,
        version,
        ["-O3", "-std=c++20", "-shared", "--cudart=static",
         "-Xcompiler=-fPIC", "-Xcompiler=-fvisibility=hidden", "-lcuda"],
        architecture="host",
    )
    lib = ctypes.CDLL(str(library))
    u64 = ctypes.c_uint64
    lib.kvh_create.argtypes = [ctypes.c_int, u64, u64, u64]
    lib.kvh_create.restype = ctypes.c_void_p
    for name, args, rest in [
        ("kvh_entry_create", [ctypes.c_void_p, u64, u64], ctypes.c_void_p),
        ("kvh_prepare", [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(u64), u64], ctypes.c_int),
        ("kvh_ensure", [ctypes.c_void_p, ctypes.c_void_p, u64], ctypes.c_int),
        ("kvh_attach", [ctypes.c_void_p, ctypes.c_void_p], ctypes.c_int),
        ("kvh_written", [ctypes.c_void_p, ctypes.c_void_p, u64], ctypes.c_int),
        ("kvh_snapshot", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, u64], ctypes.c_int),
        ("kvh_reset", [ctypes.c_void_p, ctypes.c_void_p], ctypes.c_int),
        ("kvh_append", [ctypes.c_void_p, ctypes.c_void_p, u64, ctypes.c_ubyte], ctypes.c_int),
        ("kvh_info", [ctypes.c_void_p, ctypes.POINTER(u64)], ctypes.c_int),
        ("kvh_fill", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, u64, ctypes.c_ubyte], ctypes.c_int),
        ("kvh_publish", [ctypes.c_void_p, ctypes.c_void_p, u64, ctypes.POINTER(ctypes.c_void_p), u64, u64, ctypes.c_void_p, u64], ctypes.c_int),
        ("kvh_borrow", [ctypes.c_void_p, ctypes.c_void_p, u64, ctypes.POINTER(ctypes.c_void_p), u64, ctypes.POINTER(ctypes.c_void_p), u64, u64], ctypes.c_int),
        ("kvh_forget", [ctypes.c_void_p, u64], ctypes.c_int),
        ("kvh_state_alloc", [u64, ctypes.c_ubyte], ctypes.c_void_p),
        ("kvh_state_free", [ctypes.c_void_p], ctypes.c_int),
        ("kvh_state_read", [ctypes.c_void_p, ctypes.c_void_p, u64], ctypes.c_int),
        ("kvh_state_overwrite", [ctypes.c_void_p, u64, ctypes.c_ubyte], ctypes.c_int),
    ]:
        fn = getattr(lib, name)
        fn.argtypes = args
        fn.restype = rest
    lib.kvh_error.restype = ctypes.c_char_p
    lib.kvh_destroy.argtypes = [ctypes.c_void_p]
    lib.kvh_entry_destroy.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.kvh_manager_create.argtypes = [ctypes.c_int]
    lib.kvh_manager_create.restype = ctypes.c_void_p
    lib.kvh_manager_destroy.argtypes = [ctypes.c_void_p]
    lib.kvh_manager_capacity.argtypes = [ctypes.c_void_p, u64]
    lib.kvh_manager_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(u64)]
    yield lib
    torch.cuda.synchronize()


def info(harness, pool):
    values = (ctypes.c_uint64 * 18)()
    assert harness.kvh_info(pool, values) == 0, harness.kvh_error().decode()
    return dict(zip(FIELDS, values))


def prepare(harness, pool, plan):
    entries = (ctypes.c_void_p * len(plan))(*[e for e, _ in plan])
    goals = (ctypes.c_uint64 * len(plan))(*[g for _, g in plan])
    assert harness.kvh_prepare(pool, entries, goals, len(plan)) == 0, harness.kvh_error().decode()


def snapshot(harness, pool, entry, kind, size):
    out = (ctypes.c_ubyte * size)()
    assert harness.kvh_snapshot(pool, entry, kind, out, size) == 0, harness.kvh_error().decode()
    return bytes(out)


def pattern(seed, size):
    return bytes((seed + i) & 0xFF for i in range(size))


GIB = 2**30
PLANE = 16 * 2**20  # 8 pages of 2 MiB per plane on V100
HALF = PLANE // 2


def test_spill_reload_round_trip(harness):
    # GPU budget holds only ONE entry's K+V pages at a time: growing the
    # second entry must evict the first to host backing and back.
    pool = harness.kvh_create(0, PLANE, GIB, 1 << 20)
    assert pool
    first = harness.kvh_entry_create(pool, PLANE, 4096)
    second = harness.kvh_entry_create(pool, PLANE, 4096)
    prepare(harness, pool, [(first, HALF), (second, HALF)])
    assert harness.kvh_ensure(pool, first, HALF) == 0
    assert harness.kvh_attach(pool, first) == 0
    assert harness.kvh_fill(pool, first, 0, HALF, 0x11) == 0
    # Growing the second entry forces eviction of the first to host backing.
    assert harness.kvh_ensure(pool, second, HALF) == 0
    assert harness.kvh_attach(pool, second) == 0
    assert harness.kvh_fill(pool, second, 0, HALF, 0x22) == 0
    stats = info(harness, pool)
    assert stats["ram_backing"] > 0 and stats["evictions"] >= 1 and stats["d2h"] > 0
    # Reload the first entry: async issue, attach, byte-exact snapshot.
    assert harness.kvh_ensure(pool, first, HALF) == 0
    assert harness.kvh_attach(pool, first) == 0
    assert snapshot(harness, pool, first, 0, HALF) == pattern(0x11, HALF)
    assert snapshot(harness, pool, second, 0, HALF) == pattern(0x22, HALF)
    assert info(harness, pool)["reloads"] >= 1
    harness.kvh_entry_destroy(pool, first)
    harness.kvh_entry_destroy(pool, second)
    harness.kvh_destroy(pool)


class State:
    """ctypes view over the layout-compatible KvhState array."""

    class Pair(ctypes.Structure):
        _fields_ = [("src", ctypes.c_void_p), ("bytes", ctypes.c_uint64)]


def test_prefix_share_snapshots_and_survives_donor(harness):
    manager = harness.kvh_manager_create(0)
    pool = harness.kvh_create(0, PLANE * 4, GIB, 1 << 20)
    donor = harness.kvh_entry_create(pool, PLANE, 4096)
    prepare(harness, pool, [(donor, HALF)])
    assert harness.kvh_ensure(pool, donor, HALF) == 0
    assert harness.kvh_attach(pool, donor) == 0
    assert harness.kvh_fill(pool, donor, 0, HALF, 0x33) == 0
    before = info(harness, pool)["physical"]

    # Two auxiliary recurrent states are frozen at publish time.
    states = (State.Pair * 2)(
        State.Pair(harness.kvh_state_alloc(4096, 0x55), 4096),
        State.Pair(harness.kvh_state_alloc(8192, 0x77), 8192),
    )
    entries = (ctypes.c_void_p * 1)(donor)
    assert harness.kvh_publish(manager, pool, 42, entries, 1, HALF, states, 2) == 0, harness.kvh_error().decode()

    # The donor keeps mutating its own auxiliary buffers after publishing.
    assert harness.kvh_state_overwrite(states[0].src, 4096, 0xEE) == 0

    borrower = harness.kvh_entry_create(pool, PLANE, 4096)
    prepare(harness, pool, [(donor, HALF)])
    dest_buffers = (ctypes.c_void_p * 2)(
        harness.kvh_state_alloc(4096, 0x00), harness.kvh_state_alloc(8192, 0x00)
    )
    fresh = (ctypes.c_void_p * 1)(borrower)
    assert harness.kvh_borrow(manager, pool, 42, fresh, 1, dest_buffers, 2, HALF) == 1
    assert snapshot(harness, pool, borrower, 0, HALF) == pattern(0x33, HALF)
    # Shared pages are reused, not duplicated.
    assert info(harness, pool)["physical"] == before

    # Frozen auxiliary state is restored byte-exactly (pre-mutation values).
    out = (ctypes.c_ubyte * 4096)()
    assert harness.kvh_state_read(dest_buffers[0], out, 4096) == 0
    assert bytes(out) == pattern(0x55, 4096)
    out = (ctypes.c_ubyte * 8192)()
    assert harness.kvh_state_read(dest_buffers[1], out, 8192) == 0
    assert bytes(out) == pattern(0x77, 8192)

    # A borrower against a missing key is rejected without side effects.
    stranger = harness.kvh_entry_create(pool, PLANE, 4096)
    stranger_ref = (ctypes.c_void_p * 1)(stranger)
    assert harness.kvh_borrow(manager, pool, 999, stranger_ref, 1, dest_buffers, 2, HALF) == 0

    # After the donor entry is destroyed the record is no longer usable...
    harness.kvh_entry_destroy(pool, donor)
    again = harness.kvh_entry_create(pool, PLANE, 4096)
    again_ref = (ctypes.c_void_p * 1)(again)
    assert harness.kvh_borrow(manager, pool, 42, again_ref, 1, dest_buffers, 2, HALF) == 0

    # ...but the earlier borrower keeps its shared pages and data.
    assert snapshot(harness, pool, borrower, 0, HALF) == pattern(0x33, HALF)
    harness.kvh_state_free(states[0].src)
    harness.kvh_state_free(states[1].src)
    harness.kvh_state_free(dest_buffers[0])
    harness.kvh_state_free(dest_buffers[1])
    harness.kvh_entry_destroy(pool, borrower)
    harness.kvh_entry_destroy(pool, stranger)
    harness.kvh_entry_destroy(pool, again)
    harness.kvh_destroy(pool)
    harness.kvh_manager_destroy(manager)


def manager_info(harness, manager):
    values=(ctypes.c_uint64*3)()
    assert harness.kvh_manager_info(manager,values)==0
    return tuple(values)


class PrefixFixture:
    """Small real-GPU pool; all owned handles close even on assertion failure."""
    def __init__(self,harness,budget_pages=32):
        self.h=harness;self.manager=harness.kvh_manager_create(0)
        # Discover actual device granularity, not a hard-coded V100 page size.
        probe=harness.kvh_create(0,0,GIB,1<<20)
        assert probe
        self.g=info(harness,probe)['granule'];harness.kvh_destroy(probe)
        self.pool=harness.kvh_create(0,budget_pages*self.g,GIB,1<<20)
        assert self.pool
        self.entries=[];self.states=[]
    def entry(self):
        e=self.h.kvh_entry_create(self.pool,8*self.g,8);assert e
        self.entries.append(e);return e
    def fill(self,e,n,seed=0x31):
        prepare(self.h,self.pool,[(e,n)])
        assert self.h.kvh_ensure(self.pool,e,n)==0,self.h.kvh_error().decode()
        assert self.h.kvh_attach(self.pool,e)==0
        for k in(0,1):assert self.h.kvh_fill(self.pool,e,k,n,seed+k)==0
    def publish(self,key,e,n,state_bytes=0):
        sources=[]
        if state_bytes:
            ptr=self.h.kvh_state_alloc(state_bytes,0x71);assert ptr
            self.states.append(ptr);sources=[State.Pair(ptr,state_bytes)]
        extra=(State.Pair*len(sources))(*sources)
        return self.h.kvh_publish(self.manager,self.pool,key,(ctypes.c_void_p*1)(e),1,n,extra,len(sources))
    def borrow(self,key,e,n,dests=()):
        return self.h.kvh_borrow(self.manager,self.pool,key,(ctypes.c_void_p*1)(e),1,
            (ctypes.c_void_p*len(dests))(*dests),len(dests),n)
    def close(self):
        for e in reversed(self.entries):self.h.kvh_entry_destroy(self.pool,e)
        for state in self.states:self.h.kvh_state_free(state)
        self.h.kvh_manager_destroy(self.manager);self.h.kvh_destroy(self.pool)


def test_prefix_reset_invalidates_key_without_mutating_live_borrower(harness):
    f=PrefixFixture(harness)
    try:
        donor,borrower,later=(f.entry()for _ in range(3));f.fill(donor,32)
        assert f.publish(1,donor,32)==0 and f.borrow(1,borrower,32)==1
        assert harness.kvh_reset(f.pool,donor)==0
        f.fill(donor,32,0x55)
        assert snapshot(harness,f.pool,borrower,0,32)==pattern(0x31,32)
        assert f.borrow(1,later,32)==0
        # Old borrower remains usable; no replay of the donor's new prompt.
        prepare(harness,f.pool,[(borrower,40)])
        assert harness.kvh_append(f.pool,borrower,8,0x66)==0
        assert snapshot(harness,f.pool,borrower,0,40)==pattern(0x31,32)+bytes([0x66])*8
    finally:f.close()


def test_prefix_COW_tight_budget_reload_and_flags(harness):
    f=PrefixFixture(harness,budget_pages=2)
    try:
        donor,borrower=f.entry(),f.entry();f.fill(donor,32)
        assert f.publish(1,donor,32)==0 and f.borrow(1,borrower,32)==1
        prepare(harness,f.pool,[(borrower,40)])
        assert harness.kvh_append(f.pool,borrower,8,0x77)==0,harness.kvh_error().decode()
        for e,n,expected in((donor,32,pattern(0x31,32)),(borrower,40,pattern(0x31,32)+bytes([0x77])*8)):
            prepare(harness,f.pool,[(e,n)])
            assert harness.kvh_ensure(f.pool,e,n)==0
            assert harness.kvh_attach(f.pool,e)==0
            assert snapshot(harness,f.pool,e,0,n)==expected
        prepare(harness,f.pool,[(borrower,48)])
        assert harness.kvh_append(f.pool,borrower,8,0x88)==0
    finally:f.close()


def test_prefix_COW_keeps_complete_pages_shared_and_counts_host_copies(harness):
    f=PrefixFixture(harness)
    try:
        donor,borrower=f.entry(),f.entry();n=2*f.g+32;f.fill(donor,n)
        assert f.publish(1,donor,n)==0 and f.borrow(1,borrower,n)==1
        before=info(harness,f.pool)
        prepare(harness,f.pool,[(borrower,n+8)])
        assert harness.kvh_append(f.pool,borrower,8,0x99)==0
        after=info(harness,f.pool)
        assert after['physical']==8*f.g  # six original pages, two private tails
        assert after['d2h']-before['d2h']==after['h2d']-before['h2d']==2*f.g
        assert snapshot(harness,f.pool,borrower,0,n+8)==pattern(0x31,n)+bytes([0x99])*8
        assert snapshot(harness,f.pool,donor,0,n)==pattern(0x31,n)
    finally:f.close()


def test_prefix_LRU_closed_and_reset_borrowers_release_pins(harness):
    f=PrefixFixture(harness)
    try:
        donor=f.entry();f.fill(donor,32)
        assert harness.kvh_manager_capacity(f.manager,2)==0
        for key in range(70):
            assert f.publish(key,donor,32)==0,harness.kvh_error().decode()
            borrower=f.entry();assert f.borrow(key,borrower,32)==1
            assert manager_info(harness,f.manager)[1]==1
            if key%2:assert harness.kvh_reset(f.pool,borrower)==0
            else:
                harness.kvh_entry_destroy(f.pool,borrower);f.entries.remove(borrower)
            assert manager_info(harness,f.manager)[1]==0
        assert manager_info(harness,f.manager)[0]==2
    finally:f.close()


def test_prefix_rejection_is_atomic_and_does_not_allocate_snapshots(harness):
    import torch
    f=PrefixFixture(harness)
    try:
        donor,borrower=f.entry(),f.entry();f.fill(donor,32)
        assert harness.kvh_manager_capacity(f.manager,1)==0
        assert f.publish(1,donor,32,state_bytes=2**20)==0
        destination=harness.kvh_state_alloc(2**20,0);f.states.append(destination)
        # Wrong length and missing destination fail before the first mapping.
        assert f.borrow(1,borrower,24,(destination,))==-1
        assert f.borrow(1,borrower,32,(None,))==-1
        assert f.borrow(1,borrower,32,(destination,))==1
        extra=(State.Pair*1)(State.Pair(f.states[0],2**20));entries=(ctypes.c_void_p*1)(donor)
        torch.cuda.synchronize();before=torch.cuda.mem_get_info()[0]
        for key in range(2,18):
            assert harness.kvh_publish(f.manager,f.pool,key,entries,1,32,extra,1)==-1
        torch.cuda.synchronize();after=torch.cuda.mem_get_info()[0]
        assert after>=before-2**20  # rejects precede cudaMalloc, not sixteen leaked MiB
        assert manager_info(harness,f.manager)==(1,1,2**20)
    finally:f.close()
