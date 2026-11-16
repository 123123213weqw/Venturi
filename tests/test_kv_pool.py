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
        ("kvh_info", [ctypes.c_void_p, ctypes.POINTER(u64)], ctypes.c_int),
        ("kvh_fill", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, u64, ctypes.c_ubyte], ctypes.c_int),
        ("kvh_publish", [ctypes.c_void_p, ctypes.c_void_p, u64, ctypes.POINTER(ctypes.c_void_p), u64, u64, ctypes.c_void_p, u64], ctypes.c_int),
        ("kvh_borrow", [ctypes.c_void_p, ctypes.c_void_p, u64, ctypes.POINTER(ctypes.c_void_p), u64, ctypes.POINTER(ctypes.c_void_p), u64], ctypes.c_int),
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
    assert harness.kvh_borrow(manager, pool, 42, fresh, 1, dest_buffers, 2) == 1
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
    assert harness.kvh_borrow(manager, pool, 999, stranger_ref, 1, dest_buffers, 2) == 0

    # After the donor entry is destroyed the record is no longer usable...
    harness.kvh_entry_destroy(pool, donor)
    again = harness.kvh_entry_create(pool, PLANE, 4096)
    again_ref = (ctypes.c_void_p * 1)(again)
    assert harness.kvh_borrow(manager, pool, 42, again_ref, 1, dest_buffers, 2) == 0

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
