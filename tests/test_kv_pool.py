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
        ("kvh_publish", [ctypes.c_void_p, u64, ctypes.c_void_p, u64], ctypes.c_int),
        ("kvh_borrow", [ctypes.c_void_p, ctypes.c_void_p, u64, ctypes.c_void_p], ctypes.c_int),
    ]:
        fn = getattr(lib, name)
        fn.argtypes = args
        fn.restype = rest
    lib.kvh_error.restype = ctypes.c_char_p
    lib.kvh_destroy.argtypes = [ctypes.c_void_p]
    lib.kvh_entry_destroy.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.kvh_index_create.argtypes = []
    lib.kvh_index_create.restype = ctypes.c_void_p
    lib.kvh_index_destroy.argtypes = [ctypes.c_void_p]
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


def test_prefix_share_is_read_only_and_stable(harness):
    index = harness.kvh_index_create()
    pool = harness.kvh_create(0, PLANE * 4, GIB, 1 << 20)
    donor = harness.kvh_entry_create(pool, PLANE, 4096)
    prepare(harness, pool, [(donor, HALF)])
    assert harness.kvh_ensure(pool, donor, HALF) == 0
    assert harness.kvh_attach(pool, donor) == 0
    assert harness.kvh_fill(pool, donor, 0, HALF, 0x33) == 0
    before = info(harness, pool)["physical"]
    assert harness.kvh_publish(index, 42, donor, HALF) == 0

    borrower = harness.kvh_entry_create(pool, PLANE, 4096)
    prepare(harness, pool, [(donor, HALF)])  # borrower not yet a goal: borrow only
    assert harness.kvh_borrow(pool, index, 42, borrower) == 1
    assert snapshot(harness, pool, borrower, 0, HALF) == pattern(0x33, HALF)
    # Shared pages are reused, not duplicated.
    assert info(harness, pool)["physical"] == before

    # The borrower can be dropped without disturbing the donor.
    harness.kvh_entry_destroy(pool, borrower)
    assert snapshot(harness, pool, donor, 0, HALF) == pattern(0x33, HALF)
    # A second borrower with a missing key is rejected.
    fresh = harness.kvh_entry_create(pool, PLANE, 4096)
    assert harness.kvh_borrow(pool, index, 999, fresh) == 0
    harness.kvh_entry_destroy(pool, fresh)
    harness.kvh_entry_destroy(pool, donor)
    harness.kvh_destroy(pool)
    harness.kvh_index_destroy(index)
