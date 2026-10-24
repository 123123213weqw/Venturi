"""Static residency must never be represented as achieved occupancy."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sqlite3

import pytest


@pytest.fixture
def tool():
    path = Path(__file__).resolve().parents[1] / "tools/inspect_cuda_occupancy.py"
    spec = spec_from_file_location("cuda_occupancy", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resource_symbols_deduplicate(tool):
    assert tool.resource_symbols(
        " Function abc:\n REG:64\n Function abc:\n Function def:\n"
    ) == ["abc", "def"]


def test_grid_ceiling_is_not_residency_or_achieved(tool):
    hw = dict(sm_count=80, warp_size=32, threads_per_sm=2048)
    full = tool.grid_limits(320, 256, 4, hw)
    assert full["theoretical_occupancy_fraction"] == 0.5
    assert full["grid_capacity_waves"] == 1
    small = tool.grid_limits(16, 256, 4, hw)
    assert small["theoretical_occupancy_fraction"] == 0.5
    assert small["grid_limited_device_average_residency_upper_bound"] == 0.025
    assert "achieved_occupancy" not in small
    # Partial warps reserve whole-warp slots, not a fractional warp.
    assert tool.grid_limits(640, 33, 8, hw)["theoretical_active_warps_per_sm"] == 16


@pytest.mark.parametrize(
    "grid,threads,blocks", [(0, 256, 4), (1, 0, 4), (1, 256, 0), (True, 256, 4)]
)
def test_grid_reject_invalid(tool, grid, threads, blocks):
    with pytest.raises(ValueError):
        tool.grid_limits(
            grid, threads, blocks, dict(sm_count=80, warp_size=32, threads_per_sm=2048)
        )


def test_hash_mismatch_stops_before_cuda(tool, tmp_path, monkeypatch):
    library = tmp_path / "library.so"
    library.write_bytes(b"not a real library")
    monkeypatch.setattr(tool, "Driver", lambda: pytest.fail("CUDA must not be touched"))
    with pytest.raises(ValueError, match="hash mismatch"):
        tool.inspect(library, {}, [0], "nonexistent", "0" * 64)


def test_trace_preserves_thread_ownership_and_launch_geometry(tool, tmp_path):
    path = tmp_path / "trace.sqlite"
    db = sqlite3.connect(path)
    db.executescript("""
    create table StringIds(id integer,value text);
    insert into StringIds values (1,'native_decode'),(2,'kernel'),(3,'mangled');
    create table NVTX_EVENTS(start integer,end integer,text text,textId integer,globalTid integer);
    insert into NVTX_EVENTS values (0,500,null,1,11),(0,500,null,1,12);
    create table CUPTI_ACTIVITY_KIND_RUNTIME(start integer,end integer,globalTid integer,correlationId integer);
    insert into CUPTI_ACTIVITY_KIND_RUNTIME values (10,11,11,1),(12,13,12,2),(14,15,22,3);
    create table CUPTI_ACTIVITY_KIND_KERNEL(start integer,end integer,globalPid integer,correlationId integer,
      mangledName integer,demangledName integer,deviceId integer,blockX integer,blockY integer,blockZ integer,
      gridX integer,gridY integer,gridZ integer,registersPerThread integer,staticSharedMemory integer,
      dynamicSharedMemory integer,localMemoryPerThread integer,cacheConfig integer);
    insert into CUPTI_ACTIVITY_KIND_KERNEL values
      (100,200,9,1,3,2,0,256,1,1,16,1,1,64,128,0,0,0),
      (200,350,9,2,3,2,1,256,1,1,1280,1,1,64,128,0,0,0),
      (10,490,9,3,3,2,0,256,1,1,16,1,1,64,128,0,0,0);
    """)
    db.commit()
    db.close()
    cohort = tool.trace_launches(path)[0]
    assert cohort["summed_kernel_nanoseconds"] == 250
    assert len(cohort["launches"]) == 2
    a, b = cohort["launches"]
    assert a["grid_blocks"] == 1280 and a["trace_device_id"] == 1
    assert b["grid_blocks"] == 16 and b["trace_device_id"] == 0
    assert a["threads"] == 256 and a["trace_registers_per_thread"] == 64
