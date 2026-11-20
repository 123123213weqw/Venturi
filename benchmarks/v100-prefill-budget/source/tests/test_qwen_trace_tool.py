"""Trace attribution and theoretical limits, independent of CUDA or Nsight installs."""

import json
from pathlib import Path
import sqlite3
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from analyze_qwen_trace import summarize  # noqa: E402


def test_graph_kernels_follow_the_host_launch_nvtx_phase(tmp_path):
    db = tmp_path / "trace.sqlite"
    c = sqlite3.connect(db)
    c.executescript("""
      create table StringIds (id integer, value text);
      insert into StringIds values (1,'prefill'), (2,'decode'), (3,'q4_rowsplit_gemm_mma_kernel'),
                                  (4,'recurrent_batch_update_kernel'), (5,'cudaGraphLaunch');
      create table NVTX_EVENTS(start integer,end integer,globalTid integer,text text,textId integer);
      insert into NVTX_EVENTS values (0,40,1,null,1),(50,90,1,null,2);
      create table CUPTI_ACTIVITY_KIND_RUNTIME(start integer,end integer,globalTid integer,
                                              correlationId integer,nameId integer);
      insert into CUPTI_ACTIVITY_KIND_RUNTIME values (1,2,1,11,5),(51,52,1,12,5);
      create table CUPTI_ACTIVITY_KIND_KERNEL(start integer,end integer,correlationId integer,
        shortName integer,blockX integer,blockY integer,blockZ integer,staticSharedMemory integer,
        dynamicSharedMemory integer,registersPerThread integer,localMemoryPerThread integer,
        gridX integer,gridY integer,gridZ integer);
      insert into CUPTI_ACTIVITY_KIND_KERNEL values (10,30,11,3,128,1,1,0,25000,40,0,100,1,1),
                                                   (60,70,12,4,256,1,1,0,0,128,0,100,1,1);
    """)
    c.commit()
    c.close()
    result = summarize(
        db,
        dict(
            blocks_per_sm=24,
            threads_per_sm=1536,
            registers_per_sm=65536,
            shared_bytes_per_sm=100000,
        ),
    )
    assert result["phases"]["prefill"]["projection_fraction"] == 1
    assert result["phases"]["decode"]["projection_fraction"] == 0
    geometry = result["phases"]["prefill"]["top_kernels"][0]["launch_geometry"][0]
    assert geometry["resource_only_resident_blocks_upper_bound"] == 4
    assert geometry["resource_only_thread_occupancy_upper_bound"] == pytest.approx(
        1 / 3
    )
    assert result["kernel_interval_coverage_fraction"] == 0.5
    json.dumps(result, allow_nan=False)
