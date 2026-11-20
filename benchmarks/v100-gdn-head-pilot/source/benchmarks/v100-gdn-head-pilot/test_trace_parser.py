"""Synthetic correlation checks, including asynchronous GPU work past host ranges."""

import importlib.util
from pathlib import Path
import sqlite3

import pytest


def load():
    path = Path(__file__).with_name("summarize_trace.py")
    spec = importlib.util.spec_from_file_location("gdn_pilot_trace", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(tmp_path):
    p = tmp_path / "trace.sqlite"
    db = sqlite3.connect(p)
    db.executescript("""
      create table StringIds(id integer,value text);
      insert into StringIds values (1,'projection'),(2,'cudaLaunchKernel');
      create table NVTX_EVENTS(start integer,end integer,text text,textId integer,globalTid integer);
      create table CUPTI_ACTIVITY_KIND_RUNTIME(start integer,end integer,globalTid integer,correlationId integer,nameId integer);
      create table CUPTI_ACTIVITY_KIND_KERNEL(start integer,end integer,globalPid integer,correlationId integer,demangledName integer,deviceId integer,blockX integer,blockY integer,blockZ integer,gridX integer,gridY integer,gridZ integer,registersPerThread integer,staticSharedMemory integer,dynamicSharedMemory integer,localMemoryPerThread integer);
      create table CUPTI_ACTIVITY_KIND_MEMCPY(start integer,end integer,correlationId integer,copyKind integer,bytes integer);
    """)
    labels = [
        f"gdn_0_{work}_{mode}"
        for work in ["1x1", "1x3", "8x1"]
        for mode in ["row", "head"]
    ]
    for i, name in enumerate(labels):
        start = i * 1000
        db.execute(
            "insert into NVTX_EVENTS values (?,?,?,?,?)",
            (start, start + 100, name, None, 11),
        )
        for device in (0, 1):
            corr = i * 10 + device
            db.execute(
                "insert into CUPTI_ACTIVITY_KIND_RUNTIME values (?,?,?,?,?)",
                (start + 10 + device, start + 12 + device, 11, corr, 2),
            )
            db.execute(
                "insert into CUPTI_ACTIVITY_KIND_KERNEL values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    start + 200 + device * 50,
                    start + 300 + device * 50,
                    9,
                    corr,
                    1,
                    device,
                    128,
                    1,
                    1,
                    400,
                    1,
                    1,
                    48,
                    0,
                    9216,
                    0,
                ),
            )
        # Same host time but different submitting thread must be excluded.
        db.execute(
            "insert into CUPTI_ACTIVITY_KIND_RUNTIME values (?,?,?,?,?)",
            (start + 14, start + 15, 22, i * 10 + 5, 2),
        )
        db.execute(
            "insert into CUPTI_ACTIVITY_KIND_KERNEL values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                start + 10,
                start + 900,
                9,
                i * 10 + 5,
                1,
                0,
                256,
                1,
                1,
                1,
                1,
                1,
                32,
                0,
                0,
                0,
            ),
        )
        db.execute(
            "insert into CUPTI_ACTIVITY_KIND_MEMCPY values (?,?,?,?,?)",
            (start + 180, start + 200, i * 10, 1, 512),
        )
    db.commit()
    db.close()
    return p


def test_launch_ownership_overlap_geometry_and_async_span(tmp_path):
    report = load().summarize(fixture(tmp_path))
    assert not report["achieved_hardware_counters"]
    assert len(report["operators"]) == 6
    for item in report["operators"]:
        assert item["kernels"] == 2 and item["kernel_nanoseconds"] == 200
        assert item["host_span_nanoseconds"] == 100
        assert item["activity_span_nanoseconds"] == 170
        assert item["cross_device_kernel_union_nanoseconds"] == 150
        assert item["simultaneous_two_device_kernel_nanoseconds"] == 50
        assert item["transfers"]["host_to_device"]["bytes"] == 512
        assert all(
            x["gridX"] == 400 and x["registersPerThread"] == 48
            for x in item["actual_launch_geometry"]
        )
        assert item["api_duration_sums_not_critical_path"][0]["count"] == 2


def test_incomplete_profile_rejected(tmp_path):
    path = fixture(tmp_path)
    with sqlite3.connect(path) as db:
        db.execute("delete from NVTX_EVENTS where start=5000")
    with pytest.raises(ValueError, match="six unique"):
        load().summarize(path)
