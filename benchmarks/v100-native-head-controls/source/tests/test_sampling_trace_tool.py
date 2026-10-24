"""Trace attribution must use launch correlation and validate actual payloads."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sqlite3

import pytest


def load():
    spec = spec_from_file_location(
        "greedy_trace",
        Path(__file__).resolve().parents[1] / "tools/summarize_greedy_trace.py",
    )
    tool = module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def fixture(path):
    db = sqlite3.connect(path)
    for sql in (
        "create table StringIds(id integer,value text)",
        "create table NVTX_EVENTS(start integer,end integer,text text,globalTid integer)",
        "create table CUPTI_ACTIVITY_KIND_RUNTIME(start integer,end integer,globalTid integer,correlationId integer)",
        "create table CUPTI_ACTIVITY_KIND_KERNEL(start integer,end integer,globalPid integer,correlationId integer,demangledName integer)",
        "create table CUPTI_ACTIVITY_KIND_MEMCPY(start integer,end integer,correlationId integer,bytes integer,copyKind integer)",
    ):
        db.execute(sql)
    db.executemany(
        "insert into StringIds values(?,?)",
        [(1, "greedy_partials<float>"), (2, "greedy_finish")],
    )
    for i, (rows, native) in enumerate(((1, True), (1, False), (8, True), (8, False))):
        offset = i * 10000
        mode = "native" if native else "cpu"
        db.execute(
            "insert into NVTX_EVENTS values(?,?,?,?)",
            (offset, offset + 5000, f"{mode}_greedy_rows_{rows}", 99),
        )
        if native:
            for j in (1, 2):
                corr = i * 10 + j
                db.execute(
                    "insert into CUPTI_ACTIVITY_KIND_RUNTIME values(?,?,?,?)",
                    (offset + j * 100, offset + j * 100 + 10, 99, corr),
                )
                db.execute(
                    "insert into CUPTI_ACTIVITY_KIND_KERNEL values(?,?,?,?,?)",
                    (offset + 6000 + j * 100, offset + 6000 + j * 100 + 50, 7, corr, j),
                )
        corr = i * 10 + 3
        db.execute(
            "insert into CUPTI_ACTIVITY_KIND_RUNTIME values(?,?,?,?)",
            (offset + 300, offset + 350, 99, corr),
        )
        db.execute(
            "insert into CUPTI_ACTIVITY_KIND_MEMCPY values(?,?,?,?,2)",
            (offset + 7000, offset + 7500, corr, rows * (16 if native else 248320 * 4)),
        )
    db.commit()
    db.close()


def test_launch_attribution_not_gpu_timestamp_overlap(tmp_path):
    p = tmp_path / "trace.sqlite"
    fixture(p)
    result = load().summarize(p)
    assert len(result["ranges"]) == 4
    assert [r["copy_bytes"] for r in result["ranges"]] == [16, 993280, 128, 7946240]
    # GPU activity is deliberately queued beyond its submitting host range.
    assert result["ranges"][0]["partials_us"] == 0.05
    assert result["ranges"][0]["interkernel_gap_us"] == 0.05
    assert result["ranges"][0]["host_between_launch_api_gap_us"] == 0.09
    assert not result["achieved_hardware_counters"]


def test_reject_wrong_copy_size_and_multiple_processes(tmp_path):
    p = tmp_path / "trace.sqlite"
    fixture(p)
    db = sqlite3.connect(p)
    db.execute("update CUPTI_ACTIVITY_KIND_MEMCPY set bytes=32 where correlationId=3")
    db.commit()
    with pytest.raises(AssertionError, match="payload"):
        load().summarize(p)
    db.execute(
        "update CUPTI_ACTIVITY_KIND_MEMCPY set bytes=16,copyKind=1 where correlationId=3"
    )
    db.commit()
    with pytest.raises(AssertionError, match="D2H"):
        load().summarize(p)
    db.execute(
        "update CUPTI_ACTIVITY_KIND_KERNEL set globalPid=8 where correlationId=1"
    )
    db.commit()
    with pytest.raises(AssertionError, match="single profiled"):
        load().summarize(p)
    db.close()
