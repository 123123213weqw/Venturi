"""Cohort correlation across threads and overlapping devices is not summed wall."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sqlite3


def test_parallel_trace_cohort_union_and_request_thread_selection(
    tmp_path, monkeypatch
):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "tools"))
    spec = spec_from_file_location(
        "parallel_trace", root / "tools/summarize_native_parallel_trace.py"
    )
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "trace.sqlite"
    db = sqlite3.connect(path)
    db.executescript("""
    create table StringIds(id integer,value text);
    insert into StringIds values (1,'native_unsplit'),(2,'q8');
    create table NVTX_EVENTS(start integer,end integer,text text,textId integer,globalTid integer);
    insert into NVTX_EVENTS values (0,500,null,1,11),(0,500,null,1,12);
    create table CUPTI_ACTIVITY_KIND_RUNTIME(start integer,end integer,globalTid integer,correlationId integer);
    insert into CUPTI_ACTIVITY_KIND_RUNTIME values (10,11,11,1),(12,13,12,2),(14,15,22,3);
    create table CUPTI_ACTIVITY_KIND_KERNEL(start integer,end integer,globalPid integer,correlationId integer,demangledName integer,deviceId integer);
    insert into CUPTI_ACTIVITY_KIND_KERNEL values (100,300,9,1,2,0),(200,400,9,2,2,1),(20,490,9,3,2,0);
    """)
    db.commit()
    db.close()
    r = module.summarize(path)["cohorts"][0]
    assert r["requests"] == 2
    assert r["kernels"] == 2 and r["kernel_nanoseconds"] == 400
    assert r["cross_device_kernel_union_nanoseconds"] == 300
    assert r["simultaneous_two_device_kernel_nanoseconds"] == 100
    assert r["activity_union_nanoseconds"] == 300
    assert r["gaps_between_this_process_activities_nanoseconds"] == 0
