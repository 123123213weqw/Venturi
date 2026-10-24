"""Driver launches and asynchronous execution must not disappear in attribution."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sqlite3


def test_native_trace_joins_launches_not_gpu_timestamp_window(tmp_path):
    file = tmp_path / "trace.sqlite"
    db = sqlite3.connect(file)
    db.executescript("""
    create table StringIds(id integer,value text);
    insert into StringIds values (1,'owned_q8'),(2,'vendor_driver_kernel'),(3,'native_decode');
    create table NVTX_EVENTS(start integer,end integer,text text,textId integer,globalTid integer);
    insert into NVTX_EVENTS values (200,700,null,3,11);
    create table CUPTI_ACTIVITY_KIND_RUNTIME(start integer,end integer,globalTid integer,correlationId integer);
    insert into CUPTI_ACTIVITY_KIND_RUNTIME values (250,260,11,1),(500,510,11,3),(250,260,12,4);
    create table CUPTI_ACTIVITY_KIND_DRIVER(start integer,end integer,globalTid integer,correlationId integer);
    insert into CUPTI_ACTIVITY_KIND_DRIVER values (270,280,11,2);
    create table CUPTI_ACTIVITY_KIND_KERNEL(start integer,end integer,globalPid integer,correlationId integer,demangledName integer);
    insert into CUPTI_ACTIVITY_KIND_KERNEL values (300,400,9,1,1),(720,820,9,2,2),(350,450,9,4,1);
    create table CUPTI_ACTIVITY_KIND_MEMCPY(start integer,end integer,correlationId integer,copyKind integer,bytes integer);
    insert into CUPTI_ACTIVITY_KIND_MEMCPY values (550,570,3,2,16);
    """)
    db.commit()
    db.close()
    path = (
        Path(__file__).resolve().parents[1] / "tools/summarize_native_runtime_trace.py"
    )
    spec = spec_from_file_location("native_trace_tool", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.summarize(file)
    assert not report["achieved_hardware_counters"]
    item = report["ranges"][0]
    assert item["kernels"] == 2
    assert item["kernel_nanoseconds"] == 200
    assert item["activity_span_nanoseconds"] == 520
    assert item["activity_union_nanoseconds"] == 220
    assert item["gaps_between_this_process_activities_nanoseconds"] == 300
    assert item["transfers"]["device_to_host"]["bytes"] == 16
    assert item["transfers"]["host_to_device"]["bytes"] == 0
