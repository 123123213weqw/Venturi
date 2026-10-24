"""Cohort CUDA timelines joined by each submitting request's NVTX/thread range.

Per-device kernel unions reveal two-GPU overlap, not achieved SM occupancy.
Duration sums are never substituted for request/cohort wall time. Timeline gaps
refer only to this process's captured activities, not device-wide idle time.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3

from summarize_native_runtime_trace import activity_summary


def union_nanoseconds(intervals):
    if not intervals:
        return 0
    intervals = sorted(intervals)
    first, last = intervals[0]
    covered = 0
    for a, b in intervals[1:]:
        if a > last:
            covered += last - first
            first, last = a, b
        else:
            last = max(last, b)
    return covered + last - first


def summarize(path):
    db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        tables = {
            r[0]
            for r in db.execute("select name from sqlite_master where type='table'")
        }
        strings = dict(db.execute("select id,value from StringIds"))
        apis = []
        for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
            if table in tables:
                apis.extend(db.execute(f"select * from {table}"))
        kernels = list(db.execute("select * from CUPTI_ACTIVITY_KIND_KERNEL"))
        if len({r["globalPid"] for r in kernels}) != 1:
            raise ValueError("Only a single profiled process is supported")
        copies = (
            list(db.execute("select * from CUPTI_ACTIVITY_KIND_MEMCPY"))
            if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables
            else []
        )
        memset = (
            list(db.execute("select * from CUPTI_ACTIVITY_KIND_MEMSET"))
            if "CUPTI_ACTIVITY_KIND_MEMSET" in tables
            else []
        )
        events = defaultdict(list)
        for e in db.execute("select * from NVTX_EVENTS where end is not null"):
            label = e["text"] or strings.get(e["textId"])
            if label in ("native_ttft", "native_decode", "native_unsplit"):
                events[label].append(e)
        result = []
        for label, ranges in events.items():
            ids = {
                a["correlationId"]
                for a in apis
                for e in ranges
                if a["globalTid"] == e["globalTid"]
                and e["start"] <= a["start"]
                and a["end"] <= e["end"]
            }
            ks = [k for k in kernels if k["correlationId"] in ids]
            cs = [k for k in copies if k["correlationId"] in ids]
            ms = [k for k in memset if k["correlationId"] in ids]
            devices = sorted({k["deviceId"] for k in ks})
            if len(devices) != 2:
                raise ValueError("Expected two trace-local CUDA device IDs")
            unions = {
                d: union_nanoseconds(
                    [(k["start"], k["end"]) for k in ks if k["deviceId"] == d]
                )
                for d in devices
            }
            all_union = union_nanoseconds([(k["start"], k["end"]) for k in ks])
            result.append(
                dict(
                    name=label,
                    requests=len(ranges),
                    host_cohort_span_nanoseconds=max(e["end"] for e in ranges)
                    - min(e["start"] for e in ranges),
                    **activity_summary(ks, cs, ms, strings),
                    per_device_kernel_union_nanoseconds=unions,
                    cross_device_kernel_union_nanoseconds=all_union,
                    simultaneous_two_device_kernel_nanoseconds=sum(unions.values())
                    - all_union,
                )
            )
        if not result:
            raise ValueError("No native per-request ranges found")
        return dict(scope=__doc__, achieved_hardware_counters=False, cohorts=result)
    finally:
        db.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sqlite")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    Path(args.output).write_text(json.dumps(summarize(args.sqlite), indent=2) + "\n")


if __name__ == "__main__":
    main()
