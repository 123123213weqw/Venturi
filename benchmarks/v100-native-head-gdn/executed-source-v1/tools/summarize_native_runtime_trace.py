"""Profiler-only CUDA attribution by native NVTX launch/API correlation.

Includes runtime and driver launches, including cuBLAS driver API launches.
GPU activity can execute beyond its submitting CPU range. Timeline gaps are
gaps between this process's traced activities, not device-wide idle time or an
explanation of every CPU delay. No achieved hardware counters are inferred.
Kernel-duration sums across GPUs can exceed wall time because devices overlap;
per-device attribution and the cross-device activity union are reported separately.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3


def activity_summary(kernels, copies, memset, strings):
    by_name = defaultdict(lambda: dict(count=0, nanoseconds=0))
    for k in kernels:
        name = strings[k["demangledName"]]
        by_name[name]["count"] += 1
        by_name[name]["nanoseconds"] += k["end"] - k["start"]
    intervals = sorted((r["start"], r["end"]) for r in kernels + copies + memset)
    if not intervals:
        raise ValueError("Expected CUDA activity in native range")
    covered = 0
    begin, end = intervals[0]
    for a, b in intervals[1:]:
        if a > end:
            covered += end - begin
            begin, end = a, b
        else:
            end = max(end, b)
    covered += end - begin
    span = max(b for _, b in intervals) - intervals[0][0]
    transfers = {}
    for kind, name in ((1, "host_to_device"), (2, "device_to_host")):
        chosen = [c for c in copies if c["copyKind"] == kind]
        transfers[name] = dict(
            count=len(chosen),
            bytes=sum(c["bytes"] for c in chosen),
            nanoseconds=sum(c["end"] - c["start"] for c in chosen),
        )
    return dict(
        kernels=len(kernels),
        kernel_nanoseconds=sum(k["end"] - k["start"] for k in kernels),
        activity_span_nanoseconds=span,
        activity_union_nanoseconds=covered,
        gaps_between_this_process_activities_nanoseconds=span - covered,
        transfers=transfers,
        kernel_by_name=sorted(
            [dict(name=name, **v) for name, v in by_name.items()],
            key=lambda x: x["nanoseconds"],
            reverse=True,
        ),
    )


def device_id(row):
    # Older exports may omit this column: preserve an unknown bucket rather
    # than incorrectly attributing its activities to device zero.
    return row["deviceId"] if "deviceId" in row.keys() else None


def summarize(path):
    db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        tables = {
            r[0]
            for r in db.execute("select name from sqlite_master where type='table'")
        }
        strings = dict(db.execute("select id,value from StringIds"))
        kernels = list(db.execute("select * from CUPTI_ACTIVITY_KIND_KERNEL"))
        if len({r["globalPid"] for r in kernels}) != 1:
            raise ValueError("Only a single profiled process is supported")
        apis = []
        for name in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
            if name in tables:
                apis.extend(db.execute(f"select * from {name}"))
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
        result = []
        for event in db.execute("select * from NVTX_EVENTS where end is not null"):
            label = event["text"]
            if label is None and "textId" in event.keys():
                label = strings.get(event["textId"])
            if label not in ("native_ttft", "native_decode", "native_unsplit"):
                continue
            launches = [
                a
                for a in apis
                if a["globalTid"] == event["globalTid"]
                and event["start"] <= a["start"]
                and a["end"] <= event["end"]
            ]
            ids = {a["correlationId"] for a in launches}
            ks = [k for k in kernels if k["correlationId"] in ids]
            cs = [c for c in copies if c["correlationId"] in ids]
            ms = [m for m in memset if m["correlationId"] in ids]
            devices = sorted(
                {device_id(r) for r in ks + cs + ms},
                key=lambda d: (d is None, d if d is not None else 0),
            )
            result.append(
                dict(
                    name=label,
                    host_range_nanoseconds=event["end"] - event["start"],
                    **activity_summary(ks, cs, ms, strings),
                    by_device=[
                        dict(
                            device_id=d,
                            **activity_summary(
                                [k for k in ks if device_id(k) == d],
                                [c for c in cs if device_id(c) == d],
                                [m for m in ms if device_id(m) == d],
                                strings,
                            ),
                        )
                        for d in devices
                    ],
                )
            )
        if not result:
            raise ValueError("No expected native NVTX ranges found")
        return dict(scope=__doc__, achieved_hardware_counters=False, ranges=result)
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
