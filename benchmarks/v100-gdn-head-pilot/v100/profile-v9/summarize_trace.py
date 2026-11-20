"""Separate research GDN timeline attribution; no achieved counters or model TPS."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from summarize_native_parallel_trace import union_nanoseconds  # noqa: E402
from summarize_native_runtime_trace import activity_summary  # noqa: E402


def summarize(path):
    db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        tables = {
            r[0]
            for r in db.execute("select name from sqlite_master where type='table'")
        }
        strings = dict(db.execute("select id,value from StringIds"))
        apis = [
            r
            for t in ["CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"]
            if t in tables
            for r in db.execute("select * from " + t)
        ]
        kernels = list(db.execute("select * from CUPTI_ACTIVITY_KIND_KERNEL"))
        if len({r["globalPid"] for r in kernels}) != 1:
            raise ValueError("Exactly one captured process is required")
        copies = list(db.execute("select * from CUPTI_ACTIVITY_KIND_MEMCPY"))
        result = []
        seen = set()
        for e in db.execute("select * from NVTX_EVENTS where end is not null"):
            label = e["text"] or strings.get(e["textId"])
            if not label or not re.fullmatch(
                r"gdn_\d+_(1x1|1x3|8x1)_(row|head)", label
            ):
                continue
            if label in seen:
                raise ValueError("Repeated measurement range: " + label)
            seen.add(label)
            owned = [
                a
                for a in apis
                if a["globalTid"] == e["globalTid"]
                and e["start"] <= a["start"]
                and a["end"] <= e["end"]
            ]
            ids = {a["correlationId"] for a in owned}
            ks = [k for k in kernels if k["correlationId"] in ids]
            cs = [c for c in copies if c["correlationId"] in ids]
            devices = sorted({k["deviceId"] for k in ks})
            if len(devices) != 2:
                raise ValueError("Two trace-local devices required")
            unions = {
                str(d): union_nanoseconds(
                    [(k["start"], k["end"]) for k in ks if k["deviceId"] == d]
                )
                for d in devices
            }
            combined = union_nanoseconds([(k["start"], k["end"]) for k in ks])
            launches = defaultdict(lambda: dict(count=0, nanoseconds=0))
            for k in ks:
                fields = (
                    "deviceId",
                    "blockX",
                    "blockY",
                    "blockZ",
                    "gridX",
                    "gridY",
                    "gridZ",
                    "registersPerThread",
                    "staticSharedMemory",
                    "dynamicSharedMemory",
                    "localMemoryPerThread",
                )
                key = (strings[k["demangledName"]],) + tuple(k[f] for f in fields)
                launches[key]["count"] += 1
                launches[key]["nanoseconds"] += k["end"] - k["start"]
            geometry = [
                dict(zip(("name",) + fields, key), **v)
                for key, v in sorted(
                    launches.items(), key=lambda kv: -kv[1]["nanoseconds"]
                )
            ]
            by_api = defaultdict(lambda: dict(count=0, nanoseconds=0))
            for api in owned:
                name = strings[api["nameId"]]
                by_api[name]["count"] += 1
                by_api[name]["nanoseconds"] += api["end"] - api["start"]
            result.append(
                dict(
                    name=label,
                    host_span_nanoseconds=e["end"] - e["start"],
                    **activity_summary(ks, cs, [], strings),
                    per_device_kernel_union_nanoseconds=unions,
                    cross_device_kernel_union_nanoseconds=combined,
                    simultaneous_two_device_kernel_nanoseconds=sum(unions.values())
                    - combined,
                    actual_launch_geometry=geometry,
                    api_duration_sums_not_critical_path=[
                        dict(name=n, **v)
                        for n, v in sorted(
                            by_api.items(), key=lambda kv: -kv[1]["nanoseconds"]
                        )
                    ],
                    d2d=dict(
                        count=sum(c["copyKind"] == 8 for c in cs),
                        bytes=sum(c["bytes"] for c in cs if c["copyKind"] == 8),
                    ),
                )
            )
        if len(result) != 6:
            raise ValueError(
                "Expected six unique layer0 pilot ranges, got " + str(len(result))
            )
        return dict(scope=__doc__, achieved_hardware_counters=False, operators=result)
    finally:
        db.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sqlite")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    Path(a.output).write_text(json.dumps(summarize(a.sqlite), indent=2) + "\n")
