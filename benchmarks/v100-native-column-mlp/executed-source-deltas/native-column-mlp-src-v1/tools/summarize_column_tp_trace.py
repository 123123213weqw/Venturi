"""Attribute graph-node MLP operator work; no model, TPS or achieved counters."""

import argparse
import json
from pathlib import Path
import sqlite3

from summarize_native_parallel_trace import union_nanoseconds
from summarize_native_runtime_trace import activity_summary


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
            for table in ["CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"]
            if table in tables
            for r in db.execute("select * from " + table)
        ]
        kernels = list(db.execute("select * from CUPTI_ACTIVITY_KIND_KERNEL"))
        if len({r["globalPid"] for r in kernels}) != 1:
            raise ValueError("One profiled process required")
        copies = list(db.execute("select * from CUPTI_ACTIVITY_KIND_MEMCPY"))
        result = []
        for e in db.execute("select * from NVTX_EVENTS where end is not null"):
            name = e["text"] or strings.get(e["textId"])
            if not name or not name.startswith("column_tp/"):
                continue
            ids = {
                a["correlationId"]
                for a in apis
                if a["globalTid"] == e["globalTid"]
                and e["start"] <= a["start"]
                and a["end"] <= e["end"]
            }
            ks = [k for k in kernels if k["correlationId"] in ids]
            cs = [c for c in copies if c["correlationId"] in ids]
            if len({k["deviceId"] for k in ks}) != 2:
                raise ValueError("Two trace-local devices required")
            missing = sum(
                1
                for k in kernels + copies
                if "graphNodeId" in k.keys()
                and k["graphNodeId"]
                and e["start"] <= k["start"]
                and k["end"] <= e["end"]
                and k["correlationId"] not in ids
            )
            if missing:
                raise ValueError("Unattributed graph nodes in operator range")
            result.append(
                dict(
                    name=name,
                    host_span_nanoseconds=e["end"] - e["start"],
                    **activity_summary(ks, cs, [], strings),
                    graph_nodes_actually_attributed=sum(
                        bool(k["graphNodeId"]) for k in ks if "graphNodeId" in k.keys()
                    ),
                    per_device_kernel_union_nanoseconds={
                        str(d): union_nanoseconds(
                            [(k["start"], k["end"]) for k in ks if k["deviceId"] == d]
                        )
                        for d in sorted({k["deviceId"] for k in ks})
                    },
                )
            )
        if not result:
            raise ValueError("No column-TP operator ranges found")
        return dict(scope=__doc__, achieved_hardware_counters=False, operators=result)
    finally:
        db.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sqlite")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    Path(a.output).write_text(json.dumps(summarize(a.sqlite), indent=2) + "\n")


if __name__ == "__main__":
    main()
