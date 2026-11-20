import pathlib
import json
import sqlite3
import argparse

p = argparse.ArgumentParser(
    description="Standalone projection attribution by submitting NVTX range/correlation, not throughput or achieved occupancy"
)
p.add_argument("trace")
p.add_argument("--output", required=True)
a = p.parse_args()
db = sqlite3.connect(pathlib.Path(a.trace).resolve().as_uri() + "?mode=ro", uri=True)
db.row_factory = sqlite3.Row
strings = dict(db.execute("select id,value from StringIds"))
ks = list(db.execute("select * from CUPTI_ACTIVITY_KIND_KERNEL"))
apis = []
for table in ["CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"]:
    try:
        apis += list(db.execute("select * from " + table))
    except sqlite3.OperationalError:
        pass
result = []
for event in db.execute("select * from NVTX_EVENTS where end is not null"):
    label = event["text"] or strings.get(event["textId"], "")
    if not label.startswith("projection_"):
        continue
    ids = {
        a["correlationId"]
        for a in apis
        if a["globalTid"] == event["globalTid"]
        and event["start"] <= a["start"]
        and a["end"] <= event["end"]
    }
    kernels = [k for k in ks if k["correlationId"] in ids]
    groups = {}
    last = -1
    union = 0
    for k in sorted(kernels, key=lambda k: k["start"]):
        union += max(0, k["end"] - max(k["start"], last))
        last = max(last, k["end"])
        key = (
            strings[k["demangledName"]],
            k["gridX"],
            k["gridY"],
            k["blockX"],
            k["registersPerThread"],
            k["staticSharedMemory"],
            k["dynamicSharedMemory"],
        )
        d = groups.setdefault(key, dict(count=0, kernel_nanoseconds=0))
        d["count"] += 1
        d["kernel_nanoseconds"] += k["end"] - k["start"]
    span = max(k["end"] for k in kernels) - min(k["start"] for k in kernels)
    result.append(
        dict(
            label=label,
            launches=len(kernels),
            kernel_nanoseconds=sum(k["end"] - k["start"] for k in kernels),
            activity_span_nanoseconds=span,
            activity_union_nanoseconds=union,
            gaps_between_this_process_kernels_nanoseconds=span - union,
            kernels=[
                dict(
                    zip(
                        [
                            "name",
                            "grid_x",
                            "grid_y",
                            "block_x",
                            "registers_per_thread",
                            "static_shared_bytes",
                            "dynamic_shared_bytes",
                        ],
                        key,
                    ),
                    **value,
                )
                for key, value in groups.items()
            ],
        )
    )
pathlib.Path(a.output).write_text(
    json.dumps(
        dict(
            scope="Separate instrumented standalone projection calls. No serving TPS, achieved occupancy or hardware-counter bandwidth claim.",
            ranges=result,
        ),
        indent=2,
    )
    + "\n"
)
