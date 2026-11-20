"""Summarize Nsight SQLite kernel traces using native prefill/decode NVTX ranges.

Duration attribution is not a hardware-counter measurement. Launch-resource
bounds are theoretical ceilings, not measured occupancy or evidence of spills.
"""

import argparse
from bisect import bisect_right
from collections import defaultdict
import json
from pathlib import Path
import sqlite3


def projection(name):
    return any(
        s in name
        for s in (
            "q4_",
            "q5_",
            "q6_",
            "q8_",
            "ada_small_t_mma",
            "rowsplit_grouped_mma",
            "bf16_gdn_gating_proj_gemm",
        )
    )


def summarize(database, hardware):
    c = sqlite3.connect(f"file:{Path(database).resolve()}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    strings = dict(c.execute("select id,value from StringIds"))
    phases = defaultdict(list)
    for r in c.execute("select * from NVTX_EVENTS where end is not null"):
        name = r["text"] or strings.get(r["textId"])
        if name in ("prefill", "decode"):
            phases[r["globalTid"]].append((r["start"], r["end"], name))
    for ranges in phases.values():
        ranges.sort()
    starts = {tid: [r[0] for r in ranges] for tid, ranges in phases.items()}
    runtime = {}
    api = defaultdict(lambda: [0, 0])
    for r in c.execute("select * from CUPTI_ACTIVITY_KIND_RUNTIME"):
        tid = r["globalTid"]
        i = bisect_right(starts.get(tid, []), r["start"]) - 1
        phase = (
            phases[tid][i][2]
            if i >= 0 and r["start"] <= phases[tid][i][1]
            else "unattributed"
        )
        runtime[r["correlationId"]] = phase
        name = strings[r["nameId"]]
        api[name][0] += 1
        api[name][1] += r["end"] - r["start"]
    groups = defaultdict(lambda: dict(count=0, nanoseconds=0, geometry={}))
    intervals = []
    for r in c.execute("select * from CUPTI_ACTIVITY_KIND_KERNEL"):
        phase = runtime.get(r["correlationId"], "unattributed")
        name = strings[r["shortName"]]
        g = groups[(phase, name)]
        g["count"] += 1
        g["nanoseconds"] += r["end"] - r["start"]
        intervals.append((r["start"], r["end"]))
        threads = r["blockX"] * r["blockY"] * r["blockZ"]
        shared = r["staticSharedMemory"] + r["dynamicSharedMemory"]
        regs = r["registersPerThread"]
        blocks = min(
            hardware["blocks_per_sm"],
            hardware["threads_per_sm"] // threads,
            hardware["registers_per_sm"] // (regs * threads)
            if regs
            else hardware["blocks_per_sm"],
            hardware["shared_bytes_per_sm"] // shared
            if shared
            else hardware["blocks_per_sm"],
        )
        geometry = dict(
            threads=threads,
            registers_per_thread=regs,
            shared_bytes_per_block=shared,
            local_memory_bytes_per_thread=r["localMemoryPerThread"],
            grid_blocks=r["gridX"] * r["gridY"] * r["gridZ"],
            resource_only_resident_blocks_upper_bound=blocks,
            resource_only_thread_occupancy_upper_bound=blocks
            * threads
            / hardware["threads_per_sm"],
        )
        key = json.dumps(geometry, sort_keys=True)
        g["geometry"].setdefault(key, dict(geometry, count=0))["count"] += 1
    by_phase = {}
    for phase in ("prefill", "decode", "unattributed"):
        selected = [(name, g) for (p, name), g in groups.items() if p == phase]
        total = sum(g["nanoseconds"] for _, g in selected)
        if not total:
            continue
        by_phase[phase] = dict(
            kernel_count=sum(g["count"] for _, g in selected),
            summed_kernel_seconds=total / 1e9,
            projection_fraction=sum(
                g["nanoseconds"] for name, g in selected if projection(name)
            )
            / total,
            top_kernels=[
                dict(
                    name=name,
                    count=g["count"],
                    seconds=g["nanoseconds"] / 1e9,
                    fraction=g["nanoseconds"] / total,
                    launch_geometry=list(g["geometry"].values()),
                )
                for name, g in sorted(selected, key=lambda x: -x[1]["nanoseconds"])
            ],
        )
    intervals.sort()
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    span = max(end for _, end in intervals) - intervals[0][0] if intervals else 0
    return dict(
        database=str(database),
        phases=by_phase,
        api={
            name: dict(count=count, seconds=ns / 1e9)
            for name, (count, ns) in api.items()
        },
        kernel_timeline_span_seconds=span / 1e9,
        kernel_interval_coverage_fraction=sum(b - a for a, b in merged) / span
        if span
        else None,
        limitations=[
            "Instrumented durations are not unprofiled throughput.",
            "Resource-only ceilings ignore register-allocation granularity, grid waves and runtime stalls; not achieved occupancy.",
            "Local memory includes compiler stack/storage; a nonzero field alone does not establish spills.",
            "Kernel interval coverage is not SM utilization; no DRAM throughput or tensor-core utilization counters.",
        ],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("database")
    p.add_argument("--hardware", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    result = summarize(a.database, json.loads(Path(a.hardware).read_text()))
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: {key: value for key, value in v.items() if key != "top_kernels"}
                for k, v in result["phases"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
