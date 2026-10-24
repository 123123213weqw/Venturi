"""Attribute this single-process Nsight trace by CPU-side NVTX launch ranges.

Profiler-only durations are not unprofiled latency, achieved occupancy or DRAM
bandwidth. Join kernels/copies to their launch API correlation IDs, not just GPU
timestamps: queued activity may execute after the host range submitted it.
"""

import argparse
import json
from pathlib import Path
import sqlite3
import statistics


def summarize(path):
    db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        assert (
            db.execute(
                "select count(distinct globalPid) from CUPTI_ACTIVITY_KIND_KERNEL"
            ).fetchone()[0]
            == 1
        ), "Only a single profiled process is supported"
        strings = dict(db.execute("select id,value from StringIds"))
        results = []
        ranges = db.execute(
            "select start,end,text,globalTid from NVTX_EVENTS where end is not null and (text like 'native_greedy_rows_%' or text like 'cpu_greedy_rows_%') order by start"
        ).fetchall()
        for event in ranges:
            native = event["text"].startswith("native_")
            rows = int(event["text"].rsplit("_", 1)[1])
            assert rows in (1, 8)
            params = (event["start"], event["end"], event["globalTid"])
            launches = db.execute(
                "select * from CUPTI_ACTIVITY_KIND_RUNTIME where start>=? and end<=? and globalTid=?",
                params,
            ).fetchall()
            by_id = {r["correlationId"]: r for r in launches}
            kernels = [
                k
                for k in db.execute(
                    "select * from CUPTI_ACTIVITY_KIND_KERNEL order by start"
                )
                if k["correlationId"] in by_id
            ]
            copies = [
                c
                for c in db.execute(
                    "select * from CUPTI_ACTIVITY_KIND_MEMCPY order by start"
                )
                if c["correlationId"] in by_id
            ]
            assert len(copies) == 1, (
                "Expected exactly one result/full-logit D2H copy per range"
            )
            assert copies[0]["copyKind"] == 2, "Expected CUPTI D2H copy kind"
            expected_bytes = rows * (16 if native else 248320 * 4)
            assert copies[0]["bytes"] == expected_bytes, (
                "Copy payload does not match the declared path"
            )
            item = dict(
                mode="native" if native else "cpu",
                rows=rows,
                profiled_host_range_us=(event["end"] - event["start"]) / 1000,
                copy_bytes=expected_bytes,
                copy_us=(copies[0]["end"] - copies[0]["start"]) / 1000,
            )
            if native:
                assert (
                    len(kernels) == 2
                    and "greedy_partials" in strings[kernels[0]["demangledName"]]
                    and "greedy_finish" in strings[kernels[1]["demangledName"]]
                ), "Unexpected native kernel attribution"
                a, b = kernels
                gap = b["start"] - a["end"]
                assert gap >= 0 and copies[0]["start"] >= b["end"]
                item.update(
                    partials_us=(a["end"] - a["start"]) / 1000,
                    finish_us=(b["end"] - b["start"]) / 1000,
                    interkernel_gap_us=gap / 1000,
                    final_kernel_to_copy_gap_us=(copies[0]["start"] - b["end"]) / 1000,
                    host_between_launch_api_gap_us=(
                        by_id[b["correlationId"]]["start"]
                        - by_id[a["correlationId"]]["end"]
                    )
                    / 1000,
                )
            else:
                assert not kernels, "CPU policy should not launch GPU selection kernels"
            results.append(item)
        assert results, "No expected profiler ranges found"
        aggregates = []
        for mode in ("native", "cpu"):
            for rows in (1, 8):
                chosen = [r for r in results if r["mode"] == mode and r["rows"] == rows]
                if chosen:
                    aggregates.append(
                        dict(
                            mode=mode,
                            rows=rows,
                            samples=len(chosen),
                            median={
                                key: statistics.median(r[key] for r in chosen)
                                for key in chosen[0]
                                if key not in ("mode", "rows")
                            },
                        )
                    )
        return dict(
            scope=__doc__,
            achieved_hardware_counters=False,
            aggregates=aggregates,
            ranges=results,
        )
    finally:
        db.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sqlite")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    Path(args.output).write_text(json.dumps(summarize(args.sqlite), indent=2) + "\n")


if __name__ == "__main__":
    main()
