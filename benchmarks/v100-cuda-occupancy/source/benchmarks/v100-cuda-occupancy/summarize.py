"""Recompute CUDA residency and matched engine measurements without mixing scopes."""

import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def compiled(path):
    result = {}
    report = read(path)
    for device in report["devices"]:
        rows = {}
        for r in device["compiled_q8_functions"]:
            if "q8_tiledI6__halfLi" not in r["symbol"]:
                continue
            kind = "decode" if "Li1ELi4" in r["symbol"] else "prefill"
            rows[kind] = {k: v for k, v in r.items() if k != "symbol"}
        result[str(device["hardware"]["ordinal"])] = rows
    return result


def aggregate(waves, count, each):
    assert all(w["emitted_tokens"] == count * each for w in waves)
    mean_wall = statistics.mean(w["wall_seconds"] for w in waves)
    return dict(
        trials=len(waves),
        mean_cohort_wall_seconds=mean_wall,
        min_cohort_wall_seconds=min(w["wall_seconds"] for w in waves),
        max_cohort_wall_seconds=max(w["wall_seconds"] for w in waves),
        mean_process_cpu_seconds=statistics.mean(
            w["process_cpu_seconds"] for w in waves
        ),
        aggregate_output_tokens_per_wall_second=count * each / mean_wall,
        mean_request_completion_seconds=statistics.mean(
            r["completion_latency_seconds"] for w in waves for r in w["requests"]
        ),
        max_request_completion_seconds=max(
            r["completion_latency_seconds"] for w in waves for r in w["requests"]
        ),
    )


def telemetry(rows, count, gpu):
    samples = []
    for directory, report in rows:
        cohort = next(c for c in report["concurrency"] if c["request_count"] == count)
        windows = [
            (t[s]["start_unix"], t[s]["end_unix"])
            for t in cohort["trials"]
            for s in ("ttft", "decode", "unsplit")
        ]
        tz = timezone(timedelta(seconds=report["utc_offset_seconds"]))
        for row in csv.reader(
            (directory / f"gpu-{gpu}-samples.csv").read_text().splitlines()
        ):
            if len(row) != 8:
                continue
            stamp = (
                datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
                .replace(tzinfo=tz)
                .timestamp()
            )
            if any(a <= stamp <= b for a, b in windows):
                samples.append([float(v.strip()) for v in row[1:]])
    assert samples
    names = (
        "gpu_busy_percent",
        "memory_controller_busy_percent",
        "device_used_mib",
        "power_watts",
        "temperature_c",
        "sm_clock_mhz",
        "memory_clock_mhz",
    )
    return dict(
        samples=len(samples),
        values={
            name: dict(
                mean=statistics.mean(r[i] for r in samples),
                min=min(r[i] for r in samples),
                max=max(r[i] for r in samples),
            )
            for i, name in enumerate(names)
        },
    )


def main():
    raw = ROOT / "v100"
    result = dict(
        achieved_occupancy=None,
        counter_policy=(raw / "probe/policy.txt").read_text().strip(),
        occupancy_api_scope="Theoretical resident warps, not achieved occupancy",
        baseline=compiled(raw / "probe/occupancy.json"),
        forced_six=compiled(raw / "variants/minblocks-6/occupancy.json"),
        forced_eight=compiled(raw / "variants/minblocks-8/occupancy.json"),
        token_aware=compiled(raw / "token-aware/occupancy.json"),
        trace_kernel_shares={},
        unprofiled={},
    )
    for label, cohorts in read(raw / "probe/occupancy.json")["devices"][0][
        "traces"
    ].items():
        c = next(c for c in cohorts if c["name"] == "native_decode")
        total = c["summed_kernel_nanoseconds"]
        result["trace_kernel_shares"][label] = dict(
            q8_decode_summed_kernel_time_fraction=sum(
                r["kernel_nanoseconds"]
                for r in c["launches"]
                if "q8_tiledI6__half" in r["symbol"]
            )
            / total,
            scope="Existing separate instrumented traces, not throughput or achieved occupancy",
        )
    for mode in ("tp", "pp"):
        for count in (1, 8):
            selected = {}
            for variant, phases in (("baseline", (0, 3)), ("token_aware", (1, 2))):
                rows = [
                    (
                        raw / f"token-aware/phase-{p}-{mode}",
                        read(raw / f"token-aware/phase-{p}-{mode}/report.json"),
                    )
                    for p in phases
                ]
                assert all(r["completed"] and not r["profiled"] for _, r in rows)
                trials = [
                    t
                    for _, r in rows
                    for c in r["concurrency"]
                    if c["request_count"] == count
                    for t in c["trials"]
                ]
                selected[variant] = {
                    scope: aggregate([t[scope] for t in trials], count, each)
                    for scope, each in (("ttft", 1), ("decode", 31), ("unsplit", 32))
                }
                selected[variant]["telemetry"] = {
                    str(g): telemetry(rows, count, g) for g in (1, 0)
                }
            selected["speedups"] = {
                s: selected["baseline"][s]["mean_cohort_wall_seconds"]
                / selected["token_aware"][s]["mean_cohort_wall_seconds"]
                for s in ("ttft", "decode", "unsplit")
            }
            result["unprofiled"][f"{mode}-c{count}"] = selected
    (ROOT / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    for label, rows in result["unprofiled"].items():
        print(
            json.dumps(
                dict(
                    label=label,
                    **{k: rows[k] for k in ("baseline", "token_aware", "speedups")},
                )
            )
        )


if __name__ == "__main__":
    main()
