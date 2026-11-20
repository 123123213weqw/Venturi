"""Recompute matched MTP results; keep cohort, native timing and profiler scopes separate."""

import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "v100"


def read(path):
    return json.loads(path.read_text())


def telemetry(directory, report, count, variant, gpu, offset):
    windows = [
        (t["cohort"]["start_unix"], t["cohort"]["end_unix"])
        for t in report["trials"]
        if t["request_count"] == count and t["variant"] == variant
    ]
    tz = timezone(timedelta(seconds=offset))
    samples = []
    for row in csv.reader((directory / f"gpu-{gpu}-samples.csv").read_text().splitlines()):
        if len(row) != 8:
            continue
        stamp = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
        stamp = stamp.replace(tzinfo=tz).timestamp()
        if any(a <= stamp <= b for a, b in windows):
            samples.append([float(v.strip()) for v in row[1:]])
    assert samples, (count, variant, gpu)
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
                mean=statistics.mean(row[i] for row in samples),
                min=min(row[i] for row in samples),
                max=max(row[i] for row in samples),
            )
            for i, name in enumerate(names)
        },
    )


def main():
    import ast
    from collections import defaultdict

    # Compile only the archived pure aggregation functions; no Torch dependency.
    path = ROOT / "source/tools/benchmark_qwen_mtp.py"
    if not path.exists():
        path = ROOT.parent.parent / "tools/benchmark_qwen_mtp.py"
    tree = ast.parse(path.read_text())
    selected = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in ("acceptance", "summarize")
    ]
    namespace = dict(defaultdict=defaultdict, statistics=statistics)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    summarize = namespace["summarize"]

    offset = read(RAW / "diagnostics-v1/server-clock.json")["utc_offset_seconds"]
    result = dict(
        achieved_occupancy=None,
        achieved_compute_dram_cache_stall_counters=None,
        counter_policy=(RAW / "diagnostics-v1/policy.txt").read_text().strip(),
        telemetry_utc_offset_seconds=offset,
        modes={},
        trained_acceptance={},
        profiler={},
    )
    for mode in ("tp", "pp"):
        directory = RAW / f"final-v4/comparison-{mode}"
        report = read(directory / "report.json")
        assert report["completed"] and not report["profiled"]
        rows = summarize(report["trials"])
        assert rows == report["summary"]
        for row in rows:
            count, variant = row["request_count"], row["variant"]
            row["telemetry"] = {
                str(g): telemetry(directory, report, count, variant, g, offset)
                for g in (1, 0)
            }
            trial = next(
                t for t in report["trials"]
                if t["request_count"] == count and t["variant"] == variant
            )
            row["one_slot_initial_memory"] = trial["initial_memory"][0]
        result["modes"][mode] = rows
        qual = read(RAW / f"final-v4/qualification-{mode}/report.json")
        assert qual["completed"] and all(r["passed"] for r in qual["block_reference"])
        result["trained_acceptance"][mode] = []
        for case in qual["cases"]:
            assert case["output_ids_equal"] and case["committed_target_state_bytes_equal"]
            assert case["position"] == case["draft_position"]
            stats = case["mtp_stats"]
            assert stats["forced_proposals"] == 0
            result["trained_acceptance"][mode].append(dict(
                name=case["case"]["name"], emitted_tokens=len(case["generated_ids"]),
                proposed=stats["proposed_tokens"], accepted=stats["accepted_tokens"],
                acceptance=stats["acceptance_rate"], replayed=stats["replayed_input_tokens"],
            ))
    for label in ("tp-c1", "pp-c8"):
        report = read(RAW / f"diagnostics-v1/{label}/cohort-attribution.json")
        c, = report["cohorts"]
        result["profiler"][label] = dict(
            scope="Separate profiled 23-prompt/8-output complete wave; not unprofiled TPS or achieved occupancy",
            host_cohort_seconds=c["host_cohort_span_nanoseconds"] / 1e9,
            summed_kernel_seconds=c["kernel_nanoseconds"] / 1e9,
            q8_fraction_of_summed_kernel_time=sum(
                k["nanoseconds"] for k in c["kernel_by_name"] if "q8_tiled" in k["name"]
            ) / c["kernel_nanoseconds"],
            activity_span_seconds=c["activity_span_nanoseconds"] / 1e9,
            process_activity_gap_seconds=c["gaps_between_this_process_activities_nanoseconds"] / 1e9,
            two_device_simultaneous_kernel_seconds=c["simultaneous_two_device_kernel_nanoseconds"] / 1e9,
            transfers=c["transfers"],
        )
    occ = read(RAW / "final-v4/occupancy.json")
    result["theoretical_residency"] = [
        row for row in occ["devices"][0]["compiled_q8_functions"]
        if "q8_tiledI6__half" in row["symbol"]
    ]
    micro = read(RAW / "final-v4/projection-comparison/report.json")
    assert micro["completed"] and all(c["exact_baseline_output_bytes"] for c in micro["cases"])
    result["two_token_microbenchmark"] = [
        dict(shape=c["shape"], role=c["role"], median_gpu_event_ms=c["median_gpu_event_ms"])
        for c in micro["cases"] if c["tokens"] == 2
    ]
    control = read(RAW / "final-v4/original-pair-tile-control/report.json")
    assert control["completed"] and not control["profiled"]
    result["initial_four_token_tile_control_tp1"] = control["summary"]
    (ROOT / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Matched reports, trained gates, telemetry, microbenchmarks and profiler scopes verified")


if __name__ == "__main__":
    main()
