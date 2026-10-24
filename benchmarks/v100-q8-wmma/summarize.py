"""Recompute fixed-work HTTP rates, CPU/latency and timed-window GPU telemetry."""

import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def telemetry(directory, benchmark, count, gpu, offset):
    windows = [
        (w["start_unix"], w["end_unix"])
        for w in benchmark["waves"]
        if w["request_count"] == count
    ]
    samples = []
    tz = timezone(timedelta(seconds=offset))
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
            samples.append([float(x.strip()) for x in row[1:]])
    if not samples:
        return dict(samples=0)
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
                mean=statistics.mean(x[i] for x in samples),
                min=min(x[i] for x in samples),
                max=max(x[i] for x in samples),
            )
            for i, name in enumerate(names)
        },
    )


def main():
    raw = ROOT / "v100/campaign-v5"
    campaign = read(raw / "campaign-report.json")
    result = dict(
        scope="Unprofiled HTTP timings. Screening has one repeat; promotion and external matrix cells have three. Outputs/global cohort wall, never summed concurrent latencies. Diagnostics kept separately.",
        achieved_hardware_counters=None,
        external_performance_goal_complete=False,
        experiments=[],
    )
    entries = [(raw, item) for item in campaign["experiments"]]
    diagnostics = ROOT / "v100/final-v6"
    result["failed_experiments"] = [
        dict(
            directory=str(base.relative_to(ROOT)),
            label=item["label"],
            failure=item.get("failure"),
        )
        for base, item in entries
        if not item["completed"]
    ]
    for base, item in entries:
        directory = base / item["label"]
        benchmark = read(directory / "timed/report.json")
        if not benchmark["waves"]:
            continue
        rows = []
        for count in sorted({w["request_count"] for w in benchmark["waves"]}):
            waves = [w for w in benchmark["waves"] if w["request_count"] == count]
            requests = [q for w in waves for q in w["requests"]]
            assert all(w["emitted_tokens"] == 32 * count for w in waves)
            assert all(
                q["response"]["tokens"] == benchmark["case"]["expected_generated_ids"]
                for q in requests
            )
            wall = statistics.mean(w["wall_seconds"] for w in waves)
            row = dict(
                request_count=count,
                repeats=len(waves),
                mean_global_cohort_wall_seconds=wall,
                aggregate_complete_request_output_tps=32 * count / wall,
                trial_output_tps=[32 * count / w["wall_seconds"] for w in waves],
                mean_server_cpu_core_seconds=statistics.mean(
                    w["server_cpu_core_seconds"] for w in waves
                ),
                mean_client_completion_seconds=statistics.mean(
                    q["client_completion_seconds"] for q in requests
                ),
                min_client_completion_seconds=min(
                    q["client_completion_seconds"] for q in requests
                ),
                max_client_completion_seconds=max(
                    q["client_completion_seconds"] for q in requests
                ),
                all_measured_ids_equal_pinned=True,
                telemetry={
                    str(g): telemetry(
                        directory, benchmark, count, g, campaign["utc_offset_seconds"]
                    )
                    for g in (1, 0)
                },
            )
            responses = [q["response"] for q in requests]
            if all("native_timings" in r for r in responses):
                times = [r["native_timings"] for r in responses]
                row["mean_native_time_to_first_token_seconds"] = statistics.mean(
                    t["native_time_to_first_token_seconds"] for t in times
                )
                row["native_ttft_scope"] = (
                    "Native call internals; excludes HTTP and prior reset/queue time; scalar C-entry timing can include its model-mutex wait. Batched TTFT starts at the common cohort C-call and includes all cohort prefill. Not streaming client TTFT."
                )
                if count == 1:
                    row["native_single_post_first_tps"] = 31 / statistics.mean(
                        t["native_generation_seconds"]
                        - t["native_time_to_first_token_seconds"]
                        for t in times
                    )
            else:
                row["engine_reported_timings"] = [r["timings"] for r in responses]
                row["client_ttft"] = None
            if item["mtp_depth"]:
                field = "mtp_stats" if item["engine"] == "venturi" else "timings"
                row["mtp_counters"] = [r[field] for r in responses]
            if item["engine"] == "llama" and count == 1:
                row["llama_reported_post_first_tps"] = 31 / statistics.mean(
                    r["timings"]["predicted_ms"] / 1000 for r in responses
                )
                row["llama_decode_scope"] = (
                    "31 post-first outputs / mean engine-reported predicted_ms. Not client TTFT or HTTP wall."
                )
            if item["engine"] == "venturi" and item["mode"] == "batch":
                assert all(r["batch_request_count"] == count for r in responses)
                row["cohort_admission_not_split"] = True
                durations = []
                for wave in waves:
                    ts = [q["response"]["native_timings"] for q in wave["requests"]]
                    first = {t["native_time_to_first_token_seconds"] for t in ts}
                    assert len(first) == 1, (
                        "Batch native first-token timestamp is not shared"
                    )
                    durations.append(
                        max(t["native_generation_seconds"] for t in ts)
                        - next(iter(first))
                    )
                row["batch_native_post_first_cohort_tps"] = (
                    (32 - 1) * count / statistics.mean(durations)
                )
                row["batch_native_post_first_scope"] = (
                    "(all emitted outputs minus one per request) / (last native cohort finish - shared native first-token timestamp); no summed request durations. Not an HTTP or competitor decode metric."
                )
                row["mean_admission_delay_seconds"] = statistics.mean(
                    r["batch_admission_delay_seconds"] for r in responses
                )
                row["shared_counters_per_wave"] = [
                    w["requests"][0]["response"]["batch_stats"] for w in waves
                ]
            rows.append(row)
        result["experiments"].append(
            dict(
                label=item["label"],
                engine=item["engine"],
                mode=item["mode"],
                mtp_depth=item["mtp_depth"],
                projection_profile=item.get("projection_profile"),
                whole_arm_completed=item["completed"],
                benchmark_completed=benchmark["completed"],
                summary=rows,
            )
        )
    indexed = {
        e["label"]: {r["request_count"]: r for r in e["summary"]}
        for e in result["experiments"]
    }
    metric = "aggregate_complete_request_output_tps"
    if "repeat-t8" in indexed:
        result["t8_vs_control_speedup"] = {
            str(c): r[metric] / indexed["repeat-control"][c][metric]
            for c, r in indexed["repeat-t8"].items()
        }
        if "repeat-llama-tensor-off" in indexed:
            result["t8_vs_fresh_llama_ratio"] = {
                str(c): r[metric] / indexed["repeat-llama-tensor-off"][c][metric]
                for c, r in indexed["repeat-t8"].items()
            }
    result["trained_qualification"] = "v100/guarded-v4/qualification/report.json"
    result["trained_same_profile_bytes_equal"] = read(
        ROOT / result["trained_qualification"]
    )["completed"]
    result["tensor_core_model_dispatch"] = False
    result["arithmetic_scope"] = (
        "Guarded DP4A T8 preserves each original/residual profile's bytes. "
        "WMMA is a separate slow research operator, not selected by native generation. "
        "Native/GGUF Q8 labels still do not establish exact arithmetic parity."
    )
    result["projection_microbenchmarks"] = {}
    for name, directory in (
        ("wmma", "projection-v2"),
        ("unguarded_t8", "t8-v3"),
        ("guarded_t8", "guarded-v4"),
    ):
        report = read(ROOT / "v100" / directory / "projection/report.json")
        assert report["completed"] and len(report["cases"]) == 32
        summary = {}
        for t in (1, 2, 8, 32):
            cases = [c for c in report["cases"] if c["tokens"] == t]
            summary[str(t)] = {
                p: dict(
                    median_vs_original_fp16=statistics.median(
                        c["median_gpu_event_ms"]["original-fp16"]
                        / c["median_gpu_event_ms"][p]
                        for c in cases
                    ),
                    median_ms=statistics.median(
                        c["median_gpu_event_ms"][p] for c in cases
                    ),
                )
                for p in report["profiles"]
            }
            if name == "guarded_t8":
                summary[str(t)]["median_control_over_candidate"] = statistics.median(
                    c["median_gpu_event_ms"]["control-w8-a8-g32-f32scale-residual-a8"]
                    / c["median_gpu_event_ms"]["w8-a8-g32-f32scale-residual-a8"]
                    for c in cases
                )
        result["projection_microbenchmarks"][name] = dict(
            scope=report["scope"],
            row_mode=report["row_mode"],
            summary=summary,
            original_profiles=report["profiles"],
            source=f"v100/{directory}/projection/report.json",
            inference_tps_claim=False,
        )
    result["profiles"] = {}
    for c in (1, 8):
        for label, directory in (
            ("control", ROOT.parent / "v100-shared-inputs/v100/final-v4"),
            ("candidate", diagnostics),
        ):
            path = directory / f"q8-residual-c{c}-attribution.json"
            if not path.exists():
                continue
            data = read(path)
            cohort = data["cohorts"][0]
            parts = {
                key: sum(
                    k["nanoseconds"]
                    for k in cohort["kernel_by_name"]
                    if key in k["name"]
                )
                for key in ("vdq::dot", "vdq::quantize", "gdn_recurrent")
            }
            result["profiles"][f"{label}-c{c}"] = dict(
                source=str(path.relative_to(ROOT.parent.parent)),
                reused=label == "control",
                instrumented=True,
                logical_requests=c,
                native_call_ranges=cohort["requests"],
                kernel_part_nanoseconds=parts,
                projection_fraction_of_summed_kernel_time=(
                    parts["vdq::dot"] + parts["vdq::quantize"]
                )
                / cohort["kernel_nanoseconds"],
                attribution=data,
            )
    for filename in (
        "projection-attribution.json",
        "resources.json",
        "instruction-audit.json",
    ):
        path = diagnostics / filename
        if path.exists():
            result[filename.removesuffix(".json")] = read(path)
    result["profile_scope"] = (
        "Separate 23-prompt/8-output model traces and standalone projection traces. "
        "Control model traces reused hash-pinned from the preceding capsule, not fresh throughput. "
        "Legacy requests counts native ranges, not logical batch requests; summed kernel time "
        "and static resource/instruction counts are not achieved occupancy, TFLOPS or GB/s."
    )
    (ROOT / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    for e in result["experiments"]:
        print(
            e["label"],
            [
                (
                    r["request_count"],
                    round(r["aggregate_complete_request_output_tps"], 2),
                )
                for r in e["summary"]
            ],
        )


if __name__ == "__main__":
    main()
