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
    raw = ROOT / "v100/campaign-v3"
    campaign = read(raw / "campaign-report.json")
    result = dict(
        scope="Unprofiled HTTP timings. Screening has one repeat; promotion and external matrix cells have three. Outputs/global cohort wall, never summed concurrent latencies. Diagnostics kept separately.",
        achieved_hardware_counters=None,
        external_performance_goal_complete=False,
        experiments=[],
    )
    entries = [(raw, item) for item in campaign["experiments"]]
    diagnostics = ROOT / "v100/final-v4"
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
    result["residual_vs_original_speedup"] = {
        str(c): r[metric] / indexed["repeat-original"][c][metric]
        for c, r in indexed["repeat-residual"].items()
    }
    result["residual_vs_llama_non_speculative_ratio"] = {
        str(c): r[metric] / indexed["repeat-llama-tensor-off"][c][metric]
        for c, r in indexed["repeat-residual"].items()
    }
    result["residual_mtp_vs_original_speedup"] = (
        indexed["repeat-mtp-residual"][1][metric]
        / indexed["repeat-mtp-original"][1][metric]
    )
    result["precision_scope"] = (
        "Same original-derived Q8 weights and FP16 KV. fp16 retains original activation arithmetic; "
        "q8-residual explicitly uses two group32 A8 passes with FP32 scales and changed FP32 reduction. "
        "llama.cpp uses different GGUF Q8_0 encoding, selected small-parameter precision and traced Q8_1 "
        "activation packing. Practical engine comparison, not identical arithmetic or original-BF16 quality. "
        "Native MTP target uses the named profile, draft remains original FP16."
    )
    qualification = read(ROOT / "v100/qualification-v2/qualification/report.json")
    result["qualification"] = {
        k: v for k, v in qualification.items() if k != "profiles"
    }
    q = qualification["profiles"]["q8-residual"]
    checks = [
        check for case in q["trained_cases"]
        for check in case["teacher_forced_logit_checks"]
    ]
    result["qualification"]["additional_trained_steps"] = dict(
        count=len(checks),
        all_passed=all(c["passed"] for c in checks),
        max_relative_l2=max(c["relative_l2"] for c in checks),
        min_centered_cosine=min(c["centered_cosine"] for c in checks),
        max_kl=max(c["reference_kl_to_native"] for c in checks),
        all_case_outputs_and_same_profile_states_equal=all(
            c["original_ids"] == c["generated_ids"]
            and c["same_profile_diagnostic_generate_state_bytes_exact"]
            for c in q["trained_cases"]
        ),
    )
    result["mtp_qualification"] = read(raw / "mtp-qualification/report.json")
    result["profile_scope"] = (
        "Separate instrumented 23-prompt/8-output CUDA/NVTX runs, not the 32-output speed campaign. "
        "Each trace has one native-call range; C8 means eight logical requests in that one range. "
        "Legacy attribution requests counts ranges, not logical requests. Kernel duration sums are "
        "attribution, not throughput, achieved occupancy or DRAM GB/s."
    )
    result["profiles"] = {
        name: read(diagnostics / f"{name}-attribution.json")
        for name in ("fp16-c1", "fp16-c8", "q8-residual-c1", "q8-residual-c8")
        if (diagnostics / f"{name}-attribution.json").exists()
    }
    result["projection_attribution"] = {}
    for name, profile in result["profiles"].items():
        assert len(profile["cohorts"]) == 1
        cohort = profile["cohorts"][0]
        parts = {
            key: sum(
                k["nanoseconds"]
                for k in cohort["kernel_by_name"]
                if key in k["name"]
            )
            for key in ("q8_tiled", "vdq::dot", "vdq::quantize", "gdn_recurrent")
        }
        projection = sum(parts[k] for k in ("q8_tiled", "vdq::dot", "vdq::quantize"))
        result["projection_attribution"][name] = dict(
            instrumented=True,
            logical_requests=8 if name.endswith("c8") else 1,
            native_call_ranges=cohort["requests"],
            kernel_part_nanoseconds=parts,
            projection_fraction_of_summed_kernel_time=projection
            / cohort["kernel_nanoseconds"],
        )
    if (diagnostics / "resources.json").exists():
        result["compiled_resources"] = read(diagnostics / "resources.json")
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
