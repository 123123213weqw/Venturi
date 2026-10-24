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
    raw = ROOT / "v100/campaign-v2"
    campaign = read(raw / "campaign-report.json")
    result = dict(
        scope="Unprofiled HTTP timings. Screening has one repeat; promotion and external matrix cells have three. Outputs/global cohort wall, never summed concurrent latencies. Diagnostics kept separately.",
        achieved_hardware_counters=None,
        external_performance_goal_complete=False,
        experiments=[],
    )
    entries = [(raw, item) for item in campaign["experiments"]]
    prior = ROOT.parent / "v100-q8-wmma/v100/campaign-v5"
    entries += [
        (prior, item)
        for item in read(prior / "campaign-report.json")["experiments"]
        if item["engine"] == "llama" and item["label"].startswith("repeat")
    ]
    external = ROOT.parent / "v100-strided-output/v100/external-depth-v9"
    entries += [
        (external, item)
        for item in read(external / "campaign-report.json")["experiments"]
    ]
    diagnostics = ROOT / "v100/final-v3"
    result["failed_experiments"] = [
        dict(
            directory=str(base.relative_to(ROOT.parent)),
            label=item["label"],
            failure=item.get("failure"),
        )
        for base, item in entries
        if not item["completed"]
    ]
    for base, item in entries:
        directory = base / item["label"]
        path = directory / "timed/report.json"
        if not path.exists():
            continue
        benchmark = read(path)
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
                reused=base in (prior, external),
                summary=rows,
            )
        )
    indexed = {
        e["label"]: {r["request_count"]: r for r in e["summary"]}
        for e in result["experiments"]
    }
    metric = "aggregate_complete_request_output_tps"
    result["candidate_promoted"] = campaign["promoted"]
    result["promoted_variant"] = (
        "rows" + str(campaign["promoted_projection_rows"])
        if campaign["promoted"]
        else None
    )
    result["screen_gains"] = campaign["screen_gains"]
    result["mtp_screen_gains"] = campaign.get("mtp_screen_gains", {})
    result["qualification_gates"] = campaign["qualification_gates"]
    tag = (
        "rows" + str(campaign["promoted_projection_rows"])
        if campaign["promoted"]
        else None
    )
    if tag:
        selected = indexed["repeat-" + tag]
        result["same_profile_speedup"] = {
            str(c): row[metric] / indexed["repeat-rows32"][c][metric]
            for c, row in selected.items()
        }
        result["candidate_vs_reused_external_non_spec_ratio"] = {
            str(c): row[metric] / indexed["repeat-llama-tensor-off"][c][metric]
            for c, row in selected.items()
        }
    mtp_arms = [
        (e["label"], e["summary"][0])
        for e in result["experiments"]
        if e["engine"] == "llama"
        and e["mtp_depth"]
        and e["whole_arm_completed"]
        and e["benchmark_completed"]
        and len(e["summary"]) == 1
        and e["summary"][0]["request_count"] == 1
    ]
    best = max(mtp_arms, key=lambda e: e[1][metric])
    result["best_tested_llama_mtp_c1"] = dict(
        label=best[0],
        summary=best[1],
        selection_scope="Highest successful tested HTTP TPS among depths 1-4; not exhaustive, explicitly reused.",
    )
    result["arithmetic_scope"] = (
        "Same exact qualified native library and arithmetic. Only projection-row workspace/prefill schedule is changed; slot context, tokens, KV and numerical gates remain fixed. This tunes an existing native max_tokens parameter, not a new kernel or model architecture."
    )
    result["projection_budget"] = dict(
        control=32, candidate=campaign.get("promoted_projection_rows")
    )
    result["unchanged_mtp_reuse"] = campaign["unchanged_mtp_reuse"]
    result["tensor_core_model_dispatch"] = False
    result["profiles"] = {}
    old = ROOT.parent / "v100-gdn-compact/v100/final-v5"
    for label in ["control", "candidate"]:
        for suffix in ["c1", "c8"]:
            name = label + "-" + suffix
            path = (
                (old / ("candidate-" + suffix + "-attribution.json"))
                if label == "control"
                else diagnostics / (name + "-attribution.json")
            )
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
                for key in [
                    "vdq::dot",
                    "vds::dot",
                    "vdt::dot",
                    "vqf::",
                    "gdn_recurrent",
                    "gdn_compact",
                ]
            }
            result["profiles"][name] = dict(
                source=str(path.relative_to(ROOT.parent)),
                reused=(label == "control"),
                instrumented=True,
                kernel_part_nanoseconds=parts,
                attribution=data,
            )
    for filename in ["resources.json", "reused-control-profiles.json"]:
        path = diagnostics / filename
        if path.exists():
            result[filename.removesuffix(".json")] = read(path)
    result["profile_scope"] = (
        "Separate 23-prompt/8-output candidates at the selected projection budget, with unchanged 32-row controls explicitly reused from the preceding compact-GDN capsule. Not HTTP speed or achieved occupancy. Kernel sums may overlap across devices."
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
