"""Recompute native MTP2 replay/prefix and explicitly reused control evidence."""

import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent


def read(p):
    return json.loads(p.read_text())


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
    paths = [
        (ROOT / "v100/campaign-v3", False),
        (ROOT / "v100/prefix-campaign-v10", False),
        (ROOT.parent / "v100-prefill-budget/v100/campaign-v2", True),
        (ROOT.parent / "v100-q8-wmma/v100/campaign-v5", True),
        (ROOT.parent / "v100-strided-output/v100/external-depth-v9", True),
    ]
    result = dict(
        scope="32 fixed greedy outputs per request/global cohort wall, including HTTP handler reset, queueing and client response archival/validation; no summed concurrent request times. Screens one repeat; full matrix three. Explicitly reused strongest ordinary batch/external controls retained. FP16 KV; native residual-Q8 target and original-FP16 draft; external Q8_0/Q8_1 encoding and small-parameter arithmetic differ.",
        external_speed_goal_complete=False,
        default_enabled=False,
        achieved_hardware_counters=None,
        experiments=[],
    )
    for base, reused in paths:
        campaign = read(base / "campaign-report.json")
        assert campaign["completed"]
        for item in campaign["experiments"]:
            if reused and not item["label"].startswith("repeat"):
                continue
            if "q8-wmma" in str(base) and item["engine"] != "llama":
                continue
            p = base / item["label"] / "timed/report.json"
            if not item["completed"] or not p.exists():
                continue
            benchmark = read(p)
            assert benchmark["completed"]
            rows = []
            for count in sorted({w["request_count"] for w in benchmark["waves"]}):
                waves = [w for w in benchmark["waves"] if w["request_count"] == count]
                requests = [q for w in waves for q in w["requests"]]
                responses = [q["response"] for q in requests]
                assert all(w["emitted_tokens"] == 32 * count for w in waves)
                assert all(
                    r["tokens"] == benchmark["case"]["expected_generated_ids"]
                    for r in responses
                )
                wall = statistics.mean(w["wall_seconds"] for w in waves)
                row = dict(
                    request_count=count,
                    repeats=len(waves),
                    mean_global_cohort_wall_seconds=wall,
                    complete_request_output_tps=32 * count / wall,
                    trial_output_tps=[32 * count / w["wall_seconds"] for w in waves],
                    mean_server_cpu_core_seconds=statistics.mean(
                        w["server_cpu_core_seconds"] for w in waves
                    ),
                    mean_client_completion_seconds=statistics.mean(
                        q["client_completion_seconds"] for q in requests
                    ),
                    max_client_completion_seconds=max(
                        q["client_completion_seconds"] for q in requests
                    ),
                    all_measured_ids_equal_pinned=True,
                    telemetry={
                        str(g): telemetry(
                            base / item["label"],
                            benchmark,
                            count,
                            g,
                            campaign["utc_offset_seconds"],
                        )
                        for g in [1, 0]
                    },
                )
                if all("native_timings" in r for r in responses):
                    times = [r["native_timings"] for r in responses]
                    row["mean_native_time_to_first_token_seconds"] = statistics.mean(
                        t["native_time_to_first_token_seconds"] for t in times
                    )
                    row["native_ttft_scope"] = (
                        "Native call internals, not streaming client TTFT. Includes own C++ model-lock waits, excludes reset/submission/HTTP and prior pre-call queues."
                    )
                    if count == 1:
                        row["native_single_post_first_tps"] = 31 / statistics.mean(
                            t["native_generation_seconds"]
                            - t["native_time_to_first_token_seconds"]
                            for t in times
                        )
                else:
                    row["engine_reported_timings"] = [r["timings"] for r in responses]
                    if count == 1:
                        row["llama_reported_post_first_tps"] = 31 / statistics.mean(
                            r["timings"]["predicted_ms"] / 1000 for r in responses
                        )
                if item["mtp_depth"]:
                    field = "mtp_stats" if item["engine"] == "venturi" else "timings"
                    row["mtp_counters"] = [r[field] for r in responses]
                rows.append(row)
            result["experiments"].append(
                dict(
                    label=item["label"],
                    capsule=str(base.relative_to(ROOT.parent)),
                    engine=item["engine"],
                    mode=item["mode"],
                    mtp_depth=item["mtp_depth"],
                    recovery=item.get("mtp_recovery", "replay"),
                    reused=reused,
                    summary=rows,
                )
            )
    indexed = {
        (e["capsule"], e["label"]): {r["request_count"]: r for r in e["summary"]}
        for e in result["experiments"]
    }
    current = "v100-native-mtp2/v100/prefix-campaign-v10"
    campaign = read(ROOT / "v100/prefix-campaign-v10/campaign-report.json")
    result["full_matrix_executed"] = campaign["promoted"]
    if campaign["promoted"]:
        a = indexed[current, "repeat-control-mtp1"]
        b = indexed[current, "repeat-candidate-mtp2-prefix"]
        result["prefix_vs_fresh_mtp1_http_ratio"] = {
            str(c): b[c]["complete_request_output_tps"]
            / a[c]["complete_request_output_tps"]
            for c in b
        }
        ordinary = indexed["v100-prefill-budget/v100/campaign-v2", "repeat-rows256"]
        external = indexed["v100-q8-wmma/v100/campaign-v5", "repeat-llama-tensor-off"]
        result["prefix_vs_reused_best_ordinary_batch_http_ratio"] = {
            str(c): b[c]["complete_request_output_tps"]
            / ordinary[c]["complete_request_output_tps"]
            for c in b
        }
        result["prefix_vs_reused_external_non_spec_http_ratio"] = {
            str(c): b[c]["complete_request_output_tps"]
            / external[c]["complete_request_output_tps"]
            for c in b
        }
        ext = indexed[
            "v100-strided-output/v100/external-depth-v9", "repeat-llama-tensor-mtp2-c1"
        ][1]
        result["single_prefix_vs_reused_best_tested_external_mtp2_http_ratio"] = (
            b[1]["complete_request_output_tps"] / ext["complete_request_output_tps"]
        )
        result["single_prefix_vs_reused_best_tested_external_mtp2_post_first_ratio"] = (
            b[1]["native_single_post_first_tps"] / ext["llama_reported_post_first_tps"]
        )
    for stage in ["model-v2", "prefix-model-v6"]:
        q = read(ROOT / f"v100/{stage}/qualification/report.json")
        assert q["completed"]
        result[stage + "_qualification"] = dict(
            recovery=q.get("target_recovery", "replay"),
            recursive_reference=q["recurrence_reference"],
            cases=[
                dict(
                    name=c["case"]["name"],
                    exact_target=c["committed_target_state_bytes_equal"],
                    exact_draft=c["committed_draft_state_equal_mtp1"],
                    statistics=c["mtp_stats"],
                )
                for c in q["cases"]
            ],
        )
    result["profiles"] = {}
    for stage, names in [
        ("final-v4", ["mtp1-c1", "mtp1-c8", "mtp2-c1", "mtp2-c8"]),
        ("prefix-final-v11", ["mtp2-prefix-c1", "mtp2-prefix-c8"]),
        (
            "full-profile-v14",
            [
                "mtp2-replay-full-c1",
                "mtp2-replay-full-c8",
                "mtp2-prefix-full-c1",
                "mtp2-prefix-full-c8",
            ],
        ),
    ]:
        for name in names:
            meta = read(ROOT / f"v100/{stage}/{name}-metadata.json")
            attr = read(ROOT / f"v100/{stage}/{name}-attribution.json")["cohorts"][0]
            result["profiles"][stage + "/" + name] = dict(
                profiled=True,
                logical_requests=meta["logical_requests"],
                library_sha256=meta["library_sha256"],
                native_call_ranges=meta["native_call_ranges"],
                attribution=attr,
                no_achieved_occupancy_or_bandwidth_claim=True,
            )
    (ROOT / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if "ratio" in k
                or k in ("full_matrix_executed", "external_speed_goal_complete")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
