"""Recompute T3 operator and matched HTTP timings; profiles are separate."""

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
    result = dict(
        external_speed_goal_complete=False,
        experiments=[],
        profiles={},
        achieved_occupancy=None,
        scope="32 fixed greedy outputs/request divided by global HTTP cohort wall; native/HTTP scopes separate. Shared cohort times/counters never summed across requests. Unprofiled repeats only.",
    )
    paths = [
        (ROOT / "v100/campaign-v3", False),
        (ROOT / "v100/sdk-campaign-v8", False),
        (ROOT.parent / "v100-q8-wmma/v100/campaign-v5", True),
        (ROOT.parent / "v100-strided-output/v100/external-depth-v9", True),
    ]
    for base, reused in paths:
        c = read(base / "campaign-report.json")
        assert c["completed"]
        for e in c["experiments"]:
            if reused and (
                e["engine"] != "llama" or not e["label"].startswith("repeat")
            ):
                continue
            p = base / e["label"] / "timed/report.json"
            if not e["completed"] or not p.exists():
                continue
            b = read(p)
            rows = []
            for count in sorted({w["request_count"] for w in b["waves"]}):
                waves = [w for w in b["waves"] if w["request_count"] == count]
                requests = [r for w in waves for r in w["requests"]]
                responses = [r["response"] for r in requests]
                assert all(w["emitted_tokens"] == 32 * count for w in waves)
                assert all(
                    r["tokens"] == b["case"]["expected_generated_ids"]
                    for r in responses
                )
                row = dict(
                    request_count=count,
                    repeats=len(waves),
                    complete_request_output_tps=32
                    * count
                    / statistics.mean(w["wall_seconds"] for w in waves),
                    trial_output_tps=[32 * count / w["wall_seconds"] for w in waves],
                    mean_cohort_wall_seconds=statistics.mean(
                        w["wall_seconds"] for w in waves
                    ),
                    mean_server_cpu_core_seconds=statistics.mean(
                        w["server_cpu_core_seconds"] for w in waves
                    ),
                    mean_client_completion_seconds=statistics.mean(
                        r["client_completion_seconds"] for r in requests
                    ),
                    max_client_completion_seconds=max(
                        r["client_completion_seconds"] for r in requests
                    ),
                    telemetry={
                        str(g): telemetry(
                            base / e["label"], b, count, g, c["utc_offset_seconds"]
                        )
                        for g in [1, 0]
                    },
                )
                if all("native_timings" in r for r in responses):
                    tt = [r["native_timings"] for r in responses]
                    row["mean_native_ttft_seconds"] = statistics.mean(
                        t["native_time_to_first_token_seconds"] for t in tt
                    )
                    row["min_native_ttft_seconds"] = min(
                        t["native_time_to_first_token_seconds"] for t in tt
                    )
                    row["max_native_ttft_seconds"] = max(
                        t["native_time_to_first_token_seconds"] for t in tt
                    )
                    row["mean_native_generation_seconds"] = statistics.mean(
                        t["native_generation_seconds"] for t in tt
                    )
                    if count == 1:
                        row["native_single_post_first_tps"] = 31 / statistics.mean(
                            t["native_generation_seconds"]
                            - t["native_time_to_first_token_seconds"]
                            for t in tt
                        )
                else:
                    row["engine_timings"] = [r["timings"] for r in responses]
                    if count == 1:
                        row["llama_reported_post_first_tps"] = 31 / statistics.mean(
                            r["timings"]["predicted_ms"] / 1000 for r in responses
                        )
                if e["mtp_depth"] and e["engine"] == "venturi":
                    stats = [r["mtp_stats"] for r in responses]
                    row["mtp_counters"] = stats
                    row["acceptance_ratio"] = sum(
                        s["accepted_tokens"] for s in stats
                    ) / sum(s["proposed_tokens"] for s in stats)
                    row["target_replayed_inputs"] = sum(
                        s["replayed_input_tokens"] for s in stats
                    )
                    if e["mode"] == "mtp-batch" or e["mtp_recovery"] == "prefix":
                        assert row["target_replayed_inputs"] == 0
                if all("mtp_batch_stats" in r for r in responses):
                    shared = [
                        w["requests"][0]["response"]["mtp_batch_stats"] for w in waves
                    ]
                    for w in waves:
                        assert (
                            len(
                                {
                                    r["response"]["batch_cohort_id"]
                                    for r in w["requests"]
                                }
                            )
                            == 1
                        )
                        assert all(
                            r["response"]["mtp_batch_stats"]
                            == w["requests"][0]["response"]["mtp_batch_stats"]
                            for r in w["requests"]
                        )
                    row["shared_mtp_batch_stats_once_per_wave"] = shared
                    row["mean_native_prefill_complete_seconds"] = statistics.mean(
                        s["shared_prefill_complete_seconds"] for s in shared
                    )
                    row["mean_native_cohort_seconds"] = statistics.mean(
                        s["shared_native_generation_seconds"] for s in shared
                    )
                    row["native_cohort_after_all_first_outputs_tps"] = (
                        31
                        * count
                        / statistics.mean(
                            s["shared_native_generation_seconds"]
                            - s["shared_prefill_complete_seconds"]
                            for s in shared
                        )
                    )
                rows.append(row)
            result["experiments"].append(
                dict(
                    capsule=str(base.relative_to(ROOT.parent)),
                    label=e["label"],
                    engine=e["engine"],
                    mode=e["mode"],
                    depth=e["mtp_depth"],
                    reused=reused,
                    summary=rows,
                )
            )
    index = {
        (e["capsule"], e["label"]): {r["request_count"]: r for r in e["summary"]}
        for e in result["experiments"]
    }
    native = "v100-batched-mtp/v100/campaign-v3"
    sdk = "v100-batched-mtp/v100/sdk-campaign-v8"
    old = index[native, "repeat-native-mtp-batch"]
    new = index[sdk, "repeat-new-sdk"]
    sdk_old = index[sdk, "repeat-old-sdk"]
    ordinary = index[native, "repeat-control-ordinary256"]
    single = index[native, "repeat-control-scalar-mtp2-c1"]
    external = index["v100-q8-wmma/v100/campaign-v5", "repeat-llama-tensor-off"]
    ext_single = index[
        "v100-strided-output/v100/external-depth-v9", "repeat-llama-tensor-mtp2-c1"
    ][1]
    result["native_target_batch_http_vs_stronger_existing_ratio"] = {
        str(n): old[n]["complete_request_output_tps"]
        / (single if n == 1 else ordinary)[n]["complete_request_output_tps"]
        for n in old
    }
    result["sdk_http_ratio"] = {
        str(n): new[n]["complete_request_output_tps"]
        / sdk_old[n]["complete_request_output_tps"]
        for n in new
    }
    result["new_sdk_target_batch_vs_ordinary_ratio"] = {
        str(n): new[n]["complete_request_output_tps"]
        / ordinary[n]["complete_request_output_tps"]
        for n in new
    }
    result["new_sdk_target_batch_vs_reused_llama_non_spec_ratio"] = {
        str(n): new[n]["complete_request_output_tps"]
        / external[n]["complete_request_output_tps"]
        for n in new
    }
    result["best_native_single_vs_reused_external_decode_ratio"] = (
        single[1]["native_single_post_first_tps"]
        / ext_single["llama_reported_post_first_tps"]
    )
    result["best_native_single_vs_reused_external_http_ratio"] = (
        single[1]["complete_request_output_tps"]
        / ext_single["complete_request_output_tps"]
    )
    for base, names, reused in [
        (
            ROOT / "v100/final-v4",
            ["mtp-batch-c1", "mtp-batch-c8", "ordinary256-control-c8"],
            False,
        ),
        (
            ROOT.parent / "v100-native-mtp2/v100/full-profile-v14",
            ["mtp2-prefix-full-c1", "mtp2-prefix-full-c8"],
            True,
        ),
    ]:
        for name in names:
            result["profiles"][base.name + "/" + name] = dict(
                reused=reused,
                metadata=read(base / (name + "-metadata.json")),
                attribution=read(base / (name + "-attribution.json")),
            )
    for name in ["mtp-batch-c1", "mtp-batch-c8"]:
        base = ROOT / "v100/sdk-final-v9"
        result["profiles"][base.name + "/" + name] = dict(
            reused=False,
            metadata=read(base / (name + "-metadata.json")),
            attribution=read(base / (name + "-attribution.json")),
        )
    (ROOT / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if "ratio" in k or k == "external_speed_goal_complete"
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
