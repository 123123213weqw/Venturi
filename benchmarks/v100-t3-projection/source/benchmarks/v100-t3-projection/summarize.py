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
    campaign = read(ROOT / "v100/campaign-v3/campaign-report.json")
    assert campaign["completed"]
    paths = [
        (ROOT / "v100/campaign-v3", False),
        (ROOT.parent / "v100-native-mtp2/v100/prefix-campaign-v10", True),
        (ROOT.parent / "v100-prefill-budget/v100/campaign-v2", True),
        (ROOT.parent / "v100-q8-wmma/v100/campaign-v5", True),
        (ROOT.parent / "v100-strided-output/v100/external-depth-v9", True),
    ]
    result = dict(
        external_speed_goal_complete=False,
        scope="32 greedy output tokens per request / global cohort wall, including HTTP reset, queueing, archival and validation; native generation/TTFT excludes reset and HTTP. No summed concurrent request durations. Profiling is separate.",
        promoted=campaign["promoted"],
        batch_promoted=campaign["batch_promoted"],
        experiments=[],
        achieved_occupancy=None,
    )
    for base, reused in paths:
        c = read(base / "campaign-report.json")
        for e in c["experiments"]:
            if reused and not e["label"].startswith("repeat"):
                continue
            if "v100-q8-wmma" in str(base) and e["engine"] != "llama":
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
                    row["native_ttft_seconds"] = statistics.mean(
                        t["native_time_to_first_token_seconds"] for t in tt
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
                if e["mtp_depth"]:
                    row["mtp_counters"] = [
                        r["mtp_stats" if e["engine"] == "venturi" else "timings"]
                        for r in responses
                    ]
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
    indexed = {
        (e["capsule"], e["label"]): {r["request_count"]: r for r in e["summary"]}
        for e in result["experiments"]
    }
    current = "v100-t3-projection/v100/campaign-v3"
    if campaign["promoted"]:
        a = indexed[current, "repeat-control-mtp2"]
        b = indexed[current, "repeat-candidate-mtp2"]
        result["same_profile_http_ratio"] = {
            str(n): b[n]["complete_request_output_tps"]
            / a[n]["complete_request_output_tps"]
            for n in b
        }
        result["same_profile_single_decode_ratio"] = (
            b[1]["native_single_post_first_tps"] / a[1]["native_single_post_first_tps"]
        )
        external = indexed[
            "v100-strided-output/v100/external-depth-v9", "repeat-llama-tensor-mtp2-c1"
        ][1]
        result["single_vs_best_tested_reused_external_http_ratio"] = (
            b[1]["complete_request_output_tps"]
            / external["complete_request_output_tps"]
        )
        result["single_vs_best_tested_reused_external_decode_ratio"] = (
            b[1]["native_single_post_first_tps"]
            / external["llama_reported_post_first_tps"]
        )
    op = read(ROOT / "v100/study-v1/operators/report.json")
    assert op["completed"]
    result["operators"] = op
    result["profiles"] = {}
    for base, names, reused in [
        (ROOT / "v100/final-v4", ["mtp2-prefix-c1", "mtp2-prefix-c8"], False),
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
    (ROOT / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if "ratio" in k
                or k in ("promoted", "batch_promoted", "external_speed_goal_complete")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
