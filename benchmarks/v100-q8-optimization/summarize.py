"""Recompute matched native-only measurements; do not mix profiler timings."""

import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parent
REMOTE = ROOT / "v100"


def read(path):
    return json.loads(path.read_text())


def aggregate(reports):
    rows = [row for report in reports for row in report["trials"]]
    assert all(r["completed"] and not r["profiled"] for r in reports)
    assert all(r["weight_mode"] == "resident" and r["dtype"] == "fp16" for r in reports)
    assert all(len(row["generated_ids"]) == 32 for row in rows)
    for row in rows:
        assert (
            row["before_decode"]["model_weight_upload_bytes"]
            == row["after_decode"]["model_weight_upload_bytes"]
        )
    result = dict(trials=len(rows))
    for name in ("ttft", "decode", "unsplit"):
        walls = [row[name]["wall_seconds"] for row in rows]
        item = dict(
            mean_wall_seconds=statistics.mean(walls),
            min_wall_seconds=min(walls),
            max_wall_seconds=max(walls),
            mean_process_cpu_seconds=statistics.mean(
                row[name]["process_cpu_seconds"] for row in rows
            ),
        )
        if name != "ttft":
            item["output_tokens_per_second"] = (31 if name == "decode" else 32) / item[
                "mean_wall_seconds"
            ]
        result[name] = item
    return result


def telemetry(reports, directories):
    tz = timezone(
        timedelta(seconds=read(REMOTE / "server-clock.json")["utc_offset_seconds"])
    )
    samples = []
    for report, directory in zip(reports, directories):
        intervals = [
            (r[scope]["start_unix"], r[scope]["end_unix"])
            for r in report["trials"]
            for scope in ("ttft", "decode", "unsplit")
        ]
        for row in csv.reader((directory / "gpu-samples.csv").read_text().splitlines()):
            if len(row) != 8:
                continue
            timestamp = (
                datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
                .replace(tzinfo=tz)
                .timestamp()
            )
            if any(a <= timestamp <= b for a, b in intervals):
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
        scope="Device-wide sampled NVML proxies during timed ranges; not achieved occupancy/bandwidth",
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
    assert (REMOTE / "final-exit-code.txt").read_text().strip() == "0"
    assert (REMOTE / "deploy-exit-code.txt").read_text().strip() == "0"
    qualifiers = [
        read(REMOTE / f"{name}/report.json")
        for name in ("resident-qualification", "tiled-qualification")
    ]
    assert all(r["native_full_runtime_case_passed"] for r in qualifiers)
    phases = [read(REMOTE / f"matched-phase-{i}/report.json") for i in range(4)]
    baseline = [phases[0], phases[3]]
    tiled = [phases[1], phases[2]]
    assert all(r["case"] == phases[0]["case"] for r in phases)
    assert all(r["artifact_id"] == phases[0]["artifact_id"] for r in phases)
    assert all(r["warmup_ids"] == phases[0]["warmup_ids"] for r in phases)
    assert all(
        row["generated_ids"] == phases[0]["warmup_ids"]
        for r in phases
        for row in r["trials"]
    )
    metrics = dict(
        artifact_sha256=qualifiers[0]["artifact_sha256"],
        artifact_id=phases[0]["artifact_id"],
        workload=dict(
            prompt_tokens=26,
            output_tokens=32,
            decode_tokens=31,
            eos_stopping=False,
            order=["baseline", "tiled", "tiled", "baseline"],
            scope="Single native request; load/preparation/reset/warmup excluded; EOS-disabled fixed work includes post-EOS tokens",
        ),
        baseline=aggregate(baseline),
        tiled=aggregate(tiled),
        exact_all_matched_generated_ids=True,
        exact_trained_logits_hidden_gdn_kv_bytes=True,
        fixed_cpu_reference_logit_gates_passed=True,
        full_original_bf16_corpus_quality_qualified=False,
        independent_upstream_architecture_qualified=False,
        external_engine_performance_comparison=False,
        native_batching_paged_kv_qualified=False,
        whole_model_graphs_qualified=False,
        achieved_hardware_counters_available=False,
    )
    metrics["speedup"] = {
        key: metrics["baseline"][key]["mean_wall_seconds"]
        / metrics["tiled"][key]["mean_wall_seconds"]
        for key in ("ttft", "decode", "unsplit")
    }
    metrics["telemetry"] = {
        "baseline": telemetry(
            baseline, [REMOTE / f"matched-phase-{i}" for i in (0, 3)]
        ),
        "tiled": telemetry(tiled, [REMOTE / f"matched-phase-{i}" for i in (1, 2)]),
    }
    story = [read(REMOTE / f"story-phase-{i}/report.json") for i in range(2)]
    assert story[0]["warmup_ids"] == story[1]["warmup_ids"]
    assert all(
        row["generated_ids"] == story[0]["warmup_ids"]
        for r in story
        for row in r["trials"]
    )
    assert not set(story[0]["warmup_ids"]) & {248046, 248044}
    metrics["story"] = dict(
        prompt_tokens=23,
        output_tokens=32,
        emitted_eos=False,
        exact_matched_ids=True,
        baseline=aggregate([story[0]]),
        tiled=aggregate([story[1]]),
    )
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: metrics[key]
                for key in ("baseline", "tiled", "speedup", "telemetry")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
