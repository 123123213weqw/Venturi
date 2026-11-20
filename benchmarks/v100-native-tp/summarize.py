"""Recompute single-request two-V100 comparisons from preserved raw evidence.

Two-GPU projection sharding is not concurrent request serving. No MTP or
external-engine timing is mixed into these internal, unprofiled measurements.
"""

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
    assert all(r["completed"] and not r["profiled"] for r in reports)
    assert all(r["dtype"] == "fp16" and r["weight_mode"] == "resident" for r in reports)
    rows = [r for report in reports for r in report["trials"]]
    assert all(len(r["generated_ids"]) == 32 for r in rows)
    assert all(
        r["before_decode"]["model_weight_upload_bytes"]
        == r["after_decode"]["model_weight_upload_bytes"]
        for r in rows
    )
    result = dict(trials=len(rows))
    for scope, tokens in (("ttft", 1), ("decode", 31), ("unsplit", 32)):
        walls = [r[scope]["wall_seconds"] for r in rows]
        item = dict(
            mean_wall_seconds=statistics.mean(walls),
            min_wall_seconds=min(walls),
            max_wall_seconds=max(walls),
            mean_process_cpu_seconds=statistics.mean(
                r[scope]["process_cpu_seconds"] for r in rows
            ),
        )
        if scope != "ttft":
            item["output_tokens_per_wall_second"] = tokens / item["mean_wall_seconds"]
        result[scope] = item
    return result


def telemetry(reports, directories, filename):
    tz = timezone(timedelta(seconds=read(REMOTE / "server-clock.json")["utc_offset_seconds"]))
    samples = []
    for report, directory in zip(reports, directories):
        windows = [
            (r[s]["start_unix"], r[s]["end_unix"])
            for r in report["trials"]
            for s in ("ttft", "decode", "unsplit")
        ]
        for row in csv.reader((directory / filename).read_text().splitlines()):
            if len(row) != 8:
                continue
            timestamp = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=tz).timestamp()
            if any(a <= timestamp <= b for a, b in windows):
                samples.append([float(v.strip()) for v in row[1:]])
    assert samples
    names = (
        "gpu_busy_percent", "memory_controller_busy_percent", "device_used_mib",
        "power_watts", "temperature_c", "sm_clock_mhz", "memory_clock_mhz",
    )
    return dict(
        samples=len(samples),
        scope="Device-wide NVML samples during timed ranges; not achieved occupancy/bandwidth",
        values={
            name: dict(mean=statistics.mean(row[i] for row in samples), min=min(row[i] for row in samples), max=max(row[i] for row in samples))
            for i, name in enumerate(names)
        },
    )


def same_ids(reports):
    assert len({r["artifact_id"] for r in reports}) == 1
    assert all(r["case"] == reports[0]["case"] for r in reports)
    expected = reports[0]["warmup_ids"]
    assert all(r["warmup_ids"] == expected for r in reports)
    assert all(row["generated_ids"] == expected for r in reports for row in r["trials"])
    return expected


def main():
    for job in ("full", "diagnostics", "validation-v2", "warp"):
        assert (REMOTE / job / "exit-code.txt").read_text().strip() == "0", job
    qualification = read(REMOTE / "full/qualification/report.json")
    assert qualification["native_full_runtime_case_passed"]
    assert qualification["all_native_logit_hidden_state_bytes_match_previous"]
    assert qualification["all_fixed_cpu_logit_gates_passed"]
    phases = [read(REMOTE / f"full/matched-phase-{i}/report.json") for i in range(4)]
    same_ids(phases)
    assert len({r["library_sha256"] for r in phases}) == 1
    singles, pairs = [phases[0], phases[3]], [phases[1], phases[2]]
    assert all(r["tensor_parallel_device"] is None for r in singles)
    assert all(r["tensor_parallel_device"] == "cuda:1" and r["tensor_parallel_transport"] == "host" for r in pairs)
    story = [read(REMOTE / f"full/story-phase-{i}/report.json") for i in range(2)]
    story_ids = same_ids(story)
    assert not set(story_ids) & {248046, 248044}
    result = dict(
        artifact_id=phases[0]["artifact_id"],
        artifact_sha256=qualification["artifact_sha256"],
        library_sha256=phases[0]["library_sha256"],
        workload=dict(prompt_tokens=26, output_tokens=32, decode_tokens=31, concurrency=1,
                      order=["single", "parallel", "parallel", "single"], eos_stopping=False,
                      scope="Native single request; model load/reset/warmup excluded; fixed work includes post-EOS tokens"),
        single=aggregate(singles), parallel=aggregate(pairs),
        exact_all_generated_ids=True, exact_trained_logits_hidden_gdn_kv=True,
        fixed_cpu_reference_gates_passed=True,
        story=dict(prompt_tokens=23, output_tokens=32, emitted_eos=False, exact_generated_ids=True,
                   single=aggregate([story[0]]), parallel=aggregate([story[1]])),
        initial_native_memory=dict(single=singles[0]["initial_native_memory"], parallel=pairs[0]["initial_native_memory"]),
        peer_probe=read(REMOTE / "diagnostics/peer-probe.json"),
        hardware_counters_available=False, driver_policy_changed=False,
        native_mtp_implemented=False, native_continuous_batching_paged_kv_qualified=False,
        whole_model_graphs_qualified=False, full_original_bf16_corpus_quality_qualified=False,
        independent_upstream_architecture_qualified=False, matched_external_engine_comparison=False,
        reported_50_tps=dict(source="User confirmation in this chat", gpus=2, mtp=True,
                             checkpoint="Stopped service used Qwen3.8-27B-Uncensored-Q8_0.gguf; native uses official Qwen3.5-27B",
                             independently_reproduced=False, comparable_same_checkpoint_workload=False),
    )
    result["speedup"] = {s: result["single"][s]["mean_wall_seconds"] / result["parallel"][s]["mean_wall_seconds"] for s in ("ttft", "decode", "unsplit")}
    result["telemetry"] = dict(
        single_primary_physical_gpu_1=telemetry(singles, [REMOTE / f"full/matched-phase-{i}" for i in (0, 3)], "gpu-samples.csv"),
        parallel_primary_physical_gpu_1=telemetry(pairs, [REMOTE / f"full/matched-phase-{i}" for i in (1, 2)], "gpu-samples.csv"),
        parallel_secondary_physical_gpu_0=telemetry(pairs, [REMOTE / f"full/matched-phase-{i}" for i in (1, 2)], "peer-gpu-samples.csv"),
    )
    result["profiler_only"] = {s: read(REMOTE / f"diagnostics/{s}-profiler/attribution.json") for s in ("single", "parallel")}
    result["profiled_host_apis"] = {s: read(REMOTE / f"diagnostics/{s}-profiler/api-attribution.json") for s in ("single", "parallel")}
    warp = read(REMOTE / "warp/ablation/report.json")
    assert warp["completed"] and all(c["exact_baseline_output_bytes"] for c in warp["cases"])
    result["unpromoted_warp_experiment"] = dict(
        promoted=False,
        scope="Projection-only graph microbenchmark; not a full-model speed result",
        cases=[dict(shape=c["shape"], median_gpu_event_ms=c["median_gpu_event_ms"]) for c in warp["cases"]],
    )
    (ROOT / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("single", "parallel", "speedup", "story")}, indent=2))


if __name__ == "__main__":
    main()
