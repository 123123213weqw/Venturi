"""Recompute PP/TP cohort comparisons; never sum request latencies for TPS."""

import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
REMOTE = ROOT / "v100"


def read(path):
    return json.loads(path.read_text())


def aggregate(rows, count):
    result = dict(trials=len(rows), requests_per_trial=count)
    for scope, each in (("ttft", 1), ("decode", 31), ("unsplit", 32)):
        waves = [r[scope] for r in rows]
        assert all(w["emitted_tokens"] == count * each for w in waves)
        mean_wall = statistics.mean(w["wall_seconds"] for w in waves)
        result[scope] = dict(
            mean_cohort_wall_seconds=mean_wall,
            min_cohort_wall_seconds=min(w["wall_seconds"] for w in waves),
            max_cohort_wall_seconds=max(w["wall_seconds"] for w in waves),
            mean_process_cpu_seconds=statistics.mean(
                w["process_cpu_seconds"] for w in waves
            ),
            aggregate_output_tokens_per_wall_second=count * each / mean_wall,
            mean_request_completion_latency_seconds=statistics.mean(
                q["completion_latency_seconds"] for w in waves for q in w["requests"]
            ),
            max_request_completion_latency_seconds=max(
                q["completion_latency_seconds"] for w in waves for q in w["requests"]
            ),
            mean_request_thread_cpu_seconds=statistics.mean(
                q["thread_cpu_seconds"] for w in waves for q in w["requests"]
            ),
        )
    return result


def telemetry(phases, count, physical_gpu):
    samples = []
    for directory, report in phases:
        tz = timezone(timedelta(seconds=report["utc_offset_seconds"]))
        row = next(c for c in report["concurrency"] if c["request_count"] == count)
        windows = [
            (r[s]["start_unix"], r[s]["end_unix"])
            for r in row["trials"]
            for s in ("ttft", "decode", "unsplit")
        ]
        for r in csv.reader(
            (directory / f"gpu-{physical_gpu}-samples.csv").read_text().splitlines()
        ):
            if len(r) != 8:
                continue
            timestamp = (
                datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
                .replace(tzinfo=tz)
                .timestamp()
            )
            if any(a <= timestamp <= b for a, b in windows):
                samples.append([float(x.strip()) for x in r[1:]])
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
        scope="Device-wide samples in timed windows, not achieved occupancy/bandwidth",
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
    for job in ("final", "diagnostics", "validation-v2", "initial-comparison"):
        assert (REMOTE / job / "exit-code.txt").read_text().strip() == "0", job
    qualification = read(REMOTE / "final/qualification/report.json")
    assert qualification["native_full_runtime_case_passed"]
    assert qualification["all_fixed_cpu_logit_gates_passed"]
    assert qualification["all_native_logit_hidden_state_bytes_match_previous"]
    phases = [
        (
            REMOTE / f"final/phase-{i}",
            read(REMOTE / f"final/phase-{i}/report.json"),
        )
        for i in range(4)
    ]
    assert [r["mode"] for _, r in phases] == ["tp", "pp", "pp", "tp"]
    assert all(r["completed"] and not r["profiled"] for _, r in phases)
    assert len({r["library_sha256"] for _, r in phases}) == 1
    assert len({r["artifact_id"] for _, r in phases}) == 1
    expected = phases[0][1]["case"]["expected_generated_ids"]
    assert len(expected) == 32 and not set(expected) & {248046, 248044}
    for _, report in phases:
        assert (
            report["dtype"] == "fp16"
            and not report["mtp"]
            and report["case"] == phases[0][1]["case"]
        )
        assert [c["request_count"] for c in report["concurrency"]] == [1, 2, 4, 8]
        for cohort in report["concurrency"]:
            assert cohort["completed"]
            assert all(r["ids"] == expected for r in cohort["warmup"]["requests"])
            for trial in cohort["trials"]:
                for i, full in enumerate(trial["unsplit"]["requests"]):
                    assert full["ids"] == expected
                    assert (
                        trial["ttft"]["requests"][i]["ids"]
                        + trial["decode"]["requests"][i]["ids"]
                        == expected
                    )
                assert all(
                    a["model_weight_upload_bytes"] == b["model_weight_upload_bytes"]
                    for a, b in zip(trial["before_decode"], trial["after_decode"])
                )
    result = dict(
        scope=__doc__,
        artifact_id=phases[0][1]["artifact_id"],
        artifact_sha256=qualification["artifact_sha256"],
        library_sha256=phases[0][1]["library_sha256"],
        workload=dict(
            prompt_tokens=23,
            output_tokens=32,
            decode_tokens=31,
            eos_stopping=False,
            emitted_eos=False,
            order=["tp", "pp", "pp", "tp"],
            split_layer=32,
        ),
        correctness=dict(
            exact_logits_hidden_gdn_history_kv=True,
            unchanged_fixed_cpu_gates=True,
            all_concurrent_generated_ids_match_pinned_native=True,
        ),
        semantics=dict(
            tp="Existing serialized whole-model request lock, resident row-sharded projections",
            pp="Native stage-local locks/state, two contiguous layer partitions, requests overlap across stages",
            both="Explicit host DMA, FP16 profile, Q8 group32 weights; no MTP/batching/paged KV/whole-model graphs",
        ),
        achieved_hardware_counters_available=False,
        original_bf16_corpus_quality_qualified=False,
        independent_upstream_architecture_qualified=False,
        matched_external_engine_comparison=False,
        counts={},
        profiler_only={},
    )
    for count in (1, 2, 4, 8):
        by_mode = {}
        for mode in ("tp", "pp"):
            chosen = [(p, r) for p, r in phases if r["mode"] == mode]
            rows = [
                trial
                for _, r in chosen
                for c in r["concurrency"]
                if c["request_count"] == count
                for trial in c["trials"]
            ]
            by_mode[mode] = aggregate(rows, count)
            by_mode[mode]["telemetry"] = {
                str(gpu): telemetry(chosen, count, gpu) for gpu in (1, 0)
            }
            by_mode[mode]["initial_native_memory"] = next(
                c["initial_native_memory"]
                for c in chosen[0][1]["concurrency"]
                if c["request_count"] == count
            )
        by_mode["pp_over_tp_speedup"] = {
            s: by_mode["tp"][s]["mean_cohort_wall_seconds"]
            / by_mode["pp"][s]["mean_cohort_wall_seconds"]
            for s in ("ttft", "decode", "unsplit")
        }
        result["counts"][str(count)] = by_mode
    for mode in ("tp", "pp"):
        for count in (1, 8):
            result["profiler_only"][f"{mode}-c{count}"] = read(
                REMOTE / f"diagnostics/{mode}-c{count}/cohort-attribution.json"
            )
    initial = [
        read(REMOTE / f"initial-comparison/phase-{i}/report.json") for i in (1, 2)
    ]
    assert all(
        r["completed"] and not r["profiled"] and r["mode"] == "pp" for r in initial
    )
    result["initial_scheduler"] = dict(
        scope="Initial separate final-stage selection lock, preserved rather than promoted",
        counts={
            str(count): aggregate(
                [
                    t
                    for r in initial
                    for c in r["concurrency"]
                    if c["request_count"] == count
                    for t in c["trials"]
                ],
                count,
            )
            for count in (1, 2, 4, 8)
        },
    )
    result["final_stage_selection_atomic"] = True
    (ROOT / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    for count, modes in result["counts"].items():
        print(
            json.dumps(
                dict(
                    requests=count,
                    **{
                        m: {s: modes[m][s] for s in ("ttft", "decode", "unsplit")}
                        for m in ("tp", "pp")
                    },
                    speedup=modes["pp_over_tp_speedup"],
                )
            )
        )


if __name__ == "__main__":
    main()
