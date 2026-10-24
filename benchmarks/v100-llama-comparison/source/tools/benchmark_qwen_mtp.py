"""Matched native non-speculative/MTP complete-request throughput and acceptance.

Both target and draft weights stay resident in both variants. Request slots,
reset, model load and full-length warmup are outside timing. Aggregate TPS is
all emitted tokens / global cohort wall. Single-request post-first-token rate
uses same-call native timestamps; no concurrent decode TPS is manufactured from
summed/averaged request durations or from subtracting unrelated prefill trials.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
from time import perf_counter

import torch

from benchmark_qwen_native_parallel import run_wave
from venturi.models import (
    NativeQwenModel,
    NativeQwenPipelineModel,
    NativeQwenMTPModel,
    QwenQ8Weights,
    QwenMTPWeights,
)
from venturi.ninfer import NInferArtifact
from venturi.ops import CudaBackend


def acceptance(rows):
    proposed = sum(r["proposed_tokens"] for r in rows)
    accepted = sum(r["accepted_tokens"] for r in rows)
    return dict(
        proposed_tokens=proposed,
        accepted_tokens=accepted,
        acceptance_rate=accepted / proposed if proposed else None,
        accepted_by_depth=[accepted],
        proposed_by_depth=[proposed],
        verification_calls=sum(r["verification_calls"] for r in rows),
        replayed_input_tokens=sum(r["replayed_input_tokens"] for r in rows),
        denominator="Matching draft tokens / all proposed draft tokens; warmup excluded",
    )


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["request_count"], r["variant"])].append(r)
    result = []
    for (count, variant), trials in groups.items():
        outputs = {r["cohort"]["emitted_tokens"] for r in trials}
        if len(outputs) != 1:
            raise ValueError("Unequal emitted work")
        wall = statistics.mean(r["cohort"]["wall_seconds"] for r in trials)
        item = dict(
            request_count=count,
            variant=variant,
            trials=len(trials),
            mean_cohort_wall_seconds=wall,
            min_cohort_wall_seconds=min(r["cohort"]["wall_seconds"] for r in trials),
            max_cohort_wall_seconds=max(r["cohort"]["wall_seconds"] for r in trials),
            aggregate_complete_request_output_tps=next(iter(outputs)) / wall,
            mean_process_cpu_seconds=statistics.mean(
                r["cohort"]["process_cpu_seconds"] for r in trials
            ),
            mean_request_completion_seconds=statistics.mean(
                q["completion_latency_seconds"]
                for r in trials
                for q in r["cohort"]["requests"]
            ),
            max_request_completion_seconds=max(
                q["completion_latency_seconds"]
                for r in trials
                for q in r["cohort"]["requests"]
            ),
            mean_native_ttft_seconds=statistics.mean(
                t["native_time_to_first_token_seconds"]
                for r in trials
                for t in r["native_timings"]
            ),
        )
        if count == 1:
            periods = [
                t["native_generation_seconds"] - t["native_time_to_first_token_seconds"]
                for r in trials
                for t in r["native_timings"]
            ]
            item["single_request_native_post_first_output_tps"] = (
                next(iter(outputs)) - 1
            ) / statistics.mean(periods)
        if variant == "mtp":
            item["acceptance"] = acceptance([s for r in trials for s in r["mtp_stats"]])
        result.append(item)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact")
    p.add_argument("--mtp-artifact", required=True)
    p.add_argument("--library", required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--mode", choices=("tp", "pp"), required=True)
    p.add_argument("--output-directory", required=True)
    p.add_argument("--concurrency", default="1,2,4,8")
    p.add_argument("--profile-nvtx", action="store_true")
    p.add_argument("--profile-variant", choices=("baseline", "mtp"), default="mtp")
    p.add_argument("--new-tokens", type=int, default=32)
    args = p.parse_args()
    counts = [int(n) for n in args.concurrency.split(",")]
    if (
        not counts
        or any(c not in (1, 2, 4, 8) for c in counts)
        or len(set(counts)) != len(counts)
    ):
        p.error("Unique concurrency values 1,2,4,8 required")
    case = json.loads(Path(args.case).read_text())
    ids, expected = (
        case["prompt_token_ids"],
        case["expected_generated_ids"][: args.new_tokens],
    )
    if not 2 <= args.new_tokens <= 256 or len(expected) != args.new_tokens:
        p.error("Use fixed 2..256 output IDs from a pinned case")
    out = Path(args.output_directory)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    report = dict(
        scope=__doc__,
        mode=args.mode,
        case=case,
        new_tokens=args.new_tokens,
        dtype="fp16",
        eos_token_ids=[],
        physical_gpus=[1, 0],
        cuda_visible_devices="1,0",
        library_sha256=hashlib.sha256(Path(args.library).read_bytes()).hexdigest(),
        profiled=args.profile_nvtx,
        achieved_occupancy=None,
        model_weights_resident_in_both_variants=True,
        trials=[],
        completed=False,
    )

    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()

    samplers = []
    logs = []
    target = model = None
    sessions = []
    try:
        for gpu in (1, 0):
            f = (out / f"gpu-{gpu}-samples.csv").open("wb")
            logs.append(f)
            samplers.append(
                subprocess.Popen(
                    [
                        "nvidia-smi",
                        "-i",
                        str(gpu),
                        "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu,clocks.sm,clocks.mem",
                        "--format=csv,noheader,nounits",
                        "-lms",
                        "100",
                    ],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                )
            )
        backend = CudaBackend(args.library)
        weights = QwenQ8Weights.from_ninfer(NInferArtifact(args.artifact))
        draft = QwenMTPWeights.from_ninfer(NInferArtifact(args.mtp_artifact))
        draft.validate_target(weights)
        report.update(
            target_artifact_id=weights.artifact.artifact_id,
            mtp_artifact_id=draft.artifact.artifact_id,
        )
        kw = dict(
            weight_mode="resident",
            dtype=torch.float16,
            device="cuda:0",
            max_tokens=max(2, len(ids)),
        )
        start = perf_counter()
        target = (
            NativeQwenPipelineModel.from_q8(
                weights, backend, pipeline_device="cuda:1", split_layer=32, **kw
            )
            if args.mode == "pp"
            else NativeQwenModel.from_q8(
                weights,
                backend,
                tensor_parallel_device="cuda:1",
                tensor_parallel_transport="host",
                **kw,
            )
        )
        model = NativeQwenMTPModel.from_q8(target, draft)
        report["model_load_seconds"] = perf_counter() - start
        schedule = (
            (args.profile_variant,)
            if args.profile_nvtx
            else ("baseline", "mtp", "mtp", "baseline")
        )
        from concurrent.futures import ThreadPoolExecutor

        for count in counts:
            for phase, variant in enumerate(schedule):
                selected = target if variant == "baseline" else model
                sessions = [
                    selected.create_session(len(ids) + args.new_tokens)
                    for _ in range(count)
                ]
                try:
                    with ThreadPoolExecutor(max_workers=count) as pool:
                        calls = [
                            lambda s=s: s.generate(
                                ids, max_new_tokens=args.new_tokens, eos_token_ids=[]
                            )
                            for s in sessions
                        ]
                        warm = run_wave(pool, calls, args.new_tokens)
                        assert all(q["ids"] == expected for q in warm["requests"])
                        for s in sessions:
                            s.reset()
                        initial = [s.info() for s in sessions]
                        wave = run_wave(
                            pool,
                            calls,
                            args.new_tokens,
                            "native_unsplit" if args.profile_nvtx else None,
                        )
                        assert all(q["ids"] == expected for q in wave["requests"])
                    times = [s.generation_timings() for s in sessions]
                    assert all(
                        0
                        < t["native_time_to_first_token_seconds"]
                        < t["native_generation_seconds"]
                        for t in times
                    )
                    stats = [s.stats() for s in sessions] if variant == "mtp" else []
                    assert all(s["forced_proposals"] == 0 for s in stats)
                    report["trials"].append(
                        dict(
                            request_count=count,
                            phase=phase,
                            variant=variant,
                            cohort=wave,
                            native_timings=times,
                            mtp_stats=stats,
                            initial_memory=initial,
                            final_memory=[s.info() for s in sessions],
                        )
                    )
                    save()
                    print(
                        json.dumps(
                            dict(
                                requests=count,
                                variant=variant,
                                phase=phase,
                                wall=wave["wall_seconds"],
                                tps=wave["aggregate_output_tokens_per_wall_second"],
                                acceptance=acceptance(stats) if stats else None,
                            )
                        ),
                        flush=True,
                    )
                finally:
                    for s in sessions:
                        s.close()
                    sessions = []
        report["summary"] = summarize(report["trials"])
        report["completed"] = True
        save()
    finally:
        for s in sessions:
            s.close()
        if model:
            model.close()
        if target:
            target.close()
        for s in samplers:
            s.terminate()
            s.wait(timeout=15)
        for f in logs:
            f.close()


if __name__ == "__main__":
    main()
