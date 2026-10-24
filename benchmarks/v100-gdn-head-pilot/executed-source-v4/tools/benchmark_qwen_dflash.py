"""Eight-request DFlash2/off comparison through the resident production batch API.

One generate_many RPC submits eight requests; this is not eight HTTP clients.
Aggregate throughput uses shared client wall time including prefill and JSON
transport, never summed overlapping request durations. Startup/warmup excluded.
Native per-request timings are not streamed client TTFT. No precision changes,
new models, external builds, or kernel edits are made.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
from time import time

from benchmark_qwen_cpu import measure
from benchmark_qwen_gpu import FIELDS, telemetry
from run_qwen_validation import idle_gpu
from venturi.backends.qwen import QwenNativeSession


PROMPTS = [
    "Explain why the sky looks blue. Give a detailed answer.",
    "Write a Python function to merge two sorted lists and explain its complexity.",
    "Explain how a CPU cache works and why locality matters.",
    "Write a Python binary search function and describe its edge cases.",
    "Explain how rain forms, from evaporation to precipitation.",
    "Compare TCP and UDP, with examples of when to use each.",
    "Write a Python function that counts word frequencies and explain its complexity.",
    "Explain photosynthesis and its role in the carbon cycle.",
]
ORDER = (
    ("baseline-a", "none"),
    ("dflash-a", "dflash2"),
    ("dflash-b", "dflash2"),
    ("baseline-b", "none"),
)


def summarize(waves):
    rows = [r for w in waves for r in w["result"]["results"]]
    tokens = sum(len(r["generated_ids"]) for r in rows)
    wall = sum(w["wall_seconds"] for w in waves)
    rounds = sum(r["speculative"]["rounds"] for r in rows)
    drafted = sum(r["speculative"]["drafted_tokens"] for r in rows)
    accepted = sum(r["speculative"]["accepted_tokens"] for r in rows)
    decode_rounds = sum(w["result"]["runtime"]["decode_rounds"] for w in waves)
    return dict(
        output_tokens=tokens,
        client_wall_seconds=wall,
        aggregate_output_tokens_per_second=tokens / wall,
        median_batch_client_wall_seconds=statistics.median(
            w["wall_seconds"] for w in waves
        ),
        native_cpu_seconds=sum(w["native_cpu_seconds"] for w in waves),
        python_cpu_seconds=sum(w["python_cpu_seconds"] for w in waves),
        native_cpu_percent=100 * sum(w["native_cpu_seconds"] for w in waves) / wall,
        median_native_first_token_seconds=statistics.median(
            r["timings"]["first_token_seconds"] for r in rows
        ),
        median_native_request_total_seconds=statistics.median(
            r["timings"]["total_seconds"] for r in rows
        ),
        mean_actual_decode_batch=sum(
            w["result"]["runtime"]["decode_row_rounds"] for w in waves
        )
        / decode_rounds
        if decode_rounds
        else None,
        speculative_rounds=rounds,
        drafted_tokens=drafted,
        accepted_draft_tokens=accepted,
        accepted_draft_fraction=accepted / drafted if drafted else None,
        accepted_drafts_per_round=accepted / rounds if rounds else None,
        accepted_per_position=[
            sum(r["speculative"]["accepted_per_position"][i] for r in rows)
            for i in range(8)
        ]
        if rounds
        else [],
        fallback_steps=sum(r["speculative"]["fallback_steps"] for r in rows),
    )


def compare(reference, actual):
    prefix = next(
        (i for i, (a, b) in enumerate(zip(reference, actual)) if a != b),
        min(len(reference), len(actual)),
    )
    return dict(
        equal=reference == actual,
        matching_prefix=prefix,
        reference_length=len(reference),
        actual_length=len(actual),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("--runner", required=True)
    p.add_argument("--output-directory", required=True)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    if not 1 <= args.repeats <= 6:
        raise ValueError("repeats must be 1..6")
    out = Path(args.output_directory)
    out.mkdir(parents=True, exist_ok=False)
    generation = dict(max_new_tokens=128, ignore_eos=True)
    sessions = []
    for name, speculative in ORDER:
        directory = out / name
        directory.mkdir()
        background = idle_gpu()
        waves = []
        with QwenNativeSession(
            args.checkpoint,
            runner=args.runner,
            mode="parallel",
            max_context=2048,
            max_concurrency=8,
            kv_capacity=16384,
            prefill_chunk=512,
            kv_dtype="bf16",
            cuda_graph=True,
            cpu_wait="blocking",
            speculative=speculative,
            draft_tokens=8,
            verify_tree=False,
            optimized_proposal=False,
            timeout=600,
        ) as model:
            ready = model.ready
            warm = model.generate_many(PROMPTS, **generation)
            with (directory / "telemetry.csv").open("w") as sink:
                sampler = subprocess.Popen(
                    [
                        "nvidia-smi",
                        "--query-gpu=" + ",".join(FIELDS),
                        "--format=csv,noheader,nounits",
                        "-lms",
                        "200",
                    ],
                    stdout=sink,
                    stderr=subprocess.DEVNULL,
                )
                start = time()
                try:
                    for repeat in range(args.repeats):
                        measured = measure(
                            model, lambda: model.generate_many(PROMPTS, **generation)
                        )
                        if any(
                            len(r["generated_ids"]) != 128
                            for r in measured["result"]["results"]
                        ):
                            raise AssertionError(
                                "Fixed-length workload did not produce 128 tokens"
                            )
                        waves.append(dict(repeat=repeat, **measured))
                        print(
                            name,
                            repeat,
                            measured["result"]["output_tokens_per_second"],
                            flush=True,
                        )
                    end = time()
                finally:
                    sampler.terminate()
                    try:
                        sampler.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        sampler.kill()
                        sampler.wait()
            if sampler.returncode not in (0, -15):
                raise RuntimeError(f"Telemetry sampler failed: {sampler.returncode}")
        replay = [
            [
                compare(a["generated_ids"], b["generated_ids"])
                for a, b in zip(warm["results"], w["result"]["results"])
            ]
            for w in waves
        ]
        session = dict(
            name=name,
            speculative=speculative,
            ready=ready,
            background=background,
            warm=warm,
            waves=waves,
            replay=replay,
            all_replays_match_warm=all(c["equal"] for row in replay for c in row),
            process_exit=model._process.returncode,
            summary=summarize(waves),
            measured_phase=dict(start_unix=start, end_unix=end),
            telemetry=telemetry(
                directory / "telemetry.csv",
                [dict(mode="parallel", start_unix=start, end_unix=end)],
            ),
        )
        (directory / "results.json").write_text(json.dumps(session, indent=2) + "\n")
        sessions.append(session)
    baseline = sessions[0]["warm"]["results"]
    comparisons = [
        dict(
            session=s["name"],
            repeat=w["repeat"],
            request=i,
            **compare(baseline[i]["generated_ids"], r["generated_ids"]),
        )
        for s in sessions
        for w in s["waves"]
        for i, r in enumerate(w["result"]["results"])
    ]
    summaries = {
        mode: summarize(
            [w for s in sessions if s["speculative"] == mode for w in s["waves"]]
        )
        for mode in ("none", "dflash2")
    }
    report = dict(
        method=__doc__,
        prompts=PROMPTS,
        generation=generation,
        runner_sha256=hashlib.sha256(Path(args.runner).read_bytes()).hexdigest(),
        checkpoint=args.checkpoint,
        session_order=[n for n, _ in ORDER],
        summaries=summaries,
        output_comparisons=comparisons,
        all_replays_match_warm=all(s["all_replays_match_warm"] for s in sessions),
        all_native_processes_exited_cleanly=all(
            s["process_exit"] == 0 for s in sessions
        ),
        throughput_speedup=summaries["dflash2"]["aggregate_output_tokens_per_second"]
        / summaries["none"]["aggregate_output_tokens_per_second"],
        limitations=[
            "Fixed 128 outputs, EOS ignored; not an EOS-respecting chat or HTTP load test.",
            "Greedy differences are retained, not declared lossless.",
            "GPU telemetry is coarse whole-device measurement, not achieved occupancy or bandwidth utilization.",
            "Prior counter restriction unchanged; achieved occupancy/compute/DRAM/cache/stall counters unavailable.",
        ],
    )
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(summaries=summaries, speedup=report["throughput_speedup"]), indent=2
        )
    )
    if not report["all_native_processes_exited_cleanly"]:
        raise AssertionError("Native process failed; evidence retained")
    if not report["all_replays_match_warm"]:
        raise AssertionError("Within-session replay differed; counterexamples retained")


if __name__ == "__main__":
    main()
