"""Finite single-request prefill benchmark through the resident public Qwen API.

First-token timing is the native engine's internal timestamp, not streamed
client time-to-first-byte. Client wall includes the complete generation RPC.
Weights, precision, context and chunk size stay fixed between runner variants.
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


def make_prompts(tokenize, lengths):
    unit = tokenize("The quick brown fox jumps over the lazy dog. ")
    suffix = tokenize("\nContinue the explanation:")
    if not unit:
        raise ValueError("Tokenizer returned an empty fixture unit")
    prompts = {}
    for length in lengths:
        prefix = tokenize(f"Document with {length} tokens. ")
        if length <= len(prefix) + len(suffix):
            raise ValueError("Prompt length cannot contain the fixture prefix/suffix")
        body = unit * (length // len(unit) + 1)
        prompts[length] = (prefix + body)[: length - len(suffix)] + suffix
        if len(prompts[length]) != length:
            raise ValueError("Fixture token count mismatch")
    return prompts


def summarize(runs):
    result = {}
    for length in sorted({r["prompt_length"] for r in runs}):
        chosen = [r for r in runs if r["prompt_length"] == length]
        wall = sum(r["wall_seconds"] for r in chosen)
        prefill = sum(r["result"]["timings"]["prefill_seconds"] for r in chosen)
        result[str(length)] = dict(
            calls=len(chosen),
            median_native_prefill_seconds=statistics.median(
                r["result"]["timings"]["prefill_seconds"] for r in chosen
            ),
            native_prefill_tokens_per_second=length * len(chosen) / prefill,
            median_native_first_token_seconds=statistics.median(
                r["result"]["timings"]["first_token_seconds"] for r in chosen
            ),
            median_native_decode_seconds=statistics.median(
                r["result"]["timings"]["decode_seconds"] for r in chosen
            ),
            client_wall_seconds=wall,
            native_cpu_seconds=sum(r["native_cpu_seconds"] for r in chosen),
            python_cpu_seconds=sum(r["python_cpu_seconds"] for r in chosen),
            native_cpu_percent=100
            * sum(r["native_cpu_seconds"] for r in chosen)
            / wall,
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--runner", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=[128, 509, 1021, 2048, 8192]
    )
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if (
        not 1 <= args.repeats <= 8
        or not args.lengths
        or any(n < 1 or n + 16 > 16384 for n in args.lengths)
    ):
        raise ValueError(
            "Require 1..8 repeats and prompt lengths fitting 16384 with 16 output tokens"
        )
    out = Path(args.output_directory)
    out.mkdir(parents=True, exist_ok=False)
    background = idle_gpu()
    runs = []
    generation = dict(max_new_tokens=16, ignore_eos=True)
    with QwenNativeSession(
        args.checkpoint,
        runner=args.runner,
        max_context=16384,
        max_concurrency=1,
        prefill_chunk=512,
        cuda_graph=True,
        cpu_wait="blocking",
        timeout=600,
    ) as model:
        ready = model.ready
        prompts = make_prompts(model.tokenize, sorted(set(args.lengths)))
        warm = {
            n: model.generate(token_ids=ids, **generation) for n, ids in prompts.items()
        }
        with (out / "telemetry.csv").open("w") as sink:
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
                    order = (
                        list(prompts) if repeat % 2 == 0 else list(reversed(prompts))
                    )
                    for length in order:
                        measured = measure(
                            model,
                            lambda: model.generate(
                                token_ids=prompts[length], **generation
                            ),
                        )
                        if measured["result"]["prompt_tokens"] != length:
                            raise AssertionError("Native prompt token count changed")
                        runs.append(
                            dict(prompt_length=length, repeat=repeat, **measured)
                        )
                        print(
                            length,
                            measured["result"]["timings"]["prefill_seconds"],
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
    repeat_equal = all(
        r["result"]["generated_ids"] == warm[r["prompt_length"]]["generated_ids"]
        for r in runs
    )
    report = dict(
        method=__doc__,
        ready=ready,
        background=background,
        generation=generation,
        prompt_ids=prompts,
        warm=warm,
        runs=runs,
        summaries=summarize(runs),
        replay_matches_warm=repeat_equal,
        process_exit=model._process.returncode,
        runner_sha256=hashlib.sha256(Path(args.runner).read_bytes()).hexdigest(),
        measured_phase=dict(start_unix=start, end_unix=end),
        telemetry=telemetry(
            out / "telemetry.csv", [dict(mode="single", start_unix=start, end_unix=end)]
        ),
    )
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summaries"], indent=2))
    if not repeat_equal:
        raise AssertionError(
            "State replay differs; counterexamples retained in results.json"
        )


if __name__ == "__main__":
    main()
