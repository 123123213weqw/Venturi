"""Finite Linux process-CPU/wall measurements through Venturi's resident Qwen API.

CPU percent uses 100% = one logical core. The owned native process includes its
threads; Python CPU is measured separately. Startup/warmup are excluded. No
claim about CPU power or energy follows from these process-time counters.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
from time import perf_counter

from run_qwen_validation import idle_gpu
from venturi.backends.qwen import QwenNativeSession


def process_cpu(pid):
    # comm may contain spaces or ')' characters; the final ')' closes field 2.
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


def python_cpu():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def measure(model, call):
    pid = model._process.pid
    before = process_cpu(pid)
    py_before = python_cpu()
    started = perf_counter()
    result = call()
    seconds = perf_counter() - started
    cpu = process_cpu(pid) - before
    py_cpu = python_cpu() - py_before
    return dict(
        wall_seconds=seconds,
        native_cpu_seconds=cpu,
        python_cpu_seconds=py_cpu,
        native_cpu_percent=100 * cpu / seconds,
        result=result,
    )


def summary(runs):
    seconds = sum(r["wall_seconds"] for r in runs)
    cpu = sum(r["native_cpu_seconds"] for r in runs)
    py_cpu = sum(r["python_cpu_seconds"] for r in runs)
    tokens = sum(
        len(x["generated_ids"])
        for r in runs
        for x in (r["result"].get("results", [r["result"]]))
    )
    return dict(
        wall_seconds=seconds,
        native_cpu_seconds=cpu,
        python_cpu_seconds=py_cpu,
        native_cpu_percent=100 * cpu / seconds,
        total_cpu_seconds_per_output_token=(cpu + py_cpu) / tokens,
        output_tokens=tokens,
        output_tokens_per_second=tokens / seconds,
        median_first_token_seconds=statistics.median(
            x["timings"]["first_token_seconds"]
            for r in runs
            for x in r["result"].get("results", [r["result"]])
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("--runner", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--cpu-wait", choices=("auto", "blocking"))
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--max-concurrency", type=int, default=1)
    args = p.parse_args()
    if not 1 <= args.repeats <= 20:
        raise ValueError("repeats must be 1..20")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite retained evidence")
    idle_gpu()
    options = dict(
        runner=args.runner,
        max_context=2048,
        max_concurrency=args.max_concurrency,
        prefill_chunk=512,
        timeout=600,
    )
    if args.cpu_wait is not None:
        options["cpu_wait"] = args.cpu_wait
    prompts = [
        f"Document {i}. "
        + "The quick brown fox jumps over the lazy dog. " * 50
        + "\nContinue the explanation:"
        for i in range(4)
    ]
    generation = dict(max_new_tokens=64, raw_prompt=True, ignore_eos=True)
    with QwenNativeSession(args.checkpoint, **options) as model:
        ready = model.ready
        warm = model.generate(prompts[0], **generation)
        runs = [
            measure(model, lambda: model.generate(prompts[i % 4], **generation))
            for i in range(args.repeats)
        ]
        results = dict(single=dict(runs=runs, summary=summary(runs)))
        if args.max_concurrency > 1:
            model.set_mode("parallel")
            model.generate_many(prompts, **generation)
            batches = [
                measure(model, lambda: model.generate_many(prompts, **generation))
                for _ in range(args.repeats)
            ]
            results["parallel"] = dict(runs=batches, summary=summary(batches))
        # A changed prompt followed by replay must not leak state.
        again = model.generate(prompts[0], **generation)
        if again["generated_ids"] != warm["generated_ids"]:
            raise AssertionError("Serial state replay changed generated IDs")
    report = dict(
        ready=ready,
        runner_sha256=hashlib.sha256(Path(args.runner).read_bytes()).hexdigest(),
        prompts=prompts,
        generation=generation,
        results=results,
        process_exit=model._process.returncode,
        state_replay_equal=True,
        method=__doc__,
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: v["summary"] for k, v in results.items()}, indent=2), flush=True
    )


if __name__ == "__main__":
    main()
