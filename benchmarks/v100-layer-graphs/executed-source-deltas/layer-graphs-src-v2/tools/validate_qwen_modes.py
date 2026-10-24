"""Finite production-API single/parallel acceptance and wall-clock comparison.

Run on the selected idle CUDA server with an existing checkpoint and protocol-2
runner. No downloads, precision fallback, services or external source changes.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

from run_qwen_validation import idle_gpu
from venturi.backends.qwen import QwenNativeSession


def ids(batch):
    return [item["generated_ids"] for item in batch["results"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--runner", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--kv-dtype", choices=("bf16", "k8v4"), default="bf16")
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--kv-capacity", type=int)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite retained evidence")
    idle_gpu()
    prompts = [
        f"Document {i}. "
        + "The quick brown fox jumps over the lazy dog. " * 50
        + "\nContinue the explanation:"
        for i in range(8)
    ]
    runs = []
    with QwenNativeSession(
        args.checkpoint,
        runner=args.runner,
        max_context=2048,
        max_concurrency=args.max_concurrency,
        kv_capacity=args.kv_capacity,
        prefill_chunk=512,
        kv_dtype=args.kv_dtype,
        timeout=600,
    ) as model:
        ready = model.ready
        pid = model._process.pid
        generation = dict(raw_prompt=True, max_new_tokens=64, ignore_eos=True)
        # Warm both native paths before timing; shared weights stay resident.
        reference = model.generate_many(prompts, **generation)
        model.set_mode("parallel")
        warm_parallel = model.generate_many(prompts, **generation)
        comparisons = [
            dict(
                stage="parallel_warmup",
                equal_by_request=[
                    a == b for a, b in zip(ids(reference), ids(warm_parallel))
                ],
            )
        ]
        for mode in ("single", "parallel", "parallel", "single"):
            model.set_mode(mode)
            run = model.generate_many(prompts, **generation)
            comparisons.append(
                dict(
                    stage=f"measured_{len(runs)}_{mode}",
                    equal_by_request=[a == b for a, b in zip(ids(reference), ids(run))],
                )
            )
            if mode == "single" and ids(reference) != ids(run):
                raise AssertionError("Serial reset changed outputs after a mode switch")
            if model._process.pid != pid:
                raise AssertionError("Mode switch restarted the model process")
            runs.append(run)
            print(
                f"{mode}: {run['output_tokens_per_second']:.3f} total output tokens/s; "
                f"actual decode batch {run['runtime']['mean_decode_batch']:.3f}",
                flush=True,
            )
        # Uneven EOS termination and order, then another serial reset check.
        chat = ["Reply only YES.", "What is 2+2?", "Explain gravity in one sentence."]
        serial_chat = model.generate_many(chat, mode="single", max_new_tokens=32)
        parallel_chat = model.generate_many(chat, mode="parallel", max_new_tokens=32)
        comparisons.append(
            dict(
                stage="chat",
                equal_by_request=[
                    a == b for a, b in zip(ids(serial_chat), ids(parallel_chat))
                ],
            )
        )
        # A list containing one row must remain the exact single-row native route.
        one_parallel = model.generate_many([prompts[0]], mode="parallel", **generation)
        if ids(one_parallel)[0] != ids(reference)[0]:
            raise AssertionError(
                "One-row parallel dispatch changed the single-row route"
            )
        isolated = model.generate_many(
            [
                "Reply with exactly the word APPLE and nothing else.",
                "Reply with exactly the word RIVER and nothing else.",
                "Reply with exactly the word CLOUD and nothing else.",
                "Reply with exactly the word MUSIC and nothing else.",
            ],
            mode="parallel",
            max_new_tokens=16,
        )
        literal_answers = [r["text"].strip() for r in isolated["results"]]
        if literal_answers != ["APPLE", "RIVER", "CLOUD", "MUSIC"]:
            raise AssertionError(
                f"Parallel literal-answer oracle failed: {literal_answers}"
            )
        model.set_mode("single")
        reset = model.generate(prompts[0], **generation)
        if reset["generated_ids"] != reference["results"][0]["generated_ids"]:
            raise AssertionError("Recurrent/KV state leaked across requests")
    summaries = {}
    for mode in ("single", "parallel"):
        selected = [r for r in runs if r["mode"] == mode]
        summaries[mode] = dict(
            output_tokens_per_second=sum(
                len(x["generated_ids"]) for r in selected for x in r["results"]
            )
            / sum(r["wall_seconds"] for r in selected),
            mean_decode_batch=sum(r["runtime"]["decode_row_rounds"] for r in selected)
            / sum(r["runtime"]["decode_rounds"] for r in selected),
            median_first_token_seconds=statistics.median(
                x["timings"]["first_token_seconds"]
                for r in selected
                for x in r["results"]
            ),
        )
    report = dict(
        ready=ready,
        runner_sha256=hashlib.sha256(Path(args.runner).read_bytes()).hexdigest(),
        prompts=prompts,
        generation=generation,
        warmup_serial=reference,
        warmup_parallel=warm_parallel,
        runs=runs,
        summaries=summaries,
        serial_chat=serial_chat,
        parallel_chat=parallel_chat,
        comparisons=comparisons,
        literal_answer_oracle=isolated,
        greedy_ids_equal=all(all(c["equal_by_request"]) for c in comparisons),
        same_resident_pid=pid,
        process_exit=model._process.returncode,
        scope="Eight prompts per wave; one resident process, two measured waves per mode in ABBA order. 64 fixed output tokens/request, all output tokens divided by elapsed batch wall time including preparation/prefill/queueing, startup/warmup excluded. No cross-precision or universal numerical equivalence claim. Native Qwen scheduler/kernels are external NInfer/Cinference.",
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
