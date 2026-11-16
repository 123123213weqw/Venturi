"""Explicit native Qwen replica replay, not an HTTP or a decode-only benchmark.

Requires separately admitted idle GPUs, prebuilt libraries, a local Q8 checkpoint
and its tokenizer. Does not download, compile, pick devices or change services.
The standard-library pool itself has no Qwen/Torch dependency.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import time


def percentile(values, quantile):
    ranked = sorted(values)
    return ranked[round((len(ranked)-1)*quantile)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--ops-library", default=os.environ.get("VENTURI_OPS_LIBRARY"))
    p.add_argument("--qwen-library", default=os.environ.get("VENTURI_QWEN_LIBRARY"))
    p.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    p.add_argument("--concurrencies", nargs="+", type=int, default=[4, 8, 16, 32])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output-tokens", type=int, default=24)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if (not a.ops_library or not a.qwen_library or a.repeats < 3 or
            not 1 <= len(a.devices) <= 8 or len(set(a.devices)) != len(a.devices) or
            not all(1 <= n <= 256 for n in a.concurrencies) or
            not 1 <= a.output_tokens <= 64):
        p.error("Explicit libraries, distinct devices, bounded cohorts and >=3 repeats required")
    out = Path(a.output)
    if out.exists():
        raise FileExistsError("Never overwrite a frozen result")
    # Deliberately lazy: --help does not import a tensor framework or touch GPUs.
    import torch
    from tokenizers import Tokenizer
    from venturi.models import NativeQwenModel, QwenWeights
    from venturi.models.qwen.native_runtime import QwenRuntime
    from venturi.ops import CudaBackend
    from venturi.runtime import ContinuousReplica, ContinuousReplicaPool, ContinuousScheduler

    def digest(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(16*2**20), b""):
                h.update(b)
        return h.hexdigest()

    identity = digest(a.checkpoint)+":"+digest(a.tokenizer)+":q8-fp16-sealed"
    tok = Tokenizer.from_file(a.tokenizer)
    texts = [
        "The capital of France is", "Explain why the sky is blue in one sentence.",
        "Write a Python function that adds two integers.", "用一句话解释什么是连续合批。",
        "What is 17 plus 25?", "List three ways to save electricity at home.",
        "Translate good morning into Chinese.", "A triangle has how many sides?",
    ]
    cases = []
    for i in range(max(a.concurrencies)):
        # Distinct, heterogeneous real text. No random token IDs or shared KV.
        text = f"Question {i+1}: "+texts[i % len(texts)]
        text += " Please give a clear and concise answer."*(i % 4)
        prompt = "<|im_start|>user\n"+text+"<|im_end|>\n<|im_start|>assistant\n"
        cases.append(dict(text=text, input_ids=tok.encode(prompt).ids))
    capacity = max(len(c["input_ids"]) for c in cases)+a.output_tokens+1
    runtime = QwenRuntime(a.qwen_library, CudaBackend(a.ops_library))
    weights = QwenWeights.open(a.checkpoint)
    models, batches = [], []
    report = dict(identity=identity, checkpoint=str(Path(a.checkpoint).resolve()),
        devices=a.devices, precision="Q8/G32 weights, FP16 activation/scales, native fp16 projection profile",
        slots_per_replica=16, token_budget=32, prefill_chunk=16, capacity=capacity,
        output_budget=a.output_tokens, eos_policy="disabled for fixed-work replay",
        cases=cases, runs=[], correctness={}, torch=torch.__version__,
        scope="Two or more independent native model replicas, common-wall end-to-end aggregate output TPS; NOT decode-only TPS, HTTP, streaming, MTP, TP/PP/PD or prefix sharing")

    def save():
        temp = out.with_suffix(out.suffix+".tmp")
        temp.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
        temp.replace(out)

    def fence():
        for d in a.devices:
            torch.cuda.synchronize(d)

    def run(policy, group, expected):
        pool = ContinuousReplicaPool(tuple(ContinuousReplica(str(i), identity,
            ContinuousScheduler(b, token_budget=32, prefill_chunk_size=16))
            for i,b in enumerate(batches)), routing=policy, max_requests=256)
        pool.start()
        try:
            fence()
            cpu = time.process_time(); began = time.perf_counter(); unix = time.time()
            tickets = [pool.submit(c["input_ids"], max_new_tokens=a.output_tokens) for c in group]
            results = [t.future.result(timeout=300) for t in tickets]
            fence()
            wall = time.perf_counter()-began; cpu = time.process_time()-cpu
            # Exact arithmetic-trajectory gate, not a quality benchmark. Preserve
            # the first counterexample instead of relaxing the gate for speed.
            actual = [list(r.output_ids) for r in results]
            if actual != expected:
                report["counterexample"] = dict(policy=policy, actual=actual, expected=expected)
                save()
                raise AssertionError("Pooled greedy output differs from independent native B1")
            return dict(policy=policy, concurrency=len(group), started_unix=unix,
                ended_unix=time.time(), wall_seconds=wall, cpu_seconds=cpu,
                aggregate_e2e_output_tps=sum(len(r.output_ids) for r in results)/wall,
                ttft_p50=statistics.median(r.time_to_first_token_seconds for r in results),
                ttft_p95=percentile([r.time_to_first_token_seconds for r in results], .95),
                latency_p50=statistics.median(r.wall_seconds for r in results),
                latency_p95=percentile([r.wall_seconds for r in results], .95),
                max_inter_token_seconds=max(r.max_inter_token_seconds for r in results),
                max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                assignments=[t.replica for t in tickets], completions=[asdict(r) for r in results],
                scheduling=pool.info())
        finally:
            pool.close(timeout=180)

    try:
        load_began = time.perf_counter()
        for device in a.devices:
            models.append(NativeQwenModel.from_q8(weights, runtime, weight_mode="resident",
                dtype=torch.float16, device=device, max_tokens=32, reserve_bytes=2**30,
                projection_profile="fp16", metadata_mode="sealed"))
            batches.append(models[-1].create_batch([capacity]*16))
        fence()
        report["load_and_prepare_seconds"] = time.perf_counter()-load_began
        expected = []
        ref = models[0].create_session(capacity)
        try:
            for c in cases:
                ref.reset()
                expected.append(ref.generate(c["input_ids"], max_new_tokens=a.output_tokens))
        finally:
            ref.close()
        report["reference_outputs"] = [dict(ids=ids, text=tok.decode(ids)) for ids in expected]
        # Qualify each device's B1 trajectory before timing the routing policies.
        for i,model in enumerate(models):
            ref = model.create_session(capacity)
            try:
                for c,want in zip(cases,expected):
                    ref.reset()
                    assert ref.generate(c["input_ids"], max_new_tokens=a.output_tokens) == want, i
            finally:
                ref.close()
        report["correctness"]["independent_devices_exact"] = True
        save()
        for count in a.concurrencies:
            for repeat in range(a.repeats):
                policies = ["round-robin", "least-work"]
                if repeat % 2:
                    policies.reverse()
                for policy in policies:
                    # Startup, warmup, teardown and correctness checks are not
                    # in the common-wall timed scope. No profiler is enabled.
                    run(policy, cases[:count], expected[:count])
                    row = run(policy, cases[:count], expected[:count])
                    row["repeat"] = repeat
                    report["runs"].append(row); save()
                    print(json.dumps({k:row[k] for k in ("policy","concurrency","repeat","wall_seconds","aggregate_e2e_output_tps","ttft_p50")}), flush=True)
        report["correctness"]["all_pooled_outputs_exact"] = True
        report["complete"] = True
        save()
    finally:
        # run() must have joined every owner before freeing borrowed models.
        # If an owner times out, process-level guard is the final cleanup owner.
        if any(t.name == "venturi-continuous" and t.is_alive()
               for t in __import__("threading").enumerate()):
            raise RuntimeError("Retain native buffers: inference owner still live")
        for b in batches:
            b.close()
        for m in models:
            m.close()


if __name__ == "__main__":
    main()
