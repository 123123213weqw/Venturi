"""Local-checkpoint CPU/CUDA decode trace; profiling timings are not benchmarks."""

import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from venturi import load_checkpoint
from venturi.generation import LocalTokenizer, _greedy_token
from venturi.models import GPTNeoXModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--execution", choices=("eager", "cuda-graph"), default="eager")
    parser.add_argument("--nsys", action="store_true", help="Bracket decode with cudaProfilerStart/Stop instead of Kineto")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    loaded = load_checkpoint(args.checkpoint, require_float_dtype="F16")
    model = GPTNeoXModel(loaded)
    tokenizer = LocalTokenizer(
        Path(args.checkpoint) / "tokenizer.json", vocab_size=model.config.vocab_size
    )
    ids = tokenizer.encode("Once upon a time")
    session = (
        model.create_cuda_graph_session(
            len(ids) + args.steps + 1, decode_start=len(ids), decode_steps=args.steps
        ) if args.execution == "cuda-graph" else model.create_session(len(ids) + args.steps + 1)
    )
    select = getattr(session, "select_token", _greedy_token)
    for _ in range(3):
        session.reset()
        token = select(session.forward(ids))
        for _ in range(args.steps):
            token = select(session.forward([token]))
    session.reset()
    token = select(session.forward(ids))
    torch.cuda.synchronize()
    if args.nsys:
        torch.cuda.cudart().cudaProfilerStart()
        for _ in range(args.steps):
            token = select(session.forward([token]))
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        print("Nsight decode-only capture completed")
        return
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as profile:
        for _ in range(args.steps):
            with torch.profiler.record_function("venturi.decode_step"):
                token = select(session.forward([token]))
        torch.cuda.synchronize()
    profile.export_chrome_trace(str(out / "trace.json"))
    averages = profile.key_averages()
    (out / "cpu.txt").write_text(averages.table(sort_by="self_cpu_time_total", row_limit=35))
    (out / "cuda.txt").write_text(averages.table(sort_by="self_device_time_total", row_limit=35))
    trace = json.loads((out / "trace.json").read_text())
    kernels = [e for e in trace["traceEvents"] if e.get("cat") == "kernel"]
    runtime = [e for e in trace["traceEvents"] if e.get("cat") == "cuda_runtime"]
    report = {
        "steps": args.steps,
        "execution": args.execution,
        "cuda_trace_available": bool(kernels),
        "kernel_count": len(kernels) if kernels else None,
        "kernels_per_decode_step": len(kernels) / args.steps if kernels else None,
        "summed_kernel_seconds": sum(e.get("dur", 0) for e in kernels) / 1e6 if kernels else None,
        "cuda_runtime_calls": dict(Counter(e["name"] for e in runtime)),
        "operators": [{
            "name": e.key, "calls": e.count,
            "self_cpu_seconds": e.self_cpu_time_total / 1e6,
            "self_device_seconds": e.self_device_time_total / 1e6,
        } for e in sorted(averages, key=lambda x: x.self_cpu_time_total, reverse=True)[:40]],
        "scope": "16-step default decode-only instrumented trace after warmup; loading and prefill excluded. Profiler overhead affects CPU/GPU timings; use unprofiled paired benchmarks for throughput. Summed kernel time is not wall time or an additive bottleneck attribution.",
    }
    (out / "profile.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "operators"}, indent=2))


if __name__ == "__main__":
    main()
