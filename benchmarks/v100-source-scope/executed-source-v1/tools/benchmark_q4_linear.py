"""Synthetic operator measurements, not model throughput or a llama.cpp comparison."""

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile
import time

import torch

from venturi.ops import AdaBackend, Q4RowSplitWeight


def measure(call, iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def benchmark(backend, n, k, tokens, repeats, iterations, device):
    generator = torch.Generator().manual_seed(92 + n + k)
    integers = torch.randint(-8, 8, (n, k), dtype=torch.int8, generator=generator)
    scales = (torch.rand((n, (k + 63) // 64), generator=generator) * 0.2 + 0.001).half()
    weight = Q4RowSplitWeight.from_codes(integers, scales)
    x = torch.randn((tokens, k), device=device, dtype=torch.bfloat16)
    plans = {
        name: backend.prepare_linear(weight, tokens=tokens, device=device, route=name)
        for name in ("auto", "generic")
    }
    outputs = {name: plan.empty_output() for name, plan in plans.items()}
    # Exact integer/scales oracle is independent of the packed decoder. Full-shape
    # correctness is exercised in tests; timing checks every token of 16 rows.
    reference = (
        x.cpu().double()
        @ (
            integers[:16].double()
            * scales[:16].double().repeat_interleave(64, 1)[:, :k]
        ).T
    )
    records = {}
    graphs = {}
    for name, plan in plans.items():
        for _ in range(5):
            plan.run(x, out=outputs[name])
        actual = outputs[name][:, :16].cpu().double()
        error = (actual - reference.to(torch.bfloat16).double()).abs()
        if not bool(
            (
                error <= 2 * torch.finfo(torch.bfloat16).eps * reference.abs() + 2e-4
            ).all()
        ):
            raise RuntimeError(f"Numerical check failed for {name}")
        # Capture many launches to amortize graph launch overhead and isolate GPU work.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(iterations):
                plan.run(x, out=outputs[name])
        graphs[name] = graph
        records[name] = {
            "route": plan.route,
            "eager_ms": [],
            "graph_kernel_ms": [],
            "workspace_bytes": plan.workspace_bytes,
            "max_sample_error_vs_rounded_fp64": float(error.max()),
        }
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    before = torch.cuda.memory_allocated(device)
    for repeat in range(repeats):
        order = list(plans) if repeat % 2 == 0 else list(reversed(plans))
        for name in order:
            plan = plans[name]
            records[name]["eager_ms"].append(
                measure(lambda: plan.run(x, out=outputs[name]), iterations)
            )
            records[name]["graph_kernel_ms"].append(
                measure(graphs[name].replay, 5) / iterations
            )
    after = torch.cuda.memory_allocated(device)
    peak = torch.cuda.max_memory_allocated(device)
    if (before, before) != (after, peak):
        raise RuntimeError("Execution allocated PyTorch device memory")
    for record in records.values():
        record["median_eager_ms"] = statistics.median(record["eager_ms"])
        record["median_graph_kernel_ms"] = statistics.median(record["graph_kernel_ms"])
    return {
        "shape": {"N": n, "K": k, "T": tokens},
        "activation_dtype": "bfloat16",
        "physical_weight_bytes": weight.nbytes,
        "routes": records,
        "execution_allocated_bytes_delta": after - before,
        "execution_peak_extra_bytes": peak - before,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if args.repeats <= 0 or args.iterations <= 0:
        parser.error("repeats and iterations must be positive")
    device = torch.device(args.device)
    with torch.cuda.device(device):
        torch.manual_seed(17)
        backend = AdaBackend(args.library)
        build_record = backend.library.with_name("build.json")
        report = {
            "scope": "Synthetic Q4 Linear microbenchmark, not model inference",
            "timestamp_unix": time.time(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cpu_threads": torch.get_num_threads(),
            "gpu": torch.cuda.get_device_name(device),
            "capability": torch.cuda.get_device_capability(device),
            "build": json.loads(build_record.read_text())
            if build_record.exists()
            else None,
            "repeats": args.repeats,
            "iterations": args.iterations,
            "notes": [
                "Random signed Q4 codes and FP16 scales, not pretrained weights.",
                "Alternating auto/generic run order; five warm-up calls per route.",
                "Eager event times include Python launch gaps; graph times amortize graph launch.",
                "Resident weights; no disk I/O, quantizer, full-weight decode, or model scheduler.",
                "Allocator figures cover PyTorch only, not driver/context allocations.",
            ],
            "results": [],
        }
        telemetry = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,temperature.gpu,power.limit,clocks.sm,memory.used",
                "--format=csv",
            ],
            text=True,
            capture_output=True,
        )
        report["gpu_telemetry_before"] = telemetry.stdout
        for n, k in [(4096, 4096), (5120, 5120), (5120, 25600)]:
            for tokens in (1, 8, 16, 128):
                print(f"Benchmark N={n} K={k} T={tokens}", flush=True)
                report["results"].append(
                    benchmark(
                        backend, n, k, tokens, args.repeats, args.iterations, device
                    )
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=args.output.parent, delete=False
    ) as file:
        json.dump(report, file, indent=2)
        file.write("\n")
        temporary = file.name
    os.replace(temporary, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
