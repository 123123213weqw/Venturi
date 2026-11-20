"""Remote-only resident-shape preparation and independent attention qualification.

Synthetic Qwen/Llama-sized operators, not pretrained model generation or TPS.
Resident storage comparison is identical BF16/Q4 data and three token shapes.
Attention is a correctness-first dense-prefix online-softmax baseline, not a
FlashAttention competitor. Timed phases are unprofiled; telemetry is coarse.
"""

import argparse
import gc
from functools import partial
from importlib.util import module_from_spec, spec_from_file_location
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
from time import perf_counter, time

import torch

from gpu_telemetry import FIELDS, telemetry
from benchmark_shared_mlp import cpu_seconds, make_weights, phase
from gpu_telemetry import idle_gpu
from venturi.ops import AdaBackend, CudaBackend, TorchBackend


def storage_report(plans):
    tensors = [t for p in plans for t in p.storage_tensors]
    unique = {t.data_ptr(): t for t in tensors}
    return dict(
        unique_weight_allocations=len(unique),
        unique_weight_bytes=sum(t.numel() * t.element_size() for t in unique.values()),
        logical_plan_weight_bytes=sum(t.numel() * t.element_size() for t in tensors),
        all_shape_pointers_identical=all(
            tuple(t.data_ptr() for t in p.storage_tensors)
            == tuple(t.data_ptr() for t in plans[0].storage_tensors)
            for p in plans
        ),
    )


def build_shapes(weights, backend, shared):
    resident = (
        weights.upload(backend, dtype=torch.bfloat16, device="cuda:0")
        if shared
        else None
    )
    plans = [
        resident.prepare(tokens=t)
        if shared
        else weights.prepare(backend, tokens=t, dtype=torch.bfloat16, device="cuda:0")
        for t in (1, 17, 128)
    ]
    return resident, plans


def release(plans):
    for plan in plans:
        plan.close()
    plans.clear()
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def run_shape_sequence(plans, inputs, targets, workspaces):
    for plan, x, target, workspace in zip(plans, inputs, targets, workspaces):
        plan.run(x, out=target, workspace=workspace)


def check(actual, reference):
    error = (actual.cpu().double() - reference.double()).abs()
    bound = 2 * torch.finfo(actual.dtype).eps * reference.double().abs() + 2e-5
    return dict(
        passed=bool((error <= bound).all()),
        max_abs_error=error.max().item(),
        failed_elements=int((~(error <= bound)).sum()),
        criterion="abs(actual-rounded_FP64) <= 2*eps*abs(rounded_FP64) + 2e-5",
    )


def time_paths(calls, seconds):
    reports = []
    # ABBA reduces simple ordering bias; this is still a microbenchmark.
    for backend in ("reference", "native", "native", "reference"):
        measured = phase(calls[backend], seconds)
        measured["backend"] = backend
        reports.append(measured)
    return dict(
        phases=reports,
        medians={
            name: {
                key: statistics.median(p[key] for p in reports if p["backend"] == name)
                for key in (
                    "wall_us_per_call",
                    "event_us_per_call",
                    "process_cpu_percent",
                )
            }
            for name in ("reference", "native")
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--q4-library", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--seconds-per-phase", type=float, default=2.0)
    args = parser.parse_args()
    if not 1.5 <= args.seconds_per_phase <= 10:
        raise ValueError("seconds-per-phase must be in [1.5,10]")
    out = Path(args.output_directory)
    out.mkdir(parents=True, exist_ok=False)
    background = idle_gpu()
    torch.set_num_threads(4)
    torch.manual_seed(20261018)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    spec = spec_from_file_location(
        "independent_attention_reference",
        Path(__file__).resolve().parents[1] / "tests/attention_reference.py",
    )
    oracle = module_from_spec(spec)
    spec.loader.exec_module(oracle)
    native = CudaBackend(args.library, q4_backend=AdaBackend(args.q4_library))
    reference = TorchBackend()
    results = dict(
        scope="Synthetic operator qualification only. No model TPS or full-model acceleration claim.",
        background=background,
        residency=[],
        attention=[],
        rope=[],
        hardware=dict(
            device=torch.cuda.get_device_name(),
            capability=torch.cuda.get_device_capability(),
        ),
        libraries={
            str(Path(p)): hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in (args.library, args.q4_library)
        },
        counters="Unavailable: unchanged RmProfilingAdminOnly=1 policy (prior ERR_NVGPUCTRPERM). No achieved occupancy, DRAM bandwidth, cache/stall counters measured.",
        timing="Unprofiled ABBA, 3 warmups, CUDA graph replay, 100 calls/chunk, >=seconds-per-phase per phase. Preparation is separately wall/process-CPU timed; GPU allocator bytes are live tensors, not whole-device NVML VRAM.",
    )
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
        try:
            for q4 in (False, True):
                weights = make_weights(5120, 17408, q4)
                # Establish context/cuBLAS startup before preparation comparisons.
                resident, plans = build_shapes(weights, native, True)
                del resident
                release(plans)
                rows = []
                previous = None
                for shared in (False, True, True, False):
                    torch.cuda.reset_peak_memory_stats()
                    initial = torch.cuda.memory_allocated()
                    cpu, wall, stamp = cpu_seconds(), perf_counter(), time()
                    resident, plans = build_shapes(weights, native, shared)
                    torch.cuda.synchronize()
                    measured = dict(
                        shared=shared,
                        start_unix=stamp,
                        end_unix=time(),
                        wall_seconds=perf_counter() - wall,
                        process_cpu_seconds=cpu_seconds() - cpu,
                        allocated_delta_bytes=torch.cuda.memory_allocated() - initial,
                        peak_allocated_delta_bytes=torch.cuda.max_memory_allocated()
                        - initial,
                        **storage_report(plans),
                    )
                    # Same represented input must produce bit-identical results for
                    # independently owned and resident plans, not just close text.
                    outputs = []
                    for plan in plans:
                        x = torch.full(
                            (plan.tokens, 5120),
                            0.125,
                            device="cuda",
                            dtype=torch.bfloat16,
                        )
                        target, ws = torch.empty_like(x), plan.empty_workspace()
                        plan.run(x, out=target, workspace=ws)
                        outputs.append(target.cpu())
                        del x, target, ws
                    del plan
                    inputs = [
                        torch.full(
                            (p.tokens, 5120), 0.125, device="cuda", dtype=torch.bfloat16
                        )
                        for p in plans
                    ]
                    targets = [torch.empty_like(x) for x in inputs]
                    workspaces = [p.empty_workspace() for p in plans]

                    run_shapes = partial(
                        run_shape_sequence, plans, inputs, targets, workspaces
                    )

                    for _ in range(3):
                        run_shapes()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        run_shapes()
                    graph.replay()
                    torch.cuda.synchronize()
                    assert all(
                        torch.equal(a, b.cpu()) for a, b in zip(outputs, targets)
                    )
                    measured["execution"] = dict(
                        scope="One graph replay runs T=1,17,128 sequentially, same synthetic block/input; not model generation.",
                        eager_graph_bit_identical=True,
                        **phase(graph.replay, args.seconds_per_phase),
                    )
                    del graph, inputs, targets, workspaces, run_shapes
                    if previous is not None:
                        assert all(torch.equal(a, b) for a, b in zip(outputs, previous))
                    previous = outputs
                    rows.append(measured)
                    del resident
                    release(plans)
                report = dict(
                    format="q4_g64_fp16" if q4 else "bf16",
                    hidden=5120,
                    intermediate=17408,
                    token_shapes=[1, 17, 128],
                    trials=rows,
                    all_four_trials_outputs_bit_identical=True,
                )
                results["residency"].append(report)
                print("residency", report["format"], rows, flush=True)
                del weights, previous
                gc.collect()
                torch.cuda.empty_cache()

            for t, s, hq, hkv, d in (
                (1, 129, 24, 4, 256),
                (128, 128, 24, 4, 256),
                (128, 640, 24, 4, 256),
                (128, 128, 32, 8, 128),
            ):
                q = torch.randn(t, hq, d, device="cuda", dtype=torch.bfloat16)
                k = torch.randn(s, hkv, d, device="cuda", dtype=q.dtype)
                v = torch.randn_like(k)
                config = dict(
                    tokens=t,
                    kv_tokens=s,
                    query_heads=hq,
                    kv_heads=hkv,
                    head_dim=d,
                    dtype=q.dtype,
                    device="cuda:0",
                )
                ref = oracle.attention_reference(q, k, v, d**-0.5)
                calls, graphs, buffers, qualifications = {}, {}, {}, {}
                for name, backend in (("reference", reference), ("native", native)):
                    plan = backend.prepare_attention(**config)
                    target = torch.empty_like(q)
                    for _ in range(3):
                        plan.run(q, k, v, out=target)
                    torch.cuda.synchronize()
                    qualifications[name] = check(target, ref)
                    assert qualifications[name]["passed"], qualifications
                    eager = target.clone()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        plan.run(q, k, v, out=target)
                    graph.replay()
                    torch.cuda.synchronize()
                    assert torch.equal(eager, target)
                    calls[name], graphs[name], buffers[name] = (
                        graph.replay,
                        graph,
                        target,
                    )
                report = dict(
                    configuration={
                        k: str(v) if isinstance(v, torch.dtype) else v
                        for k, v in config.items()
                    },
                    qualification=qualifications,
                    eager_graph_bit_identical=True,
                    **time_paths(calls, args.seconds_per_phase),
                )
                results["attention"].append(report)
                print("attention", (t, s, hq, hkv, d), report["medians"], flush=True)
                del calls, graphs, buffers, graph, target, eager, q, k, v, ref
                gc.collect()
                torch.cuda.empty_cache()

            for t in (1, 128):
                x = torch.randn(t, 24, 256, device="cuda", dtype=torch.bfloat16)
                angle = torch.randn(t, 128, device="cuda")
                cos, sin = angle.cos(), angle.sin()
                ref = oracle.rope_reference(x, cos, sin, 256, "split_half")
                calls, graphs, buffers, qualifications = {}, {}, {}, {}
                for name, backend in (("reference", reference), ("native", native)):
                    plan = backend.prepare_rope(
                        tokens=t, heads=24, head_dim=256, dtype=x.dtype, device="cuda:0"
                    )
                    target = torch.empty_like(x)
                    for _ in range(3):
                        plan.run(x, cos, sin, out=target)
                    torch.cuda.synchronize()
                    qualifications[name] = check(target, ref)
                    assert qualifications[name]["passed"], qualifications
                    eager = target.clone()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        plan.run(x, cos, sin, out=target)
                    graph.replay()
                    torch.cuda.synchronize()
                    assert torch.equal(eager, target)
                    calls[name], graphs[name], buffers[name] = (
                        graph.replay,
                        graph,
                        target,
                    )
                report = dict(
                    tokens=t,
                    heads=24,
                    head_dim=256,
                    dtype=str(x.dtype),
                    qualification=qualifications,
                    eager_graph_bit_identical=True,
                    **time_paths(calls, args.seconds_per_phase),
                )
                results["rope"].append(report)
                print("rope", t, report["medians"], flush=True)
                del calls, graphs, buffers, graph, target, eager, x, cos, sin, ref
                gc.collect()
        finally:
            sampler.terminate()
            try:
                sampler.wait(timeout=5)
            except subprocess.TimeoutExpired:
                sampler.kill()
                sampler.wait()
        if sampler.returncode not in (0, -15):
            raise RuntimeError(f"GPU sampler failed: {sampler.returncode}")
    for report in results["residency"]:
        for trial in report["trials"]:
            measured = trial["execution"]
            measured["gpu"] = telemetry(out / "telemetry.csv", [measured])["single"]
    for kind in ("attention", "rope"):
        for report in results[kind]:
            for measured in report["phases"]:
                measured["gpu"] = telemetry(out / "telemetry.csv", [measured])["single"]
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
