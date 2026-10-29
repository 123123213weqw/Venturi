"""Remote-only qualification and matched benchmarks for the independent MLP block.

Synthetic Llama/Qwen-sized weights, not checkpoint inference or model TPS.
Dense compares the same stored BF16 weights. Q4 compares packed execution with
an explicitly materialized FP32 PyTorch reference, not another Q4 kernel.
FP64 qualification is independent of both execution backends. No model downloads.
"""

import argparse
from importlib.util import module_from_spec, spec_from_file_location
from functools import partial
import hashlib
import json
from pathlib import Path
import resource
import statistics
import subprocess
from time import perf_counter, time

import torch

from gpu_telemetry import FIELDS, telemetry
from gpu_telemetry import idle_gpu
from venturi.models.gated_mlp import GatedMLPWeights
from venturi.ops import (
    AdaBackend,
    CudaBackend,
    DenseWeight,
    Q4RowSplitWeight,
    TorchBackend,
)


def make_weights(h, intermediate, q4):
    generator = torch.Generator().manual_seed(812 + h + intermediate)

    def weight(n, k):
        if q4:
            codes = torch.randint(-8, 8, (n, k), generator=generator, dtype=torch.int8)
            scales = torch.full((n, (k + 63) // 64), 0.2 / k**0.5, dtype=torch.float16)
            return Q4RowSplitWeight.from_codes(codes, scales)
        return DenseWeight((torch.randn(n, k, generator=generator) / k**0.5).bfloat16())

    return GatedMLPWeights(
        torch.ones(h, dtype=torch.bfloat16),
        weight(intermediate, h),
        weight(intermediate, h),
        weight(h, intermediate),
    )


def cpu_seconds():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def phase(call, seconds):
    durations = []
    calls = 0
    gpu_ms = 0
    start = time()
    cpu_start = cpu_seconds()
    wall_start = perf_counter()
    while perf_counter() - wall_start < seconds:
        a, b = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        a.record()
        for _ in range(100):
            call()
        b.record()
        b.synchronize()
        measured = a.elapsed_time(b)
        durations.append(measured / 100)
        gpu_ms += measured
        calls += 100
    wall = perf_counter() - wall_start
    cpu = cpu_seconds() - cpu_start
    return dict(
        mode="single",
        start_unix=start,
        end_unix=time(),
        calls=calls,
        wall_seconds=wall,
        process_cpu_seconds=cpu,
        process_cpu_percent=100 * cpu / wall,
        wall_us_per_call=wall / calls * 1e6,
        event_us_per_call=gpu_ms / calls * 1000,
        median_chunk_event_us_per_call=statistics.median(durations) * 1000,
        chunk_event_ms_per_call=durations,
    )


def qualify(actual, expected):
    error = (actual.cpu().double() - expected.double()).abs()
    scale = max(expected.double().square().mean().sqrt().item(), 1e-12)
    maximum = error.max().item() / scale
    relative_rms = error.square().mean().sqrt().item() / scale
    eps = torch.finfo(torch.bfloat16).eps
    return dict(
        max_abs_error=error.max().item(),
        normalized_max_error=maximum,
        relative_rms_error=relative_rms,
        diagnostic_budget_met=maximum <= 2 * eps and relative_rms <= eps / 2,
        scope="Composite diagnostic only, not a correctness oracle for intermediate FP32 rounding. The strict diagnostic budget can fail even on the PyTorch reference; every individual Op is independently qualified on its actual inputs.",
    )


def qualify_stages(plan, weights, x, oracle):
    """Do not blame a late rounding-boundary difference on an earlier kernel.

    Each Op is checked against independent FP64 math using its own actual
    represented inputs; the composite FP64 comparison is recorded separately.
    """
    selected = x.clone()
    h, i = plan.hidden_size, plan.intermediate_size
    norm = torch.empty((plan.tokens, h), device=x.device, dtype=x.dtype)
    gate = torch.empty((plan.tokens, i), device=x.device, dtype=x.dtype)
    up = torch.empty_like(gate)
    down = torch.empty_like(norm)
    out = torch.empty_like(norm)
    scratch = (
        torch.empty(plan.linear_workspace_bytes, device=x.device, dtype=torch.uint8)
        if plan.linear_workspace_bytes
        else None
    )
    stages = {}

    def check_stage(name, actual, reference, atol=2e-5):
        expected = reference.to(actual.dtype)
        error = (actual.cpu().double() - expected.double()).abs()
        bound = 2 * torch.finfo(actual.dtype).eps * reference.abs() + atol
        failed = (error > bound).nonzero()[:8]
        stages[name] = dict(
            max_abs_error=error.max().item(),
            passed=bool((error <= bound).all()),
            criterion=f"abs(error) <= 2*eps*abs(FP64 reference) + {atol}",
            first_failures=[
                dict(
                    token=int(index[0]),
                    channel=int(index[1]),
                    actual=float(actual.cpu()[tuple(index)]),
                    reference=float(reference[tuple(index)]),
                    rounded_reference=float(expected[tuple(index)]),
                    error=float(error[tuple(index)]),
                    bound=float(bound[tuple(index)]),
                )
                for index in failed
            ],
        )

    plan.norm.run(selected, out=norm)
    source = selected.cpu().double()
    check_stage(
        "rmsnorm",
        norm,
        source
        * (source.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
        * weights.norm.double(),
    )
    for name, linear, weight, target in [
        ("gate", plan.gate, weights.gate, gate),
        ("up", plan.up, weights.up, up),
    ]:
        linear.run(
            norm, out=target, workspace=scratch if linear.workspace_bytes else None
        )
        check_stage(
            name,
            target,
            norm.cpu().double() @ oracle.decode(weight).T,
            atol=2e-4 if isinstance(weight, Q4RowSplitWeight) else 2e-5,
        )
    reference = (
        gate.cpu().double() * torch.sigmoid(gate.cpu().double()) * up.cpu().double()
    )
    plan.swiglu.run(gate, up, out=gate)
    check_stage("swiglu", gate, reference)
    plan.down.run(
        gate, out=down, workspace=scratch if plan.down.workspace_bytes else None
    )
    check_stage(
        "down",
        down,
        gate.cpu().double() @ oracle.decode(weights.down).T,
        atol=2e-4 if isinstance(weights.down, Q4RowSplitWeight) else 2e-5,
    )
    plan.add.run(selected, down, out=out)
    check_stage("residual", out, selected.cpu().double() + down.cpu().double())
    return stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--q4-library", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--seconds-per-phase", type=float, default=2.0)
    args = parser.parse_args()
    if not 1.5 <= args.seconds_per_phase <= 10:
        raise ValueError("Require 1.5..10 seconds per phase")
    out = Path(args.output_directory)
    out.mkdir(parents=True, exist_ok=False)
    background = idle_gpu()
    torch.set_num_threads(4)
    torch.manual_seed(20261018)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    spec = spec_from_file_location(
        "independent_mlp_reference",
        Path(__file__).resolve().parents[1] / "tests/block_reference.py",
    )
    oracle = module_from_spec(spec)
    spec.loader.exec_module(oracle)
    native = CudaBackend(args.library, q4_backend=AdaBackend(args.q4_library))
    reports = []
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
            for name, h, intermediate in [
                ("llama_style", 4096, 14336),
                ("qwen_style", 5120, 17408),
            ]:
                for q4 in (False, True):
                    weights = make_weights(h, intermediate, q4)
                    for tokens in (1, 128):
                        key = f"{name}-" + ("q4" if q4 else "bf16") + f"-t{tokens}"
                        x = (torch.randn(tokens, h, device="cuda") * 0.5).bfloat16()
                        indices = sorted(set([0, tokens // 2, tokens - 1]))
                        expected = oracle.gated_mlp(weights, x[indices], 1e-6)
                        row = dict(
                            name=key,
                            hidden=h,
                            intermediate=intermediate,
                            tokens=tokens,
                            format="q4_g64_fp16" if q4 else "bf16",
                            sampled_token_rows=indices,
                            qualification={},
                            phases={},
                        )
                        for label, backend in [
                            ("torch", TorchBackend()),
                            ("native", native),
                        ]:
                            plan = weights.prepare(
                                backend,
                                tokens=tokens,
                                dtype=torch.bfloat16,
                                device="cuda:0",
                            )
                            workspace = plan.empty_workspace()
                            output = torch.empty_like(x)

                            run = partial(plan.run, x, out=output, workspace=workspace)

                            for _ in range(3):
                                run()
                            torch.cuda.synchronize()
                            if not bool(torch.isfinite(output).all()):
                                raise AssertionError("Nonfinite block output")
                            row["qualification"][label] = qualify(
                                output[indices], expected
                            )
                            row["qualification"][label + "_stages"] = qualify_stages(
                                plan, weights, x, oracle
                            )
                            checks = row["qualification"][label + "_stages"]
                            if not all(v["passed"] for v in checks.values()):
                                failure = dict(
                                    row=row,
                                    input_seed=20261018,
                                    selected_inputs=x[indices].cpu().float().tolist(),
                                    actual=output[indices].cpu().float().tolist(),
                                    reference=expected.float().tolist(),
                                )
                                (out / (key + "-counterexample.json")).write_text(
                                    json.dumps(failure, indent=2) + "\n"
                                )
                                raise AssertionError(
                                    "Independent qualification failed; counterexample retained"
                                )
                            before_graph = output.cpu().clone()
                            if label == "torch":
                                baseline_output = before_graph
                            else:
                                row["native_vs_torch"] = dict(
                                    equal_elements=int(
                                        (before_graph == baseline_output).sum()
                                    ),
                                    total_elements=before_graph.numel(),
                                    max_abs_difference=(
                                        before_graph.float() - baseline_output.float()
                                    )
                                    .abs()
                                    .max()
                                    .item(),
                                    scope="Compatibility comparison only; independent FP64 stage qualification is separate.",
                                )
                            row[label + "_workspace_bytes"] = workspace.nbytes
                            row[label + "_routes"] = [
                                p.route for p in (plan.gate, plan.up, plan.down)
                            ]
                            row["phases"][label + "_eager"] = phase(
                                run, args.seconds_per_phase
                            )
                            stream = torch.cuda.Stream()
                            stream.wait_stream(torch.cuda.current_stream())
                            with torch.cuda.stream(stream):
                                run()
                                graph = torch.cuda.CUDAGraph()
                                with torch.cuda.graph(graph, stream=stream):
                                    run()
                            stream.synchronize()
                            row["phases"][label + "_graph"] = phase(
                                graph.replay, args.seconds_per_phase
                            )
                            torch.cuda.synchronize()
                            row["qualification"][label + "_graph"] = qualify(
                                output[indices], expected
                            )
                            row[label + "_graph_replay_equal"] = torch.equal(
                                before_graph, output.cpu()
                            )
                            del run, graph, stream, plan, workspace, output
                        reports.append(row)
                        (out / (key + ".json")).write_text(
                            json.dumps(row, indent=2) + "\n"
                        )
                        print(
                            key,
                            {
                                k: round(v["wall_us_per_call"], 2)
                                for k, v in row["phases"].items()
                            },
                            flush=True,
                        )
                    del weights
        finally:
            sampler.terminate()
            try:
                sampler.wait(timeout=5)
            except subprocess.TimeoutExpired:
                sampler.kill()
                sampler.wait()
    if sampler.returncode not in (0, -15):
        raise RuntimeError("Telemetry failed")
    for row in reports:
        for value in row["phases"].values():
            value["gpu"] = telemetry(out / "telemetry.csv", [value])["single"]
    report = dict(
        method=__doc__,
        device=torch.cuda.get_device_name(),
        background=background,
        cuda=torch.version.cuda,
        torch_version=torch.__version__,
        libraries={
            name: dict(
                path=path, sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()
            )
            for name, path in [("blocks", args.library), ("q4", args.q4_library)]
        },
        cases=reports,
        limitations=[
            "Synthetic sublayer benchmark, not full-model generation or an upstream-engine comparison.",
            "FP64 qualifies every individual Op output on actual represented inputs. Composite diagnostics cover T1 and three token rows at T128, but are not used as a substitute for stage qualification; all outputs scanned finite.",
            "Q4 composition uses the pre-existing attributed Ada operator, not a newly invented Q4 GEMM.",
            "CPU time includes this Python process and its native library work; 100% is one core.",
            "CUDA event elapsed includes gaps between launches; not profiler-summed kernel time.",
            "GPU data are 200ms whole-device samples; memory-controller busy is not bandwidth utilization.",
            "Achieved occupancy/compute/DRAM/cache/stall counters remain unavailable under the existing driver policy.",
            "Torch reference allocates intermediates; native caller buffers and 4MiB dense cuBLAS workspace are explicit.",
        ],
    )
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
