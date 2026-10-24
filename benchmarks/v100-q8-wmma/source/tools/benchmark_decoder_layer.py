"""Remote-only two-layer decoder qualification and workspace/backend comparisons.

Synthetic standard GQA decoder layers, not the hybrid Qwen checkpoint, pretrained
quality evaluation, generation TPS, or a replacement for the external runner.
Compare identical native Ops with per-layer vs shared sequential workspaces, then
explicit PyTorch vs native using shared workspaces. All phases are unprofiled.
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
import sys

import torch

from benchmark_qwen_gpu import FIELDS, telemetry
from benchmark_shared_mlp import phase
from run_qwen_validation import idle_gpu
from venturi.models import AttentionDimensions, DecoderLayerWeights, GatedMLPWeights
from venturi.ops import (
    AdaBackend,
    CudaBackend,
    DenseWeight,
    Q4RowSplitWeight,
    TorchBackend,
)


def synthetic_weights(h, intermediate, dimensions, q4, seed):
    gen = torch.Generator().manual_seed(seed)

    def weight(n, k):
        if q4:
            codes = torch.randint(-8, 8, (n, k), generator=gen, dtype=torch.int8)
            scales = torch.full((n, (k + 63) // 64), 0.15 / k**0.5, dtype=torch.float16)
            return Q4RowSplitWeight.from_codes(codes, scales)
        return DenseWeight((torch.randn(n, k, generator=gen) / k**0.5).bfloat16())

    return DecoderLayerWeights(
        torch.ones(h, dtype=torch.bfloat16),
        weight(dimensions.query_width, h),
        weight(dimensions.kv_width, h),
        weight(dimensions.kv_width, h),
        weight(h, dimensions.query_width),
        GatedMLPWeights(
            torch.ones(h, dtype=torch.bfloat16),
            weight(intermediate, h),
            weight(intermediate, h),
            weight(h, intermediate),
        ),
    )


def tables(length, dims):
    positions = torch.arange(length, dtype=torch.float64)[:, None]
    frequency = 10000 ** (
        -torch.arange(dims.rotary_dim // 2, dtype=torch.float64)
        / (dims.rotary_dim // 2)
    )
    angles = positions * frequency
    return angles.cos().float().cuda(), angles.sin().float().cuda()


def metrics(phases):
    return {
        name: {
            key: statistics.median(p[key] for p in phases if p["condition"] == name)
            for key in ("wall_us_per_call", "event_us_per_call", "process_cpu_percent")
        }
        for name in dict.fromkeys(p["condition"] for p in phases)
    }


def replay_sequence(caches, graphs, prefix):
    for cache in caches:
        cache.truncate(prefix)
    for graph in graphs:
        graph.run()


def run_trial(plans, steps, caches, x, cos, sin, prefix, shared, seconds):
    unique_arenas = [plans[0].empty_workspace()]
    if not shared:
        unique_arenas.append(plans[1].empty_workspace())
    arenas = unique_arenas * 2 if shared else unique_arenas
    buffers = [torch.empty_like(x) for _ in plans]
    graphs = [
        step.capture(inp, cos, sin, out=target, workspace=arena, cache=cache)
        for step, inp, target, arena, cache in zip(
            steps, [x, buffers[0]], buffers, arenas, caches
        )
    ]

    call = partial(replay_sequence, caches, graphs, prefix)
    call()
    torch.cuda.synchronize()
    measured = phase(call, seconds)
    measured.update(
        workspace_allocations=len(unique_arenas),
        active_workspace_bytes=sum(a.nbytes for a in unique_arenas),
        single_lane_workspace_bytes=unique_arenas[0].nbytes,
        linear_scratch_bytes=plans[0].linear_workspace_bytes,
        full_sequence_calls_per_second=measured["calls"] / measured["wall_seconds"],
        output=buffers[-1].cpu(),
    )
    for cache in caches:
        cache.truncate(prefix)
    del graphs, buffers, arenas, unique_arenas, call
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return measured


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
    # Explicit independent test oracle, never an inference fallback.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    spec = spec_from_file_location(
        "independent_decoder_checks",
        Path(__file__).resolve().parents[1] / "tests/decoder_checks.py",
    )
    oracle = module_from_spec(spec)
    spec.loader.exec_module(oracle)
    native = CudaBackend(args.library, q4_backend=AdaBackend(args.q4_library))
    backends = dict(native=native, reference=TorchBackend())
    dimensions = AttentionDimensions(24, 4, 256)
    reports = []
    results = dict(
        scope=__doc__,
        background=background,
        hardware=dict(
            name=torch.cuda.get_device_name(),
            capability=torch.cuda.get_device_capability(),
        ),
        libraries={
            str(Path(p)): hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in (args.library, args.q4_library)
        },
        protocol="Two sequential synthetic layers, BF16 activations/KV. H5120/I17408/Hq24/Hkv4/D256. Three capture warmups, >=seconds-per-phase unprofiled phases, 100 wrapper calls/chunk; fixed-position replay truncates only the suffix before each call. CPU includes validation/dispatch/cache bookkeeping/event waiting. Workspace ABBA (separate/shared/shared/separate); backend ABBA (reference/native/native/reference). No sustained autoregressive rollout, checkpoint, batching or model TPS.",
        counters="Unavailable: unchanged RmProfilingAdminOnly=1 policy; prior ERR_NVGPUCTRPERM. No achieved occupancy, bandwidth, cache/stall counters or new kernel attribution.",
        cases=reports,
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
                weights = [
                    synthetic_weights(5120, 17408, dimensions, q4, 20261018 + i)
                    for i in range(2)
                ]
                residents = {
                    name: [w.upload(backend, dimensions=dimensions) for w in weights]
                    for name, backend in backends.items()
                }
                token_plans = {
                    name: {t: [r.prepare(tokens=t) for r in items] for t in (1, 128)}
                    for name, items in residents.items()
                }
                for name, plans in token_plans.items():
                    assert all(
                        tuple(t.data_ptr() for t in a.storage_tensors)
                        == tuple(t.data_ptr() for t in b.storage_tensors)
                        for a, b in zip(plans[1], plans[128])
                    )
                source = (torch.randn(129, 5120, device="cuda") * 0.25).bfloat16()
                cos, sin = tables(129, dimensions)
                for count, prefix in ((128, 0), (1, 128)):
                    key = ("q4" if q4 else "bf16") + f"-t{count}-p{prefix}"
                    print("qualifying", key, flush=True)
                    x = source[prefix : prefix + count]
                    c, s = cos[prefix : prefix + count], sin[prefix : prefix + count]
                    caches = {
                        name: [r.empty_cache(capacity=129) for r in items]
                        for name, items in residents.items()
                    }
                    # Decode caches contain the real preceding two-layer prefill,
                    # not zero/random prefix data. Each backend's actual prefix is
                    # independently observed by the qualification below.
                    if prefix:
                        for name in backends:
                            current = source[:prefix]
                            for plan, cache in zip(
                                token_plans[name][128], caches[name]
                            ):
                                target = torch.empty_like(current)
                                plan.at(position=0).run(
                                    current,
                                    cos[:prefix],
                                    sin[:prefix],
                                    out=target,
                                    workspace=plan.empty_workspace(),
                                    cache=cache,
                                )
                                current = target
                            del current, target
                    steps = {
                        name: [p.at(position=prefix) for p in token_plans[name][count]]
                        for name in backends
                    }
                    qualification = {}
                    expected = {}
                    for name in backends:
                        current = x
                        checks = []
                        for step, cache, weight in zip(
                            steps[name], caches[name], weights
                        ):
                            prior = (cache.keys, cache.values) if prefix else None
                            qualified, kv, stage_checks = (
                                oracle.qualified_decoder_composition(
                                    step, weight, current, c, s, prefix=prior
                                )
                            )
                            target = torch.empty_like(current)
                            step.run(
                                current,
                                c,
                                s,
                                out=target,
                                workspace=step.layer.empty_workspace(),
                                cache=cache,
                            )
                            torch.cuda.synchronize()
                            assert torch.equal(target, qualified)
                            assert torch.equal(cache.keys, kv[0]) and torch.equal(
                                cache.values, kv[1]
                            )
                            checks.append(stage_checks)
                            cache.truncate(prefix)
                            current = target
                        expected[name] = current.cpu()
                        qualification[name] = dict(
                            per_layer_checks=checks,
                            composition_and_cache_bit_identical=True,
                            independently_qualified_ops=30,
                        )
                    report = dict(
                        name=key,
                        stored_format="q4_g64_fp16" if q4 else "bf16",
                        tokens=count,
                        prefix=prefix,
                        layers=2,
                        qualification=qualification,
                        weight_pointers_shared_across_token_shapes=True,
                        native_weight_bytes=sum(
                            r.storage_nbytes for r in residents["native"]
                        ),
                        cache_bytes=sum(cache.nbytes for cache in caches["native"]),
                        workspace_phases=[],
                        backend_phases=[],
                    )
                    for shared in (False, True, True, False):
                        row = run_trial(
                            token_plans["native"][count],
                            steps["native"],
                            caches["native"],
                            x,
                            c,
                            s,
                            prefix,
                            shared,
                            args.seconds_per_phase,
                        )
                        actual = row.pop("output")
                        assert torch.equal(actual, expected["native"])
                        row.update(
                            condition="shared" if shared else "separate",
                            eager_graph_bit_identical=True,
                        )
                        report["workspace_phases"].append(row)
                    for name in ("reference", "native", "native", "reference"):
                        row = run_trial(
                            token_plans[name][count],
                            steps[name],
                            caches[name],
                            x,
                            c,
                            s,
                            prefix,
                            True,
                            args.seconds_per_phase,
                        )
                        actual = row.pop("output")
                        assert torch.equal(actual, expected[name])
                        row.update(condition=name, eager_graph_bit_identical=True)
                        report["backend_phases"].append(row)
                    error = (
                        expected["native"].double() - expected["reference"].double()
                    ).abs()
                    scale = expected["reference"].double().square().mean().sqrt().item()
                    report["cross_backend_diagnostic"] = dict(
                        max_abs_error=error.max().item(),
                        normalized_rms_error=error.square().mean().sqrt().item()
                        / scale,
                        matching_elements=int(
                            (expected["native"] == expected["reference"]).sum()
                        ),
                        elements=error.numel(),
                        scope="Diagnostic, not a per-Op correctness gate or lossless/quality claim.",
                    )
                    report["workspace_medians"], report["backend_medians"] = (
                        metrics(report["workspace_phases"]),
                        metrics(report["backend_phases"]),
                    )
                    reports.append(report)
                    (out / (key + ".json")).write_text(
                        json.dumps(report, indent=2) + "\n"
                    )
                    print(
                        key,
                        report["workspace_medians"],
                        report["backend_medians"],
                        flush=True,
                    )
                    del (
                        caches,
                        steps,
                        expected,
                        qualified,
                        kv,
                        current,
                        target,
                        actual,
                        prior,
                        cache,
                        weight,
                        step,
                    )
                    gc.collect()
                    torch.cuda.empty_cache()
                for plans in token_plans.values():
                    for items in plans.values():
                        for plan in items:
                            plan.close()
                del (
                    residents,
                    token_plans,
                    weights,
                    source,
                    cos,
                    sin,
                    x,
                    c,
                    s,
                    plan,
                    plans,
                    items,
                )
                gc.collect()
                torch.cuda.empty_cache()
        finally:
            sampler.terminate()
            try:
                sampler.wait(timeout=5)
            except subprocess.TimeoutExpired:
                sampler.kill()
                sampler.wait()
        if sampler.returncode not in (0, -15):
            raise RuntimeError(f"GPU sampler failed: {sampler.returncode}")
    for report in reports:
        for kind in ("workspace_phases", "backend_phases"):
            for measured in report[kind]:
                measured["gpu"] = telemetry(out / "telemetry.csv", [measured])["single"]
        (out / (report["name"] + ".json")).write_text(
            json.dumps(report, indent=2) + "\n"
        )
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
