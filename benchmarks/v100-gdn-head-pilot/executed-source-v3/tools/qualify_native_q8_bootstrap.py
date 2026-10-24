"""Remote-only GDN frontend and generic Q8 primitive qualification/timing.

Actual small checkpoint constants with synthetic activations; synthetic generic
Q8 projection weights. Not full Qwen execution, trained-model quality, model TPS,
or a comparison with an optimized quantized engine. No model download here.
"""

import argparse
from functools import partial
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import subprocess

import torch

from benchmark_qwen_gpu import FIELDS, telemetry
from qualify_checkpoint_gdn import bf16_check, comparison, timed_pair, write
from run_qwen_validation import idle_gpu
from venturi.models import GDNFrontendWeights
from venturi.ninfer import NInferArtifact
from venturi.ops import CudaBackend, Q8RowSplitWeight, TorchBackend


def capture(call):
    stream, graph = torch.cuda.Stream(), torch.cuda.CUDAGraph()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
        with torch.cuda.graph(graph, stream=stream):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    return graph


def oracle_module():
    spec = spec_from_file_location(
        "state_space_oracle",
        Path(__file__).resolve().parents[1] / "tests/test_state_space.py",
    )
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--blocks-library", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--skip-projection", action="store_true")
    args = parser.parse_args(argv)
    if args.seconds < 2:
        raise ValueError("Use >=2 second unprofiled phases")
    torch.set_num_threads(4)
    out = Path(args.output_directory)
    out.mkdir(parents=True, exist_ok=False)
    write(out / "background.json", {"compute_processes": idle_gpu()})
    artifact = NInferArtifact(args.checkpoint)
    weights = GDNFrontendWeights.from_ninfer(artifact, layer=0)
    backends = {
        "native": CudaBackend(args.blocks_library),
        "torch-reference": TorchBackend(),
    }
    constants = {name: weights.upload(backend) for name, backend in backends.items()}
    oracle = oracle_module()
    cfg = artifact.manifest["components"]["text"]["config"]
    report = dict(
        scope=__doc__,
        artifact_id=artifact.artifact_id,
        cuda=torch.version.cuda,
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        native_library=str(backends["native"].library),
        cases=[],
        counters="Achieved occupancy/SM/DRAM/cache/stall counters not measured; RmProfilingAdminOnly=1. No profiler in timed phases.",
    )
    waves = []
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
            for tokens in (1, 128):
                channels = weights.conv.shape[1]
                x = oracle.sample((tokens, channels), torch.bfloat16, "cuda")
                history = oracle.sample(
                    (weights.conv.shape[0] - 1, channels), torch.bfloat16, "cuda", 70
                )
                a, b = (
                    oracle.sample(
                        (tokens, weights.value_heads), torch.bfloat16, "cuda", seed
                    )
                    for seed in (1, 2)
                )
                hidden, z = (
                    oracle.sample(
                        (tokens, weights.value_heads, weights.value_dim),
                        torch.bfloat16,
                        "cuda",
                        seed,
                    )
                    for seed in (3, 4)
                )
                norm_x = oracle.sample(
                    (tokens, cfg["hidden_size"]), torch.bfloat16, "cuda", 4
                )
                conv_ideal, history_ideal = oracle.conv_oracle(x, weights.conv, history)
                value = hidden.cpu().double()
                normal = (
                    (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6))
                    .bfloat16()
                    .double()
                )
                weighted = (normal * weights.gated_norm.double()).bfloat16().double()
                norm_ideal = weighted * oracle.silu64(z.cpu().double())
                value = norm_x.cpu().double()
                gamma = (1 + weights.input_norm.float()).double()
                zero_ideal = (
                    value
                    * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
                    * gamma
                )
                for kind in ("conv", "gates", "gated-norm", "zero-norm"):
                    print(kind, "T=", tokens, flush=True)
                    calls, buffers, checks, plans = {}, {}, {}, {}
                    case = dict(
                        name=f"{kind}-t{tokens}",
                        tokens=tokens,
                        source="Actual layer-0 checkpoint constants; synthetic BF16 activation inputs",
                        timing="Fixed-state graph replay wrapper; not autoregressive decoding",
                    )
                    for name, resident in constants.items():
                        if kind == "conv":
                            plan = resident.conv.prepare(tokens=tokens)
                            target, final = (
                                torch.empty_like(x),
                                torch.empty_like(history),
                            )
                            call = partial(
                                plan.run,
                                x,
                                history=history,
                                out=target,
                                history_out=final,
                            )
                            call()
                            check = bf16_check(target, conv_ideal)
                            check["history_exact"] = torch.equal(
                                final.cpu(), history_ideal
                            )
                            assert check["history_exact"]
                            buffers[name] = (target, final)
                        elif kind == "gates":
                            plan = resident.gates.prepare(tokens=tokens)
                            g, beta = (
                                torch.empty_like(a, dtype=torch.float32)
                                for _ in range(2)
                            )
                            call = partial(plan.run, a, b, log_decay=g, beta=beta)
                            call()
                            v = a.cpu().double() + resident.gates._bias.cpu().double()
                            expected_g = (
                                -resident.gates._rate.cpu().double()
                                * torch.where(v > 20, v, torch.log1p(v.exp()))
                            )
                            expected_beta = (
                                b.cpu().double().sigmoid().bfloat16().double()
                            )
                            error_g = (g.cpu().double() - expected_g).abs()
                            error_beta = (beta.cpu().double() - expected_beta).abs()
                            check = dict(
                                passed=bool(
                                    (error_g <= 2e-7 + 2e-6 * expected_g.abs()).all()
                                )
                                and bool(
                                    (
                                        error_beta <= 2e-7 + 2e-6 * expected_beta.abs()
                                    ).all()
                                ),
                                max_g_abs_error=error_g.max().item(),
                                max_beta_abs_error=error_beta.max().item(),
                            )
                            buffers[name] = (g, beta)
                        elif kind == "gated-norm":
                            plan = resident.gated_norm.prepare(
                                tokens=tokens, heads=weights.value_heads
                            )
                            target = torch.empty_like(hidden)
                            call = partial(plan.run, hidden, z, out=target)
                            call()
                            check = bf16_check(target, norm_ideal)
                            buffers[name] = (target,)
                        else:
                            plan = resident.input_norm.prepare(tokens=tokens)
                            target = torch.empty_like(norm_x)
                            call = partial(plan.run, norm_x, out=target)
                            call()
                            check = bf16_check(target, zero_ideal)
                            buffers[name] = (target,)
                        checks[name], calls[name], plans[name] = check, call, plan
                    case["qualification"] = checks
                    write(out / (case["name"] + "-qualification.json"), case)
                    assert all(c["passed"] for c in checks.values()), case
                    graphs = {name: capture(call) for name, call in calls.items()}
                    case["phases"] = timed_pair(
                        {name: graph.replay for name, graph in graphs.items()},
                        out / (case["name"] + "-phases.json"),
                        args.seconds,
                    )
                    case["medians"] = comparison(case["phases"])
                    waves.extend(case["phases"])
                    report["cases"].append(case)
                    write(out / "results.json", report)
                    del graphs, calls, buffers, plans
                if args.skip_projection:
                    continue
                # Projection size matches the Qwen up/gate operator. Input codes
                # are explicitly synthetic, not repacked Q4/checkpoint weights.
                n, k = cfg["intermediate_size"], cfg["hidden_size"]
                generator = torch.Generator().manual_seed(38)
                quantized = Q8RowSplitWeight(
                    torch.randint(
                        -127, 128, (n, k), generator=generator, dtype=torch.int8
                    ),
                    torch.full((n, k // 32), 0.001, dtype=torch.float16),
                    k,
                )
                xq = oracle.sample((tokens, k), torch.bfloat16, "cuda", 91)
                exact = (
                    quantized.codes.double()
                    * quantized.scales.double().repeat_interleave(32, -1)
                )
                ideal = xq.cpu().double() @ exact.T
                residents = {
                    name: backend.upload_linear(
                        quantized, dtype=torch.bfloat16, device="cuda:0"
                    )
                    for name, backend in backends.items()
                }
                plans = {
                    name: resident.prepare(tokens=tokens)
                    for name, resident in residents.items()
                }
                targets = {
                    name: torch.empty(tokens, n, dtype=torch.bfloat16, device="cuda")
                    for name in plans
                }
                calls = {
                    name: (
                        lambda plan=plan, target=targets[name]: plan.run(xq, out=target)
                    )
                    for name, plan in plans.items()
                }
                for call in calls.values():
                    call()
                case = dict(
                    name=f"q8-projection-t{tokens}",
                    tokens=tokens,
                    source="Synthetic Q8 [-127,127]/group-32 FP16 planes, not trained weights",
                    shape=[n, k],
                    native_weight_bytes=residents["native"].storage_nbytes,
                    timing="Graph replay wrapper. Reference materializes weights as FP32 during preparation, not each call; not another optimized Q8 implementation",
                    qualification={
                        name: bf16_check(target, ideal, floor=2e-4)
                        for name, target in targets.items()
                    },
                )
                write(out / (case["name"] + "-qualification.json"), case)
                assert all(c["passed"] for c in case["qualification"].values()), case
                graphs = {name: capture(call) for name, call in calls.items()}
                case["phases"] = timed_pair(
                    {name: graph.replay for name, graph in graphs.items()},
                    out / (case["name"] + "-phases.json"),
                    args.seconds,
                )
                case["medians"] = comparison(case["phases"])
                waves.extend(case["phases"])
                report["cases"].append(case)
                write(out / "results.json", report)
                del graphs, calls, targets, plans, residents, exact, ideal, quantized
        finally:
            sampler.terminate()
            sampler.wait(timeout=10)
    for case in report["cases"]:
        case["telemetry"] = {
            name: telemetry(
                out / "telemetry.csv",
                [p for p in case["phases"] if p["condition"] == name],
            )["single"]
            for name in backends
        }
    write(out / "results.json", report)
    print(
        json.dumps({c["name"]: c["medians"] for c in report["cases"]}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
