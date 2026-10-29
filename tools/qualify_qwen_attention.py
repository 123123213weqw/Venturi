"""Real-geometry Qwen attention helpers with original small norm parameters.

Synthetic activations, not trained model execution or engine TPS. Native versus
explicit allocating Torch reference, fixed-input captured-graph ABBA timing.
"""

import argparse
from functools import partial
import json
from pathlib import Path
import subprocess

import torch

from gpu_telemetry import FIELDS, telemetry
from qualify_checkpoint_gdn import timed_pair, comparison, write, bf16_check
from qualify_native_q8_bootstrap import capture
from gpu_telemetry import idle_gpu
from venturi.models import QwenTextRoPE
from venturi.ops import CudaBackend, TorchBackend


def sample(shape, seed):
    return (
        (
            torch.randn(
                shape,
                generator=torch.Generator().manual_seed(seed),
                dtype=torch.float64,
            )
            * 0.3
        )
        .bfloat16()
        .cuda()
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parameters", required=True)
    p.add_argument("--blocks-library", required=True)
    p.add_argument("--output-directory", required=True)
    p.add_argument("--seconds", type=float, default=2)
    a = p.parse_args(argv)
    if a.seconds < 2:
        raise ValueError("Use >=2 second unprofiled phases")
    torch.set_num_threads(4)
    out = Path(a.output_directory)
    out.mkdir(parents=True, exist_ok=False)
    write(out / "background.json", dict(compute_processes=idle_gpu()))
    parameters = json.loads(Path(a.parameters).read_text())
    weights = {
        k: torch.tensor(
            v["values"],
            dtype=torch.float32 if v["dtype"] == "torch.float32" else torch.bfloat16,
        )
        for k, v in parameters["parameters"].items()
    }
    backends = {
        "native": CudaBackend(a.blocks_library),
        "torch-reference": TorchBackend(),
    }
    qk = {
        name: b.upload_qwen_qk_norm(
            weights["query_norm"],
            weights["key_norm"],
            dtype=torch.bfloat16,
            device="cuda:0",
        )
        for name, b in backends.items()
    }
    gdn = {
        name: b.upload_gated_rmsnorm(
            weights["gdn_norm"], dtype=torch.bfloat16, device="cuda:0"
        )
        for name, b in backends.items()
    }
    rope = QwenTextRoPE.from_config(
        dict(
            head_dim=256,
            max_position_embeddings=262144,
            rope_parameters=dict(
                rope_type="default",
                rope_theta=10000000,
                partial_rotary_factor=0.25,
                mrope_interleaved=True,
                mrope_section=[11, 11, 10],
            ),
        )
    )
    report = dict(
        scope=__doc__,
        source_parameters=parameters,
        cuda=torch.version.cuda,
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        library=a.blocks_library,
        cases=[],
        counters="No achieved occupancy/SM/DRAM/cache/stall counters; RmProfilingAdminOnly=1. No profiler in timed phases.",
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
            for t in (1, 128):
                qg, key = sample((t, 24 * 512), 13), sample((t, 4, 256), 14)
                co, si = rope.tables(
                    backends["native"],
                    position=17,
                    tokens=t,
                    dtype=torch.bfloat16,
                    device="cuda:0",
                )
                x, z = sample((t, 48, 128), 15), sample((t, 48, 128), 16)
                for kind in ("qk-rope-gate", "fp32-gated-norm"):
                    calls, keeps, checks = {}, [], {}
                    for name, b in backends.items():
                        if kind == "qk-rope-gate":
                            op = qk[name].prepare(
                                tokens=t, query_heads=24, kv_heads=4, rotary_dim=64
                            )
                            gate = b.prepare_qwen_output_gate(
                                tokens=t,
                                heads=24,
                                head_dim=256,
                                dtype=torch.bfloat16,
                                device="cuda:0",
                            )
                            qo = torch.empty(
                                t, 24, 256, dtype=torch.bfloat16, device="cuda"
                            )
                            ko = torch.empty_like(key)
                            qn, kn = torch.empty_like(qo), torch.empty_like(key)
                            op.run_normalized(qg, key, query_out=qn, key_out=kn)
                            results = []
                            for raw, gamma, norm in (
                                (
                                    qg.reshape(t, 24, 512)[..., :256],
                                    qk[name].query_gamma,
                                    qn,
                                ),
                                (key, qk[name].key_gamma, kn),
                            ):
                                v = raw.cpu().double()
                                ideal = (
                                    v
                                    * torch.rsqrt(
                                        v.square().mean(-1, keepdim=True) + 1e-6
                                    )
                                    * gamma.cpu().double()
                                )
                                results.append(bf16_check(norm, ideal))
                            op.run(qg, key, co, si, query_out=qo, key_out=ko)
                            for norm, actual in ((qn, qo), (kn, ko)):
                                v = norm.cpu().double()
                                c, s = (v.cpu().double()[:, None, :] for v in (co, si))
                                expected = v.clone()
                                expected[..., :32] = (
                                    (
                                        (v[..., :32] * c).bfloat16().double()
                                        - (v[..., 32:64] * s).bfloat16().double()
                                    )
                                    .bfloat16()
                                    .double()
                                )
                                expected[..., 32:64] = (
                                    (
                                        (v[..., 32:64] * c).bfloat16().double()
                                        + (v[..., :32] * s).bfloat16().double()
                                    )
                                    .bfloat16()
                                    .double()
                                )
                                assert torch.equal(actual.cpu(), expected.bfloat16())
                                results.append(
                                    dict(
                                        passed=True,
                                        fused_equals_staged_rounded_rope=True,
                                    )
                                )
                            value = qo.cpu().double()
                            gates = (
                                qg.cpu()
                                .reshape(t, 24, 512)[..., 256:]
                                .double()
                                .sigmoid()
                                .bfloat16()
                                .double()
                            )
                            gate.run(qo, qg, out=qo)
                            results.append(bf16_check(qo, value * gates))

                            def call(
                                op=op,
                                gate=gate,
                                qo=qo,
                                ko=ko,
                                qg=qg,
                                key=key,
                                co=co,
                                si=si,
                            ):
                                op.run(qg, key, co, si, query_out=qo, key_out=ko)
                                gate.run(qo, qg, out=qo)

                            keeps.append((op, gate, qo, ko, qn, kn))
                            calls[name] = call
                        else:
                            op = gdn[name].prepare(tokens=t, heads=48)
                            target = torch.empty_like(x)
                            value = x.cpu().double()
                            norm = (
                                (
                                    value
                                    * torch.rsqrt(
                                        value.square().mean(-1, keepdim=True) + 1e-6
                                    )
                                )
                                .bfloat16()
                                .double()
                            )
                            zg = z.cpu().double()
                            ideal = (
                                norm
                                * weights["gdn_norm"].double()
                                * (zg * zg.sigmoid())
                            )
                            call = partial(op.run, x, z, out=target)
                            call()
                            results = [bf16_check(target, ideal)]
                            assert (
                                gdn[name]._weight.dtype == torch.float32
                                and not gdn[name].round_weighted
                            )
                            keeps.append((op, target))
                            calls[name] = call
                        checks[name] = results
                    assert all(c["passed"] for cs in checks.values() for c in cs), (
                        checks
                    )
                    case = dict(
                        name=f"{kind}-t{t}",
                        tokens=t,
                        qualification=checks,
                        timing="Fixed-input graph replay wrapper; CPU includes event wait; CUDA event intervals can include host gaps; not model TPS",
                    )
                    graphs = {name: capture(call) for name, call in calls.items()}
                    case["phases"] = timed_pair(
                        {name: g.replay for name, g in graphs.items()},
                        out / (case["name"] + "-phases.json"),
                        a.seconds,
                    )
                    case["medians"] = comparison(case["phases"])
                    report["cases"].append(case)
                    write(out / "results.json", report)
                    del calls, graphs, keeps
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
    print(json.dumps({c["name"]: c["medians"] for c in report["cases"]}, indent=2))


if __name__ == "__main__":
    main()
