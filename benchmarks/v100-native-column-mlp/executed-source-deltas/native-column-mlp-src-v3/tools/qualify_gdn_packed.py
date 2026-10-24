"""Matched native packed-GDN versus explicit-unpack recurrence qualification.

Synthetic BF16 inputs at checkpoint-defined GDN geometry. Fixed-state graph
replays, not complete model execution or tokens/s. The baseline copies strided
Q/K/V views into reusable contiguous buffers for T>1, then uses the same native
recurrence. At T=1 the views are already contiguous and no copies are needed.
"""

import argparse
from functools import partial
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import subprocess

import torch

from benchmark_qwen_gpu import FIELDS, telemetry
from qualify_checkpoint_gdn import bf16_check, phase, write
from qualify_native_q8_bootstrap import capture
from run_qwen_validation import idle_gpu
from venturi.ninfer import NInferArtifact
from venturi.ops import CudaBackend


def input_oracle():
    spec = spec_from_file_location(
        "independent_gdn_oracle",
        Path(__file__).resolve().parents[1] / "tests/test_gdn.py",
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
    args = parser.parse_args(argv)
    if args.seconds < 2:
        raise ValueError("Use >=2 second unprofiled ABBA phases")
    torch.set_num_threads(4)
    out = Path(args.output_directory)
    out.mkdir(parents=True, exist_ok=False)
    write(out / "background.json", {"compute_processes": idle_gpu()})
    artifact = NInferArtifact(args.checkpoint)
    cfg = artifact.manifest["components"]["text"]["config"]
    geometry = dict(
        qk_heads=cfg["linear_num_key_heads"],
        value_heads=cfg["linear_num_value_heads"],
        key_dim=cfg["linear_key_head_dim"],
        value_dim=cfg["linear_value_head_dim"],
    )
    backend, oracle = CudaBackend(args.blocks_library), input_oracle()
    report = dict(
        scope=__doc__,
        geometry=geometry,
        artifact_id=artifact.artifact_id,
        library=str(backend.library),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        cases=[],
        counters="Not measured: achieved occupancy, SM/DRAM throughput, cache/stall. RmProfilingAdminOnly=1; no profiler in timed phases.",
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
            for tokens in (1, 128):
                print("qualifying packed GDN T=", tokens, flush=True)
                q, k, v, g, beta, state = oracle.gdn_inputs(
                    tokens=tokens,
                    hq=geometry["qk_heads"],
                    hv=geometry["value_heads"],
                    dk=geometry["key_dim"],
                    dv=geometry["value_dim"],
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                packed = torch.cat([x.flatten(1) for x in (q, k, v)], 1)
                width = geometry["qk_heads"] * geometry["key_dim"]
                views = (
                    packed[:, :width].reshape_as(q),
                    packed[:, width : 2 * width].reshape_as(k),
                    packed[:, 2 * width :].reshape_as(v),
                )
                separate = tuple(view.contiguous() for view in views)
                copies_required = not all(view.is_contiguous() for view in views)
                plan = backend.prepare_gdn(tokens=tokens, **geometry)
                buffers = {
                    name: (torch.empty_like(v), torch.empty_like(state))
                    for name in ("unpack-native", "packed-native")
                }

                def baseline(
                    separate=separate,
                    views=views,
                    buffers=buffers,
                    copies_required=copies_required,
                    plan=plan,
                    g=g,
                    beta=beta,
                    state=state,
                ):
                    if copies_required:
                        for target, source in zip(separate, views):
                            target.copy_(source)
                    plan.run(
                        *separate,
                        g,
                        beta,
                        state=state,
                        out=buffers["unpack-native"][0],
                        state_out=buffers["unpack-native"][1],
                    )

                calls = {
                    "unpack-native": baseline,
                    "packed-native": partial(
                        plan.run_packed,
                        packed,
                        g,
                        beta,
                        state=state,
                        out=buffers["packed-native"][0],
                        state_out=buffers["packed-native"][1],
                    ),
                }
                for call in calls.values():
                    call()
                expected, expected_state = oracle.gdn_oracle(q, k, v, g, beta, state)
                checks = {}
                for name, (output, final) in buffers.items():
                    check = bf16_check(output, expected)
                    torch.testing.assert_close(
                        final.cpu().double(), expected_state, atol=2e-6, rtol=2e-4
                    )
                    check["max_state_abs_error"] = (
                        (final.cpu().double() - expected_state).abs().max().item()
                    )
                    checks[name] = check
                exact = all(torch.equal(a, b) for a, b in zip(*buffers.values()))
                assert exact and all(c["passed"] for c in checks.values()), checks
                graphs = {name: capture(call) for name, call in calls.items()}
                case = dict(
                    tokens=tokens,
                    normalization="FP32 q/k L2 with epsilon=1e-6",
                    dtype="BF16 activations / FP32 state/gates",
                    source="Synthetic inputs, checkpoint geometry only; no trained logits/quality claim",
                    qualification=checks,
                    packed_vs_unpacked_bit_exact=exact,
                    baseline_copies_per_replay=3 if copies_required else 0,
                    unpack_workspace_bytes=sum(
                        t.numel() * t.element_size() for t in separate
                    )
                    if copies_required
                    else 0,
                    packed_unpack_workspace_bytes=0,
                    phases=[],
                    timing="Fixed-state graph replay wrapper; separate state_out; CUDA event intervals can include host submission gaps; CPU time includes event wait",
                )
                write(out / f"t{tokens}-qualification.json", case)
                for name in (
                    "unpack-native",
                    "packed-native",
                    "packed-native",
                    "unpack-native",
                ):
                    measured = phase(graphs[name].replay, args.seconds)
                    measured["condition"] = name
                    case["phases"].append(measured)
                    write(out / f"t{tokens}-phases.json", case["phases"])
                report["cases"].append(case)
                write(out / "results.json", report)
                del calls, graphs, buffers, separate, views, packed, baseline
        finally:
            sampler.terminate()
            sampler.wait(timeout=10)
    for case in report["cases"]:
        case["telemetry"] = {
            name: telemetry(
                out / "telemetry.csv",
                [p for p in case["phases"] if p["condition"] == name],
            )["single"]
            for name in ("unpack-native", "packed-native")
        }
    write(out / "results.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
