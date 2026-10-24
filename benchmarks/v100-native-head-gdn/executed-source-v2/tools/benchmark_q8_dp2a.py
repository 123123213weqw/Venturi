"""Bounded actual-Q8-plane operator prerequisite, not complete-model TPS/quality.

Both arms include quantization plus projection on the same resident weights.
Candidate uses one INT16/F32 activation pass; control uses two residual A8 passes.
FP16 input, FP16 full outputs or FP32 column down partials; arithmetic differs.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import resource
import statistics
import subprocess
import sys
from time import perf_counter, time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from q8_dp2a_reference import projection_reference as ref16  # noqa: E402
from q8_dp4a_reference import projection_reference as ref8  # noqa: E402
from venturi.models import QwenQ8Weights  # noqa: E402
from venturi.ninfer import NInferArtifact  # noqa: E402
from venturi.ops import CudaBackend, Q8RowSplitWeight, prepare_q8_dp4a  # noqa: E402
from venturi.ops.q8_column_tp import prepare_q8_column_partial, slice_q8_columns  # noqa: E402
from venturi.ops.q8_dp2a import DP2APilot  # noqa: E402
from venturi.ops.q8_fused import prepare_q8_fused  # noqa: E402
from venturi.ops.q8_shared import prepare_q8_tiled_shared  # noqa: E402


def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def error(actual, ref):
    a, b = actual.double(), ref.double()
    return dict(
        relative_l2=float((a - b).norm() / (b.norm() + 1e-20)),
        max_absolute=float((a - b).abs().max()),
    )


def make_cells(weights):
    for label, suffixes in [
        ("fused-mlp", ["mlp/gate", "mlp/up"]),
        ("fused-gdn", ["gdn/qkv", "gdn/z"]),
        ("fused-attention", ["attention/q", "attention/k", "attention/v"]),
        ("column-down-fp32", ["mlp/down"]),
        ("row-gdn-output", ["gdn/output"]),
        ("row-attention-output", ["attention/o"]),
    ]:
        binding = next(
            s.binding
            for s in weights.plan.bindings
            if s.binding.endswith("/" + suffixes[0])
        )
        prefix = binding[: -len(suffixes[0])]
        sources = []
        for suffix in suffixes:
            w = weights.read_linear(prefix + suffix)
            if label == "column-down-fp32":
                w = slice_q8_columns(w, 0, w.in_features // 2)
            else:
                n = (w.out_features + 1) // 2
                w = Q8RowSplitWeight(
                    w.codes[:n].clone(), w.scales[:n].clone(), w.in_features
                )
            sources.append(w)
        yield label, [prefix + s for s in suffixes], sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact")
    parser.add_argument("--library", required=True)
    parser.add_argument("--baseline-library", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--profile", action="store_true")
    a = parser.parse_args()
    out = Path(a.output_directory)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    if torch.cuda.get_device_capability(0) != (7, 0):
        raise ValueError("Explicit SM70 pilot required")
    weights = QwenQ8Weights.from_ninfer(NInferArtifact(a.artifact))
    pilot, backend = DP2APilot(a.library), CudaBackend(a.baseline_library)
    report = dict(
        scope=__doc__,
        profiled=a.profile,
        artifact_id=weights.artifact.artifact_id,
        pilot_library_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),
        control_library_sha256=hashlib.sha256(
            Path(a.baseline_library).read_bytes()
        ).hexdigest(),
        complete_model_quality_qualified=False,
        model_dispatch_changed=False,
        achieved_hardware_counters=None,
        graph_nodes=8,
        replays=128,
        cases=[],
        completed=False,
    )

    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    sampler = None
    log = (out / "gpu-samples.csv").open("wb")
    try:
        if not a.profile:
            sampler = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu,clocks.sm,clocks.mem",
                    "--format=csv,noheader,nounits",
                    "-lms",
                    "100",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        for label, bindings, sources in make_cells(weights):
            if a.profile and label not in ("fused-mlp", "column-down-fp32"):
                continue
            residents = [
                backend.upload_linear(w, dtype=torch.float16, device="cuda:0")
                for w in sources
            ]
            for t in [1, 3, 8]:
                k = sources[0].in_features
                dtype = torch.float32 if label == "column-down-fp32" else torch.float16
                candidate = pilot.prepare(residents, tokens=t, dtype=dtype)
                if len(residents) > 1:
                    control = prepare_q8_fused(
                        residents, tokens=t, residual_compensation=True
                    )
                elif dtype == torch.float32:
                    control = prepare_q8_column_partial(residents[0], tokens=t)
                elif t >= 3:
                    control = prepare_q8_tiled_shared(
                        residents[0],
                        tokens=t,
                        rows=16 if t >= 8 else 8,
                        token_tile=8 if t >= 8 else 4,
                        residual_compensation=True,
                    )
                else:
                    control = prepare_q8_dp4a(
                        residents[0], tokens=t, residual_compensation=True
                    )
                xcpu = torch.randn(
                    t, k, generator=torch.Generator().manual_seed(732 + t)
                ).half()
                x = xcpu.cuda()
                outputs = [
                    [
                        torch.empty(t, w.out_features, dtype=dtype, device="cuda:0")
                        for w in sources
                    ]
                    for _ in range(2)
                ]
                work = [
                    torch.empty(p.workspace_bytes, dtype=torch.uint8, device="cuda:0")
                    for p in [control, candidate]
                ]

                def call(
                    arm,
                    candidate=candidate,
                    x=x,
                    outputs=outputs,
                    work=work,
                    control=control,
                    group_count=len(residents),
                ):
                    if arm:
                        candidate.run(x, outputs=outputs[arm], workspace=work[arm])
                    elif group_count > 1:
                        control.run(x, outputs=outputs[arm], workspace=work[arm])
                    else:
                        control.run(x, out=outputs[arm][0], workspace=work[arm])

                for i in [0, 1]:
                    call(i)
                torch.cuda.synchronize()
                checks = []
                for j, w in enumerate(sources):
                    rows = list(
                        dict.fromkeys(
                            [0, w.out_features - 1]
                            + torch.randint(
                                w.out_features,
                                (30,),
                                generator=torch.Generator().manual_seed(985),
                            ).tolist()
                        )
                    )
                    code, scale = w.codes[rows], w.scales[rows]
                    refs = [ref8(xcpu, code, scale, 2), ref16(xcpu, code, scale)]
                    full = (
                        xcpu.double()
                        @ (code.double() * scale.double().repeat_interleave(32, -1))[
                            :, :k
                        ].T
                    )
                    for arm in [0, 1]:
                        got = outputs[arm][j].cpu()[:, rows]
                        check = error(got, refs[arm])
                        check["versus_unquantized_fp16_activation_fp64"] = error(
                            got, full
                        )
                        assert check["relative_l2"] < (
                            0.001 if dtype == torch.float16 else 2e-6
                        ), "Independent profile reconstruction failed"
                        assert (
                            check["versus_unquantized_fp16_activation_fp64"][
                                "relative_l2"
                            ]
                            < 0.002
                        ), "Actual encoded-plane sampled quality failed"
                        checks.append(dict(arm=arm, matrix=j, rows=rows, **check))
                case = dict(
                    label=label,
                    bindings=bindings,
                    tokens=t,
                    shapes=[[w.out_features, w.in_features] for w in sources],
                    output_dtype=str(dtype),
                    workspace_bytes=[p.workspace_bytes for p in [control, candidate]],
                    reference_checks=checks,
                    waves=[],
                )
                if a.profile:
                    with torch.cuda.nvtx.range("native_unsplit"):
                        for arm in [0, 1]:
                            with torch.cuda.nvtx.range(
                                f"projection_{label}_t{t}_arm{arm}"
                            ):
                                for _ in range(16):
                                    call(arm)
                                torch.cuda.synchronize()
                    report["cases"].append(case)
                    save()
                    continue
                graphs = []
                for arm in [0, 1]:
                    g, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.graph(g, stream=stream):
                        for _ in range(report["graph_nodes"]):
                            call(arm)
                    torch.cuda.current_stream().wait_stream(stream)
                    graphs.append(g)
                    for _ in range(3):
                        g.replay()
                torch.cuda.synchronize()
                for repeat in range(3):
                    for arm in [0, 1] if repeat % 2 == 0 else [1, 0]:
                        start, end = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        torch.cuda.synchronize()
                        began, wall, proc = time(), perf_counter(), cpu()
                        start.record()
                        for _ in range(report["replays"]):
                            graphs[arm].replay()
                        end.record()
                        end.synchronize()
                        calls = report["replays"] * report["graph_nodes"]
                        case["waves"].append(
                            dict(
                                arm=arm,
                                repeat=repeat,
                                start_unix=began,
                                end_unix=time(),
                                calls=calls,
                                gpu_event_ms_per_call=start.elapsed_time(end) / calls,
                                wall_seconds=perf_counter() - wall,
                                cpu_core_seconds=cpu() - proc,
                            )
                        )
                case["median_gpu_event_ms"] = [
                    statistics.median(
                        w["gpu_event_ms_per_call"]
                        for w in case["waves"]
                        if w["arm"] == i
                    )
                    for i in [0, 1]
                ]
                case["operator_throughput_ratio"] = (
                    case["median_gpu_event_ms"][0] / case["median_gpu_event_ms"][1]
                )
                report["cases"].append(case)
                save()
                del graphs, call, outputs, work, control, candidate, x
                gc.collect()
                torch.cuda.empty_cache()
            del residents, sources
            gc.collect()
            torch.cuda.empty_cache()
        if not a.profile:
            fused = [
                c["operator_throughput_ratio"]
                for c in report["cases"]
                if c["label"].startswith("fused")
            ]
            assert len(report["cases"]) == 18 and len(fused) == 9
            report["median_fused_operator_ratio"] = statistics.median(fused)
            report["all_cells_at_least_098"] = all(
                c["operator_throughput_ratio"] >= 0.98 for c in report["cases"]
            )
            report["promote_to_model"] = (
                report["median_fused_operator_ratio"] >= 1.05
                and report["all_cells_at_least_098"]
            )
        report["completed"] = True
        save()
    finally:
        if sampler is not None:
            sampler.terminate()
            try:
                sampler.wait(timeout=10)
            except subprocess.TimeoutExpired:
                sampler.kill()
                sampler.wait()
        log.close()


if __name__ == "__main__":
    main()
