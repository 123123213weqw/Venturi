"""Explicit SM70 group32 WMMA screen against original and residual projections.

Same original-source Q8 planes and FP16 inputs. WMMA retains input precision
but changes group/split-K reduction. Independent sampled-row FP64 gates apply.
Graph microbenchmarks are not serving TPS or full-model quality qualification.
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
from q8_dp4a_reference import projection_reference  # noqa: E402
from venturi.models import QwenQ8Weights  # noqa: E402
from venturi.ninfer import NInferArtifact  # noqa: E402
from venturi.ops import CudaBackend, Q8RowSplitWeight, prepare_q8_dp4a, prepare_q8_wmma  # noqa: E402


def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def error(actual, ref):
    a, b = actual.double(), ref.double()
    return dict(
        relative_l2=float((a - b).norm() / (b.norm() + 1e-20)),
        max_absolute=float((a - b).abs().max()),
        cosine=float((a.flatten() @ b.flatten()) / (a.norm() * b.norm() + 1e-20)),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact")
    p.add_argument("--library", required=True)
    p.add_argument("--output-directory", required=True)
    p.add_argument("--token-counts", default="1,2,8,32")
    p.add_argument("--row-mode", choices=("full", "tp-first"), default="tp-first")
    a = p.parse_args()
    counts = [int(n) for n in a.token_counts.split(",")]
    if not counts or any(n not in (1, 2, 4, 8, 23, 32) for n in counts):
        p.error("Use explicit token counts 1,2,4,8,23,32")
    out = Path(a.output_directory)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    backend = CudaBackend(a.library)
    weights = QwenQ8Weights.from_ninfer(NInferArtifact(a.artifact))
    roles = {
        s.shape: s.binding for s in weights.plan.bindings if s.format == "q8_g32_fp16"
    }
    report = dict(
        scope=__doc__,
        library_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),
        artifact_id=weights.artifact.artifact_id,
        dtype="FP16 input/output; original Q8/F16-scale weights. Original FP16, residual A8 and group32 FP32-scaled FP16 WMMA are separate arithmetic profiles.",
        row_mode=a.row_mode,
        profiles=[
            "original-fp16",
            "w8-a8-g32-f32scale-residual-a8",
            *[f"w8-f16tc-g32-f32scale-splitk{n}" for n in (1, 2, 4, 8)],
        ],
        graph_nodes=3,
        graph_replays=3,
        measured_repeats=3,
        cases=[],
        full_model_quality_qualified=False,
        hardware_counters=None,
        completed=False,
    )

    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    log = (out / "gpu-samples.csv").open("wb")
    sampler = subprocess.Popen(
        [
            "nvidia-smi",
            "-i",
            "1",
            "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu,clocks.sm,clocks.mem",
            "--format=csv,noheader,nounits",
            "-lms",
            "100",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        for original_shape, role in roles.items():
            source = weights.read_linear(role)
            if a.row_mode == "tp-first":
                half = (original_shape[0] + 1) // 2
                source = Q8RowSplitWeight(
                    source.codes[:half].clone(),
                    source.scales[:half].clone(),
                    original_shape[1],
                )
            shape = (source.codes.shape[0], original_shape[1])
            resident = backend.upload_linear(
                source, dtype=torch.float16, device="cuda:0"
            )
            for tokens in counts:
                x_cpu = torch.randn(
                    tokens,
                    shape[1],
                    generator=torch.Generator().manual_seed(812 + tokens),
                ).half()
                x = x_cpu.cuda()
                plans = [
                    resident.prepare(tokens=tokens),
                    prepare_q8_dp4a(
                        resident, tokens=tokens, residual_compensation=True
                    ),
                ] + [
                    prepare_q8_wmma(resident, tokens=tokens, split_k_warps=warps)
                    for warps in (1, 2, 4, 8)
                ]
                scratch = [
                    None,
                    torch.empty(
                        plans[1].workspace_bytes, dtype=torch.uint8, device="cuda:0"
                    ),
                ] + [None] * 4
                outputs = [
                    torch.empty(tokens, shape[0], dtype=torch.float16, device="cuda:0")
                    for _ in plans
                ]

                def call(i, plans=plans, x=x, outputs=outputs, scratch=scratch):
                    if i == 1:
                        plans[i].run(x, out=outputs[i], workspace=scratch[i])
                    else:
                        plans[i].run(x, out=outputs[i])

                for i in range(len(plans)):
                    call(i)
                torch.cuda.synchronize()
                values = [y.cpu() for y in outputs]
                assert all(torch.isfinite(v).all() for v in values)
                rows = list(
                    dict.fromkeys(
                        [0, shape[0] - 1]
                        + torch.randint(
                            shape[0],
                            (30,),
                            generator=torch.Generator().manual_seed(883),
                        ).tolist()
                    )
                )
                code, scale = source.codes[rows], source.scales[rows]
                dense = code.double() * scale.double().repeat_interleave(32, -1)
                ideals = [x_cpu.double() @ dense[:, : shape[1]].T] + [
                    projection_reference(x_cpu, code, scale, passes) for passes in (2,)
                ]
                ideals += [ideals[0]] * 4
                checks = [error(v[:, rows], ref) for v, ref in zip(values, ideals)]
                assert all(c["relative_l2"] < 0.001 for c in checks), (
                    "Independent per-profile sampled-row gate failed"
                )
                case = dict(
                    shape=shape,
                    original_shape=original_shape,
                    role=role,
                    tokens=tokens,
                    independent_sample_rows=rows,
                    sampled_reference=checks,
                    original_profile_comparison=[error(v, values[0]) for v in values],
                    same_original_top1=[
                        bool((v.argmax(-1) == values[0].argmax(-1)).all())
                        for v in values
                    ],
                    workspace_bytes=[p.workspace_bytes for p in plans],
                    waves=[],
                )
                graphs = []
                for i in range(len(plans)):
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, stream=stream):
                        for _ in range(report["graph_nodes"]):
                            call(i)
                    torch.cuda.current_stream().wait_stream(stream)
                    graphs.append(g)
                for g in graphs:
                    for _ in range(3):
                        g.replay()
                torch.cuda.synchronize()
                for repeat in range(3):
                    for i in (
                        range(len(plans))
                        if repeat % 2 == 0
                        else reversed(range(len(plans)))
                    ):
                        start, end = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        torch.cuda.synchronize()
                        began, wall, process = time(), perf_counter(), cpu()
                        start.record()
                        for _ in range(report["graph_replays"]):
                            graphs[i].replay()
                        end.record()
                        end.synchronize()
                        case["waves"].append(
                            dict(
                                profile=report["profiles"][i],
                                repeat=repeat,
                                start_unix=began,
                                end_unix=time(),
                                calls=9,
                                gpu_event_ms_per_call=start.elapsed_time(end) / 9,
                                wall_seconds=perf_counter() - wall,
                                cpu_core_seconds=cpu() - process,
                            )
                        )
                case["median_gpu_event_ms"] = {
                    name: statistics.median(
                        w["gpu_event_ms_per_call"]
                        for w in case["waves"]
                        if w["profile"] == name
                    )
                    for name in report["profiles"]
                }
                report["cases"].append(case)
                save()
                print(
                    json.dumps(
                        {
                            k: case[k]
                            for k in (
                                "shape",
                                "tokens",
                                "median_gpu_event_ms",
                                "original_profile_comparison",
                                "same_original_top1",
                            )
                        }
                    ),
                    flush=True,
                )
                del (
                    plans,
                    outputs,
                    scratch,
                    graphs,
                    g,
                    stream,
                    values,
                    ideals,
                    dense,
                    x,
                    x_cpu,
                )
                gc.collect()
            del source, resident
            gc.collect()
            torch.cuda.empty_cache()
        report["completed"] = True
        save()
    finally:
        sampler.terminate()
        sampler.wait(timeout=15)
        log.close()


if __name__ == "__main__":
    main()
