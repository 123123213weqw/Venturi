"""GDN geometry microbenchmark; warmed primitive calls, not serving TPS."""

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
from test_gdn import gdn_inputs, gdn_oracle  # noqa: E402
from venturi.ops import CudaBackend  # noqa: E402
from venturi.ops.gdn_compact import prepare_gdn_compact  # noqa: E402


def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def same(a, b):
    return torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", required=True)
    p.add_argument("--baseline-library", required=True)
    p.add_argument("--output-directory", required=True)
    p.add_argument("--profile", action="store_true")
    a = p.parse_args()
    out = Path(a.output_directory)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    backends = [CudaBackend(a.baseline_library), CudaBackend(a.library)]
    arms = ["control-256", "candidate-256", "candidate-compact"]
    cases = [(128, t) for t in ([1, 2, 8] if a.profile else [1, 2, 4, 8, 23, 32])]
    if not a.profile:
        cases += [(dk, t) for dk in [64, 256] for t in [1, 8]]
    report = dict(
        scope=__doc__,
        profiled=a.profile,
        completed=False,
        precision="FP16 packed QKV/output; FP32 decay/beta/state; original eight logical warp reduction order",
        arms=arms,
        library_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),
        baseline_library_sha256=hashlib.sha256(
            Path(a.baseline_library).read_bytes()
        ).hexdigest(),
        cases=[],
    )
    f = (out / "gpu-samples.csv").open("wb")
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
        stdout=f,
        stderr=subprocess.STDOUT,
    )

    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    try:
        for dk, t in cases:
            cfg = dict(
                tokens=t,
                qk_heads=16,
                value_heads=48,
                key_dim=dk,
                value_dim=128,
                normalize_qk=True,
                dtype=torch.float16,
                device="cuda:0",
            )
            data = gdn_inputs(
                tokens=t,
                hq=16,
                hv=48,
                dk=dk,
                dv=128,
                dtype=torch.float16,
                device="cuda:0",
            )
            q, k, v, g, beta, state = data
            qkv = torch.cat([q.flatten(1), k.flatten(1), v.flatten(1)], 1)
            plans = [b.prepare_gdn(**cfg) for b in backends] + [
                prepare_gdn_compact(backends[1], **cfg)
            ]
            outputs = [(torch.empty_like(v), torch.empty_like(state)) for _ in arms]

            def call(i, pp=plans, xx=qkv, gg=g, bb=beta, ss=state, yy=outputs):
                pp[i].run_packed(xx, gg, bb, state=ss, out=yy[i][0], state_out=yy[i][1])

            for i in range(3):
                call(i)
            torch.cuda.synchronize()
            assert all(
                same(x, y) for pair in outputs[1:] for x, y in zip(outputs[0], pair)
            )
            ideal, expected_state = gdn_oracle(*data)
            actual, final = [x.cpu().double() for x in outputs[-1]]
            assert (
                (actual - ideal.half().double()).abs()
                <= 2 * torch.finfo(torch.float16).eps * ideal.abs() + 2e-5
            ).all()
            torch.testing.assert_close(final, expected_state, atol=2e-6, rtol=2e-4)
            case = dict(
                tokens=t,
                qk_heads=16,
                value_heads=48,
                key_dim=dk,
                value_dim=128,
                qwen27b_geometry=(dk == 128),
                output_state_bytes_equal=True,
                independent_fp64_output_relative_l2=float(
                    (actual - ideal).norm() / (ideal.norm() + 1e-12)
                ),
                independent_fp64_state_relative_l2=float(
                    (final - expected_state).norm() / (expected_state.norm() + 1e-12)
                ),
                waves=[],
            )
            if a.profile:
                with torch.cuda.nvtx.range("native_unsplit"):
                    for i, name in enumerate(arms):
                        with torch.cuda.nvtx.range(f"projection_gdn_{name}_t{t}"):
                            for _ in range(16):
                                call(i)
                        torch.cuda.synchronize()
            else:
                graphs = []
                for i in range(3):
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        for _ in range(8):
                            call(i)
                    torch.cuda.current_stream().wait_stream(stream)
                    graphs.append(graph)
                for graph in graphs:
                    for _ in range(3):
                        graph.replay()
                torch.cuda.synchronize()
                for repeat in range(3):
                    for i in range(3) if repeat % 2 == 0 else reversed(range(3)):
                        start, end = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        torch.cuda.synchronize()
                        unix, w, c = time(), perf_counter(), cpu()
                        start.record()
                        for _ in range(4):
                            graphs[i].replay()
                        end.record()
                        end.synchronize()
                        case["waves"].append(
                            dict(
                                arm=arms[i],
                                repeat=repeat,
                                calls=32,
                                start_unix=unix,
                                end_unix=time(),
                                gpu_event_ms_per_call=start.elapsed_time(end) / 32,
                                wall_seconds=perf_counter() - w,
                                cpu_core_seconds=cpu() - c,
                            )
                        )
                case["median_gpu_event_ms"] = {
                    arm: statistics.median(
                        x["gpu_event_ms_per_call"]
                        for x in case["waves"]
                        if x["arm"] == arm
                    )
                    for arm in arms
                }
            report["cases"].append(case)
            save()
            print(json.dumps(case), flush=True)
            del call, outputs, plans, data, qkv
            if not a.profile:
                del graphs, graph, stream
            gc.collect()
            torch.cuda.empty_cache()
        report["completed"] = True
        save()
    finally:
        sampler.terminate()
        sampler.wait(timeout=10)
        f.close()


if __name__ == "__main__":
    main()
