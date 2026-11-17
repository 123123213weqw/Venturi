"""SM70/Q5 two-lane benchmark. Never compiles or modifies a running service.

Primary control includes per-call bounded decode+cuBLAS. Resident FP16 cuBLAS
is reported separately as a stronger single-operator control, not a model-fit
claim. Calibration, held-out validation, timing and profiling stay separate.
"""

import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import threading
import time

import torch

from venturi.ops import CudaBackend, Q5RowSplitWeight
from venturi.ops.q5_linear import (
    BLAS_BYTES,
    Q5Dispatcher,
    prepare_q5,
    time_q5,
    tune_q5,
)


class Telemetry:
    def __init__(self):
        self.samples, self.errors = [], []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def loop(self):
        while not self.stop.is_set():
            try:
                data = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,uuid,utilization.gpu,memory.used,power.draw,clocks.sm,temperature.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=5,
                )
                self.samples.append(dict(wall_unix=time.time(), csv=data.strip()))
            except Exception as e:
                self.errors.append(str(e))
            self.stop.wait(0.2)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=6)


class DenseControl:
    """Explicit diagnostic using the exact Q5-decoded FP16 matrix."""

    workspace_bytes = BLAS_BYTES

    def __init__(self, backend, weight):
        self.b = backend
        self.w = weight.dequantize_reference().half().cuda()
        self.handle = backend._native.venturi_dense_create(0)
        if not self.handle:
            backend._check(-1)
        self.fn = backend._native.venturi_q5_dense_reference_run
        self.fn.argtypes = (
            [C.c_void_p] * 4 + [C.c_int] * 3 + [C.c_void_p, C.c_size_t, C.c_void_p]
        )
        self.fn.restype = C.c_int

    def run(self, x, *, out, workspace):
        self.b._check(
            self.fn(
                self.handle,
                x.data_ptr(),
                self.w.data_ptr(),
                out.data_ptr(),
                len(x),
                self.w.shape[0],
                self.w.shape[1],
                workspace.data_ptr(),
                workspace.numel(),
                torch.cuda.current_stream(x.device).cuda_stream,
            )
        )

    def close(self):
        self.b._native.venturi_dense_destroy(self.handle)


def repeated(plan, x, out, ws, iterations, repeats):
    for _ in range(3):
        plan.run(x, out=out, workspace=ws)
    samples = [time_q5(plan, x, out, ws, iterations=iterations) for _ in range(repeats)]
    return dict(
        samples=samples,
        **{
            metric: statistics.median(s[metric] for s in samples)
            for metric in ("gpu_ms", "wall_ms", "cpu_ms")
        },
    )


def dynamic_trace(resident, xs, *, budget, iterations, repeats):
    rng = random.Random(731)
    trace = list(xs) * 3
    rng.shuffle(trace)
    ds = {
        mode: Q5Dispatcher(resident, xs, route=mode, workspace_limit_bytes=budget)
        for mode in ("cublas", "adaptive")
    }
    ws = {
        mode: torch.empty(d.workspace_bytes, device="cuda", dtype=torch.uint8)
        if d.workspace_bytes
        else None
        for mode, d in ds.items()
    }
    outs = {
        m: torch.empty(m, resident.out_features, device="cuda", dtype=torch.float16)
        for m in xs
    }
    results = {mode: [] for mode in ds}
    try:
        for d in ds.values():
            for m in trace:
                d.run(
                    xs[m],
                    out=outs[m],
                    workspace=ws["cublas"] if d.workspace_bytes else None,
                )
        for rep in range(repeats):
            for mode in (
                ("cublas", "adaptive") if rep % 2 == 0 else ("adaptive", "cublas")
            ):
                d = ds[mode]
                torch.cuda.synchronize()
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                wall, cpu = time.perf_counter(), time.process_time()
                start.record()
                for _ in range(iterations):
                    for m in trace:
                        d.run(xs[m], out=outs[m], workspace=ws[mode])
                end.record()
                end.synchronize()
                calls = iterations * len(trace)
                results[mode].append(
                    dict(
                        gpu_ms=start.elapsed_time(end) / calls,
                        wall_ms=(time.perf_counter() - wall) * 1000 / calls,
                        cpu_ms=(time.process_time() - cpu) * 1000 / calls,
                    )
                )
        return dict(
            trace=trace,
            routes=ds["adaptive"].routes,
            samples=results,
            medians={
                mode: {
                    metric: statistics.median(s[metric] for s in samples)
                    for metric in ("gpu_ms", "wall_ms", "cpu_ms")
                }
                for mode, samples in results.items()
            },
        )
    finally:
        torch.cuda.synchronize()
        for d in ds.values():
            d.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tokens", default="1,2,4,8,16,32,64,128,256")
    p.add_argument("--shapes", default="5120x5120,17408x5120,5120x17408")
    p.add_argument("--workspace-mib", type=int, default=64)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--profile-only", action="store_true")
    p.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional existing Q5 safetensors; use matching real matrices without requantizing",
    )
    a = p.parse_args()
    if min(a.iterations, a.repeats, a.workspace_mib) <= 0:
        p.error("positive benchmark counts and workspace required")
    torch.set_num_threads(4)
    torch.manual_seed(731)
    backend = CudaBackend(a.library)
    if backend.architecture != 70 or torch.cuda.get_device_capability() != (7, 0):
        raise ValueError("This campaign is SM70 only")
    budget = a.workspace_mib * 1024**2
    report = dict(
        status="running",
        policy="f16-decode/f16-activation/f32-accum/f16-output",
        seed=731,
        torch=torch.__version__,
        device=str(torch.cuda.get_device_properties(0)),
        library_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),
        workspace_limit_bytes=budget,
        iterations=a.iterations,
        repeats=a.repeats,
        checkpoint="synthetic reproducible Q5 matrices, not a real-model TPS campaign",
        main_baseline="bounded per-call Q5 reconstruction + Tensor Core cuBLAS",
        resident_control="same decoded Q5 FP16 values resident; reconstruction excluded",
        profile_only=a.profile_only,
        cases=[],
    )
    if a.checkpoint:
        from safetensors import safe_open

        with safe_open(a.checkpoint, framework="pt", device="cpu") as f:
            if (f.metadata() or {}).get("venturi.matrix_format") != "q5_g64_fp16":
                raise ValueError("Explicit Q5 checkpoint required")
            report["checkpoint"] = dict(
                path=str(a.checkpoint.resolve()),
                bytes=a.checkpoint.stat().st_size,
                metadata_sha256=hashlib.sha256(
                    json.dumps(f.metadata(), sort_keys=True).encode()
                ).hexdigest(),
            )
    a.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        a.output.write_text(json.dumps(report, indent=2) + "\n")

    with Telemetry() as telemetry:
        try:
            for shape in a.shapes.split(","):
                n, k = map(int, shape.split("x"))
                matrix_key = None
                if a.checkpoint:
                    with safe_open(a.checkpoint, framework="pt", device="cpu") as f:
                        candidates = [
                            key
                            for key in f.keys()
                            if key.endswith(".codes")
                            and f.get_slice(key).get_shape()
                            == [n, (k + 127) // 128 * 64]
                        ]
                        if not candidates:
                            raise ValueError(f"No real Q5 matrix matching N={n}, K={k}")
                        matrix_key = candidates[0][:-6]
                        weight = Q5RowSplitWeight(
                            f.get_tensor(matrix_key + ".codes"),
                            f.get_tensor(matrix_key + ".high"),
                            f.get_tensor(matrix_key + ".scales"),
                            k,
                        )
                else:
                    source = torch.randn(n, k) * 0.025
                    weight = Q5RowSplitWeight.from_dense(source)
                    del source
                resident = backend.upload_linear(
                    weight, dtype=torch.float16, device="cuda:0"
                )
                dense = DenseControl(backend, weight)
                dense_ws = torch.empty(BLAS_BYTES, device="cuda", dtype=torch.uint8)
                case = dict(
                    n=n,
                    k=k,
                    weight_bytes=weight.nbytes,
                    resident_fp16_bytes=n * k * 2,
                    matrix_key=matrix_key,
                    identity=resident._payload.identity,
                    loads=[],
                )
                report["cases"].append(case)
                xs = {}
                try:
                    for m in map(int, a.tokens.split(",")):
                        calibration_x = (torch.randn(m, k) * 0.2).half().cuda()
                        record = tune_q5(
                            resident,
                            calibration_x,
                            workspace_limit_bytes=budget,
                            warmup=3,
                            iterations=a.iterations,
                            repeats=a.repeats,
                        )
                        del calibration_x
                        # Held-out data: selection and final performance never use calibration timing.
                        x = (torch.randn(m, k) * 0.2).half().cuda()
                        xs[m] = x
                        out = torch.empty(m, n, device="cuda", dtype=torch.float16)
                        oracle = torch.empty_like(out)
                        dense.run(x, out=oracle, workspace=dense_ws)
                        torch.cuda.synchronize()
                        plans = {
                            mode: prepare_q5(
                                resident,
                                tokens=m,
                                route=mode,
                                workspace_limit_bytes=budget,
                            )
                            for mode in ("cublas", "adaptive")
                        }
                        scratch = {
                            mode: torch.empty(
                                v.workspace_bytes, device="cuda", dtype=torch.uint8
                            )
                            if v.workspace_bytes
                            else None
                            for mode, v in plans.items()
                        }
                        load = dict(
                            m=m,
                            calibration=record,
                            selected=plans["adaptive"].kernel,
                            timings={},
                            errors={},
                        )
                        case["loads"].append(load)
                        try:
                            for mode, plan in plans.items():
                                plan.run(x, out=out, workspace=scratch[mode])
                                torch.cuda.synchronize()
                                torch.testing.assert_close(
                                    out, oracle, rtol=0.002, atol=0.002
                                )
                                load["errors"][mode] = float(
                                    (out.float() - oracle.float()).abs().max()
                                )
                            samples = {mode: [] for mode in plans}
                            for rep in range(a.repeats):
                                for mode in (
                                    ("cublas", "adaptive")
                                    if rep % 2 == 0
                                    else ("adaptive", "cublas")
                                ):
                                    samples[mode].append(
                                        time_q5(
                                            plans[mode],
                                            x,
                                            out,
                                            scratch[mode],
                                            iterations=a.iterations,
                                        )
                                    )
                            for mode, values in samples.items():
                                load["timings"][mode] = dict(
                                    samples=values,
                                    **{
                                        metric: statistics.median(
                                            s[metric] for s in values
                                        )
                                        for metric in ("gpu_ms", "wall_ms", "cpu_ms")
                                    },
                                )
                            load["timings"]["resident_cublas_control"] = repeated(
                                dense, x, out, dense_ws, a.iterations, a.repeats
                            )
                            load["speedup_vs_bounded_cublas"] = (
                                load["timings"]["cublas"]["gpu_ms"]
                                / load["timings"]["adaptive"]["gpu_ms"]
                            )
                            load["speedup_vs_resident_cublas"] = (
                                load["timings"]["resident_cublas_control"]["gpu_ms"]
                                / load["timings"]["adaptive"]["gpu_ms"]
                            )
                            print(
                                json.dumps(
                                    dict(
                                        n=n,
                                        k=k,
                                        m=m,
                                        route=load["selected"],
                                        bounded_speedup=round(
                                            load["speedup_vs_bounded_cublas"], 3
                                        ),
                                        resident_speedup=round(
                                            load["speedup_vs_resident_cublas"], 3
                                        ),
                                    )
                                ),
                                flush=True,
                            )
                            save()
                        finally:
                            torch.cuda.synchronize()
                            for plan in plans.values():
                                plan.close()
                    case["dynamic"] = dynamic_trace(
                        resident,
                        xs,
                        budget=budget,
                        iterations=max(1, a.iterations // 2),
                        repeats=a.repeats,
                    )
                finally:
                    torch.cuda.synchronize()
                    dense.close()
                    del dense, resident, weight, xs, dense_ws
                    torch.cuda.empty_cache()
            report["status"] = "complete"
        except Exception as e:
            report["status"], report["error"] = "failed", repr(e)
            raise
        finally:
            report["telemetry"], report["telemetry_errors"] = (
                telemetry.samples,
                telemetry.errors,
            )
            save()


if __name__ == "__main__":
    main()
