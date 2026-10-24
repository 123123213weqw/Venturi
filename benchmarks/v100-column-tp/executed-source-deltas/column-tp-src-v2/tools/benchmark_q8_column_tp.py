"""Two-V100 Q8 MLP tensor-parallel research benchmark, not model/serving TPS.

Row TP broadcasts/gathers every projection; column/row TP keeps MLP halves
local and reduces FP32 down partials. Inputs are the same replicated normalized
FP16 hidden values. Outputs are primary-only for row TP, replicated for column
TP. There is no norm/residual, cache, attention, draft, scheduler or full model.
"""

import argparse
import ctypes as C
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
from probe_cuda_multigpu_graph import Runtime  # noqa: E402
from venturi.models import QwenQ8Weights  # noqa: E402
from venturi.ninfer import NInferArtifact  # noqa: E402
from venturi.ops import CudaBackend, Q8RowSplitWeight, prepare_q8_dp4a  # noqa: E402
from venturi.ops.q8_column_tp import prepare_q8_column_partial, slice_q8_columns  # noqa: E402
from venturi.ops.q8_fused import prepare_q8_fused  # noqa: E402
from venturi.ops.q8_shared import prepare_q8_tiled_shared  # noqa: E402


def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def row_slice(w, rank):
    half = w.out_features // 2
    return Q8RowSplitWeight(
        w.codes[rank * half : (rank + 1) * half].clone(),
        w.scales[rank * half : (rank + 1) * half].clone(),
        w.in_features,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact")
    p.add_argument("--library", required=True)
    p.add_argument("--output-directory", required=True)
    p.add_argument("--profile", action="store_true")
    a = p.parse_args()
    out = Path(a.output_directory)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    if torch.cuda.device_count() != 2 or any(
        torch.cuda.get_device_capability(i) != (7, 0) for i in [0, 1]
    ):
        raise ValueError("Exactly two explicitly selected V100 GPUs required")
    w = QwenQ8Weights.from_ninfer(NInferArtifact(a.artifact))
    # Layer zero is a trained GDN block, but the MLP layout is shared by both types.
    prefix = next(
        s.binding.rsplit("/", 2)[0]
        for s in w.plan.bindings
        if s.binding.endswith("mlp/gate")
    )
    gate, up, down = [
        w.read_linear(prefix + "/mlp/" + name) for name in ["gate", "up", "down"]
    ]
    h, inter = gate.in_features, gate.out_features
    assert (
        (h, inter) == (5120, 17408)
        and down.in_features == inter
        and down.out_features == h
    )
    backend = CudaBackend(a.library)
    residents = {}
    for mode in ["row", "column"]:
        residents[mode] = []
        for rank in [0, 1]:
            weights = [
                row_slice(gate, rank),
                row_slice(up, rank),
                row_slice(down, rank)
                if mode == "row"
                else slice_q8_columns(
                    down, rank * (inter // 2), (rank + 1) * (inter // 2)
                ),
            ]
            residents[mode].append(
                [
                    backend.upload_linear(v, dtype=torch.float16, device=f"cuda:{rank}")
                    for v in weights
                ]
            )
    rt = Runtime()
    rt.dll.cudaMemcpy2DAsync.argtypes = [
        C.c_void_p,
        C.c_size_t,
        C.c_void_p,
        C.c_size_t,
        C.c_size_t,
        C.c_size_t,
        C.c_int,
        C.c_void_p,
    ]
    rt.dll.cudaMemcpy2DAsync.restype = C.c_int
    streams, events, external = [], [], []
    graphs = []
    capturing = False
    report = dict(
        scope=__doc__,
        completed=False,
        profiled=a.profile,
        artifact_id=w.artifact.artifact_id,
        library_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),
        binding_prefix=prefix,
        hidden=h,
        intermediate=inter,
        precision="Stored Q8/F16 scales, two residual-A8 passes/F32 scales; column down partials+transport FP32, one sum+FP16 cast. Changed FP32 association, not prior-profile byte parity.",
        model_quality_qualified=False,
        full_model_tp_implemented=False,
        achieved_hardware_counters=None,
        cases=[],
    )
    sampler = None
    log = (out / "gpu-samples.csv").open("wb")
    try:
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
        for rank in [0, 1]:
            rt.call("cudaSetDevice", rank)
            stream, event = C.c_void_p(), C.c_void_p()
            rt.call("cudaStreamCreateWithFlags", C.byref(stream), 1)
            rt.call("cudaEventCreateWithFlags", C.byref(event), 2)
            streams.append(stream)
            events.append(event)
            external.append(torch.cuda.ExternalStream(stream.value, device=rank))
        for tokens in [1, 3, 8]:

            def empty(shape, rank=0, dtype=torch.float16, pin=False):
                return torch.empty(
                    shape,
                    dtype=dtype,
                    device="cpu" if pin else f"cuda:{rank}",
                    pin_memory=pin,
                )

            def sync():
                for rank in [0, 1]:
                    rt.call("cudaSetDevice", rank)
                    rt.call("cudaStreamSynchronize", streams[rank])

            def copy(dst, src, kind, rank, nbytes=None):
                rt.call("cudaSetDevice", rank)
                rt.call(
                    "cudaMemcpyAsync",
                    dst.data_ptr(),
                    src.data_ptr(),
                    nbytes or src.numel() * src.element_size(),
                    kind,
                    streams[rank],
                )

            def copy2(dst, src, width, rows, dpitch, spitch, kind, rank, offset=0):
                rt.call("cudaSetDevice", rank)
                rt.call(
                    "cudaMemcpy2DAsync",
                    dst.data_ptr() + offset,
                    dpitch,
                    src.data_ptr(),
                    spitch,
                    width,
                    rows,
                    kind,
                    streams[rank],
                )

            def record(rank):
                rt.call("cudaSetDevice", rank)
                rt.call("cudaEventRecord", events[rank], streams[rank])

            def wait(rank, producer):
                rt.call("cudaSetDevice", rank)
                rt.call("cudaStreamWaitEvent", streams[rank], events[producer], 0)

            xs = [empty((tokens, h), r) for r in [0, 1]]
            xcpu = torch.randn(
                tokens, h, generator=torch.Generator().manual_seed(901)
            ).half()
            for rank in [0, 1]:
                xs[rank].copy_(xcpu)
                torch.cuda.synchronize(rank)
            row_full = [empty((tokens, inter)) for _ in [0, 1]]
            row_peer = empty((2, tokens, inter // 2), 1)
            row_mid = empty((tokens, inter))
            row_peer_mid = empty((tokens, inter), 1)
            row_y = empty((tokens, h))
            row_y_peer = empty((tokens, h // 2), 1)
            row_y_local = row_y[:, : h // 2] if tokens == 1 else empty((tokens, h // 2))
            host_x = empty((tokens, h), pin=True)
            host_gu = empty((2, tokens, inter // 2), pin=True)
            host_mid = empty((tokens, inter), pin=True)
            host_y = empty((tokens, h // 2), pin=True)
            col_gu = [[empty((tokens, inter // 2), r) for _ in [0, 1]] for r in [0, 1]]
            col_mid = [empty((tokens, inter // 2), r) for r in [0, 1]]
            col_partial = [empty((tokens, h), r, torch.float32) for r in [0, 1]]
            col_remote = empty((tokens, h), dtype=torch.float32)
            col_y = [empty((tokens, h), r) for r in [0, 1]]
            host_partial = empty((tokens, h), dtype=torch.float32, pin=True)
            host_sum = empty((tokens, h), pin=True)
            plans = {}
            work = {}
            silu = {}
            for mode in ["row", "column"]:
                plans[mode] = []
                work[mode] = []
                silu[mode] = []
                for rank in [0, 1]:
                    rg, ru, rd = residents[mode][rank]
                    fused = prepare_q8_fused(
                        [rg, ru],
                        tokens=tokens,
                        residual_compensation=True,
                        output_strides=[
                            inter if mode == "row" and rank == 0 else inter // 2
                        ]
                        * 2,
                    )
                    if mode == "column":
                        partial = prepare_q8_column_partial(rd, tokens=tokens)
                    elif tokens == 1:
                        partial = prepare_q8_dp4a(
                            rd, tokens=tokens, residual_compensation=True
                        )
                    else:
                        partial = prepare_q8_tiled_shared(
                            rd,
                            tokens=tokens,
                            rows=16 if tokens >= 8 else 8,
                            token_tile=8 if tokens >= 8 else 4,
                            residual_compensation=True,
                        )
                    plans[mode].append((fused, partial))
                    work[mode].append(
                        [
                            empty((v.workspace_bytes,), rank, torch.uint8)
                            for v in [fused, partial]
                        ]
                    )
                    silu[mode].append(
                        backend.prepare_swiglu(
                            tokens=tokens,
                            features=inter
                            if mode == "row" and rank == 0
                            else inter // 2,
                            dtype=torch.float16,
                            device=f"cuda:{rank}",
                        )
                    )

            def row_flow():
                copy(host_x, xs[0], 2, 0)
                record(0)
                wait(1, 0)
                copy(xs[1], host_x, 1, 1)
                with torch.cuda.device(1), torch.cuda.stream(external[1]):
                    plans["row"][1][0].run(
                        xs[1],
                        outputs=[row_peer[0], row_peer[1]],
                        workspace=work["row"][1][0],
                    )
                copy(host_gu, row_peer, 2, 1)
                record(1)
                with torch.cuda.device(0), torch.cuda.stream(external[0]):
                    plans["row"][0][0].run(
                        xs[0], outputs=row_full, workspace=work["row"][0][0]
                    )
                wait(0, 1)
                for i in [0, 1]:
                    copy2(
                        row_full[i],
                        host_gu[i],
                        inter,
                        tokens,
                        inter * 2,
                        inter,
                        1,
                        0,
                        inter,
                    )
                with torch.cuda.device(0), torch.cuda.stream(external[0]):
                    silu["row"][0].run(*row_full, out=row_mid)
                copy(host_mid, row_mid, 2, 0)
                record(0)
                wait(1, 0)
                copy(row_peer_mid, host_mid, 1, 1)
                with torch.cuda.device(1), torch.cuda.stream(external[1]):
                    plans["row"][1][1].run(
                        row_peer_mid, out=row_y_peer, workspace=work["row"][1][1]
                    )
                copy(host_y, row_y_peer, 2, 1)
                record(1)
                with torch.cuda.device(0), torch.cuda.stream(external[0]):
                    plans["row"][0][1].run(
                        row_mid, out=row_y_local, workspace=work["row"][0][1]
                    )
                if tokens != 1:
                    copy2(row_y, row_y_local, h, tokens, h * 2, h, 3, 0)
                wait(0, 1)
                copy2(row_y, host_y, h, tokens, h * 2, h, 1, 0, h)

            def column_flow():
                record(0)
                wait(1, 0)
                for rank in [1, 0]:
                    with torch.cuda.device(rank), torch.cuda.stream(external[rank]):
                        plans["column"][rank][0].run(
                            xs[rank],
                            outputs=col_gu[rank],
                            workspace=work["column"][rank][0],
                        )
                        silu["column"][rank].run(*col_gu[rank], out=col_mid[rank])
                        plans["column"][rank][1].run(
                            col_mid[rank],
                            out=col_partial[rank],
                            workspace=work["column"][rank][1],
                        )
                    if rank == 1:
                        copy(host_partial, col_partial[1], 2, 1)
                        record(1)
                wait(0, 1)
                copy(col_remote, host_partial, 1, 0)
                with torch.cuda.device(0), torch.cuda.stream(external[0]):
                    plans["column"][0][1].reduce(
                        col_partial[0], col_remote, out=col_y[0]
                    )
                copy(host_sum, col_y[0], 2, 0)
                record(0)
                wait(1, 0)
                copy(col_y[1], host_sum, 1, 1)
                record(1)
                wait(0, 1)

            flows = [row_flow, column_flow]
            for flow in flows:
                flow()
                sync()
            expected = row_y.cpu()
            assert torch.equal(
                torch.cat([v.cpu() for v in col_mid], 1), row_mid.cpu()
            ), "Local SwiGLU halves changed bytes"
            assert torch.equal(col_y[0].cpu(), col_y[1].cpu()), (
                "Replicated reduction outputs disagree"
            )
            rows = sorted(
                {
                    0,
                    h - 1,
                    *torch.randint(
                        h, (30,), generator=torch.Generator().manual_seed(883)
                    ).tolist(),
                }
            )
            ref = projection_reference(
                row_mid.cpu(), down.codes[rows], down.scales[rows], 2
            )
            checks = []
            for name, value in [("row", expected), ("column", col_y[0].cpu())]:
                diff = value[:, rows].double() - ref
                err = float(diff.norm() / (ref.norm() + 1e-20))
                assert err < 0.001
                checks.append(
                    dict(
                        arm=name,
                        relative_l2=err,
                        max_abs=float(diff.abs().max()),
                        rows=rows,
                    )
                )
            difference = expected.double() - col_y[0].cpu().double()
            assert difference.norm() / (expected.double().norm() + 1e-20) < 0.001
            case = dict(
                tokens=tokens,
                fp64_checks=checks,
                local_gate_up_swiglu_bytes_equal=True,
                replicated_outputs_equal=True,
                different_fp16_outputs=int(difference.count_nonzero()),
                difference_l2=float(
                    difference.norm() / (expected.double().norm() + 1e-20)
                ),
                logical_copy_bytes_per_direction_per_call={
                    "row": tokens * (3 * h + 4 * inter),
                    "column": tokens * 6 * h,
                },
                waves=[],
            )
            local_graphs = []
            for flow in flows:
                g, ex = C.c_void_p(), C.c_void_p()
                rt.call("cudaSetDevice", 0)
                rt.call("cudaStreamBeginCapture", streams[0], 0)
                capturing = True
                for _ in range(4):
                    flow()
                rt.call("cudaStreamEndCapture", streams[0], C.byref(g))
                capturing = False
                rt.call("cudaGraphInstantiateWithFlags", C.byref(ex), g, 0)
                graphs.append((g, ex))
                local_graphs.append(ex)
            for ex in local_graphs:
                rt.call("cudaSetDevice", 0)
                for _ in range(3):
                    rt.call("cudaGraphLaunch", ex, streams[0])
                sync()
            for repeat in range(1 if a.profile else 3):
                for mode in [0, 1] if repeat % 2 == 0 else [1, 0]:
                    sync()
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    loops = 3 if a.profile else 100
                    rt.call("cudaSetDevice", 0)
                    with torch.cuda.device(0), torch.cuda.stream(external[0]):
                        began, wall, used = time(), perf_counter(), cpu()
                        torch.cuda.nvtx.range_push(
                            f"column_tp/t{tokens}/" + ["row", "column"][mode]
                        )
                        start.record()
                        for _ in range(loops):
                            rt.call("cudaGraphLaunch", local_graphs[mode], streams[0])
                        end.record()
                        end.synchronize()
                        torch.cuda.nvtx.range_pop()
                    case["waves"].append(
                        dict(
                            arm=["row", "column"][mode],
                            repeat=repeat,
                            calls=loops * 4,
                            start_unix=began,
                            end_unix=time(),
                            gpu_event_ms_per_call=start.elapsed_time(end) / (loops * 4),
                            wall_seconds=perf_counter() - wall,
                            cpu_core_seconds=cpu() - used,
                        )
                    )
                    assert torch.equal(row_y.cpu(), expected)
                    assert torch.equal(col_y[0].cpu(), col_y[1].cpu())
            case["median_gpu_event_ms"] = {
                name: statistics.median(
                    v["gpu_event_ms_per_call"]
                    for v in case["waves"]
                    if v["arm"] == name
                )
                for name in ["row", "column"]
            }
            case["speed_ratio"] = (
                case["median_gpu_event_ms"]["row"]
                / case["median_gpu_event_ms"]["column"]
            )
            report["cases"].append(case)
            (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(case), flush=True)
            sync()
            for g, ex in graphs:
                rt.call("cudaSetDevice", 0)
                rt.call("cudaGraphExecDestroy", ex)
                rt.call("cudaGraphDestroy", g)
            graphs.clear()
            gc.collect()
        report["completed"] = True
    finally:
        # Cleanup must not leave a periodic sampler behind after a CUDA error.
        if capturing:
            orphan = C.c_void_p()
            rt.dll.cudaSetDevice(0)
            rt.dll.cudaStreamEndCapture(streams[0], C.byref(orphan))
            rt.dll.cudaGetLastError()
            if orphan.value:
                rt.dll.cudaGraphDestroy(orphan)
        for rank, stream in enumerate(streams):
            rt.dll.cudaSetDevice(rank)
            rt.dll.cudaStreamSynchronize(stream)
        for g, ex in graphs:
            rt.dll.cudaSetDevice(0)
            rt.dll.cudaGraphExecDestroy(ex)
            rt.dll.cudaGraphDestroy(g)
        for rank, stream in enumerate(streams):
            rt.dll.cudaSetDevice(rank)
            rt.dll.cudaEventDestroy(events[rank])
            rt.dll.cudaStreamDestroy(stream)
        if sampler is not None:
            sampler.terminate()
            try:
                sampler.wait(timeout=15)
            except subprocess.TimeoutExpired:
                sampler.kill()
                sampler.wait()
        log.close()
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(completed=True, cases=len(report["cases"]))))


if __name__ == "__main__":
    main()
