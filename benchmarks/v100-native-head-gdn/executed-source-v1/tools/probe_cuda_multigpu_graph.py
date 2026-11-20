"""Probe two-GPU host-bridge graph capture; not a model or serving benchmark."""

import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import resource
from time import perf_counter

import torch
from venturi.ops import CudaBackend, Q8RowSplitWeight, prepare_q8_dp4a


class CudaError(RuntimeError):
    def __init__(self, call, code, name):
        self.code, self.name = code, name
        super().__init__(f"{call}: {name} ({code})")


class Runtime:
    def __init__(self):
        self.dll = C.CDLL("/usr/local/cuda/lib64/libcudart.so.12")
        v, i, u = C.c_void_p, C.c_int, C.c_uint
        signatures = {
            "cudaSetDevice": [i],
            "cudaStreamCreateWithFlags": [C.POINTER(v), u],
            "cudaEventCreateWithFlags": [C.POINTER(v), u],
            "cudaEventRecord": [v, v],
            "cudaStreamWaitEvent": [v, v, u],
            "cudaMemcpyAsync": [v, v, C.c_size_t, i, v],
            "cudaStreamSynchronize": [v],
            "cudaStreamBeginCapture": [v, i],
            "cudaStreamEndCapture": [v, C.POINTER(v)],
            "cudaGraphInstantiateWithFlags": [C.POINTER(v), v, C.c_ulonglong],
            "cudaGraphLaunch": [v, v],
            "cudaGraphDestroy": [v],
            "cudaGraphExecDestroy": [v],
            "cudaEventDestroy": [v],
            "cudaStreamDestroy": [v],
            "cudaGetLastError": [],
        }
        for name, args in signatures.items():
            fn = getattr(self.dll, name)
            fn.argtypes, fn.restype = args, i
        self.dll.cudaGetErrorName.argtypes = [i]
        self.dll.cudaGetErrorName.restype = C.c_char_p

    def call(self, name, *args):
        code = getattr(self.dll, name)(*args)
        if code:
            raise CudaError(name, code, self.dll.cudaGetErrorName(code).decode())


def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    if torch.cuda.device_count() != 2:
        raise ValueError("Exactly two explicitly selected CUDA devices required")
    rt = Runtime()
    streams, events, external = [], [], []
    graph, executable = C.c_void_p(), C.c_void_p()
    report = dict(
        scope=__doc__,
        completed=False,
        capture_supported=False,
        tokens=3,
        k=128,
        output_rows_per_device=17,
        logical_requests=None,
        model_quality_qualified=False,
        physical_gpus=[1, 0],
        transport="pinned host, both ownership fences",
        library_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),
        achieved_hardware_counters=None,
        trials=[],
    )
    capturing = False
    try:
        for device in [0, 1]:
            rt.call("cudaSetDevice", device)
            stream, event = C.c_void_p(), C.c_void_p()
            rt.call("cudaStreamCreateWithFlags", C.byref(stream), 1)
            streams.append(stream)
            rt.call("cudaEventCreateWithFlags", C.byref(event), 2)
            events.append(event)
            external.append(torch.cuda.ExternalStream(stream.value, device=device))
        backend = CudaBackend(a.library)
        generator = torch.Generator().manual_seed(843)
        weights = [
            Q8RowSplitWeight(
                torch.randint(
                    -127, 128, (17, 128), dtype=torch.int8, generator=generator
                ),
                torch.full((17, 4), 0.01, dtype=torch.float16),
                128,
            )
            for _ in range(2)
        ]
        resident = [
            backend.upload_linear(w, dtype=torch.float16, device=f"cuda:{i}")
            for i, w in enumerate(weights)
        ]
        plans = [
            prepare_q8_dp4a(r, tokens=3, residual_compensation=True) for r in resident
        ]
        xs = [
            torch.empty((3, 128), dtype=torch.float16, device=f"cuda:{i}")
            for i in [0, 1]
        ]
        ys = [
            torch.empty((3, 17), dtype=torch.float16, device=f"cuda:{i}")
            for i in [0, 1]
        ]
        work = [
            torch.empty(p.workspace_bytes, dtype=torch.uint8, device=f"cuda:{i}")
            for i, p in enumerate(plans)
        ]
        inbound = torch.empty((3, 128), dtype=torch.float16, pin_memory=True)
        outbound = torch.empty((3, 17), dtype=torch.float16, pin_memory=True)
        gathered = torch.empty((3, 17), dtype=torch.float16, device="cuda:0")
        xcpu = torch.randn((3, 128), generator=generator).half()
        xs[0].copy_(xcpu)
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)

        def flow():
            rt.call("cudaSetDevice", 0)
            rt.call(
                "cudaMemcpyAsync",
                inbound.data_ptr(),
                xs[0].data_ptr(),
                inbound.numel() * 2,
                2,
                streams[0],
            )
            rt.call("cudaEventRecord", events[0], streams[0])
            rt.call("cudaSetDevice", 1)
            rt.call("cudaStreamWaitEvent", streams[1], events[0], 0)
            rt.call(
                "cudaMemcpyAsync",
                xs[1].data_ptr(),
                inbound.data_ptr(),
                inbound.numel() * 2,
                1,
                streams[1],
            )
            with torch.cuda.device(1), torch.cuda.stream(external[1]):
                plans[1].run(xs[1], out=ys[1], workspace=work[1])
            rt.call(
                "cudaMemcpyAsync",
                outbound.data_ptr(),
                ys[1].data_ptr(),
                outbound.numel() * 2,
                2,
                streams[1],
            )
            rt.call("cudaEventRecord", events[1], streams[1])
            rt.call("cudaSetDevice", 0)
            with torch.cuda.stream(external[0]):
                plans[0].run(xs[0], out=ys[0], workspace=work[0])
            rt.call("cudaStreamWaitEvent", streams[0], events[1], 0)
            rt.call(
                "cudaMemcpyAsync",
                gathered.data_ptr(),
                outbound.data_ptr(),
                outbound.numel() * 2,
                1,
                streams[0],
            )

        for _ in range(3):
            flow()
            rt.call("cudaStreamSynchronize", streams[0])
        expected = [ys[0].cpu(), ys[1].cpu()]
        assert torch.equal(
            gathered.cpu().view(torch.uint8), expected[1].view(torch.uint8)
        )
        rt.call("cudaSetDevice", 0)
        rt.call("cudaStreamBeginCapture", streams[0], 0)
        capturing = True
        flow()
        rt.call("cudaStreamEndCapture", streams[0], C.byref(graph))
        capturing = False
        rt.call("cudaGraphInstantiateWithFlags", C.byref(executable), graph, 0)
        report["capture_supported"] = True
        for repeat in range(3):
            for mode in ["eager", "graph"] if repeat % 2 == 0 else ["graph", "eager"]:
                start, used = perf_counter(), cpu()
                for _ in range(20):
                    if mode == "graph":
                        rt.call("cudaGraphLaunch", executable, streams[0])
                    else:
                        flow()
                rt.call("cudaStreamSynchronize", streams[0])
                report["trials"].append(
                    dict(
                        repeat=repeat,
                        mode=mode,
                        calls=20,
                        wall_seconds=perf_counter() - start,
                        cpu_core_seconds=cpu() - used,
                    )
                )
                assert torch.equal(
                    ys[0].cpu().view(torch.uint8), expected[0].view(torch.uint8)
                )
                assert torch.equal(
                    gathered.cpu().view(torch.uint8), expected[1].view(torch.uint8)
                )
        # Input addresses are fixed, values are not: changing them must affect
        # both devices on replay without a stale host-bridge read.
        xs[0].zero_()
        torch.cuda.synchronize(0)
        rt.call("cudaGraphLaunch", executable, streams[0])
        rt.call("cudaStreamSynchronize", streams[0])
        assert not ys[0].any() and not gathered.any()
        xs[0].copy_(xcpu)
        torch.cuda.synchronize(0)
        rt.call("cudaGraphLaunch", executable, streams[0])
        rt.call("cudaStreamSynchronize", streams[0])
        assert torch.equal(
            gathered.cpu().view(torch.uint8), expected[1].view(torch.uint8)
        )
        report["exact_bytes_and_mutable_input_replay"] = True
        report["model_dynamic_kv_gdn_positions_tested"] = False
    except CudaError as error:
        report["cuda_error"] = dict(
            code=error.code, name=error.name, message=str(error)
        )
        if "StreamCapture" not in error.name:
            raise
        report["capture_supported"] = False
    finally:
        rt.dll.cudaSetDevice(0)
        if capturing:
            rt.dll.cudaStreamEndCapture(streams[0], C.byref(graph))
            rt.dll.cudaGetLastError()
        if executable.value:
            rt.dll.cudaGraphExecDestroy(executable)
        if graph.value:
            rt.dll.cudaGraphDestroy(graph)
        for device, stream in enumerate(streams):
            rt.dll.cudaSetDevice(device)
            rt.dll.cudaStreamSynchronize(stream)
            if device < len(events):
                rt.dll.cudaEventDestroy(events[device])
            rt.dll.cudaStreamDestroy(stream)
    report["completed"] = True
    Path(a.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
