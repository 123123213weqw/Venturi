"""Bounded SM70 projection experiment, not full-model/serving performance.

Reads the production GGUF Q8_0 bytes without dequantizing/requantizing weights.
All arms share encoded GPU weight/activation planes and explicit FP16/F32
outputs. The production service remains running. No default kernel dispatch is
changed. CUDA-graph microbenchmarks are separate from eager counter profiling.
"""

import argparse
import ctypes as C
from datetime import datetime, timezone
import gc
import hashlib
import json
import mmap
from pathlib import Path
import statistics
import struct
import subprocess
import threading
from time import monotonic, perf_counter, process_time, sleep
import urllib.request

import numpy as np
import torch


class Descriptor(C.Structure):
    _fields_ = [(x, C.c_void_p) for x in ("codes", "scales", "out")] + [("n", C.c_int), ("stride", C.c_int)]


def save(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


class GGUF:
    def __init__(self, path):
        self.file = open(path, "rb")
        self.path = Path(path)
        self.scalar = {0:"B", 1:"b", 2:"H", 3:"h", 4:"I", 5:"i", 6:"f", 7:"?", 10:"Q", 11:"q", 12:"d"}
        assert self.file.read(4) == b"GGUF" and self.u("I") in (2, 3)
        nt, nk = self.u("Q"), self.u("Q")
        assert nt < 100000 and nk < 10000
        self.meta = {}
        for _ in range(nk):
            key = self.string()
            self.meta[key] = self.value(self.u("I"), not key.startswith("tokenizer."))
        self.tensors = {}
        for _ in range(nt):
            name, nd = self.string(), self.u("I")
            assert 0 < nd <= 4
            dims = [self.u("Q") for _ in range(nd)]
            self.tensors[name] = dict(dimensions=dims, type=self.u("I"), offset=self.u("Q"))
        align = self.meta.get("general.alignment", 32)
        self.data = (self.file.tell() + align - 1) // align * align
        self.mapping = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)

    def u(self, fmt):
        return struct.unpack("<" + fmt, self.file.read(struct.calcsize("<" + fmt)))[0]

    def string(self):
        size = self.u("Q")
        assert size < 2**31
        return self.file.read(size).decode()

    def value(self, kind, keep):
        if kind == 8:
            if keep:
                return self.string()
            self.file.seek(self.u("Q"), 1)
            return None
        if kind == 9:
            elem, count = self.u("I"), self.u("Q")
            if not keep and elem in self.scalar:
                self.file.seek(struct.calcsize("<" + self.scalar[elem]) * count, 1)
                return None
            values = [self.value(elem, keep and count <= 256) for _ in range(count)]
            return values if keep and count <= 256 else None
        return self.u(self.scalar[kind])

    def q8(self, name, rows=None, columns=None):
        spec = self.tensors[name]
        k, n = spec["dimensions"]
        assert spec["type"] == 8 and k % 32 == 0, "Exact GGUF Q8_0 blocks required"
        blocks = np.ndarray((n, k // 32), dtype=np.dtype([("d", "<f2"), ("q", "i1", (32,))]),
                            buffer=self.mapping, offset=self.data + spec["offset"])
        rows = slice(None) if rows is None else rows
        start, end = (0, k) if columns is None else columns
        assert start % 32 == end % 32 == 0
        selected = blocks[rows, start // 32:end // 32]
        codes = torch.from_numpy(selected["q"].copy().reshape(-1, end - start))
        scales = torch.from_numpy(selected["d"].copy())
        assert codes.dtype == torch.int8 and scales.dtype == torch.float16
        return codes, scales


def bind(path):
    lib = C.CDLL(str(path))
    signatures = {
        "venturi_q8_dp4a_quantize": [C.c_void_p, C.c_void_p, C.c_uint64] + [C.c_int] * 3 + [C.c_void_p],
        "venturi_q8_fused_run": [C.c_void_p, C.c_uint64, C.POINTER(Descriptor)] + [C.c_int] * 5 + [C.c_void_p],
        "venturi_q8_column_partial_run": [C.c_void_p, C.c_uint64] + [C.c_void_p] * 3 + [C.c_int] * 4 + [C.c_void_p],
        "venturi_q8_prefetch_fused_run": [C.c_void_p, C.c_uint64, C.POINTER(Descriptor)] + [C.c_int] * 5 + [C.c_void_p],
        "venturi_q8_prefetch_partial_run": [C.c_void_p, C.c_uint64] + [C.c_void_p] * 3 + [C.c_int] * 5 + [C.c_void_p],
    }
    for name, args in signatures.items():
        if hasattr(lib, name):
            f = getattr(lib, name)
            f.argtypes, f.restype = args, C.c_int
    return lib


def check(result):
    if result:
        raise RuntimeError(f"Explicit native prefetch call failed: {result}")


def idle():
    for _ in range(60):
        slots = json.load(urllib.request.urlopen("http://127.0.0.1:8000/slots", timeout=5))
        if slots and not any(row.get("is_processing") for row in slots):
            return
        sleep(2)
    raise RuntimeError("Production busy; operator experiment stopped without interrupting it")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gguf", required=True)
    p.add_argument("--library", required=True)
    p.add_argument("--qualified-library", required=True)
    p.add_argument("--output-directory", required=True)
    p.add_argument("--tokens", default="8,23,32")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--profile-case", choices=("mlp-fused", "mlp-down", "gdn-fused", "gdn-out", "attention-fused"))
    p.add_argument("--profile-arm", choices=("qualified", "probe-control", "early", "ahead", "row-pair"), default="qualified")
    a = p.parse_args()
    out = Path(a.output_directory)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    assert torch.cuda.get_device_capability() == (7, 0) and a.repeats >= 3
    idle()
    assert torch.cuda.mem_get_info()[0] > 2 * 2**30
    gguf, probe, qualified = GGUF(a.gguf), bind(a.library), bind(a.qualified_library)
    tokens = [int(v) for v in a.tokens.split(",")]
    assert tokens and all(1 <= t <= 184 for t in tokens)
    names = ["qualified", "probe-control", "early", "ahead", "row-pair"]
    report = dict(scope=__doc__, utc=datetime.now(timezone.utc).isoformat(), args=vars(a),
                  profiled=bool(a.profile_case), cases=[], completed=False,
                  gguf_architecture=gguf.meta["general.architecture"],
                  library_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),
                  qualified_library_sha256=hashlib.sha256(Path(a.qualified_library).read_bytes()).hexdigest(),
                  device_uuid=str(torch.cuda.get_device_properties(0).uuid),
                  precision="Unmodified production Q8_0 codes/F16 scales; synthetic F16 activations, two residual A8 passes/F32 scales; output F16 or F32 partials as declared. Not native full-model Qwen3.8 qualification.")
    stop, samples = threading.Event(), []

    def sample():
        while not stop.is_set():
            try:
                r = subprocess.run(["nvidia-smi", "-i", "0", "--query-gpu=index,uuid,memory.used,utilization.gpu,power.draw,clocks.sm,clocks.mem,temperature.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
                samples.append(dict(monotonic=monotonic(), returncode=r.returncode, csv=r.stdout.strip()))
            except Exception as e:
                samples.append(dict(monotonic=monotonic(), error=repr(e)))
            stop.wait(0.2)

    sampler = threading.Thread(target=sample)
    sampler.start()
    cases = [
        ("mlp-fused", [("blk.0.ffn_gate.weight", slice(0,8704), None), ("blk.0.ffn_up.weight", slice(0,8704), None)]),
        ("mlp-down", [("blk.0.ffn_down.weight", None, (0,8704))]),
        ("gdn-fused", [("blk.0.attn_qkv.weight", list(range(1024))+list(range(2048,3072))+list(range(4096,7168)), None), ("blk.0.attn_gate.weight", slice(0,3072), None)]),
        ("gdn-out", [("blk.0.ssm_out.weight", None, (0,3072))]),
        ("attention-fused", [("blk.3.attn_q.weight", slice(0,6144), None), ("blk.3.attn_k.weight", slice(0,512), None), ("blk.3.attn_v.weight", slice(0,512), None)]),
    ]
    try:
        for label, specs in cases:
            if a.profile_case and label != a.profile_case:
                continue
            idle()
            cpu_weights = [gguf.q8(*spec) for spec in specs]
            weights = [(q.cuda(), s.cuda()) for q, s in cpu_weights]
            partial = len(specs) == 1
            k, ns = weights[0][0].shape[1], [q.shape[0] for q, _ in weights]
            for t in tokens:
                idle()
                xcpu = (torch.randn(t, k, generator=torch.Generator().manual_seed(571 + t)) * 0.25).half()
                x = xcpu.cuda()
                size = t * ((k + 31) // 32) * 36 * 2
                work = torch.empty(size, device="cuda", dtype=torch.uint8)
                ys = [[torch.empty((t,n), device="cuda", dtype=torch.float32 if partial else torch.float16) for n in ns] for _ in names]
                desc = [(Descriptor * len(ns))(*[Descriptor(q.data_ptr(), s.data_ptr(), y.data_ptr(), n, n) for (q,s),y,n in zip(weights, outputs, ns)]) for outputs in ys]
                stream = torch.cuda.current_stream().cuda_stream
                check(probe.venturi_q8_dp4a_quantize(x.data_ptr(), work.data_ptr(), size, t, k, 2, stream))

                def call(i):
                    lib = qualified if i == 0 else probe
                    if partial:
                        args = [work.data_ptr(), size, weights[0][0].data_ptr(), weights[0][1].data_ptr(), ys[i][0].data_ptr(), t, ns[0], k, k]
                        fn = lib.venturi_q8_column_partial_run if i < 2 else lib.venturi_q8_prefetch_partial_run
                        if i >= 2:
                            args.append(i - 1)
                    else:
                        args = [work.data_ptr(), size, desc[i], len(ns), t, k, k, 2 if i < 2 else i - 1]
                        fn = lib.venturi_q8_fused_run if i < 2 else lib.venturi_q8_prefetch_fused_run
                    check(fn(*args, stream))

                for i in range(len(names)):
                    call(i)
                torch.cuda.synchronize()
                assert all(torch.equal(ref.view(torch.uint8), val.view(torch.uint8)) for arm in ys[1:] for ref,val in zip(ys[0],arm)), "Output bytes differ from qualified native operator"
                from q8_dp4a_reference import projection_reference
                errors = []
                for (q,s), result in zip(cpu_weights, ys[0]):
                    rows = sorted(set([0,len(q)-1]+torch.randint(len(q),(24,),generator=torch.Generator().manual_seed(62)).tolist()))
                    ref = projection_reference(xcpu, q[rows], s[rows], 2)
                    error = float((result[:,rows].cpu().double()-ref).norm()/(ref.norm()+1e-12))
                    assert error < 0.001
                    errors.append(error)
                case = dict(name=label, tokens=t, k=k, rows=ns, partial_fp32=partial,
                            complete_output_bytes_equal=True, sampled_fp64_relative_l2=errors,
                            additional_allocated_bytes=torch.cuda.memory_allocated(), additional_reserved_bytes=torch.cuda.memory_reserved(), waves=[])
                assert case["additional_reserved_bytes"] < 1024**3
                if a.profile_case:
                    i = names.index(a.profile_arm)
                    for _ in range(16):
                        call(i)
                    torch.cuda.synchronize()
                else:
                    graphs = []
                    for i in range(len(names)):
                        side = torch.cuda.Stream()
                        side.wait_stream(torch.cuda.current_stream())
                        g = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g, stream=side):
                            for _ in range(20):
                                # Capture the explicit side stream, not the original one.
                                stream = side.cuda_stream
                                call(i)
                        torch.cuda.current_stream().wait_stream(side)
                        graphs.append(g)
                    stream = torch.cuda.current_stream().cuda_stream
                    for g in graphs:
                        for _ in range(5):
                            g.replay()
                    torch.cuda.synchronize()
                    for repeat in range(a.repeats):
                        idle()
                        order = list(range(len(names))) if repeat % 2 == 0 else list(reversed(range(len(names))))
                        for i in order:
                            torch.cuda.synchronize()
                            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            wall, cpu = perf_counter(), process_time()
                            start.record()
                            for _ in range(5):
                                graphs[i].replay()
                            end.record();end.synchronize()
                            case["waves"].append(dict(arm=names[i], repeat=repeat, calls=100, start_monotonic=wall,
                                                       end_monotonic=perf_counter(), gpu_event_ms_per_call=start.elapsed_time(end)/100,
                                                       wall_seconds=perf_counter()-wall, process_cpu_seconds=process_time()-cpu))
                    case["median_ms"] = {name:statistics.median(w["gpu_event_ms_per_call"] for w in case["waves"] if w["arm"]==name) for name in names}
                    case["speedup_vs_qualified"] = {name:case["median_ms"]["qualified"]/case["median_ms"][name] for name in names}
                    del graphs, g, side
                report["cases"].append(case)
                save(out/"report.json",report)
                print(json.dumps({key:case[key] for key in ("name","tokens","k","rows","median_ms","speedup_vs_qualified") if key in case}), flush=True)
                del call, x, xcpu, work, ys, desc
                gc.collect();torch.cuda.empty_cache()
            del weights,cpu_weights
            gc.collect();torch.cuda.empty_cache()
        report["completed"] = True
        save(out/"report.json",report)
    finally:
        stop.set();sampler.join(timeout=8)
        save(out/"gpu-telemetry.json",samples)


if __name__ == "__main__":
    main()
