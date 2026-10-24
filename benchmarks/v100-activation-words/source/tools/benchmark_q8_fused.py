"""Matched same-input projection groups; not serving TPS or full-model quality.

All unfused arms quantize exactly once, then reuse that explicit packed input.
This compares complete row dots, not partial reductions or duplicated packing.
"""

import argparse
import gc
import hashlib
import json
import resource
import statistics
import subprocess
from pathlib import Path
from time import perf_counter, time
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from q8_dp4a_reference import projection_reference  # noqa: E402
from venturi.models import QwenQ8Weights  # noqa: E402
from venturi.ninfer import NInferArtifact  # noqa: E402
from venturi.ops import CudaBackend, Q8RowSplitWeight, prepare_q8_dp4a  # noqa: E402
from venturi.ops.q8_shared import prepare_q8_shared, prepare_q8_tiled_shared  # noqa: E402
from venturi.ops.q8_fused import prepare_q8_fused  # noqa: E402


def base(p):
    return p.base if hasattr(p, "base") else p


def current_plan(r, t):
    p = prepare_q8_dp4a(r, tokens=t, residual_compensation=True)
    if t == 2 and p.workspace_bytes <= 48 * 1024:
        return prepare_q8_shared(r, tokens=t, rows=16, residual_compensation=True)
    if t >= 3:
        wide = t >= 8 and r.out_features >= 1024
        return prepare_q8_tiled_shared(
            r,
            tokens=t,
            rows=16 if wide else 8,
            token_tile=8 if wide else 4,
            residual_compensation=True,
        )
    return p


def quantize(p, x, w, stream):
    p = base(p)
    r = p.resident
    r.backend._check(
        p._native.quantize(
            x.data_ptr(),
            w.data_ptr(),
            p.workspace_bytes,
            p.tokens,
            r.hidden_size,
            p.passes,
            stream,
        )
    )


def packed(p, out, w, stream):
    b = base(p)
    r = b.resident
    args = [
        w.data_ptr(),
        b.workspace_bytes,
        r._codes.data_ptr(),
        r._scales.data_ptr(),
        out.data_ptr(),
        b.tokens,
        r.vocab_size,
        r.hidden_size,
        r._codes.shape[1],
        b.passes,
    ]
    if hasattr(p, "base"):
        args.append(p.rows)
        if p._tile is not None:
            args.append(p._tile)
        r.backend._check(p._run(*args, stream))
    else:
        r.backend._check(p._native.run(*args, stream))


def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact")
    p.add_argument("--library", required=True)
    p.add_argument("--baseline-library", required=True)
    p.add_argument("--output-directory", required=True)
    p.add_argument("--profile", action="store_true")
    a = p.parse_args()
    out = Path(a.output_directory)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    weights = QwenQ8Weights.from_ninfer(NInferArtifact(a.artifact))
    backends = [CudaBackend(a.baseline_library), CudaBackend(a.library)]
    groups = [
        ("mlp", ["mlp/gate", "mlp/up"]),
        ("gdn", ["gdn/qkv", "gdn/z"]),
        ("attention", ["attention/q", "attention/k", "attention/v"]),
    ]
    report = dict(
        scope=__doc__,
        profiled=a.profile,
        artifact_id=weights.artifact.artifact_id,
        library_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),
        baseline_library_sha256=hashlib.sha256(
            Path(a.baseline_library).read_bytes()
        ).hexdigest(),
        precision="Original Q8/F16-scale weights; unchanged residual A8/F32 arithmetic; FP16 input/output",
        tokens=[1, 2, 8] if a.profile else [1, 2, 4, 8, 23, 32],
        arms=["control-unfused", "candidate-unfused", "candidate-fused"],
        cases=[],
        completed=False,
    )

    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

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
    try:
        for label, suffixes in groups:
            spec = next(
                s
                for s in weights.plan.bindings
                if s.binding.endswith("/" + suffixes[0])
            )
            prefix = spec.binding[: -len(suffixes[0])]
            sources = []
            for suffix in suffixes:
                src = weights.read_linear(prefix + suffix)
                half = (src.codes.shape[0] + 1) // 2
                sources.append(
                    Q8RowSplitWeight(
                        src.codes[:half].clone(),
                        src.scales[:half].clone(),
                        src.in_features,
                    )
                )
            residents = [
                [
                    b.upload_linear(src, dtype=torch.float16, device="cuda:0")
                    for src in sources
                ]
                for b in backends
            ]
            k = residents[0][0].in_features
            ns = [r.out_features for r in residents[0]]
            for t in report["tokens"]:
                x_cpu = torch.randn(
                    t, k, generator=torch.Generator().manual_seed(793 + t)
                ).half()
                x = x_cpu.cuda()
                plans = [[current_plan(r, t) for r in rr] for rr in residents]
                fused = prepare_q8_fused(
                    residents[1], tokens=t, residual_compensation=True
                )
                outputs = [
                    [
                        torch.empty(t, n, dtype=torch.float16, device="cuda:0")
                        for n in ns
                    ]
                    for _ in range(3)
                ]
                work = [
                    torch.empty(
                        fused.workspace_bytes, dtype=torch.uint8, device="cuda:0"
                    )
                    for _ in range(3)
                ]

                def call(i, fp=fused, xx=x, yy=outputs, ww=work, pp=plans):
                    if i == 2:
                        fp.run(xx, outputs=yy[i], workspace=ww[i])
                    else:
                        stream = torch.cuda.current_stream().cuda_stream
                        quantize(pp[i][0], xx, ww[i], stream)
                        for plan, y in zip(pp[i], yy[i]):
                            packed(plan, y, ww[i], stream)

                for i in range(3):
                    call(i)
                torch.cuda.synchronize()
                values = [[y.cpu() for y in ys] for ys in outputs]
                assert all(
                    torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                    for arm in values[1:]
                    for a, b in zip(values[0], arm)
                )
                assert torch.equal(work[0], work[1]) and torch.equal(work[1], work[2])
                errors = []
                for s, v in zip(sources, values[2]):
                    rows = list(
                        dict.fromkeys(
                            [0, len(s.codes) - 1]
                            + torch.randint(
                                len(s.codes),
                                (30,),
                                generator=torch.Generator().manual_seed(533),
                            ).tolist()
                        )
                    )
                    ref = projection_reference(x_cpu, s.codes[rows], s.scales[rows], 2)
                    actual = v[:, rows].double()
                    error = float((actual - ref).norm() / (ref.norm() + 1e-12))
                    assert error < 0.001
                    errors.append(error)
                case = dict(
                    group=label,
                    bindings=[prefix + s for s in suffixes],
                    shard_rows=ns,
                    k=k,
                    tokens=t,
                    output_and_workspace_bytes_equal=True,
                    independent_fp64_relative_l2=errors,
                    waves=[],
                )
                if a.profile:
                    with torch.cuda.nvtx.range("native_unsplit"):
                        for i, name in enumerate(report["arms"]):
                            with torch.cuda.nvtx.range(
                                f"projection_{label}_{name}_t{t}"
                            ):
                                for _ in range(16):
                                    call(i)
                            torch.cuda.synchronize()
                else:
                    graphs = []
                    for i in range(3):
                        stream = torch.cuda.Stream()
                        stream.wait_stream(torch.cuda.current_stream())
                        g = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g, stream=stream):
                            for _ in range(3):
                                call(i)
                        torch.cuda.current_stream().wait_stream(stream)
                        graphs.append(g)
                    for g in graphs:
                        for _ in range(3):
                            g.replay()
                    torch.cuda.synchronize()
                    for repeat in range(3):
                        for i in range(3) if repeat % 2 == 0 else reversed(range(3)):
                            start, end = (
                                torch.cuda.Event(enable_timing=True),
                                torch.cuda.Event(enable_timing=True),
                            )
                            torch.cuda.synchronize()
                            unix, wall, c = time(), perf_counter(), cpu()
                            start.record()
                            for _ in range(3):
                                graphs[i].replay()
                            end.record()
                            end.synchronize()
                            case["waves"].append(
                                dict(
                                    arm=report["arms"][i],
                                    repeat=repeat,
                                    calls=9,
                                    start_unix=unix,
                                    end_unix=time(),
                                    gpu_event_ms_per_call=start.elapsed_time(end) / 9,
                                    wall_seconds=perf_counter() - wall,
                                    cpu_core_seconds=cpu() - c,
                                )
                            )
                    case["median_gpu_event_ms"] = {
                        name: statistics.median(
                            w["gpu_event_ms_per_call"]
                            for w in case["waves"]
                            if w["arm"] == name
                        )
                        for name in report["arms"]
                    }
                report["cases"].append(case)
                save()
                print(json.dumps(case), flush=True)
                del call, plans, outputs, work, fused, x, values
                if not a.profile:
                    del graphs, g, stream
                gc.collect()
            del residents, sources
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
