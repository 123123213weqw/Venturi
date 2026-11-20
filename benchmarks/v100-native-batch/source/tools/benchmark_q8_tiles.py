"""Matched Q8 projection-only tile ablation using real checkpoint matrices.

All libraries share the same immutable GPU Q8 planes and activation values.
Every candidate must match original output bytes before timing. Graph replay
amortizes Python launch overhead for this microbenchmark only; it does not imply
whole-model graph support. CUDA-event intervals include graph/timeline gaps and
are not hardware-counter bandwidth or full-engine serving measurements.
"""

import argparse
from collections import Counter
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import resource
import statistics
import subprocess
from time import perf_counter

import torch

from venturi.models import QwenQ8Weights
from venturi.ninfer import NInferArtifact
from venturi.ops import CudaBackend


def cpu_seconds():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def token_counts(role, decode_only, verification_pairs):
    if decode_only:
        return (1,)
    if "embedding" in role or "head" in role:
        return (1, 2) if verification_pairs else (1,)
    return (1, 2, 4, 26) if verification_pairs else (1, 4, 26)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("artifact")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidates", required=True, help="Build-libraries JSON")
    ap.add_argument("--output-directory", required=True)
    ap.add_argument("--gpu-index", type=int, required=True)
    ap.add_argument("--decode-only", action="store_true")
    ap.add_argument(
        "--verification-pairs",
        action="store_true",
        help="Also test two-token target verification, including the output head",
    )
    args = ap.parse_args()
    out = Path(args.output_directory)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    entries = [dict(name="baseline", library=args.baseline)] + [
        dict(
            name=e.get("name", f"t{e['token_tile']}-r{e['row_tile']}"),
            library=e["library"],
        )
        for e in json.loads(Path(args.candidates).read_text())
    ]
    backends = [CudaBackend(e["library"]) for e in entries]
    for e in entries:
        e["sha256"] = hashlib.sha256(Path(e["library"]).read_bytes()).hexdigest()
    weights = QwenQ8Weights.from_ninfer(NInferArtifact(args.artifact))
    counts = Counter(
        s.shape for s in weights.plan.bindings if s.format == "q8_g32_fp16"
    )
    roles = {}
    for spec in weights.plan.bindings:
        if spec.format == "q8_g32_fp16":
            roles[spec.shape] = spec.binding
    report = dict(
        scope=__doc__,
        artifact_id=weights.artifact.artifact_id,
        gpu=torch.cuda.get_device_name(),
        dtype="fp16",
        libraries=entries,
        graph_nodes=5,
        graph_replays_per_wave=5,
        warmup_replays=3,
        decode_only=args.decode_only,
        verification_pairs=args.verification_pairs,
        hardware_counters_available=False,
        cases=[],
    )

    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    log = (out / "gpu-samples.csv").open("wb")
    sampler = subprocess.Popen(
        [
            "nvidia-smi",
            "-i",
            str(args.gpu_index),
            "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu,clocks.sm,clocks.mem",
            "--format=csv,noheader,nounits",
            "-lms",
            "100",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        for shape, role in roles.items():
            source = weights.read_linear(role)
            resident = (
                backends[0]
                .upload_linear(source, dtype=torch.float16, device="cuda:0")
                ._payload
            )
            others = [replace(resident, backend=b) for b in backends]
            for tokens in token_counts(role, args.decode_only, args.verification_pairs):
                x = torch.randn(
                    tokens,
                    shape[1],
                    device="cuda",
                    dtype=torch.float16,
                    generator=torch.Generator(device="cuda").manual_seed(812 + tokens),
                )
                outputs = [
                    torch.empty(tokens, shape[0], device="cuda", dtype=torch.float16)
                    for _ in entries
                ]
                plans = [r.prepare(tokens=tokens) for r in others]
                for plan, target in zip(plans, outputs):
                    plan.run(x, out=target)
                torch.cuda.synchronize()
                if not torch.isfinite(outputs[0]).all() or not all(
                    torch.equal(v.view(torch.uint8), outputs[0].view(torch.uint8))
                    for v in outputs[1:]
                ):
                    raise AssertionError(
                        f"Tile output bytes differ: {shape}, T={tokens}"
                    )
                graphs = []
                for plan, target in zip(plans, outputs):
                    graph = torch.cuda.CUDAGraph()
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.graph(graph, stream=stream):
                        for _ in range(report["graph_nodes"]):
                            plan.run(x, out=target)
                    torch.cuda.current_stream().wait_stream(stream)
                    graphs.append(graph)
                for graph in graphs:
                    for _ in range(report["warmup_replays"]):
                        graph.replay()
                torch.cuda.synchronize()
                case = dict(
                    shape=shape,
                    role=role,
                    occurrences=counts[shape],
                    tokens=tokens,
                    exact_baseline_output_bytes=True,
                    waves=[],
                )
                for repeat in range(4):
                    order = (
                        range(len(entries))
                        if repeat % 2 == 0
                        else reversed(range(len(entries)))
                    )
                    for i in order:
                        start, end = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        torch.cuda.synchronize()
                        wall, cpu = perf_counter(), cpu_seconds()
                        start.record()
                        for _ in range(report["graph_replays_per_wave"]):
                            graphs[i].replay()
                        end.record()
                        end.synchronize()
                        calls = report["graph_nodes"] * report["graph_replays_per_wave"]
                        case["waves"].append(
                            dict(
                                name=entries[i]["name"],
                                repeat=repeat,
                                calls=calls,
                                gpu_event_milliseconds_per_call=start.elapsed_time(end)
                                / calls,
                                wall_seconds=perf_counter() - wall,
                                process_cpu_seconds=cpu_seconds() - cpu,
                            )
                        )
                case["median_gpu_event_ms"] = {
                    e["name"]: statistics.median(
                        w["gpu_event_milliseconds_per_call"]
                        for w in case["waves"]
                        if w["name"] == e["name"]
                    )
                    for e in entries
                }
                report["cases"].append(case)
                save()
                print(
                    json.dumps(
                        {k: case[k] for k in ("shape", "tokens", "median_gpu_event_ms")}
                    ),
                    flush=True,
                )
                del graphs, plans, outputs, graph, stream, plan, target, x
                gc.collect()
            del others, resident, source
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
