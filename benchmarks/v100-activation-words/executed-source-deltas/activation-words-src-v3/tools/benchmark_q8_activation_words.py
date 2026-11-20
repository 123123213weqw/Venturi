"""Paired activation-word planes on represented checkpoint weights, not serving TPS."""

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
from venturi.ops import CudaBackend, Q8RowSplitWeight  # noqa: E402
from venturi.ops.q8_shared import prepare_q8_tiled_shared  # noqa: E402
from venturi.ops.q8_fused import prepare_q8_fused  # noqa: E402


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
    p.add_argument(
        "--unbounded-library",
        help="Optional unchanged paired-layout ablation, not a selectable promotion arm",
    )
    p.add_argument(
        "--bounded-pairs-library",
        help="Optional already-qualified paired-five ablation",
    )
    p.add_argument(
        "--candidate-name",
        choices=("candidate-pairs", "candidate-aos-five"),
        default="candidate-pairs",
    )
    a = p.parse_args()
    out = Path(a.output_directory)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    weights = QwenQ8Weights.from_ninfer(NInferArtifact(a.artifact))
    libraries = [a.baseline_library, a.library]
    if a.unbounded_library:
        libraries.append(a.unbounded_library)
    if a.bounded_pairs_library:
        libraries.append(a.bounded_pairs_library)
    backends = [CudaBackend(p) for p in libraries]
    report = dict(
        scope=__doc__,
        profiled=a.profile,
        artifact_id=weights.artifact.artifact_id,
        token_counts=[3, 4, 8],
        row_mode="tp-first",
        baseline_sha256=hashlib.sha256(
            Path(a.baseline_library).read_bytes()
        ).hexdigest(),
        candidate_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),
        precision="Original Q8/F16-scale weight planes; unchanged two A8 residual passes/F32 scales, FP16 input/output",
        graph_nodes=3,
        graph_replays=3,
        repeats=3,
        unbounded_ablation_sha256=hashlib.sha256(
            Path(a.unbounded_library).read_bytes()
        ).hexdigest()
        if a.unbounded_library
        else None,
        candidate_arm=a.candidate_name,
        bounded_pairs_ablation_sha256=hashlib.sha256(
            Path(a.bounded_pairs_library).read_bytes()
        ).hexdigest()
        if a.bounded_pairs_library
        else None,
        cases=[],
        completed=False,
        full_model_quality_qualified=False,
        achieved_hardware_counters=None,
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

    def measure(sources, label, tokens):
        resident = [
            [b.upload_linear(s, dtype=torch.float16, device="cuda:0") for s in sources]
            for b in backends
        ]
        k = sources[0].in_features
        ns = [s.codes.shape[0] for s in sources]
        group = len(ns) > 1

        def old(rs):
            if group:
                return prepare_q8_fused(rs, tokens=tokens, residual_compensation=True)
            return prepare_q8_tiled_shared(
                rs[0],
                tokens=tokens,
                rows=16 if tokens >= 8 and ns[0] >= 1024 else 8,
                token_tile=8 if tokens >= 8 and ns[0] >= 1024 else 4,
                residual_compensation=True,
            )

        plans = [old(rs) for rs in resident]
        names = ["control-aos", a.candidate_name]
        if a.unbounded_library:
            names.append("ablation-unbounded-pairs")
        if a.bounded_pairs_library:
            names.append("ablation-bounded-pairs")
        x_cpu = torch.randn(
            tokens, k, generator=torch.Generator().manual_seed(1731)
        ).half()
        x = x_cpu.cuda()
        work = [
            torch.empty(p.workspace_bytes, dtype=torch.uint8, device="cuda:0")
            for p in plans
        ]
        ys = [
            [torch.empty(tokens, n, dtype=torch.float16, device="cuda:0") for n in ns]
            for _ in plans
        ]

        def call(i):
            if group:
                plans[i].run(x, outputs=ys[i], workspace=work[i])
            else:
                plans[i].run(x, out=ys[i][0], workspace=work[i])

        for i in range(len(plans)):
            call(i)
        torch.cuda.synchronize()
        expected = [y.cpu() for y in ys[0]]
        assert all(
            torch.equal(b.view(torch.uint8), y.cpu().view(torch.uint8))
            for outputs in ys[1:]
            for b, y in zip(expected, outputs)
        ), "Activation planes changed output bytes"
        checks = []
        for s, y in zip(sources, expected):
            rows = sorted(
                set(
                    [0, s.codes.shape[0] - 1]
                    + torch.randint(
                        s.codes.shape[0],
                        (30,),
                        generator=torch.Generator().manual_seed(424),
                    ).tolist()
                )
            )
            sampled_tokens = sorted({0, tokens // 2, tokens - 1})
            ref = projection_reference(
                x_cpu[sampled_tokens], s.codes[rows], s.scales[rows], 2
            )
            v = y[sampled_tokens][:, rows].double()
            err = float((v - ref).norm() / (ref.norm() + 1e-20))
            assert err < 0.001
            checks.append(
                dict(
                    rows=rows,
                    token_indices=sampled_tokens,
                    relative_l2=err,
                    max_absolute=float((v - ref).abs().max()),
                )
            )
        case = dict(
            label=label,
            tokens=tokens,
            k=k,
            ns=ns,
            all_output_bytes_equal=True,
            sampled_fp64=checks,
            waves=[],
        )
        graphs = []
        for i in range(len(plans)):
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
            for i in (
                range(len(plans)) if repeat % 2 == 0 else reversed(range(len(plans)))
            ):
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                torch.cuda.synchronize()
                began, wall, process = time(), perf_counter(), cpu()
                torch.cuda.nvtx.range_push(
                    "activation-words/" + label + f"/t{tokens}/" + names[i]
                )
                start.record()
                for _ in range(3):
                    graphs[i].replay()
                end.record()
                end.synchronize()
                torch.cuda.nvtx.range_pop()
                case["waves"].append(
                    dict(
                        arm=names[i],
                        repeat=repeat,
                        calls=9,
                        start_unix=began,
                        end_unix=time(),
                        gpu_event_ms_per_call=start.elapsed_time(end) / 9,
                        wall_seconds=perf_counter() - wall,
                        cpu_core_seconds=cpu() - process,
                    )
                )
        case["median_gpu_event_ms"] = {
            n: statistics.median(
                w["gpu_event_ms_per_call"] for w in case["waves"] if w["arm"] == n
            )
            for n in names
        }
        case["best_measured_arm"] = min(
            case["median_gpu_event_ms"], key=case["median_gpu_event_ms"].get
        )
        report["cases"].append(case)
        save()
        print(json.dumps(case), flush=True)

    try:
        unique = {
            s.shape: s.binding
            for s in weights.plan.bindings
            if s.format == "q8_g32_fp16"
        }

        def split(s):
            half = (s.codes.shape[0] + 1) // 2
            return Q8RowSplitWeight(
                s.codes[:half].clone(), s.scales[:half].clone(), s.in_features
            )

        for role in unique.values():
            sources = [split(weights.read_linear(role))]
            for tokens in report["token_counts"]:
                measure(sources, role, tokens)
            gc.collect()
            torch.cuda.empty_cache()
        for label, suffixes in [
            ("mlp", ["mlp/gate", "mlp/up"]),
            ("gdn", ["gdn/qkv", "gdn/z"]),
            ("attention", ["attention/q", "attention/k", "attention/v"]),
        ]:
            spec = next(
                s
                for s in weights.plan.bindings
                if s.binding.endswith("/" + suffixes[0])
            )
            prefix = spec.binding[: -len(suffixes[0])]
            sources = [split(weights.read_linear(prefix + s)) for s in suffixes]
            for tokens in report["token_counts"]:
                measure(sources, "fused-" + label, tokens)
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
