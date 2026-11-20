"""Native-only fixed-length timing with separately measured TTFT and decode.

The split run generates one token, then continues the same native request for
N-1 tokens in one C++ call. A separate unsplit run checks identical token IDs.
EOS stopping is explicitly disabled for fixed work, including tokens after EOS;
these are benchmark tokens, not a normal answer-length or quality evaluation.
Load, warmup, reset and diagnostics are outside request timing. No Python model
or per-token generation loop is used. Profiler timings are labelled separately.
"""

import argparse
import hashlib
import json
from pathlib import Path
import resource
import statistics
import subprocess
from time import perf_counter, time
from contextlib import nullcontext

import torch

from venturi.models import NativeQwenModel, QwenQ8Weights
from venturi.ninfer import NInferArtifact
from venturi.ops import CudaBackend


def cpu_seconds():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def measured(call):
    start_unix, cpu, wall = time(), cpu_seconds(), perf_counter()
    result = call()
    return result, dict(
        wall_seconds=perf_counter() - wall,
        process_cpu_seconds=cpu_seconds() - cpu,
        start_unix=start_unix,
        end_unix=time(),
    )


def aggregate(rows):
    result = {}
    for name in ("ttft", "decode", "unsplit"):
        walls = [r[name]["wall_seconds"] for r in rows]
        cpus = [r[name]["process_cpu_seconds"] for r in rows]
        item = dict(
            mean_wall_seconds=statistics.mean(walls),
            min_wall_seconds=min(walls),
            max_wall_seconds=max(walls),
            mean_process_cpu_seconds=statistics.mean(cpus),
        )
        if name != "ttft":
            count = rows[0][name]["output_tokens"]
            if any(r[name]["output_tokens"] != count for r in rows):
                raise ValueError("Cannot aggregate unequal output lengths")
            item["output_tokens_per_wall_second"] = count / statistics.mean(walls)
        result[name] = item
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact")
    p.add_argument("--library", required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--output-directory", required=True)
    p.add_argument("--new-tokens", type=int, default=32)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--dtype", choices=("fp16", "bf16"), required=True)
    p.add_argument("--weight-mode", choices=("resident", "streaming"), required=True)
    p.add_argument("--gpu-index", type=int, required=True)
    p.add_argument("--profile-nvtx", action="store_true")
    args = p.parse_args()
    if not 2 <= args.new_tokens <= 256 or not 1 <= args.repeats <= 8:
        p.error("Use 2..256 output tokens and 1..8 repeats")
    out = Path(args.output_directory)
    out.mkdir(parents=True, exist_ok=False)
    case = json.loads(Path(args.case).read_text())
    ids = case["prompt_token_ids"]
    report = dict(
        scope=__doc__,
        case=case,
        new_tokens=args.new_tokens,
        eos_token_ids=[],
        weight_mode=args.weight_mode,
        dtype=args.dtype,
        profiled=args.profile_nvtx,
        physical_gpu_index=args.gpu_index,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        library_sha256=hashlib.sha256(Path(args.library).read_bytes()).hexdigest(),
        hardware_counters_available=False,
        independent_upstream_quality_qualified=False,
        external_engine_comparison=False,
        trials=[],
    )

    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    torch.set_num_threads(4)
    telemetry = (out / "gpu-samples.csv").open("wb")
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
        stdout=telemetry,
        stderr=subprocess.STDOUT,
    )
    model = session = None
    try:
        weights = QwenQ8Weights.from_ninfer(NInferArtifact(args.artifact))
        report["artifact_id"] = weights.artifact.artifact_id
        dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

        def load():
            return NativeQwenModel.from_q8(
                weights,
                CudaBackend(args.library),
                weight_mode=args.weight_mode,
                dtype=dtype,
                device="cuda:0",
                max_tokens=len(ids),
            )

        model, report["model_load"] = measured(load)
        session, report["session_load"] = measured(
            lambda: model.create_session(len(ids) + args.new_tokens + 1)
        )
        report["initial_native_memory"] = session.info()
        save()
        warm, report["warmup"] = measured(
            lambda: session.generate(ids, max_new_tokens=args.new_tokens)
        )
        report["warmup_ids"] = warm
        save()
        for trial in range(args.repeats):
            session.reset()
            context = (
                torch.cuda.nvtx.range("native_ttft")
                if args.profile_nvtx
                else nullcontext()
            )
            with context:
                first, ttft = measured(lambda: session.generate(ids, max_new_tokens=1))
            before_decode = session.info()
            context = (
                torch.cuda.nvtx.range("native_decode")
                if args.profile_nvtx
                else nullcontext()
            )
            with context:
                rest, decode = measured(
                    lambda: session.generate(first, max_new_tokens=args.new_tokens - 1)
                )
            after_decode = session.info()
            session.reset()
            context = (
                torch.cuda.nvtx.range("native_unsplit")
                if args.profile_nvtx
                else nullcontext()
            )
            with context:
                whole, unsplit = measured(
                    lambda: session.generate(ids, max_new_tokens=args.new_tokens)
                )
            if whole != first + rest or whole != warm:
                raise AssertionError("Warmup/split/unsplit native token IDs differ")
            decode["output_tokens"] = len(rest)
            unsplit["output_tokens"] = len(whole)
            report["trials"].append(
                dict(
                    trial=trial,
                    ttft=ttft,
                    decode=decode,
                    unsplit=unsplit,
                    generated_ids=whole,
                    before_decode=before_decode,
                    after_decode=after_decode,
                    consumed_positions=session.position,
                )
            )
            report["aggregate"] = aggregate(report["trials"])
            save()
            print(json.dumps(report["aggregate"]), flush=True)
        report["completed"] = True
        save()
    finally:
        if session is not None:
            session.close()
        if model is not None:
            model.close()
        sampler.terminate()
        sampler.wait(timeout=15)
        telemetry.close()


if __name__ == "__main__":
    main()
