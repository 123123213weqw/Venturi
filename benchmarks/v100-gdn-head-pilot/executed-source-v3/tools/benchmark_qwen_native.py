"""Matched native-runtime vs Python-composed owned-operator generation.

NOT an external-engine comparison. Same Q8 checkpoint/profile/prompt/greedy/EOS
policy, state reset before each request, load/preparation/warmup excluded from
request wall/CPU time. Output/request-wall rate includes prefill, not decode TPS.
Native TTFT/per-token timing is not instrumented yet; record it missing, not zero.
Raw CUDA allocations are not included in PyTorch allocator peaks.
"""

import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
from pathlib import Path
import resource
import subprocess
from time import perf_counter, time

import torch

from qualify_qwen_streaming import planned_greedy
from venturi.models import (
    NativeQwenModel,
    ResidentQwenModel,
    StreamingQwenModel,
    QwenQ8Weights,
)
from venturi.ninfer import NInferArtifact
from venturi.ops import CudaBackend, prepare_greedy


def cpu_seconds():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("artifact")
    ap.add_argument("--library", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--output-directory", required=True)
    ap.add_argument("--weight-mode", choices=("streaming", "resident"), required=True)
    ap.add_argument("--dtype", choices=("fp16", "bf16"), required=True)
    ap.add_argument(
        "--gpu-index", type=int, required=True, help="Physical nvidia-smi index"
    )
    ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument(
        "--profile-nvtx",
        action="store_true",
        help="Profiler-only labelled run; never use these timings as unprofiled performance",
    )
    ap.add_argument(
        "--inner-requests",
        type=int,
        default=1,
        help="Serial reset requests per wave; increase for tiny-fixture telemetry",
    )
    ap.add_argument(
        "--order",
        default="python,native",
        help="Explicit sequential phases; supports python,native,python,native",
    )
    ap.add_argument(
        "--scope",
        required=True,
        help="Say synthetic fixture or full trained checkpoint explicitly",
    )
    args = ap.parse_args()
    order = args.order.split(",")
    if (
        any(x not in ("python", "native") for x in order)
        or set(order) != {"python", "native"}
        or not 2 <= args.repeats <= 8
        or not 1 <= args.inner_requests <= 100
    ):
        ap.error("Use both engines and 2..8 repeated requests per phase")
    out = Path(args.output_directory)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    case = json.loads(Path(args.case).read_text())
    ids = case["prompt_token_ids"]
    maximum = case["max_new_tokens"]
    eos = case["eos_token_ids"]
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    weights = QwenQ8Weights.from_ninfer(NInferArtifact(args.artifact))
    backend = CudaBackend(args.library)
    report = dict(
        scope=__doc__,
        model_scope=args.scope,
        case=case,
        artifact_id=weights.artifact.artifact_id,
        library_sha256=hashlib.sha256(Path(args.library).read_bytes()).hexdigest(),
        dtype=str(dtype),
        weight_mode=args.weight_mode,
        max_tokens=args.max_tokens,
        capacity=len(ids) + maximum,
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        physical_gpu_index=args.gpu_index,
        order=order,
        repeats=args.repeats,
        phases=[],
        native_ttft_and_decode_latency_measured=False,
        profiled=args.profile_nvtx,
        external_engine_performance_qualified=False,
        concurrency="One request; no batching/paged KV comparison",
        hardware_counters="No achieved occupancy/SM/DRAM/cache/stall counters; existing driver policy unchanged",
    )

    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    log = (out / "gpu-samples.csv").open("w")
    sampler = subprocess.Popen(
        [
            "nvidia-smi",
            "-i",
            str(args.gpu_index),
            "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu,clocks.sm,clocks.mem",
            "--format=csv,noheader,nounits",
            "--loop-ms=100",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    expected = None
    try:
        for phase, engine in enumerate(order):
            gc.collect()
            torch.cuda.empty_cache()
            free_before, total = torch.cuda.mem_get_info()
            load = perf_counter()
            load_cpu = cpu_seconds()
            if engine == "native":
                model = NativeQwenModel.from_q8(
                    weights,
                    backend,
                    weight_mode=args.weight_mode,
                    dtype=dtype,
                    device="cuda:0",
                    max_tokens=args.max_tokens,
                )
                session = model.create_session(len(ids) + maximum)

                def generate(request=session):
                    return request.generate(
                        ids, max_new_tokens=maximum, eos_token_ids=eos
                    )
            else:
                cls = (
                    ResidentQwenModel
                    if args.weight_mode == "resident"
                    else StreamingQwenModel
                )
                model = cls.from_q8(weights, backend, dtype=dtype, device="cuda:0")
                session = model.create_session(
                    len(ids) + maximum, max_tokens=args.max_tokens
                )
                selector = prepare_greedy(
                    backend,
                    rows=1,
                    vocab_size=weights.config["vocab_size"],
                    dtype=torch.float32,
                    device="cuda:0",
                )
                selection = torch.empty(1, 2, dtype=torch.int64, device="cuda:0")
                scratch = torch.empty(
                    selector.workspace_bytes, dtype=torch.uint8, device="cuda:0"
                )

                def generate(
                    request=session, plan=selector, result=selection, workspace=scratch
                ):
                    generated = []
                    for step in range(maximum):
                        logits = request.forward(ids if step == 0 else [generated[-1]])
                        token = planned_greedy(logits, plan, result, workspace)
                        del logits
                        generated.append(token)
                        if token in eos:
                            break
                    return generated

            load_wall, load_used = perf_counter() - load, cpu_seconds() - load_cpu
            try:
                warm = perf_counter()
                warm_tokens = generate()
                warm_wall = perf_counter() - warm
                if expected is None:
                    expected = warm_tokens
                if warm_tokens != expected:
                    raise AssertionError(
                        "Engine outputs differ; do not benchmark as qualified"
                    )
                record = dict(
                    engine=engine,
                    phase=phase,
                    model_and_session_load_wall_seconds=load_wall,
                    model_and_session_load_cpu_seconds=load_used,
                    warmup_requests=1,
                    warmup_wall_seconds=warm_wall,
                    free_before_load_bytes=free_before,
                    device_total_bytes=total,
                    waves=[],
                )
                report["phases"].append(record)
                save()
                for repeat in range(args.repeats):
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    stamp, wave_start, wave_cpu = time(), perf_counter(), cpu_seconds()
                    durations, cpu_times = [], []
                    for _ in range(args.inner_requests):
                        session.reset()
                        start, cpu = perf_counter(), cpu_seconds()
                        with (
                            torch.cuda.nvtx.range(f"runtime_{engine}")
                            if args.profile_nvtx
                            else nullcontext()
                        ):
                            tokens = generate()
                        durations.append(perf_counter() - start)
                        cpu_times.append(cpu_seconds() - cpu)
                        if tokens != expected:
                            raise AssertionError(
                                "Repeated output regression; preserve report"
                            )
                    elapsed, used = sum(durations), sum(cpu_times)
                    outer_wall, outer_cpu = (
                        perf_counter() - wave_start,
                        cpu_seconds() - wave_cpu,
                    )
                    free_after, _ = torch.cuda.mem_get_info()
                    wave = dict(
                        repeat=repeat,
                        start_unix=stamp,
                        end_unix=time(),
                        wall_seconds=elapsed,
                        process_cpu_seconds=used,
                        serial_requests=args.inner_requests,
                        per_request_wall_seconds=durations,
                        per_request_cpu_seconds=cpu_times,
                        wall_seconds_including_resets=outer_wall,
                        process_cpu_seconds_including_resets=outer_cpu,
                        output_tokens=len(tokens) * args.inner_requests,
                        generated_ids=tokens,
                        output_tokens_per_request_wall_second=len(tokens)
                        * args.inner_requests
                        / elapsed,
                        output_tokens_per_serial_wall_second_including_resets=len(
                            tokens
                        )
                        * args.inner_requests
                        / outer_wall,
                        consumed_positions=session.position,
                        free_after_request_bytes=free_after,
                        peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(),
                        torch_peak_excludes_native_cuda_buffers=True,
                    )
                    if engine == "native":
                        wave["native_owned_memory"] = session.info()
                    else:
                        wave["python_state_kv_bytes"] = session.cache_bytes
                    record["waves"].append(wave)
                    save()
                    print(
                        json.dumps(
                            dict(
                                engine=engine,
                                phase=phase,
                                repeat=repeat,
                                wall=elapsed,
                                cpu=used,
                                tokens=tokens,
                            )
                        ),
                        flush=True,
                    )
            finally:
                session.close()
                if engine == "native":
                    model.close()
            del generate, session, model
            if engine == "python":
                del selector, selection, scratch
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        sampler.terminate()
        sampler.wait(timeout=10)
        log.close()
    report["telemetry_exit_code"] = sampler.returncode
    report["all_matched_generated_ids"] = expected
    report["matched_internal_runtime_requests_completed"] = True
    save()


if __name__ == "__main__":
    main()
