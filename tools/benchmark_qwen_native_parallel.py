"""Native TP versus PP, concurrent requests and fixed output work.

TP is the existing model-serialized baseline, not continuous batching. PP uses
native per-stage locks so distinct requests may overlap on separate GPUs.
Every request uses one native call per timed phase, never a Python token loop.
Split TTFT/continuation phases are separate from complete one-call requests.
Aggregate throughput uses total emitted tokens divided by cohort wall time.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import datetime
import hashlib
import json
from pathlib import Path
import resource
import statistics
import subprocess
import threading
from time import perf_counter, thread_time, time

import torch

from venturi.models import NativeQwenModel, NativeQwenPipelineModel, QwenQ8Weights
from venturi.ninfer import NInferArtifact
from venturi.ops import CudaBackend


def cpu_seconds():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def run_wave(pool, calls, tokens_per_request, profile_label=None):
    """A common barrier includes scheduling/queue delay in request latency."""
    start = {}

    def begin():
        start.update(wall=perf_counter(), unix=time(), cpu=cpu_seconds())

    barrier = threading.Barrier(len(calls) + 1, action=begin, timeout=180)

    def worker(index, call):
        barrier.wait()
        entered, cpu = perf_counter(), thread_time()
        with torch.cuda.nvtx.range(profile_label) if profile_label else nullcontext():
            ids = call()
        end = perf_counter()
        return dict(
            request=index,
            ids=ids,
            completion_latency_seconds=end - start["wall"],
            native_call_wall_seconds=end - entered,
            thread_cpu_seconds=thread_time() - cpu,
        )

    jobs = [pool.submit(worker, i, call) for i, call in enumerate(calls)]
    barrier.wait()
    rows = [j.result(timeout=600) for j in jobs]
    wall, cpu, end = (
        perf_counter() - start["wall"],
        cpu_seconds() - start["cpu"],
        time(),
    )
    if any(len(r["ids"]) != tokens_per_request for r in rows):
        raise AssertionError("Native request returned unequal fixed output work")
    return dict(
        wall_seconds=wall,
        process_cpu_seconds=cpu,
        start_unix=start["unix"],
        end_unix=end,
        emitted_tokens=len(rows) * tokens_per_request,
        aggregate_output_tokens_per_wall_second=len(rows) * tokens_per_request / wall,
        mean_completion_latency_seconds=statistics.mean(
            r["completion_latency_seconds"] for r in rows
        ),
        max_completion_latency_seconds=max(
            r["completion_latency_seconds"] for r in rows
        ),
        requests=rows,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact")
    p.add_argument("--library", required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--mode", choices=("tp", "pp"), required=True)
    p.add_argument("--output-directory", required=True)
    p.add_argument("--concurrency", default="1,2,4,8")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--new-tokens", type=int, default=32)
    p.add_argument("--split-layer", type=int)
    p.add_argument("--profile-nvtx", action="store_true")
    args = p.parse_args()
    counts = [int(x) for x in args.concurrency.split(",")]
    if (
        not counts
        or any(c not in (1, 2, 4, 8) for c in counts)
        or not 1 <= args.repeats <= 8
        or not 2 <= args.new_tokens <= 256
    ):
        p.error("Use concurrency 1,2,4,8; repeats 1..8; output tokens 2..256")
    out = Path(args.output_directory)
    out.mkdir(parents=True, exist_ok=False)
    case = json.loads(Path(args.case).read_text())
    prompt, expected = (
        case["prompt_token_ids"],
        case["expected_generated_ids"][: args.new_tokens],
    )
    if len(expected) != args.new_tokens:
        p.error("Pinned case must contain enough expected output IDs")
    torch.set_num_threads(4)
    report = dict(
        scope=__doc__,
        mode=args.mode,
        profiled=args.profile_nvtx,
        case=case,
        new_tokens=args.new_tokens,
        eos_token_ids=[],
        dtype="fp16",
        split_layer=args.split_layer,
        physical_gpus=[1, 0],
        cuda_visible_devices="1,0",
        utc_offset_seconds=int(datetime.now().astimezone().utcoffset().total_seconds()),
        library_sha256=hashlib.sha256(Path(args.library).read_bytes()).hexdigest(),
        mtp=False,
        continuous_batching=False,
        whole_model_graphs=False,
        achieved_hardware_counters_available=False,
        matched_external_comparison=False,
        concurrency=[],
    )

    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    samplers, logs = [], []
    model = None
    sessions = []
    try:
        for gpu in (1, 0):
            log = (out / f"gpu-{gpu}-samples.csv").open("wb")
            logs.append(log)
            samplers.append(
                subprocess.Popen(
                    [
                        "nvidia-smi",
                        "-i",
                        str(gpu),
                        "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu,clocks.sm,clocks.mem",
                        "--format=csv,noheader,nounits",
                        "-lms",
                        "100",
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
        weights = QwenQ8Weights.from_ninfer(NInferArtifact(args.artifact))
        report["artifact_id"] = weights.artifact.artifact_id
        common = dict(
            weight_mode="resident",
            dtype=torch.float16,
            device="cuda:0",
            max_tokens=len(prompt),
        )
        wall, cpu = perf_counter(), cpu_seconds()
        if args.mode == "pp":
            model = NativeQwenPipelineModel.from_q8(
                weights,
                CudaBackend(args.library),
                pipeline_device="cuda:1",
                split_layer=args.split_layer,
                transport="host",
                **common,
            )
        else:
            model = NativeQwenModel.from_q8(
                weights,
                CudaBackend(args.library),
                tensor_parallel_device="cuda:1",
                tensor_parallel_transport="host",
                **common,
            )
        report["model_load"] = dict(
            wall_seconds=perf_counter() - wall, process_cpu_seconds=cpu_seconds() - cpu
        )
        save()
        for count in counts:
            sessions = [
                model.create_session(len(prompt) + args.new_tokens + 1)
                for _ in range(count)
            ]
            row = dict(
                request_count=count,
                initial_native_memory=[s.info() for s in sessions],
                trials=[],
            )
            report["concurrency"].append(row)
            save()
            with ThreadPoolExecutor(max_workers=count) as pool:
                warm = run_wave(
                    pool,
                    [
                        lambda s=s: s.generate(prompt, max_new_tokens=args.new_tokens)
                        for s in sessions
                    ],
                    args.new_tokens,
                )
                if any(r["ids"] != expected for r in warm["requests"]):
                    raise AssertionError(
                        "Warmup tokens differ from pinned single-native reference"
                    )
                row["warmup"] = warm
                for repeat in range(args.repeats):
                    for s in sessions:
                        s.reset()
                    ttft = run_wave(
                        pool,
                        [
                            lambda s=s: s.generate(prompt, max_new_tokens=1)
                            for s in sessions
                        ],
                        1,
                        "native_ttft" if args.profile_nvtx else None,
                    )
                    first = [r["ids"] for r in ttft["requests"]]
                    before = [s.info() for s in sessions]
                    decode = run_wave(
                        pool,
                        [
                            lambda s=s, ids=ids: s.generate(
                                ids, max_new_tokens=args.new_tokens - 1
                            )
                            for s, ids in zip(sessions, first)
                        ],
                        args.new_tokens - 1,
                        "native_decode" if args.profile_nvtx else None,
                    )
                    after = [s.info() for s in sessions]
                    for a, b in zip(before, after):
                        if (
                            a["model_weight_upload_bytes"]
                            != b["model_weight_upload_bytes"]
                        ):
                            raise AssertionError(
                                "Resident decode uploaded more model weights"
                            )
                    for s in sessions:
                        s.reset()
                    whole = run_wave(
                        pool,
                        [
                            lambda s=s: s.generate(
                                prompt, max_new_tokens=args.new_tokens
                            )
                            for s in sessions
                        ],
                        args.new_tokens,
                        "native_unsplit" if args.profile_nvtx else None,
                    )
                    for i in range(count):
                        if (
                            first[i] + decode["requests"][i]["ids"] != expected
                            or whole["requests"][i]["ids"] != expected
                        ):
                            raise AssertionError(
                                "Concurrent/split/whole tokens differ from pinned native reference"
                            )
                    row["trials"].append(
                        dict(
                            repeat=repeat,
                            ttft=ttft,
                            decode=decode,
                            unsplit=whole,
                            before_decode=before,
                            after_decode=after,
                        )
                    )
                    save()
                    print(
                        json.dumps(
                            dict(
                                count=count,
                                repeat=repeat,
                                mode=args.mode,
                                decode_aggregate_tps=decode[
                                    "aggregate_output_tokens_per_wall_second"
                                ],
                                whole_aggregate_tps=whole[
                                    "aggregate_output_tokens_per_wall_second"
                                ],
                            )
                        ),
                        flush=True,
                    )
            for s in sessions:
                s.close()
            sessions = []
            row["completed"] = True
            save()
        report["completed"] = True
        report["peak_process_rss_bytes_linux_including_mapped_pages"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
        save()
    finally:
        for s in sessions:
            s.close()
        if model is not None:
            model.close()
        for p in samplers:
            p.terminate()
            p.wait(timeout=15)
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
