"""Explicit PD/colocated native benchmark; not a production server or model fetcher.

One shared wall interval per repeat. Pretokenized independent requests, ordinary
greedy generation, no MTP/prefix reuse. Synthetic checkpoint scope is mandatory
metadata and must never be presented as trained-model TPS.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import threading
import time


def devices(value):
    try:
        result = [int(x) for x in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected comma-separated CUDA ordinals") from error
    if not result or len(set(result)) != len(result) or any(x < 0 for x in result):
        raise argparse.ArgumentTypeError("Device ordinals must be unique and nonnegative")
    return result


def specs(path):
    body = json.loads(path.read_text())
    if not isinstance(body, list) or not body:
        raise ValueError("Request file must be a nonempty JSON list")
    result = []
    for value in body:
        if (not isinstance(value, dict) or not isinstance(value.get("prompt"), list)
                or not value["prompt"] or any(type(i) is not int or i < 0 for i in value["prompt"])
                or type(value.get("max_new_tokens")) is not int or value["max_new_tokens"] <= 0
                or type(value.get("arrival_seconds", 0)) not in (int, float)
                or not 0 <= value.get("arrival_seconds", 0) <= 3600):
            raise ValueError("Invalid pretokenized request specification")
        result.append(dict(prompt=value["prompt"], max_new_tokens=value["max_new_tokens"],
                           arrival_seconds=value.get("arrival_seconds", 0)))
    return result


def gpu_sample():
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,utilization.memory,"
                            "memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu",
                            "--format=csv"], capture_output=True, text=True, timeout=8)
        return dict(monotonic=time.perf_counter(), exit=p.returncode, csv=p.stdout, stderr=p.stderr)
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(monotonic=time.perf_counter(), error=repr(error), scope="GPU telemetry unavailable")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--mode", choices=("pd", "colocated"), required=True)
    parser.add_argument("--checkpoint-scope", choices=("trained", "generated-fixture"), required=True)
    parser.add_argument("--prefill-devices", type=devices, default=[0])
    parser.add_argument("--decode-devices", type=devices, default=[1])
    parser.add_argument("--transport", choices=("host", "peer"), default="host")
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=512)
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--prefill-chunk-size", type=int, default=16)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if set(args.prefill_devices) & set(args.decode_devices):
        parser.error("Prefill and decode devices must be disjoint")
    if (not 1 <= args.slots <= min(32, args.rows) or not 1 <= args.rows <= 1024
            or args.capacity < 2 or args.prefill_chunk_size < 1 or args.repeats < 3
            or args.warmup_rounds < 0):
        parser.error("Invalid capacities/budgets or fewer than three scored repeats")
    if args.output.exists():
        parser.error("Refusing to overwrite an existing benchmark result")
    requests = specs(args.requests)
    # Explicit execution dependency loading happens only after argument parsing.
    import torch
    from venturi.models import NativeQwenModel, QwenQ8Weights
    from venturi.models.qwen_pd import NativeQwenPDWorker, NativeQwenStateTransport
    from venturi.ninfer import NInferArtifact
    from venturi.ops import CudaBackend
    from venturi.runtime import ContinuousScheduler, ContinuousWorker, PDScheduler, PDWorkerPool

    all_devices = args.prefill_devices + args.decode_devices
    if max(all_devices) >= torch.cuda.device_count():
        raise ValueError("Requested CUDA ordinal is unavailable")
    backend = CudaBackend(args.library)
    weights = QwenQ8Weights.from_ninfer(NInferArtifact(args.artifact))
    result = dict(mode=args.mode, checkpoint_scope=args.checkpoint_scope, configuration=vars(args).copy(),
                  artifact_id=weights.artifact.artifact_id, artifact_bytes=args.artifact.stat().st_size,
                  native_library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                  scope="Native request wall incl queue/prefill/transfer/decode. Excludes model load/tokenization. NOT decode-only TPS or HTTP/SSE latency.",
                  warmup_rounds=args.warmup_rounds, samples=[], repeats=[],
                  profiler=False, achieved_occupancy_and_DRAM_bandwidth="not measured")
    result["configuration"] = {k: str(v) if isinstance(v, Path) else v for k, v in result["configuration"].items()}
    models, batches, owners = [], [], []
    sampler_stop = threading.Event()
    sampler = None
    safe_to_free = True
    try:
        started = time.perf_counter()
        for device in all_devices:
            model = NativeQwenModel.from_q8(weights, backend, weight_mode="resident",
                        dtype=torch.float16 if args.dtype == "fp16" else torch.bfloat16,
                        device=f"cuda:{device}", max_tokens=args.rows, metadata_mode="sealed")
            models.append(model)
            batches.append(model.create_batch([args.capacity] * args.slots))
        result["model_and_request_setup_seconds"] = time.perf_counter() - started
        if args.mode == "pd":
            targets = [NativeQwenPDWorker(b) for b in batches]
            split = len(args.prefill_devices)
            scheduler = PDScheduler(targets[:split], targets[split:], NativeQwenStateTransport(mode=args.transport),
                                    prefill_chunk_size=args.prefill_chunk_size)
            owners = [PDWorkerPool(scheduler).start()]
            schedulers = [scheduler]
        else:
            schedulers = [ContinuousScheduler(batch, token_budget=args.rows,
                            prefill_chunk_size=args.prefill_chunk_size) for batch in batches]
            owners = [ContinuousWorker(s).start() for s in schedulers]

        def one_round():
            cpu, start = time.process_time(), time.perf_counter()
            tickets = {}
            for i, request in sorted(enumerate(requests), key=lambda item: item[1]["arrival_seconds"]):
                time.sleep(max(0, start + request["arrival_seconds"] - time.perf_counter()))
                tickets[i] = owners[i % len(owners)].submit(request["prompt"],
                    max_new_tokens=request["max_new_tokens"], eos_token_ids=[])
            outputs = [tickets[i].future.result(timeout=300) for i in range(len(requests))]
            wall, cpu = time.perf_counter() - start, time.process_time() - cpu
            tokens = sum(len(value.output_ids) for value in outputs)
            if any(len(value.output_ids) != request["max_new_tokens"] for value, request in zip(outputs, requests)):
                raise RuntimeError("Benchmark output budget mismatch")
            return dict(wall_seconds=wall, process_cpu_seconds=cpu,
                        process_cpu_percent_one_core_100=100 * cpu / wall,
                        output_tokens=tokens, aggregate_output_tokens_per_second=tokens / wall,
                        requests_per_second=len(outputs) / wall,
                        output_ids_sha256=hashlib.sha256(json.dumps([value.output_ids for value in outputs]).encode()).hexdigest(),
                        requests=[dict(queue_seconds=v.queue_seconds, ttft_seconds=v.time_to_first_token_seconds,
                                       wall_seconds=v.wall_seconds, max_inter_token_seconds=v.max_inter_token_seconds)
                                  for v in outputs])

        for _ in range(args.warmup_rounds):
            one_round()
        result["samples"].append(gpu_sample())
        def sample():
            while not sampler_stop.wait(1):
                result["samples"].append(gpu_sample())
        sampler = threading.Thread(target=sample, name="pd-benchmark-telemetry", daemon=False)
        sampler.start()
        result["stats_before"] = [s.info() for s in schedulers]
        for _ in range(args.repeats):
            result["repeats"].append(one_round())
        if len({r["output_ids_sha256"] for r in result["repeats"]}) != 1:
            raise RuntimeError("Greedy outputs changed between repeats; qualify correctness before scoring")
        result["stats_after"] = [s.info() for s in schedulers]
        result["median_aggregate_output_tokens_per_second"] = statistics.median(
            r["aggregate_output_tokens_per_second"] for r in result["repeats"])
        result["peak_process_rss_native_units"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["status"] = "passed"
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        sampler_stop.set()
        if sampler:
            sampler.join(15)
        cleanup_errors = []
        for owner in owners:
            try:
                owner.close(timeout=300)
            except TimeoutError as error:
                safe_to_free = False
                cleanup_errors.append(repr(error))
            except BaseException as error:
                cleanup_errors.append(repr(error))
        if safe_to_free:
            for item in list(reversed(batches)) + list(reversed(models)):
                try:
                    item.close()
                except BaseException as error:
                    cleanup_errors.append(repr(error))
        result["cleanup_errors"] = cleanup_errors
        if cleanup_errors:
            result["status"] = "failed_cleanup"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    if result["status"] != "passed":
        raise RuntimeError("Native PD benchmark cleanup did not complete")
    print(json.dumps({k: result[k] for k in ("status", "checkpoint_scope", "mode", "median_aggregate_output_tokens_per_second")}))


if __name__ == "__main__":
    main()
