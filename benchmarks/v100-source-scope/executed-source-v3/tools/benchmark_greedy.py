"""Matched greedy-selector microbenchmark; not model TPS or concurrent serving.

Compare full-logit CPU transfer/selection, allocating Torch GPU selection and
owned native GPU selection. All validate finite rows, use lowest-index ties,
return [token,status] to the host and include host result consumption. Separate
native kernel-event timing excludes host copies; it is not hardware occupancy.
"""

import argparse
import hashlib
import json
from pathlib import Path
import resource
import statistics
import subprocess
from time import perf_counter, time

import torch

from venturi.ops import CudaBackend, prepare_greedy


def cpu_seconds():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--library", required=True)
    ap.add_argument("--output-directory", required=True)
    ap.add_argument(
        "--gpu-index", required=True, type=int, help="Physical nvidia-smi index"
    )
    ap.add_argument("--iterations", type=int, default=100)
    args = ap.parse_args()
    if args.iterations < 10:
        ap.error("Use at least ten calls per wave")
    outdir = Path(args.output_directory)
    outdir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    backend = CudaBackend(args.library)
    report = dict(
        scope=__doc__,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        library_sha256=hashlib.sha256(Path(args.library).read_bytes()).hexdigest(),
        seed=8107,
        dtype="FP32",
        vocab_size=248320,
        warmup_calls_per_mode=10,
        iterations_per_wave=args.iterations,
        cases=[],
        performance_scope="Selector only, NOT full-model performance",
        counters="No achieved occupancy/SM/DRAM/cache/stall measurements; driver policy retained unchanged",
    )
    log = (outdir / "gpu-samples.csv").open("w")
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
    try:
        for rows in (1, 8):
            gen = torch.Generator().manual_seed(report["seed"] + rows)
            host = torch.randn(rows, report["vocab_size"], generator=gen)
            host[:, 1000] = host[:, -1] = 12  # Identical cross-tile ties.
            x = host.cuda()
            p = prepare_greedy(
                backend, rows=rows, vocab_size=x.shape[1], device=x.device
            )
            result = torch.empty(rows, 2, dtype=torch.int64, device=x.device)
            scratch = torch.empty(p.workspace_bytes, dtype=torch.uint8, device=x.device)

            def torch_select(value, target):
                valid = value.isfinite().all(1)
                target[:, 0].copy_(torch.where(valid, value.argmax(1), -1))
                target[:, 1].copy_(~valid)
                return target

            cpu_result = torch.empty(rows, 2, dtype=torch.int64)

            def consume(value):
                pairs = value.tolist()
                if any(status != 0 or token < 0 for token, status in pairs):
                    raise AssertionError("Invalid selector result")
                return pairs

            def cpu_transfer():
                return consume(torch_select(x.cpu(), cpu_result))

            def torch_gpu():
                return consume(torch_select(x, result).cpu())

            def native_gpu():
                p.run(x, out=result, workspace=scratch)
                return consume(result.cpu())

            calls = {
                "full-logits-cpu": cpu_transfer,
                "torch-gpu": torch_gpu,
                "native-gpu": native_gpu,
            }
            expected = [[1000, 0]] * rows
            for call in calls.values():
                for _ in range(10):
                    assert call() == expected
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            case = dict(
                rows=rows,
                independent_rows_not_serving_requests=True,
                exact_expected=expected,
                waves=[],
            )
            for repeat in range(4):
                order = list(calls) if repeat % 2 == 0 else list(reversed(calls))
                for mode in order:
                    torch.cuda.synchronize()
                    stamp, wall, cpu = time(), perf_counter(), cpu_seconds()
                    for _ in range(args.iterations):
                        actual = calls[mode]()
                    torch.cuda.synchronize()
                    elapsed, used = perf_counter() - wall, cpu_seconds() - cpu
                    assert actual == expected
                    case["waves"].append(
                        dict(
                            repeat=repeat,
                            mode=mode,
                            start_unix=stamp,
                            end_unix=time(),
                            calls=args.iterations,
                            wall_seconds=elapsed,
                            process_cpu_seconds=used,
                            wall_us_per_call=elapsed / args.iterations * 1e6,
                            cpu_us_per_call=used / args.iterations * 1e6,
                            row_selections_per_second=rows * args.iterations / elapsed,
                        )
                    )
            case["median_wall_us"] = {
                mode: statistics.median(
                    w["wall_us_per_call"] for w in case["waves"] if w["mode"] == mode
                )
                for mode in calls
            }
            # Separate pure native launch timing; no host result copies in loop.
            events = []
            for _ in range(5):
                a, b = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                a.record()
                for _ in range(100):
                    p.run(x, out=result, workspace=scratch)
                b.record()
                b.synchronize()
                events.append(a.elapsed_time(b) * 10)  # ms/100 -> us/call.
            assert consume(result.cpu()) == expected
            case["native_launch_event_us_per_call"] = events
            case["native_event_scope"] = (
                "Both greedy kernels plus launch gaps; no host-result copies, no profiler"
            )
            case["peak_torch_allocated_bytes"] = torch.cuda.max_memory_allocated()
            case["peak_torch_reserved_bytes"] = torch.cuda.max_memory_reserved()
            case["logits_host_copy_bytes_cpu_mode"] = host.numel() * host.element_size()
            case["result_host_copy_bytes_gpu_modes"] = rows * 16
            report["cases"].append(case)
    finally:
        sampler.terminate()
        sampler.wait(timeout=10)
        log.close()
    report["telemetry_exit_code"] = sampler.returncode
    (outdir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "cases": [
                    dict(rows=c["rows"], median_wall_us=c["median_wall_us"])
                    for c in report["cases"]
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
