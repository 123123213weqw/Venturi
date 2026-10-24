#!/usr/bin/env python3
"""Run synchronized FP16 loading and full-byte verification in fresh processes."""

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--loaders",
        nargs="+",
        choices=("siphon", "safetensors"),
        default=["siphon", "safetensors"],
    )
    parser.add_argument("--buffer-size", type=int, default=64 * 2**20)
    parser.add_argument("--chunk-bytes", type=int, default=16 * 2**20)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.runs, args.buffer_size, args.chunk_bytes, args.timeout_seconds) <= 0:
        parser.error(
            "runs, buffer-size, chunk-bytes, and timeout-seconds must be positive"
        )
    if len(set(args.loaders)) != len(args.loaders):
        parser.error("duplicate loaders are not allowed")
    if args.device.startswith("cpu") and "siphon" in args.loaders:
        parser.error("CPU acceptance requires --loaders safetensors")
    environment = dict(os.environ, SIPHON_DEBUG="0")
    report = {
        "schema_version": 1,
        "measurement": {
            "floating_dtype": "F16",
            "integer_and_bool_buffers": "preserved unchanged",
            "fresh_process_per_run": True,
            "cache_state": "OS page cache uncontrolled; no cache eviction",
            "load_time": "synchronized in-process API time, excluding process startup and verification",
            "memory": "PyTorch allocator only, excluding native allocations and CUDA context",
            "inference": False,
        },
        "runs": [],
    }
    if args.source_manifest:
        report["source"] = json.loads(args.source_manifest.read_text())
    for repeat in range(args.runs):
        order = args.loaders if repeat % 2 == 0 else list(reversed(args.loaders))
        for loader in order:
            command = [
                sys.executable,
                "-m",
                "venturi",
                "verify",
                args.path,
                "--device",
                args.device,
                "--loader",
                loader,
                "--require-float-dtype",
                "F16",
                "--chunk-bytes",
                str(args.chunk_bytes),
                "--json",
            ]
            if loader == "siphon":
                command += ["--buffer-size", str(args.buffer_size)]
            result = subprocess.run(
                command,
                env=environment,
                capture_output=True,
                text=True,
                timeout=args.timeout_seconds,
            )
            if result.returncode:
                raise RuntimeError(
                    f"{loader} run {repeat + 1} failed:\n{result.stderr}\n{result.stdout}"
                )
            data = json.loads(result.stdout)
            if (
                not data["verification"]["passed"]
                or data["verification"]["verified_bytes"] != data["tensor_bytes"]
            ):
                raise RuntimeError(
                    f"Incomplete verification in {loader} run {repeat + 1}"
                )
            # Keep published reports independent of machine-specific checkpoint paths.
            data["model_dir"] = "<checkpoint>"
            data["repeat"] = repeat + 1
            report["runs"].append(data)
            print(
                f"{loader} run {repeat + 1}: {data['load']['total_seconds']:.6f}s; {data['verification']['verified_bytes']} bytes verified",
                file=sys.stderr,
                flush=True,
            )
    summaries = {}
    for loader in args.loaders:
        rows = [row for row in report["runs"] if row["load"]["loader"] == loader]
        times = [row["load"]["total_seconds"] for row in rows]
        summaries[loader] = {
            "runs": len(rows),
            "all_verified": True,
            "load_seconds_min": min(times),
            "load_seconds_median": statistics.median(times),
            "load_seconds_max": max(times),
            "tensor_count": rows[0]["tensor_count"],
            "tensor_bytes": rows[0]["tensor_bytes"],
            "dtype_counts": rows[0]["dtype_counts"],
        }
    report["summary"] = summaries
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
