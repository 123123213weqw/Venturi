"""Inspect or load a local safetensors checkpoint. Does not download or run models."""

import argparse
from dataclasses import asdict
import json
import sys

import torch

from .checkpoint import inspect_checkpoint
from .loader import load_checkpoint
from .verification import verify_loaded_checkpoint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="venturi")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("inspect", "load", "verify"):
        child = commands.add_parser(command)
        child.add_argument("path")
        child.add_argument("--json", action="store_true", dest="as_json")
        if command in ("load", "verify"):
            child.add_argument("--device", default="cuda:0")
            child.add_argument(
                "--loader", choices=("siphon", "safetensors"), default="siphon"
            )
            child.add_argument("--siphon-backend")
            child.add_argument(
                "--buffer-size", type=int, help="Siphon GPU ring-buffer size in bytes"
            )
            child.add_argument(
                "--require-float-dtype",
                choices=("F16", "BF16", "F32", "F64"),
                help="Require this dtype for floating-point tensors; preserve integer/bool buffers",
            )
        if command == "verify":
            child.add_argument("--chunk-bytes", type=int, default=16 * 2**20)
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            summary = inspect_checkpoint(args.path).summary()
        else:
            model = load_checkpoint(
                args.path,
                device=args.device,
                loader=args.loader,
                siphon_backend=args.siphon_backend,
                buffer_size=args.buffer_size,
                require_float_dtype=args.require_float_dtype,
            )
            summary = model.summary()
            if model.stats.device.startswith("cuda"):
                device = torch.device(model.stats.device)
                summary["cuda_allocator"] = {
                    "allocated_bytes": torch.cuda.memory_allocated(device),
                    "reserved_bytes": torch.cuda.memory_reserved(device),
                    "process_peak_allocated_bytes": torch.cuda.max_memory_allocated(
                        device
                    ),
                    "process_peak_reserved_bytes": torch.cuda.max_memory_reserved(
                        device
                    ),
                    "scope": "PyTorch allocator only; excludes native allocations and CUDA context",
                }
            if args.command == "verify":
                stats = verify_loaded_checkpoint(model, chunk_bytes=args.chunk_bytes)
                summary["verification"] = {"passed": True, **asdict(stats)}
    except (ValueError, RuntimeError, OSError) as error:
        print(f"venturi: {error}", file=sys.stderr)
        return 1
    if args.as_json:
        print(json.dumps(summary, indent=2))
    else:
        print(
            f"{summary['tensor_count']} tensors, {summary['tensor_bytes'] / 2**30:.3f} GiB, {len(summary['files'])} shard(s)"
        )
        if "load" in summary:
            stats = summary["load"]
            print(
                f"Loaded through {stats['loader']} onto {stats['device']} in {stats['total_seconds']:.3f}s"
            )
        if "verification" in summary:
            verification = summary["verification"]
            print(
                f"Verified {verification['verified_tensors']} tensors / {verification['verified_bytes']} bytes exactly in {verification['verification_seconds']:.3f}s"
            )
    return 0
