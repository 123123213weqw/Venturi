"""Inspect or load a local safetensors checkpoint. Does not download or run models."""

import argparse
import json
import sys

from .checkpoint import inspect_checkpoint
from .loader import load_checkpoint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="venturi")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("inspect", "load"):
        child = commands.add_parser(command)
        child.add_argument("path")
        child.add_argument("--json", action="store_true", dest="as_json")
        if command == "load":
            child.add_argument("--device", default="cuda:0")
            child.add_argument(
                "--loader", choices=("siphon", "safetensors"), default="siphon"
            )
            child.add_argument("--siphon-backend")
            child.add_argument(
                "--buffer-size", type=int, help="Siphon GPU ring-buffer size in bytes"
            )
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
            )
            summary = model.summary()
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
    return 0
