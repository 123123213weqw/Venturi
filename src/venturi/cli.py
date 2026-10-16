"""Local checkpoint tools and explicit, architecture-bound text generation."""

import argparse
from dataclasses import asdict
import json
import platform
import sys

import safetensors
import torch

from .checkpoint import inspect_checkpoint
from .loader import load_checkpoint
from .verification import verify_loaded_checkpoint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="venturi")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("inspect", "load", "verify", "generate"):
        child = commands.add_parser(command)
        child.add_argument("path")
        child.add_argument("--json", action="store_true", dest="as_json")
        if command in ("load", "verify", "generate"):
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
        if command == "generate":
            child.add_argument(
                "--prompt",
                required=True,
                help="Raw continuation prompt, not a chat message",
            )
            child.add_argument(
                "--tokenizer",
                help="Local tokenizer.json; defaults to checkpoint directory",
            )
            child.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            summary = inspect_checkpoint(args.path).summary()
        else:
            tokenizer = None
            if args.command == "generate":
                from .generation import LocalTokenizer
                from .models import GPTNeoXConfig

                checkpoint = inspect_checkpoint(args.path)
                config = GPTNeoXConfig.from_dict(checkpoint.config)
                tokenizer = LocalTokenizer(
                    args.tokenizer or checkpoint.directory / "tokenizer.json",
                    vocab_size=config.vocab_size,
                )
                if args.max_new_tokens < 0:
                    raise ValueError("max_new_tokens must be nonnegative")
                prompt_length = max(1, len(tokenizer.encode(args.prompt)))
                if prompt_length + args.max_new_tokens > config.max_position_embeddings:
                    raise ValueError(
                        "Prompt plus requested output exceeds max_position_embeddings"
                    )
            model = load_checkpoint(
                args.path,
                device=args.device,
                loader=args.loader,
                siphon_backend=args.siphon_backend,
                buffer_size=args.buffer_size,
                require_float_dtype=args.require_float_dtype,
            )
            summary = model.summary()
            summary["runtime"] = {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "safetensors": safetensors.__version__,
                "torch_cuda": torch.version.cuda,
                "torch_cpu_threads": torch.get_num_threads(),
            }
            if model.stats.device.startswith("cuda"):
                device = torch.device(model.stats.device)
                summary["runtime"]["device_name"] = torch.cuda.get_device_name(device)
                summary["runtime"]["device_capability"] = list(
                    torch.cuda.get_device_capability(device)
                )
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
            if args.command == "generate":
                from .generation import generate_text
                from .models import GPTNeoXModel

                result = generate_text(
                    GPTNeoXModel(model),
                    tokenizer,
                    args.prompt,
                    max_new_tokens=args.max_new_tokens,
                )
                summary["generation"] = asdict(result)
                summary["tokenizer"] = {
                    "file": tokenizer.path.name,
                    "sha256": tokenizer.sha256,
                }
                summary["execution"] = (
                    "Venturi GPT-NeoX dense eager runner; PyTorch operators, no Transformers model"
                )
    except (ValueError, RuntimeError, OSError) as error:
        print(f"venturi: {error}", file=sys.stderr)
        return 1
    if args.as_json:
        print(json.dumps(summary, indent=2))
    elif args.command == "generate":
        print(summary["generation"]["text"])
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
