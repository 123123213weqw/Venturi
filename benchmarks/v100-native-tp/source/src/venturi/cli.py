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
    arguments = sys.argv[1:] if argv is None else argv
    if arguments and arguments[0] == "qwen-native":
        from .qwen_native_cli import main as native_qwen_main

        return native_qwen_main(arguments[1:])
    if arguments and arguments[0] == "llama":
        from .llama_cli import main as llama_main

        return llama_main(arguments[1:])
    if arguments and arguments[0] == "qwen":
        from .qwen_cli import main as qwen_main

        return qwen_main(arguments[1:])
    parser = argparse.ArgumentParser(prog="venturi")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("qwen-native", help="Owned Q8 C++/CUDA runtime: inspect or explicit greedy generation")
    commands.add_parser(
        "llama", help="Independent classic Llama checkpoint bindings and generation"
    )
    commands.add_parser("qwen", help="Explicit optional native Qwen backend: inspect/stage/generate/benchmark")
    for command in ("inspect", "load", "verify", "generate", "benchmark"):
        child = commands.add_parser(command)
        child.add_argument("path")
        child.add_argument("--json", action="store_true", dest="as_json")
        if command in ("load", "verify", "generate", "benchmark"):
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
        if command in ("generate", "benchmark"):
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
            child.add_argument(
                "--execution", choices=("eager", "cuda-graph"), default="eager",
                help="Opt-in prepared FP16 CUDA prefill/decode graphs; preparation is reported separately",
            )
        if command == "benchmark":
            child.add_argument("--warmup", type=int, default=1)
            child.add_argument("--repeats", type=int, default=5)
            child.add_argument(
                "--ignore-eos", action="store_true",
                help="Explicit fixed-length workload; normal generation still respects EOS",
            )
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            summary = inspect_checkpoint(args.path).summary()
        else:
            tokenizer = None
            if args.command in ("generate", "benchmark"):
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
                if args.execution == "cuda-graph":
                    if not args.device.startswith("cuda"):
                        raise ValueError("CUDA graph decode requires a CUDA device")
                    if args.max_new_tokens > 257:
                        raise ValueError("CUDA graph decode supports at most 257 new tokens (256 graphs)")
                if args.command == "benchmark":
                    if args.max_new_tokens == 0 or args.warmup < 0 or args.repeats < 1:
                        raise ValueError(
                            "benchmark requires max_new_tokens >= 1, warmup >= 0, repeats >= 1"
                        )
                if not tokenizer.encode(args.prompt) and config.bos_token_id is None:
                    raise ValueError("Empty prompt requires a configured BOS token")
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
            if args.command in ("generate", "benchmark"):
                from .generation import generate_text
                from .models import GPTNeoXModel

                runner = GPTNeoXModel(model)
                session = None
                if args.execution == "cuda-graph" and args.max_new_tokens:
                    ids = tokenizer.encode(args.prompt) or (runner.config.bos_token_id,)
                    session = runner.create_cuda_graph_session(
                        len(ids) + args.max_new_tokens, decode_start=len(ids),
                        decode_steps=args.max_new_tokens - 1,
                        prefill_lengths=(len(ids),),
                    )
                preparation_seconds = getattr(session, "preparation_seconds", 0.0)
                if args.command == "generate":
                    result = generate_text(
                        runner, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens,
                        session=session,
                    )
                    summary["generation"] = asdict(result)
                    summary["generation_execution"] = {
                        "mode": args.execution, "preparation_seconds": preparation_seconds,
                        "request_including_preparation_seconds": preparation_seconds + result.total_seconds,
                        "time_to_first_token_including_preparation_seconds": (
                            preparation_seconds + result.time_to_first_token_seconds
                            if result.time_to_first_token_seconds is not None else None
                        ),
                    }
                else:
                    from .benchmark import benchmark_tokens

                    ids = tokenizer.encode(args.prompt) or (runner.config.bos_token_id,)
                    summary["benchmark"] = benchmark_tokens(
                        runner, ids, max_new_tokens=args.max_new_tokens,
                        warmup=args.warmup, repeats=args.repeats,
                        ignore_eos=args.ignore_eos, session=session,
                    )
                summary["tokenizer"] = {
                    "file": tokenizer.path.name,
                    "sha256": tokenizer.sha256,
                }
                summary["execution"] = (
                    f"Venturi GPT-NeoX dense {args.execution} runner; PyTorch operators, no Transformers model"
                )
    except (ValueError, RuntimeError, OSError) as error:
        print(f"venturi: {error}", file=sys.stderr)
        return 1
    if args.as_json:
        print(json.dumps(summary, indent=2))
    elif args.command == "generate":
        print(summary["generation"]["text"])
    elif args.command == "benchmark":
        stats = summary["benchmark"]
        if stats["preparation_seconds"]:
            print(f"Graph preparation: {stats['preparation_seconds']:.3f} s (excluded from warmed timings)")
        print(f"TTFT median: {stats['time_to_first_token_seconds']['median'] * 1000:.3f} ms")
        rate = stats["decode_tokens_per_second"]
        print("Decode: n/a (no decode tokens)" if rate is None else f"Decode: {rate:.3f} tokens/s")
        if stats["peak_cuda_allocated_bytes"] is not None:
            print(f"Peak PyTorch allocated: {stats['peak_cuda_allocated_bytes'] / 2**20:.3f} MiB")
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
