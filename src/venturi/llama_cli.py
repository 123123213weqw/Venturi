"""Explicit independent classic-Llama generation, no build/download on invocation."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .generation import generate_text, generate_tokens
from .text import LocalTokenizer
from .loader import load_checkpoint
from .models.llama import LlamaWeights
from .ops import CudaBackend, TorchBackend


def main(argv=None):
    parser = argparse.ArgumentParser(prog="venturi llama")
    parser.add_argument("command", choices=("generate", "inspect"))
    parser.add_argument("path")
    parser.add_argument(
        "--backend", choices=("cuda", "torch-reference"), default="cuda"
    )
    parser.add_argument("--blocks-library")
    parser.add_argument("--device", default="cuda:0")
    prompts = parser.add_mutually_exclusive_group()
    prompts.add_argument("--prompt")
    prompts.add_argument(
        "--token-ids", help="Comma-separated IDs; bypass tokenizer only"
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args(argv)
    try:
        # Bind/validate CPU tensors first; upload only owned resident tensors.
        loaded = load_checkpoint(args.path, device="cpu", loader="safetensors")
        bound = LlamaWeights.bind(loaded)
        if args.command == "inspect":
            print(
                json.dumps(
                    {"config": asdict(bound.config), "bindings": bound.bound_names},
                    indent=2,
                )
            )
            return 0
        if args.prompt is None and args.token_ids is None:
            raise ValueError("generate requires --prompt or --token-ids")
        if args.backend == "cuda":
            if not args.blocks_library:
                raise ValueError(
                    "Native generation requires an explicit --blocks-library; build separately"
                )
            backend = CudaBackend(args.blocks_library)
        else:
            backend = TorchBackend()
        model = bound.upload(backend, device=args.device)
        del loaded, bound  # Release CPU tensors, not another GPU checkpoint copy.
        if args.token_ids is not None:
            ids = [int(x.strip()) for x in args.token_ids.split(",")]
            result = generate_tokens(model, ids, max_new_tokens=args.max_new_tokens)
        else:
            tokenizer = LocalTokenizer(
                Path(args.path) / "tokenizer.json", vocab_size=model.config.vocab_size
            )
            result = generate_text(
                model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens
            )
        print(json.dumps(asdict(result), indent=2))
        return 0
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"venturi llama: {error}\n")
