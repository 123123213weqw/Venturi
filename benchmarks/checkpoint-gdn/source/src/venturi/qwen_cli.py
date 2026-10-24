"""Explicit Qwen commands; the dense safetensors runner is not substituted."""

import argparse
import json
import os
from pathlib import Path
import sys

from .backends.qwen import QwenNativeSession
from .ninfer import NInferArtifact


def main(argv=None):
    parser = argparse.ArgumentParser(prog="venturi qwen", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "stage", "generate", "benchmark"):
        child = commands.add_parser(name)
        child.add_argument("path", help="Existing local NInfer v3 checkpoint")
        child.add_argument("--json", action="store_true", dest="as_json")
        if name == "inspect":
            continue
        if name == "stage":
            child.add_argument(
                "--directory",
                required=True,
                help="Explicit staging storage, e.g. /dev/shm/venturi",
            )
            child.add_argument(
                "--reuse",
                action="store_true",
                help="Verify and reuse a previously staged artifact",
            )
            continue
        child.add_argument(
            "--runner",
            default=os.environ.get("VENTURI_QWEN_RUNNER"),
            help="Explicit venturi-qwen-runner executable (or VENTURI_QWEN_RUNNER)",
        )
        inputs = child.add_mutually_exclusive_group(required=True)
        inputs.add_argument(
            "--prompt", action="append", help="Repeat for 1..16 ordered prompts"
        )
        inputs.add_argument(
            "--prompts-file",
            help="Local JSON array of 1..16 strings (remote path when using the SSH launcher)",
        )
        child.add_argument(
            "--mode",
            choices=("single", "parallel"),
            default="single",
            help="Serial dispatch (default) or native batched decode",
        )
        child.add_argument(
            "--max-concurrency",
            type=int,
            help="Startup lane reservation: 1..8; default single=1, parallel=4",
        )
        child.add_argument(
            "--kv-capacity",
            type=int,
            help="Shared physical KV token budget; default all lanes at full context",
        )
        child.add_argument(
            "--raw-prompt",
            action="store_true",
            help="No chat template; default is a non-thinking user message",
        )
        child.add_argument("--max-new-tokens", type=int, default=32)
        child.add_argument("--max-context", type=int, default=2048)
        child.add_argument("--prefill-chunk", type=int, default=128)
        child.add_argument("--device", type=int, default=0)
        child.add_argument(
            "--execution", choices=("eager", "cuda-graph"), default="cuda-graph"
        )
        child.add_argument(
            "--cpu-wait",
            choices=("blocking", "auto"),
            default="blocking",
            help="CUDA synchronization wait policy: blocking saves CPU; auto retains CUDA's heuristic",
        )
        child.add_argument("--kv-dtype", choices=("bf16", "k8v4"), default="bf16")
        child.add_argument(
            "--speculative",
            choices=("none", "dflash2"),
            default="none",
            help="Opt-in experimental DFlash2; universal greedy-ID equivalence is not established",
        )
        child.add_argument("--draft-tokens", type=int, default=8)
        child.add_argument("--verify-tree", action="store_true")
        child.add_argument("--optimized-proposal", action="store_true")
        child.add_argument("--timeout", type=float, default=300)
        if name == "benchmark":
            child.add_argument("--warmup", type=int, default=1)
            child.add_argument("--repeats", type=int, default=3)
            child.add_argument(
                "--ignore-eos",
                action="store_true",
                help="Explicit fixed-output benchmark only",
            )
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            report = NInferArtifact(args.path).summary()
        elif args.command == "stage":
            from .staging import stage_artifact

            report = stage_artifact(args.path, args.directory, reuse=args.reuse)
        else:
            if not args.runner:
                raise ValueError(
                    "An explicit --runner or VENTURI_QWEN_RUNNER is required; no implicit backend selection"
                )
            if args.max_new_tokens < 1:
                raise ValueError("max_new_tokens must be positive")
            if args.command == "benchmark" and (args.warmup < 0 or args.repeats < 1):
                raise ValueError("benchmark requires warmup >= 0 and repeats >= 1")
            prompts = args.prompt
            if args.prompts_file:
                prompts = json.loads(Path(args.prompts_file).read_text())
            if (
                not isinstance(prompts, list)
                or not 1 <= len(prompts) <= 16
                or any(not isinstance(p, str) for p in prompts)
            ):
                raise ValueError("Provide 1..16 prompt strings")
            batch = len(prompts) > 1 or args.mode == "parallel"
            with QwenNativeSession(
                args.path,
                runner=args.runner,
                max_context=args.max_context,
                mode=args.mode,
                max_concurrency=args.max_concurrency,
                kv_capacity=args.kv_capacity,
                prefill_chunk=args.prefill_chunk,
                device=args.device,
                cuda_graph=args.execution == "cuda-graph",
                cpu_wait=args.cpu_wait,
                kv_dtype=args.kv_dtype,
                speculative=args.speculative,
                draft_tokens=args.draft_tokens,
                verify_tree=args.verify_tree,
                optimized_proposal=args.optimized_proposal,
                timeout=args.timeout,
            ) as session:
                generation = dict(
                    max_new_tokens=args.max_new_tokens, raw_prompt=args.raw_prompt
                )
                if args.command == "generate":
                    report = {
                        "backend": session.ready,
                        ("batch" if batch else "generation"): (
                            session.generate_many(prompts, **generation)
                            if batch
                            else session.generate(prompts[0], **generation)
                        ),
                    }
                else:
                    benchmark = session.benchmark_many if batch else session.benchmark
                    report = benchmark(
                        prompts if batch else prompts[0],
                        warmup=args.warmup,
                        repeats=args.repeats,
                        ignore_eos=args.ignore_eos,
                        **generation,
                    )
        if args.as_json:
            print(json.dumps(report, indent=2))
        elif args.command == "generate":
            if "batch" in report:
                for item in report["batch"]["results"]:
                    print(f"[{item['index']}] {item['text']}")
            else:
                print(report["generation"]["text"])
        elif args.command == "benchmark":
            print(f"Backend: {report['backend']} (resident model; startup excluded)")
            if "output_tokens_per_second" in report:
                print(f"Mode: {report['mode']}")
                print(
                    f"Total output: {report['output_tokens_per_second']:.3f} tokens/s (batch wall time)"
                )
            else:
                rate = report["decode_tokens_per_second"]
                print("Decode: n/a" if rate is None else f"Decode: {rate:.3f} tokens/s")
            print(
                f"First token median: {report['median_first_token_seconds'] * 1000:.3f} ms"
            )
        elif args.command == "stage":
            print(report["path"])
        else:
            print(
                f"NInfer v3: {report['tensor_count']} tensors, {report['file_bytes'] / 2**30:.3f} GiB"
            )
            print(json.dumps(report["formats"], sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"venturi qwen: {error}", file=sys.stderr)
        return 1
