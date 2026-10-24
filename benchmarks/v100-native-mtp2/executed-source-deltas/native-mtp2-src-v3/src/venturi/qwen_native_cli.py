"""Explicit owned Q8 native runtime; raw prompts and greedy-only generation."""

import argparse
from collections import Counter
import hashlib
import json
import resource
from time import perf_counter

import torch

from .generation import LocalTokenizer
from .models import (
    NativeQwenModel,
    NativeQwenPipelineModel,
    QwenQ8Weights,
    NativeQwenMTPModel,
    QwenMTPWeights,
)
from .ninfer import NInferArtifact
from .ops import CudaBackend
from .ops.vocabulary import _ids


def _csv_ids(value):
    return [int(x.strip()) for x in value.split(",")] if value.strip() else []


def _hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(16 * 2**20):
            h.update(b)
    return h.hexdigest()


def _cpu():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def main(argv=None):
    p = argparse.ArgumentParser(prog="venturi qwen-native", description=__doc__)
    p.add_argument("command", choices=("inspect", "generate"))
    p.add_argument("path")
    p.add_argument(
        "--blocks-library",
        help="Explicit ABI-matching native library; never builds automatically",
    )
    p.add_argument(
        "--weight-mode",
        choices=("streaming", "resident"),
        help="Required for generation; never an implicit fallback",
    )
    p.add_argument(
        "--dtype",
        choices=("fp16", "bf16"),
        default="bf16",
        help="Activation/KV dtype; SM70 needs explicit fp16",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--projection-profile",
        choices=("fp16", "q8-residual"),
        default="fp16",
        help="Original typed activation math (default), or explicit experimental residual-A8 FP16 target arithmetic",
    )
    p.add_argument(
        "--tensor-parallel-transport",
        choices=("peer", "host"),
        default="peer",
        help="Explicit transport; peer mode verifies actual bytes and never silently falls back",
    )
    p.add_argument(
        "--tensor-parallel-device",
        help="Optional explicit second CUDA device; resident row shards only",
    )
    p.add_argument(
        "--pipeline-device",
        help="Explicit second CUDA device for native layer pipeline",
    )
    p.add_argument("--pipeline-split-layer", type=int)
    p.add_argument("--pipeline-transport", choices=("host",), default="host")
    p.add_argument(
        "--mtp-artifact",
        help="Explicit original-source Q8 MTP payload; experimental native greedy verification",
    )
    p.add_argument(
        "--mtp-draft-tokens",
        type=int,
        choices=(1, 2),
        default=1,
        help="Explicit MTP proposal depth; depth one remains the default",
    )
    p.add_argument(
        "--expected-sha256", help="Optional full-file pin for a single-file artifact"
    )
    prompts = p.add_mutually_exclusive_group()
    prompts.add_argument(
        "--prompt", help="Raw continuation prompt: no automatic chat template"
    )
    prompts.add_argument(
        "--token-ids",
        help="Comma-separated already formatted prompt IDs; no tokenizer dependency",
    )
    p.add_argument(
        "--tokenizer",
        help="Local tokenizer.json for raw text input/decoding; no network or model code",
    )
    p.add_argument(
        "--eos-token-ids",
        help="Comma-separated EOS IDs; default from text config, empty string explicitly disables stopping",
    )
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Native prefill chunk/arena size, not context capacity",
    )
    p.add_argument(
        "--capacity",
        type=int,
        help="Default prompt length + max-new-tokens; no truncation",
    )
    args = p.parse_args(argv)
    total = perf_counter()
    try:
        artifact = NInferArtifact(args.path)
        weights = QwenQ8Weights.from_ninfer(artifact)
        verified = None
        if args.expected_sha256:
            if len(artifact.files) != 1:
                raise ValueError(
                    "A root-file hash alone cannot verify a multipart artifact"
                )
            verified = _hash(artifact.path)
            if verified != args.expected_sha256.lower():
                raise ValueError("Artifact SHA256 mismatch")
        if args.command == "inspect":
            print(
                json.dumps(
                    dict(
                        artifact_id=artifact.artifact_id,
                        config=weights.config,
                        formats=dict(
                            Counter(o["format"] for o in artifact.objects.values())
                        ),
                        file_bytes=sum(f.path.stat().st_size for f in artifact.files),
                        verified_single_file_sha256=verified,
                        scope="Metadata inspection, not payload validation, generation or quality qualification",
                    ),
                    indent=2,
                )
            )
            return 0
        if args.mtp_draft_tokens != 1 and not args.mtp_artifact:
            raise ValueError("--mtp-draft-tokens requires --mtp-artifact")
        if args.mtp_draft_tokens == 2 and args.max_tokens < 3:
            raise ValueError("MTP2 requires --max-tokens >= 3")
        if not args.blocks_library or not args.weight_mode:
            raise ValueError(
                "generate requires explicit --blocks-library and --weight-mode"
            )
        if args.projection_profile != "fp16" and (
            args.dtype != "fp16"
            or args.weight_mode != "resident"
            or args.pipeline_device
        ):
            raise ValueError(
                "q8-residual requires resident --dtype fp16 and supports single/TP, not PP"
            )
        if args.max_new_tokens <= 0:
            raise ValueError(
                "max-new-tokens must be positive; this runtime is greedy-only"
            )
        if args.prompt is None and args.token_ids is None:
            raise ValueError("generate requires a raw --prompt or explicit --token-ids")
        tokenizer = (
            LocalTokenizer(args.tokenizer, vocab_size=weights.config["vocab_size"])
            if args.tokenizer
            else None
        )
        if args.prompt is not None:
            if tokenizer is None:
                raise ValueError("Raw text requires an explicit local --tokenizer")
            ids = list(tokenizer.encode(args.prompt))
        else:
            ids = _csv_ids(args.token_ids)
        _ids(ids, weights.config["vocab_size"])
        eos = weights.config.get("eos_token_id", [])
        eos = [eos] if type(eos) is int else eos or []
        if args.eos_token_ids is not None:
            eos = _csv_ids(args.eos_token_ids)
        if eos:
            _ids(eos, weights.config["vocab_size"])
        capacity = (
            args.capacity
            if args.capacity is not None
            else len(ids) + args.max_new_tokens
        )
        if (
            capacity < len(ids) + args.max_new_tokens
            or capacity > weights.config["max_position_embeddings"]
        ):
            raise ValueError("Requested generation exceeds explicit capacity/context")
        load_start = perf_counter()
        common = dict(
            weight_mode=args.weight_mode,
            dtype=torch.float16 if args.dtype == "fp16" else torch.bfloat16,
            device=args.device,
            max_tokens=args.max_tokens,
        )
        if args.pipeline_device:
            if args.tensor_parallel_device or args.tensor_parallel_transport != "peer":
                raise ValueError("Select TP or PP, not a two-device TP/PP hybrid")
            model = NativeQwenPipelineModel.from_q8(
                weights,
                CudaBackend(args.blocks_library),
                **common,
                pipeline_device=args.pipeline_device,
                split_layer=args.pipeline_split_layer,
                transport=args.pipeline_transport,
            )
        else:
            if args.pipeline_split_layer is not None:
                raise ValueError("Pipeline split requires --pipeline-device")
            model = NativeQwenModel.from_q8(
                weights,
                CudaBackend(args.blocks_library),
                **common,
                tensor_parallel_device=args.tensor_parallel_device,
                tensor_parallel_transport=args.tensor_parallel_transport,
                projection_profile=args.projection_profile,
            )
        if args.mtp_artifact:
            target_model = model
            try:
                model = NativeQwenMTPModel.from_q8(
                    target_model,
                    QwenMTPWeights.from_ninfer(NInferArtifact(args.mtp_artifact)),
                )
            finally:
                # The MTP native owner retains the shared target weights.
                target_model.close()
        load_seconds = perf_counter() - load_start
        try:
            init = perf_counter()
            session = model.create_session(capacity)
            prepare_seconds = perf_counter() - init
            try:
                start, cpu = perf_counter(), _cpu()
                generated = session.generate(
                    ids,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_ids=eos,
                    **({"draft_tokens": 2} if args.mtp_draft_tokens == 2 else {}),
                )
                wall, used = perf_counter() - start, _cpu() - cpu
                info = session.info()
            finally:
                session.close()
        finally:
            model.close()
        print(
            json.dumps(
                dict(
                    runtime="owned C++/CUDA Q8 forward/cache/greedy control loop",
                    weight_mode=args.weight_mode,
                    dtype=args.dtype,
                    projection_profile=info.get(
                        "projection_profile",
                        args.projection_profile
                        if args.dtype == "fp16" or args.projection_profile != "fp16"
                        else "bf16",
                    ),
                    mtp_draft_projection_profile=args.dtype
                    if args.mtp_artifact
                    else None,
                    tensor_parallel_device=args.tensor_parallel_device,
                    tensor_parallel_transport=args.tensor_parallel_transport
                    if args.tensor_parallel_device
                    else None,
                    pipeline_device=args.pipeline_device,
                    pipeline_split_layer=args.pipeline_split_layer,
                    mtp_artifact=args.mtp_artifact,
                    speculative_draft_tokens=args.mtp_draft_tokens
                    if args.mtp_artifact
                    else 0,
                    sampling="explicit greedy, lowest-index ties, nonfinite reject; not source stochastic defaults",
                    prompt_tokens=len(ids),
                    generated_ids=generated,
                    text=tokenizer.decode(generated) if tokenizer else None,
                    finish_reason="eos"
                    if generated and generated[-1] in eos
                    else "max_new_tokens",
                    tokenizer_sha256=tokenizer.sha256 if tokenizer else None,
                    verified_single_file_sha256=verified,
                    model_load_wall_seconds=load_seconds,
                    session_preparation_wall_seconds=prepare_seconds,
                    generation_wall_seconds=wall,
                    generation_process_cpu_seconds=used,
                    native_memory=info,
                    timing_scope="Generation includes prefill; not steady-state decode TPS. Load/preparation reported separately.",
                    whole_model_cuda_graphs=False,
                    paged_kv=False,
                    scheduling=info.get(
                        "scheduling",
                        "Calls serialized per model on a private stream; not continuous batching",
                    ),
                    command_wall_seconds=perf_counter() - total,
                ),
                indent=2,
            )
        )
        return 0
    except (ValueError, RuntimeError, OSError, MemoryError) as e:
        p.exit(1, f"venturi qwen-native: {e}\n")


if __name__ == "__main__":
    main()
