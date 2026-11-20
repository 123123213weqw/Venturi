"""Separate native MTP1/2 timeline; not unprofiled serving throughput."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import torch
from benchmark_qwen_native_parallel import run_wave
from venturi.models import (
    NativeQwenModel,
    NativeQwenMTPModel,
    QwenQ8Weights,
    QwenMTPWeights,
)
from venturi.ninfer import NInferArtifact
from venturi.ops import CudaBackend


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact")
    p.add_argument("--mtp-artifact", required=True)
    p.add_argument("--library", required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--depth", type=int, choices=(1, 2), required=True)
    p.add_argument("--count", type=int, choices=(1, 8), required=True)
    p.add_argument("--recovery", choices=("replay", "prefix"), default="replay")
    p.add_argument("--output", required=True)
    p.add_argument("--target-layer-graph", action="store_true")
    p.add_argument(
        "--projection-profile",
        choices=(
            "q8-residual",
            "q8-residual-column-mlp",
            "q8-residual-column-mlp-head-gdn",
        ),
        default="q8-residual",
    )
    p.add_argument(
        "--mtp-draft-projection-profile",
        choices=("fp16", "q8-residual"),
        default="fp16",
    )
    a = p.parse_args()
    if a.target_layer_graph and (a.depth != 2 or a.recovery != "prefix"):
        p.error("Target layer graph requires depth two and prefix recovery")
    torch.set_num_threads(4)
    case = json.loads(Path(a.case).read_text())
    prompt = case["prompt_token_ids"]
    expected = case["expected_generated_ids"]
    target = NativeQwenModel.from_q8(
        QwenQ8Weights.from_ninfer(NInferArtifact(a.artifact)),
        CudaBackend(a.library),
        weight_mode="resident",
        dtype=torch.float16,
        device="cuda:0",
        max_tokens=32,
        tensor_parallel_device="cuda:1",
        tensor_parallel_transport="host",
        projection_profile=a.projection_profile,
    )
    model = None
    sessions = []
    try:
        model = NativeQwenMTPModel.from_q8(
            target,
            QwenMTPWeights.from_ninfer(NInferArtifact(a.mtp_artifact)),
            draft_projection_profile=a.mtp_draft_projection_profile,
        )
        sessions = [model.create_session(256) for _ in range(8)]
        if a.recovery == "prefix":
            for s in sessions:
                s.prepare_prefix()
        if a.target_layer_graph:
            for s in sessions:
                s.enable_target_layer_graph()
        with ThreadPoolExecutor(max_workers=a.count) as pool:
            calls = [
                lambda s=s: s.generate(
                    prompt,
                    max_new_tokens=len(expected),
                    draft_tokens=a.depth,
                    recovery=a.recovery,
                )
                for s in sessions[: a.count]
            ]
            warm = run_wave(pool, calls, len(expected))
            assert all(r["ids"] == expected for r in warm["requests"])
            for s in sessions:
                s.reset()
            measured = run_wave(pool, calls, len(expected), "native_unsplit")
            assert all(r["ids"] == expected for r in measured["requests"])
        report = dict(
            scope=__doc__,
            profiled=True,
            depth=a.depth,
            target_recovery=a.recovery,
            target_layer_graph=a.target_layer_graph,
            logical_requests=a.count,
            native_call_ranges=a.count,
            projection_rows=32,
            capacity_per_slot=256,
            projection_profile=a.projection_profile,
            draft_profile=a.mtp_draft_projection_profile,
            instrumented_wave=measured,
            stats=[s.stats() for s in sessions[: a.count]],
            library_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),
        )
        Path(a.output).write_text(json.dumps(report, indent=2) + "\n")
    finally:
        for s in sessions:
            s.close()
        if model is not None:
            model.close()
        target.close()


if __name__ == "__main__":
    main()
