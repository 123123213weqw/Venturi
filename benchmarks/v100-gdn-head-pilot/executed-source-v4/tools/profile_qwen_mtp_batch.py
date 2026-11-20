"""Separate fixed-cohort MTP1 target-batch trace, not unprofiled serving TPS."""

import argparse
import hashlib
import json
from pathlib import Path
import resource
from time import perf_counter
import torch
from venturi.models import (
    NativeQwenModel,
    NativeQwenMTPModel,
    QwenQ8Weights,
    QwenMTPWeights,
)
from venturi.ninfer import NInferArtifact
from venturi.ops import CudaBackend


def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact")
    p.add_argument("--mtp-artifact", required=True)
    p.add_argument("--library", required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--count", type=int, choices=(1, 8), required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--projection-profile",
        choices=("q8-residual", "q8-residual-column-mlp"),
        default="q8-residual",
    )
    p.add_argument(
        "--mtp-draft-projection-profile",
        choices=("fp16", "q8-residual"),
        default="fp16",
    )
    a = p.parse_args()
    torch.set_num_threads(4)
    case = json.loads(Path(a.case).read_text())
    ids = case["prompt_token_ids"]
    expected = case["expected_generated_ids"]
    slots = list(range(a.count))
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
    model = batch = None
    try:
        model = NativeQwenMTPModel.from_q8(
            target,
            QwenMTPWeights.from_ninfer(NInferArtifact(a.mtp_artifact)),
            draft_projection_profile=a.mtp_draft_projection_profile,
        )
        batch = model.create_batch([256] * 8)

        def call():
            return batch.generate(
                [ids] * a.count, max_new_tokens=len(expected), slots=slots
            )

        assert call() == [expected] * a.count
        batch.reset()
        start, process = perf_counter(), cpu()
        with torch.cuda.nvtx.range("native_unsplit"):
            result = call()
        wall, used = perf_counter() - start, cpu() - process
        assert result == [expected] * a.count
        Path(a.output).write_text(
            json.dumps(
                dict(
                    scope=__doc__,
                    profiled=True,
                    logical_requests=a.count,
                    native_call_ranges=1,
                    projection_rows=32,
                    capacity_per_slot=256,
                    projection_profile=a.projection_profile,
                    draft_profile=a.mtp_draft_projection_profile,
                    selected_slots=slots,
                    outputs=result,
                    instrumented_wave=dict(
                        wall_seconds=wall,
                        process_cpu_core_seconds=used,
                        emitted_tokens=a.count * len(expected),
                    ),
                    info=batch.info(),
                    library_sha256=hashlib.sha256(
                        Path(a.library).read_bytes()
                    ).hexdigest(),
                ),
                indent=2,
            )
            + "\n"
        )
    finally:
        if batch is not None:
            batch.close()
        if model is not None:
            model.close()
        target.close()


if __name__ == "__main__":
    main()
