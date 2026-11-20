"""Separate native TP scalar/cohort timeline; never throughput evidence.

Batch has one native_unsplit C-call range for eight logical requests. Scalar has
eight native_unsplit submitting-thread ranges. Do not equate range/request count.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import torch

from benchmark_qwen_native_parallel import run_wave
from venturi.models import NativeQwenModel, QwenQ8Weights
from venturi.ninfer import NInferArtifact
from venturi.ops import CudaBackend


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact")
    p.add_argument("--library", required=True)
    p.add_argument(
        "--projection-rows",
        type=int,
        choices=(32, 64, 128, 256),
        default=32,
        help="Internal projection workspace rows; not per-slot context capacity",
    )
    p.add_argument("--case", required=True)
    p.add_argument("--mode", choices=("scalar", "batch"), required=True)
    p.add_argument(
        "--projection-profile", choices=("fp16", "q8-residual"), default="fp16"
    )
    p.add_argument("--count", type=int, default=8, choices=(1, 2, 4, 8))
    p.add_argument("--output", required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    case = json.loads(Path(a.case).read_text())
    prompt, expected = case["prompt_token_ids"], case["expected_generated_ids"]
    model = NativeQwenModel.from_q8(
        QwenQ8Weights.from_ninfer(NInferArtifact(a.artifact)),
        CudaBackend(a.library),
        weight_mode="resident",
        dtype=torch.float16,
        device="cuda:0",
        max_tokens=a.projection_rows,
        tensor_parallel_device="cuda:1",
        tensor_parallel_transport="host",
        projection_profile=a.projection_profile,
    )
    sessions, batch = [], None
    try:
        if a.mode == "batch":
            batch = model.create_batch([256] * 8)
            assert (
                batch.generate([prompt] * a.count, max_new_tokens=len(expected))
                == [expected] * a.count
            )
            batch.reset()
            with torch.cuda.nvtx.range("native_unsplit"):
                actual = batch.generate(
                    [prompt] * a.count, max_new_tokens=len(expected)
                )
            assert actual == [expected] * a.count
            result = dict(batch_info=batch.info(), native_call_ranges=1)
        else:
            sessions = [model.create_session(256) for _ in range(8)]
            with ThreadPoolExecutor(max_workers=a.count) as pool:
                calls = [
                    lambda s=s: s.generate(prompt, max_new_tokens=len(expected))
                    for s in sessions[: a.count]
                ]
                warm = run_wave(pool, calls, len(expected))
                assert all(r["ids"] == expected for r in warm["requests"])
                for s in sessions:
                    s.reset()
                measured = run_wave(pool, calls, len(expected), "native_unsplit")
                assert all(r["ids"] == expected for r in measured["requests"])
            result = dict(native_call_ranges=a.count, instrumented_wave=measured)
        Path(a.output).write_text(
            json.dumps(
                dict(
                    scope=__doc__,
                    profiled=True,
                    logical_requests=a.count,
                    projection_profile=a.projection_profile,
                    projection_rows=a.projection_rows,
                    mode=a.mode,
                    **result,
                ),
                indent=2,
            )
            + "\n"
        )
    finally:
        if batch is not None:
            batch.close()
        for s in sessions:
            s.close()
        model.close()


if __name__ == "__main__":
    main()
