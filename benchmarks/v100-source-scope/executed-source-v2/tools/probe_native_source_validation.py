"""Mark raw native MTP2 calls for a separate checkpoint-metadata syscall audit.

No source guard is disabled. Loading, SDK validation, reset and result validation
are outside the marked ctypes call; output IDs and acceptance remain checked.
"""

import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import time

import torch

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
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--instrumented", action="store_true")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if a.repeats not in (1, 3):
        p.error("This bounded diagnostic permits one or three marked calls")
    torch.set_num_threads(4)
    case = json.loads(Path(a.case).read_text())
    prompt, expected = case["prompt_token_ids"], case["expected_generated_ids"]
    target = NativeQwenModel.from_q8(
        QwenQ8Weights.from_ninfer(NInferArtifact(a.artifact)),
        CudaBackend(a.library),
        weight_mode="resident",
        dtype=torch.float16,
        device="cuda:0",
        max_tokens=32,
        tensor_parallel_device="cuda:1",
        tensor_parallel_transport="host",
        projection_profile="q8-residual-column-mlp",
    )
    model, sessions = None, []
    try:
        model = NativeQwenMTPModel.from_q8(
            target,
            QwenMTPWeights.from_ninfer(NInferArtifact(a.mtp_artifact)),
            draft_projection_profile="q8-residual",
        )
        sessions = [model.create_session(256) for _ in range(8)]
        for s in sessions:
            s.prepare_prefix()
        session = sessions[0]
        warm = session.generate(
            prompt, max_new_tokens=len(expected), draft_tokens=2, recovery="prefix"
        )
        assert warm == expected
        ids = (C.c_int64 * len(prompt))(*prompt)
        result = (C.c_int64 * len(expected))()
        count = C.c_int()
        fn = model._native.venturi_qwen_generate2_prefix
        calls = []
        for _ in range(a.repeats):
            session.reset()
            session._check()
            cpu = time.process_time()
            thread = time.thread_time()
            epoch = time.time()
            start = time.perf_counter()
            rc = fn(
                session._handle,
                ids,
                len(ids),
                len(expected),
                None,
                0,
                result,
                C.byref(count),
            )
            wall = time.perf_counter() - start
            end = time.time()
            thread = time.thread_time() - thread
            cpu = time.process_time() - cpu
            model._backend._check(rc)
            actual = list(result[: count.value])
            assert actual == expected
            stats = session.stats()
            calls.append(
                dict(
                    start_unix=epoch,
                    end_unix=end,
                    native_call_wall_seconds=wall,
                    process_cpu_seconds=cpu,
                    thread_cpu_seconds=thread,
                    output_ids=actual,
                    output_ids_equal=True,
                    stats=stats,
                )
            )
        report = dict(
            scope=__doc__,
            instrumented=a.instrumented,
            calls=calls,
            checkpoint_paths=[
                str(Path(a.artifact).resolve()),
                str(Path(a.mtp_artifact).resolve()),
            ],
            library=str(Path(a.library).resolve()),
            library_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),
            profiles=dict(
                target="q8-residual-column-mlp", draft="q8-residual", kv="fp16"
            ),
            projection_rows=32,
            prepared_slots=8,
            capacity_per_slot=256,
            source_guards_unchanged=True,
            source_validation_policy=target.projection_info["source_validation_policy"],
            python_math_in_marked_call=False,
            no_http_throughput_claim=True,
        )
        Path(a.output).write_text(json.dumps(report, indent=2) + "\n")
    finally:
        for session in sessions:
            session.close()
        if model is not None:
            model.close()
        target.close()


if __name__ == "__main__":
    main()
