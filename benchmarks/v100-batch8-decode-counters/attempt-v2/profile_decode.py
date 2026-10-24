"""Decode-only native cohort measurement; no build or service management.

The prefix call emits token 0 outside the measured range. Continuation consumes
that pending token and emits tokens 1..31 with the existing native generator.
Its first one-row-per-sequence group is labelled prefill by the API, but executes
the same group computation as decode; the boolean only increments counters.
This scope includes one API re-entry, not prompt prefill or first-token work.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
from time import perf_counter, process_time

import torch

from venturi.models import NativeQwenModel, QwenQ8Weights
from venturi.ninfer import NInferArtifact
from venturi.ops import CudaBackend


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def sync():
    for device in range(2):
        torch.cuda.synchronize(device)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact")
    p.add_argument("--case")
    p.add_argument("--library", required=True)
    p.add_argument("--output-directory", required=True)
    p.add_argument("--mode", choices=("fixture", "unprofiled", "counters"), required=True)
    p.add_argument("--selected", type=int, choices=(0, 1), default=0)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--profile-fixture", action="store_true")
    p.add_argument("--counter-fast", action="store_true",
                   help="Use pinned reference and separately validated unprofiled control; omit redundant unmeasured decode warmups")
    a = p.parse_args()
    assert a.repeats >= 3
    assert not a.profile_fixture or a.mode == "fixture"
    assert not a.counter_fast or a.mode == "counters"
    profiled = a.mode == "counters" or a.profile_fixture
    out = Path(a.output_directory)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    if a.mode == "fixture":
        from test_native_head_gdn import eligible_mtp

        weights, _, _, _ = eligible_mtp(out / f"fixture-{os.getpid()}")
        prompts = [[1, 2, 3, 4][:1 + i % 4] for i in range(8)]
        expected, maximum, rows, capacity = None, 8, 8, 64
    else:
        assert a.artifact and a.case
        weights = QwenQ8Weights.from_ninfer(NInferArtifact(a.artifact))
        case = json.loads(Path(a.case).read_text())
        prompts = [case["prompt_token_ids"]] * 8
        expected = [case["expected_generated_ids"]] * 8
        maximum, rows, capacity = len(expected[0]), 256, 256
        assert maximum == 32 and len(prompts[0]) == 23
    model = NativeQwenModel.from_q8(
        weights, CudaBackend(a.library), weight_mode="resident",
        dtype=torch.float16, device="cuda:0", max_tokens=rows,
        tensor_parallel_device="cuda:1", tensor_parallel_transport="host",
        projection_profile="q8-residual-column-mlp-head-gdn",
        reserve_bytes=0 if a.mode == "fixture" else 512 * 2**20,
    )
    batch = None
    report = dict(
        utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), args=vars(a),
        scope=__doc__, profiled=profiled, batch_size=8,
        max_new_tokens=maximum, measured_output_tokens=8 * (maximum - 1),
        prompt_lengths=list(map(len, prompts)), model_config=model.config,
        library_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),
        artifact_id=weights.artifact.artifact_id,
        devices=[dict(logical=i, name=torch.cuda.get_device_name(i),
                      uuid=str(torch.cuda.get_device_properties(i).uuid)) for i in range(2)],
        runs=[],
    )
    path = out / f"run-{os.getpid()}.json"
    try:
        batch = model.create_batch([capacity] * 8)
        if a.counter_fast:
            # The preceding standalone unprofiled control already validates the
            # uninterrupted/split paths. Still verify every profiled output ID.
            assert expected is not None
            reference_positions = [len(prompt) + maximum - 1 for prompt in prompts]
        else:
            whole = batch.generate(prompts, max_new_tokens=maximum)
            if expected is not None:
                assert whole == expected, "Pinned target trajectory mismatch"
            expected = whole
            reference_positions = [r["position"] for r in batch.info()["requests"]]
        reference_states = [batch.snapshot(i) for i in range(8)] if a.mode == "fixture" else None
        report["expected_outputs"] = expected
        report["reference_positions"] = reference_positions
        save(path, report)
        # First split run validates the exact measured path before counters start.
        for repeat in range(1 if a.counter_fast else 0, a.repeats + 1):
            batch.reset()
            first = batch.generate(prompts, max_new_tokens=1)
            assert first == [row[:1] for row in expected]
            prefix_positions = [r["position"] for r in batch.info()["requests"]]
            assert prefix_positions == list(map(len, prompts))
            pending = [[row[0]] for row in first]
            sync()
            torch.cuda.set_device(a.selected)
            collect = profiled and repeat > 0
            if collect:
                torch.cuda.profiler.start()
            wall, cpu = perf_counter(), process_time()
            try:
                tail = batch.generate(pending, max_new_tokens=maximum - 1)
                sync()
            finally:
                elapsed, used = perf_counter() - wall, process_time() - cpu
                if collect:
                    torch.cuda.profiler.stop()
            actual = [head + rest for head, rest in zip(first, tail)]
            assert actual == expected, "Continuation changed greedy target IDs"
            info = batch.info()
            assert [r["position"] for r in info["requests"]] == reference_positions
            assert info["group_forward_calls"] == maximum - 1
            assert info["head_calls"] == maximum - 1
            assert info["input_rows"] == 8 * (maximum - 1)
            assert info["max_group_rows"] == 8
            assert info["prefill_group_calls"] == 1
            assert info["decode_group_calls"] == maximum - 2
            if reference_states is not None:
                from test_qwen_tensor_parallel import same_state

                for i, state in enumerate(reference_states):
                    same_state(state, batch.snapshot(i))
            trial = dict(
                phase="split-correctness" if repeat == 0 else "measured",
                repeat=repeat, counters_enabled=collect, outputs=actual,
                start_monotonic=wall, end_monotonic=wall + elapsed,
                wall_seconds=elapsed, process_cpu_seconds=used, info=info,
                aggregate_decode_tps=None if profiled else 8 * (maximum - 1) / elapsed,
            )
            report["runs"].append(trial)
            save(path, report)
            print(json.dumps({k: trial[k] for k in (
                "phase", "repeat", "counters_enabled", "wall_seconds",
                "process_cpu_seconds", "aggregate_decode_tps")}), flush=True)
        if not profiled:
            timed = report["runs"][1:]
            mean = statistics.mean(r["wall_seconds"] for r in timed)
            report["unprofiled_summary"] = dict(
                repeats=len(timed), mean_decode_wall_seconds=mean,
                aggregate_decode_tps=8 * (maximum - 1) / mean,
                per_request_decode_tps=(maximum - 1) / mean,
                mean_process_cpu_seconds=statistics.mean(r["process_cpu_seconds"] for r in timed),
            )
        report["passed"] = True
        report["canonical_state_bitwise_checked"] = a.mode == "fixture"
        save(path, report)
    finally:
        if batch is not None:
            batch.close()
        model.close()


if __name__ == "__main__":
    main()
