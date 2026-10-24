"""Correctness-first fresh-cohort adaptive vs fixed MTP cost measurements.

One target/draft weight owner and one projection-row configuration are shared.
Cold online exploration, extra calibration, and steady measurements are separate.
No production service management, builds, downloads, or HTTP serving here.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import threading
from time import perf_counter, process_time

import torch

from venturi.adaptive_mtp import AdaptiveMTPPolicy
from venturi.models import (
    NativeQwenModel,
    NativeQwenMTPModel,
    QwenQ8Weights,
    QwenMTPWeights,
)
from venturi.ninfer import NInferArtifact
from venturi.ops import CudaBackend


def save(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact")
    p.add_argument("--mtp-artifact", required=True)
    p.add_argument("--library", required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--output-directory", required=True)
    p.add_argument("--projection-rows", type=int, default=256)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--capacity", type=int, default=256)
    p.add_argument("--profile", default="q8-residual-column-mlp-head-gdn")
    args = p.parse_args()
    if args.repeats < 3:
        p.error("At least three timed repeats required")
    out = Path(args.output_directory)
    out.mkdir(parents=True, exist_ok=False)
    case = json.loads(Path(args.case).read_text())
    prompt, maximum = case["prompt_token_ids"], case["max_new_tokens"]
    torch.set_num_threads(4)
    target = NativeQwenModel.from_q8(
        QwenQ8Weights.from_ninfer(NInferArtifact(args.artifact)),
        CudaBackend(args.library),
        weight_mode="resident",
        dtype=torch.float16,
        device="cuda:0",
        tensor_parallel_device="cuda:1",
        tensor_parallel_transport="host",
        projection_profile=args.profile,
        max_tokens=args.projection_rows,
    )
    model = adaptive = None
    stop = threading.Event()
    gpu = []

    def sample():
        while not stop.is_set():
            t = perf_counter()
            try:
                r = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,uuid,memory.used,utilization.gpu,power.draw,clocks.sm,clocks.mem,temperature.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=8,
                )
                gpu.append(
                    dict(
                        monotonic=t,
                        returncode=r.returncode,
                        csv=r.stdout.strip(),
                        error=r.stderr.strip(),
                    )
                )
            except Exception as e:
                gpu.append(dict(monotonic=t, error=type(e).__name__))
            stop.wait(0.5)

    sampler = threading.Thread(target=sample)
    rows = []
    try:
        model = NativeQwenMTPModel.from_q8(
            target,
            QwenMTPWeights.from_ninfer(NInferArtifact(args.mtp_artifact)),
            draft_projection_profile="q8-residual",
        )
        policy = AdaptiveMTPPolicy()
        adaptive = model.create_adaptive_batch([args.capacity] * 8, policy=policy)
        prepared = perf_counter()
        adaptive.prepare()
        preparation = perf_counter() - prepared
        bank_memory = {d: bank.info() for d, bank in adaptive._banks.items()}
        save(
            out / "configuration.json",
            dict(
                utc=datetime.now(timezone.utc).isoformat(),
                args=vars(args),
                case=case,
                target_artifact_id=target._weights.artifact.artifact_id,
                draft_artifact_id=model._mtp_weights.artifact.artifact_id,
                library_sha256=hashlib.sha256(
                    Path(args.library).read_bytes()
                ).hexdigest(),
                prepared_bank_info=bank_memory,
                preparation_wall_seconds=preparation,
                precision="Original Q8/group32 weights; existing residual-A8 FP16 target/draft arithmetic and FP16 KV unchanged",
                memory_scope="Banks have disjoint private requests; target/draft model bytes repeat across bank records, count shared weights once",
                counters="NVML busy is not achieved occupancy; hardware counters not collected; unchanged prior policy denies unprivileged counters",
            ),
        )
        sampler.start()
        for count in (1, 2, 4, 8):
            prompts = [prompt] * count
            depths = [0, 1, 2] if count == 1 else [0, 1]
            reference = None

            def run(phase, depth=None, learn=False):
                nonlocal reference
                start, cpu = perf_counter(), process_time()
                ids = adaptive.generate(
                    prompts, max_new_tokens=maximum, strategy=depth, learn=learn
                )
                wall, used = perf_counter() - start, process_time() - cpu
                info = adaptive.info()
                if reference is None:
                    reference = ids
                    assert ids == [case["expected_generated_ids"]] * count, (
                        "Pinned target trajectory mismatch"
                    )
                assert ids == reference, "Strategy changed generated target IDs"
                r = dict(
                    phase=phase,
                    count=count,
                    requested_depth=depth,
                    selected_depth=info["selected_depth"],
                    start_monotonic=start,
                    wall_seconds=wall,
                    process_cpu_seconds=used,
                    emitted_tokens=sum(map(len, ids)),
                    aggregate_output_tps=sum(map(len, ids)) / wall,
                    outputs=ids,
                    info=info,
                )
                rows.append(r)
                save(out / "rows.json", rows)
                print(
                    json.dumps(
                        {
                            k: r[k]
                            for k in (
                                "phase",
                                "count",
                                "requested_depth",
                                "selected_depth",
                                "wall_seconds",
                                "aggregate_output_tps",
                            )
                        }
                    ),
                    flush=True,
                )

            # All candidates complete the same correctness work before comparison.
            for depth in depths:
                run("correctness-and-warmup", depth)
            # True cold-policy sequence remains visible, including exploration cost.
            for _ in depths:
                run("cold-online-exploration", learn=True)
            # Two more labelled observations per strategy; not timed-result samples.
            for _ in range(2):
                for depth in depths:
                    run("extra-calibration", depth, learn=True)
            for repeat in range(args.repeats):
                order = depths + [None]
                if repeat % 2:
                    order.reverse()
                for depth in order:
                    run(
                        "timed-auto" if depth is None else "timed-fixed",
                        depth,
                        learn=depth is None,
                    )
        summary = []
        for count in (1, 2, 4, 8):
            for variant in ("auto", 0, 1, 2):
                trials = [
                    r
                    for r in rows
                    if r["count"] == count
                    and (
                        r["phase"] == "timed-auto"
                        if variant == "auto"
                        else r["phase"] == "timed-fixed"
                        and r["requested_depth"] == variant
                    )
                ]
                if not trials:
                    continue
                wall = statistics.mean(r["wall_seconds"] for r in trials)
                summary.append(
                    dict(
                        count=count,
                        variant=variant,
                        trials=len(trials),
                        mean_wall_seconds=wall,
                        aggregate_output_tps=trials[0]["emitted_tokens"] / wall,
                        mean_process_cpu_seconds=statistics.mean(
                            r["process_cpu_seconds"] for r in trials
                        ),
                        selected_depths=[r["selected_depth"] for r in trials],
                    )
                )
        save(
            out / "summary.json",
            dict(
                results=summary,
                correctness="Every candidate and adaptive cohort exactly matches the pinned greedy IDs",
                policy=policy.info(),
                timing_scope="Fresh fixed cohort includes Python policy/reset/native execution/metadata/return; no HTTP or loading. Aggregate tokens / global wall.",
                limitations="Short homogeneous 23-prompt/32-output workload; not 256K or arbitrary corpus qualification. No per-step adaptation or continuous admission.",
            ),
        )
    finally:
        stop.set()
        if sampler.is_alive():
            sampler.join(timeout=12)
        save(out / "gpu-samples.json", gpu)
        if adaptive is not None:
            adaptive.close()
        if model is not None:
            model.close()
        target.close()


if __name__ == "__main__":
    main()
