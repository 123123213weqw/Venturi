"""Finite public-Engine concurrency/cache probes; no upstream edits or implicit downloads.

Uses an explicitly linked research runner, not the sequential production Python API.
All throughput uses a shared wall clock, never summed concurrent request durations.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

from run_qwen_validation import idle_gpu


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def commit_intervals(events):
    intervals = []
    for previous, current in zip(events, events[1:]):
        dt = current["generation_ns"] - previous["generation_ns"]
        dn = current["tokens"] - previous["tokens"]
        if dt < 0 or dn < 0:
            raise ValueError("Non-monotonic native commit timing")
        # Do not invent per-token timings inside a multi-token speculative commit.
        if dn == 1:
            intervals.append(dt / 1e6)
    if not intervals:
        return dict(single_token_intervals=0)
    ordered = sorted(intervals)
    return dict(
        single_token_intervals=len(intervals),
        median_ms=statistics.median(intervals),
        p95_ms=ordered[math.ceil(0.95 * len(ordered)) - 1],
        max_ms=max(intervals),
        intervals_over_100ms=sum(v > 100 for v in intervals),
    )


def summarize(run):
    rows = run["results"]
    before, after = run["stats_before"], run["stats_after"]
    delta = {k: after[k] - v for k, v in before.items() if isinstance(v, int)}
    host = {k: after["host_work"][k] - v for k, v in before["host_work"].items()}
    host_active = (
        sum(
            host[k]
            for k in (
                "engine_boundary_ns",
                "program_submit_ns",
                "program_post_ns",
                "engine_commit_output_ns",
                "engine_maintenance_ns",
            )
        )
        / 1e9
    )
    durations = [r["timings"]["total_seconds"] for r in rows]
    tokens = sum(len(r["generated_ids"]) for r in rows)
    return dict(
        wall_seconds=run["wall_seconds"],
        output_tokens=tokens,
        aggregate_output_tps=tokens / run["wall_seconds"],
        mean_actual_decode_batch=delta["decode_row_rounds"] / delta["decode_rounds"]
        if delta["decode_rounds"]
        else 0,
        mean_request_total_seconds=statistics.mean(durations),
        max_request_total_seconds=max(durations),
        mean_queue_seconds=statistics.mean(
            r["engine_timing"]["queue_wait_seconds"] for r in rows
        ),
        mean_first_token_seconds=statistics.mean(
            r["timings"]["first_token_seconds"] for r in rows
        ),
        mean_prompt_wall_seconds=statistics.mean(
            r["timings"]["prompt_wall_seconds"] for r in rows
        ),
        per_request_generation_tps=[
            (len(r["generated_ids"]) - 1) / r["timings"]["generation_wall_seconds"]
            if r["timings"]["generation_wall_seconds"] > 0
            else None
            for r in rows
        ],
        host_active_seconds=host_active,
        device_wait_seconds=host["device_wait_ns"] / 1e9,
        host_active_fraction_of_batch_wall=host_active / run["wall_seconds"],
        runtime_counter_delta=delta,
        host_counter_delta=host,
        commit_interval_metrics=[
            commit_intervals(r["commit_timings"]) for r in rows if "commit_timings" in r
        ],
        reused_prompt_tokens=[r["reused_prompt_tokens"] for r in rows],
        memory=run["memory"],
    )


def compare_ids(reference, actual):
    a, b = reference["generated_ids"], actual["generated_ids"]
    common = next(
        (i for i, pair in enumerate(zip(a, b)) if pair[0] != pair[1]),
        min(len(a), len(b)),
    )
    return {
        "equal": a == b,
        "matching_prefix": common,
        "reference_length": len(a),
        "actual_length": len(b),
    }


def execute(args, out, name, config, batches, profile=False):
    if (out / f"{name}.json").exists():
        raise FileExistsError(f"Refusing to overwrite retained evidence for {name}")
    idle_gpu()
    input_text = (
        "\n".join(json.dumps(x) for x in [config, *batches, {"op": "close"}]) + "\n"
    )
    command = [args.runner, args.checkpoint]
    if profile:
        graph_trace = ["--cuda-graph-trace=node"] if args.profile_node else []
        command = [
            "nsys",
            "profile",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--force-overwrite=true",
            "-o",
            str(out / name),
            *graph_trace,
            *command,
        ]
    env = {k: v for k, v in os.environ.items() if not k.startswith("NINFER_")}
    (out / f"{name}.input.jsonl").write_text(input_text)
    started = time.time()
    with (out / f"{name}.telemetry.csv").open("w") as telemetry:
        sampler = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu,clocks.sm,clocks.mem",
                "--format=csv,nounits",
                "-lms",
                "200",
            ],
            stdout=telemetry,
            stderr=subprocess.DEVNULL,
        )
        try:
            run = subprocess.run(
                command,
                input=input_text,
                text=True,
                capture_output=True,
                env=env,
                timeout=900,
            )
        finally:
            sampler.terminate()
            try:
                sampler.wait(timeout=5)
            except subprocess.TimeoutExpired:
                sampler.kill()
                sampler.wait()
    (out / f"{name}.stderr").write_text(run.stderr)
    (out / f"{name}.stdout").write_text(run.stdout)
    # Nsight may add human-readable lines; native JSON objects are one line each.
    records = [
        json.loads(line) for line in run.stdout.splitlines() if line.startswith("{")
    ]
    report = dict(
        name=name,
        config=config,
        command=command,
        returncode=run.returncode,
        elapsed_with_startup_seconds=time.time() - started,
        records=records,
    )
    report["ok"] = (
        run.returncode == 0
        and len(records) == len(batches) + 1
        and all(r.get("ok") for r in records)
    )
    if report["ok"]:
        for request, record in zip(batches, records[1:]):
            if request.get("stream_timings") and not all(
                r.get("commit_timings") for r in record["results"]
            ):
                report["ok"] = False
    if report["ok"]:
        report["summaries"] = [summarize(r) for r in records[1:]]
    write(out / f"{name}.json", report)
    print(name, "OK" if report["ok"] else "FAILED", flush=True)
    if report["ok"]:
        print(
            json.dumps(
                {
                    k: report["summaries"][-1][k]
                    for k in (
                        "wall_seconds",
                        "aggregate_output_tps",
                        "mean_actual_decode_batch",
                        "mean_first_token_seconds",
                        "host_active_fraction_of_batch_wall",
                    )
                }
            ),
            flush=True,
        )
    return report


def build_batches(workloads, outputs=64, count=8):
    long_ids = next(w["ids"] for w in workloads if w["name"] == "long-512")
    requests = [
        dict(
            token_ids=long_ids[:504] + long_ids[i * 11 : i * 11 + 8],
            max_new_tokens=outputs,
            ignore_eos=True,
            reuse=False,
        )
        for i in range(count)
    ]
    return dict(requests=requests)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("--runner", required=True)
    p.add_argument("--workloads", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--profile", action="store_true")
    p.add_argument(
        "--profile-node",
        action="store_true",
        help="Trace graph kernels, with additional instrumentation overhead",
    )
    p.add_argument("--only", nargs="*")
    args = p.parse_args()
    if not 1 <= args.repeats <= 10:
        p.error("repeats must be 1..10")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    workloads = json.loads(Path(args.workloads).read_text())
    batch = build_batches(workloads)
    base = dict(
        max_context=2048,
        prefill_chunk=512,
        cuda_graph=True,
        kv_dtype="bf16",
        cache=False,
        checkpoint_slots=0,
    )
    cases = [
        (f"c{c}-{order}", dict(base, concurrency=c, kv_capacity=c * 2048))
        for order, cs in (("a", (1, 2, 4, 8)), ("b", (8, 4, 2, 1)))
        for c in cs
    ]
    cases += [
        ("c8-tight-kv", dict(base, concurrency=8, kv_capacity=2048)),
        ("c8-budget-kv", dict(base, concurrency=8, kv_capacity=4608)),
        ("c8-no-graph", dict(base, concurrency=8, kv_capacity=16384, cuda_graph=False)),
        (
            "c8-prefill128",
            dict(base, concurrency=8, kv_capacity=16384, prefill_chunk=128),
        ),
        (
            "c8-dflash",
            dict(base, concurrency=8, kv_capacity=16384, speculative="dflash2"),
        ),
    ]
    reports = {}
    for name, config in cases:
        if args.only and name not in args.only:
            continue
        reports[name] = execute(args, out, name, config, [batch] * (1 + args.repeats))
    if not args.only or "c8-output256" in args.only:
        reports["c8-output256"] = execute(
            args,
            out,
            "c8-output256",
            dict(base, concurrency=8, kv_capacity=16384),
            [build_batches(workloads, outputs=256)] * (1 + args.repeats),
        )
    chat_requests = [
        dict(prompt=q, max_new_tokens=32, ignore_eos=False, reuse=False)
        for q in [
            "What is 2 + 2? Answer briefly.",
            "Name one primary color. Answer briefly.",
            "Explain why the sky is blue.",
            "Write a Python function that adds two numbers.",
        ]
        * 2
    ]
    for c in (1, 8):
        name = f"chat-c{c}"
        if args.only and name not in args.only:
            continue
        reports[name] = execute(
            args,
            out,
            name,
            dict(base, concurrency=c, kv_capacity=c * 2048),
            [dict(requests=chat_requests)] * (1 + args.repeats),
        )
    for c, chunk in ((1, 512), (8, 128), (8, 256), (8, 512)):
        name = f"stream-c{c}-chunk{chunk}"
        if args.only and name not in args.only:
            continue
        reports[name] = execute(
            args,
            out,
            name,
            dict(base, concurrency=c, kv_capacity=c * 2048, prefill_chunk=chunk),
            [dict(batch, stream_timings=True)] * (1 + args.repeats),
        )
    # Exact shared system prefix, changed user suffix, and explicit no-reuse controls.
    system = "You are a precise inference-engine tutor. " + (
        "The engine loads quantized weights, tracks recurrent state, and manages attention cache pages. "
        * 45
    )

    def chat(question, reuse):
        return dict(
            requests=[
                dict(
                    system=system,
                    prompt=question,
                    reuse=reuse,
                    max_new_tokens=32,
                    ignore_eos=True,
                )
            ]
        )

    questions = [
        "Explain cache pages.",
        "Explain recurrent state.",
        "Explain cache pages.",
    ]
    for enabled in (False, True):
        name = "cache-on" if enabled else "cache-disabled"
        if args.only and name not in args.only:
            continue
        config = dict(
            base,
            concurrency=1,
            kv_capacity=2048,
            cache=enabled,
            checkpoint_slots=(4 if enabled else 0),
        )
        batches = [chat(q, enabled) for q in questions]
        # A disabled read/write control in the same cache-enabled Engine.
        if enabled:
            batches.append(chat(questions[0], False))
        reports[name] = execute(args, out, name, config, batches)
    if args.profile:
        for concurrency in (1, 8):
            name = f"profile-c{concurrency}" + ("-node" if args.profile_node else "")
            if args.only and name not in args.only:
                continue
            config = dict(base, concurrency=concurrency, kv_capacity=concurrency * 2048)
            profiled = dict(build_batches(workloads, outputs=32), profile=True)
            reports[name] = execute(
                args, out, name, config, [batch, profiled], profile=True
            )
    if (
        args.profile
        and args.profile_node
        and (not args.only or "profile-cache-node" in args.only)
    ):
        reports["profile-cache-node"] = execute(
            args,
            out,
            "profile-cache-node",
            dict(base, concurrency=1, kv_capacity=2048, cache=True, checkpoint_slots=4),
            [chat(questions[0], True), dict(chat(questions[1], True), profile=True)],
            profile=True,
        )
    write(
        out / f"invocation-{time.time_ns()}.json",
        dict(
            args=vars(args),
            case_status={name: report["ok"] for name, report in reports.items()},
            working_probe_source_sha256=hashlib.sha256(
                (Path(__file__).parent / "csrc/qwen_concurrency_probe.cpp").read_bytes()
            ).hexdigest(),
            runner_sha256=hashlib.sha256(Path(args.runner).read_bytes()).hexdigest(),
            notes=[
                "Fixed 512-token raw prompts, 64-output throughput budget ignores EOS; not normal interactive completion.",
                "Paged KV is already native and always enabled; capacity experiments are not paged-vs-contiguous A/B.",
                "Runtime counters are interval differences; exposed per-request device time must not be summed.",
                "Warmup is one eight-request wave; topology changes may still trigger later graph capture.",
                "The backend supplies all mathematics, scheduler, cache and kernels; this is a research harness.",
            ],
        ),
    )

    if any(not report["ok"] for report in reports.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
