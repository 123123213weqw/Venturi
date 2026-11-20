"""Regenerate this experiment's aggregate from retained raw JSON/CSV evidence."""

import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent


def main():
    micro = {}
    names = ("prefill_baseline", "prefill_r128_w32", "prefill_r128_w64")
    for name in names:
        samples = {}
        for path in sorted((ROOT / name).glob("stream-*.csv")):
            for row in csv.DictReader(path.open()):
                key = f"K{row['k']}/T{row['t']}"
                samples.setdefault(key, []).append(float(row["median_us"]))
        micro[name] = {
            k: dict(run_medians_us=v, median_us=statistics.median(v))
            for k, v in samples.items()
        }
    sessions = {
        name: json.loads((ROOT / ("model-" + name) / "results.json").read_text())
        for name in ("baseline-a", "r128-a", "r128-b", "baseline-b")
    }
    reference_session = sessions["baseline-a"]
    for session in sessions.values():
        if (
            session["prompt_ids"] != reference_session["prompt_ids"]
            or session["generation"] != reference_session["generation"]
            or session["ready"]["config"] != reference_session["ready"]["config"]
            or session["process_exit"] != 0
        ):
            raise AssertionError("Workload/configuration differs or runner failed")
    model = {}
    counterexamples = []
    matched = total = 0
    for length in sessions["baseline-a"]["prompt_ids"]:
        reference = sessions["baseline-a"]["warm"][length]["generated_ids"]
        row = {}
        for variant, labels in [
            ("baseline", ["baseline-a", "baseline-b"]),
            ("r128_w32", ["r128-a", "r128-b"]),
        ]:
            runs = [
                r
                for label in labels
                for r in sessions[label]["runs"]
                if str(r["prompt_length"]) == length
            ]
            values = [r["result"]["timings"]["prefill_seconds"] for r in runs]
            wall = sum(r["wall_seconds"] for r in runs)
            row[variant] = dict(
                calls=len(runs),
                prefill_seconds=values,
                median_prefill_seconds=statistics.median(values),
                native_prefill_tokens_per_second=int(length) * len(runs) / sum(values),
                median_native_first_token_seconds=statistics.median(
                    r["result"]["timings"]["first_token_seconds"] for r in runs
                ),
                median_native_decode_seconds=statistics.median(
                    r["result"]["timings"]["decode_seconds"] for r in runs
                ),
                client_wall_seconds=wall,
                native_cpu_seconds=sum(r["native_cpu_seconds"] for r in runs),
                python_cpu_seconds=sum(r["python_cpu_seconds"] for r in runs),
                native_cpu_percent=100
                * sum(r["native_cpu_seconds"] for r in runs)
                / wall,
            )
            for r in runs:
                total += 1
                if r["result"]["generated_ids"] == reference:
                    matched += 1
                else:
                    counterexamples.append(
                        dict(
                            length=length,
                            variant=variant,
                            expected_ids=reference,
                            actual_ids=r["result"]["generated_ids"],
                        )
                    )
        row["candidate_latency_change_percent"] = 100 * (
            row["r128_w32"]["median_prefill_seconds"]
            / row["baseline"]["median_prefill_seconds"]
            - 1
        )
        model[length] = row
    gpu = {}
    for variant, labels in [
        ("baseline", ["baseline-a", "baseline-b"]),
        ("r128_w32", ["r128-a", "r128-b"]),
    ]:
        groups = [sessions[label]["telemetry"]["single"] for label in labels]
        count = sum(g["samples"] for g in groups)
        gpu[variant] = dict(samples=count)
        for field in groups[0]:
            if field != "samples":
                gpu[variant][field] = dict(
                    mean=sum(g[field]["mean"] * g["samples"] for g in groups) / count,
                    minimum=min(g[field]["minimum"] for g in groups),
                    maximum=max(g[field]["maximum"] for g in groups),
                )
    report = dict(
        date="2026-10-17",
        device="School_4090D / RTX 4090 D / SM89",
        decision="Reject both 128x64 schedules; production and earlier selective candidate unchanged.",
        scope="Q4 fused LinearAdd, N=5120, K=6144/17408. Private wide-T schedule only; no precision or decode changes.",
        candidates=json.loads((ROOT / "results.json").read_text())["candidates"],
        micro_method="Public ops::linear_add, ring of eight distinct weights per K exceeding L2; 8 calls/graph, 20 warmup graphs, 40 timed graphs. Medians across interleaved baseline/forward/reverse/baseline runs. Synthetic timings, not model TPS.",
        micro=micro,
        model_method=dict(
            order=list(sessions),
            repeats_per_length_per_session=3,
            prompt_lengths=[128, 509, 512, 1021, 2048, 8192],
            new_tokens=16,
            ignore_eos=True,
            max_context=16384,
            max_concurrency=1,
            kv_dtype="bf16",
            prefill_chunk=512,
            cuda_graph=True,
            cpu_wait="blocking",
            speculation="none",
            prefix_reuse=False,
            timing_scope="Unprofiled native prefill; internal first-token timestamp is not streamed TTFB. Client wall and CPU include complete 16-token RPC. Warmup/loading/tokenization excluded.",
        ),
        model=model,
        gpu=gpu,
        correctness=dict(
            independent_oracle="Naive FP64 fused residual+W*x from independently decoded packed weights, A16 criterion unchanged.",
            dense_full_output_tokens=512,
            k_values=[6144, 17408],
            additional_checks="Sampled route/tile boundaries 192..1024, all-value finite scan, guards, input/weight preservation, zero/restored-input eager and graph replay.",
            all_three_op_suites_passed=True,
            measured_rows_matching_baseline=matched,
            measured_rows_total=total,
            session_replay_matches_warm=all(
                s["replay_matches_warm"] for s in sessions.values()
            ),
            counterexamples=counterexamples,
            limitation="Six short fixed fixtures and 16-token continuations are not broad accuracy or perplexity validation.",
        ),
        diagnostics=json.loads((ROOT / "diagnostics.json").read_text()),
        limitations=[
            "GPU telemetry covers mixed-length prefill plus 16-token decode, not an isolated kernel.",
            "Whole-device sampled power does not establish causal energy efficiency.",
            "Resource-only occupancy bounds are not achieved occupancy; DRAM bandwidth, cache/stall and compute counters unavailable under current policy.",
            "Larger row tiles structurally duplicate more weight staging/dequantization across token tiles; this is a possible tradeoff, not a counter-verified bottleneck diagnosis.",
        ],
        tests=dict(cpu_passed=329, cpu_skipped=56, ruff="passed", diff_check="passed"),
    )
    (ROOT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {n: r["candidate_latency_change_percent"] for n, r in model.items()},
            indent=2,
        )
    )
    print("Matched output rows:", matched, "/", total)


if __name__ == "__main__":
    main()
