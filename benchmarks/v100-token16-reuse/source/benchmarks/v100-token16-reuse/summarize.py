"""Recompute T16 staging ablations and engine evidence; never sum shared times."""

import json
from pathlib import Path
import runpy
import statistics

ROOT = Path(__file__).resolve().parent
telemetry = runpy.run_path(str(ROOT.parent / "v100-batched-mtp/summarize.py"))[
    "telemetry"
]


def read(p):
    return json.loads(p.read_text())


def main():
    campaign = read(ROOT / "v100/staged-campaign-v7/campaign-report.json")
    assert campaign["completed"]
    op = read(ROOT / "v100/staged-study-v5/operators/report.json")
    assert op["completed"] and len(op["cases"]) == 77
    result = dict(
        external_speed_goal_complete=False,
        scope="32 fixed greedy outputs/request / global HTTP cohort wall; reset, admission, queue/reply and client archival/validation included. Native call TTFT/generation and profiles are separate scopes.",
        operator_promotion=read(ROOT / "v100/staged-model-v6/operator-promotion.json"),
        path_promotions=campaign["path_promotions"],
        screen_ratios=campaign["screen_ratios"],
        operators=op,
        operator_phases=read(ROOT / "operator-phases.json"),
        experiments=[],
        profiles={},
        achieved_occupancy=None,
    )
    paths = [
        (ROOT / "v100/staged-campaign-v7", False),
        (ROOT.parent / "v100-batched-mtp/v100/campaign-v3", True),
        (ROOT.parent / "v100-batched-mtp/v100/sdk-campaign-v8", True),
        (ROOT.parent / "v100-q8-wmma/v100/campaign-v5", True),
        (ROOT.parent / "v100-strided-output/v100/external-depth-v9", True),
    ]
    for base, reused in paths:
        c = read(base / "campaign-report.json")
        assert c["completed"]
        for e in c["experiments"]:
            if reused and not e["label"].startswith("repeat"):
                continue
            if "v100-q8-wmma" in str(base) and e["engine"] != "llama":
                continue
            p = base / e["label"] / "timed/report.json"
            if not e["completed"] or not p.exists():
                continue
            b = read(p)
            rows = []
            for count in sorted({w["request_count"] for w in b["waves"]}):
                waves = [w for w in b["waves"] if w["request_count"] == count]
                requests = [r for w in waves for r in w["requests"]]
                responses = [r["response"] for r in requests]
                assert all(w["emitted_tokens"] == 32 * count for w in waves)
                assert all(
                    r["tokens"] == b["case"]["expected_generated_ids"]
                    for r in responses
                )
                row = dict(
                    request_count=count,
                    repeats=len(waves),
                    complete_request_output_tps=32
                    * count
                    / statistics.mean(w["wall_seconds"] for w in waves),
                    trial_output_tps=[32 * count / w["wall_seconds"] for w in waves],
                    mean_cohort_wall_seconds=statistics.mean(
                        w["wall_seconds"] for w in waves
                    ),
                    mean_server_cpu_core_seconds=statistics.mean(
                        w["server_cpu_core_seconds"] for w in waves
                    ),
                    mean_client_completion_seconds=statistics.mean(
                        r["client_completion_seconds"] for r in requests
                    ),
                    max_client_completion_seconds=max(
                        r["client_completion_seconds"] for r in requests
                    ),
                    telemetry={
                        str(g): telemetry(
                            base / e["label"], b, count, g, c["utc_offset_seconds"]
                        )
                        for g in [1, 0]
                    },
                )
                if all("native_timings" in r for r in responses):
                    tt = [r["native_timings"] for r in responses]
                    row["mean_native_ttft_seconds"] = statistics.mean(
                        t["native_time_to_first_token_seconds"] for t in tt
                    )
                    row["max_native_ttft_seconds"] = max(
                        t["native_time_to_first_token_seconds"] for t in tt
                    )
                    row["mean_native_generation_seconds"] = statistics.mean(
                        t["native_generation_seconds"] for t in tt
                    )
                    if count == 1:
                        row["native_single_post_first_tps"] = 31 / statistics.mean(
                            t["native_generation_seconds"]
                            - t["native_time_to_first_token_seconds"]
                            for t in tt
                        )
                else:
                    row["engine_timings"] = [r["timings"] for r in responses]
                    if count == 1:
                        row["llama_reported_post_first_tps"] = 31 / statistics.mean(
                            r["timings"]["predicted_ms"] / 1000 for r in responses
                        )
                if e["mtp_depth"] and e["engine"] == "venturi":
                    stats = [r["mtp_stats"] for r in responses]
                    row["mtp_counters"] = stats
                    row["acceptance_ratio"] = sum(
                        s["accepted_tokens"] for s in stats
                    ) / sum(s["proposed_tokens"] for s in stats)
                    row["target_replayed_inputs"] = sum(
                        s["replayed_input_tokens"] for s in stats
                    )
                    if e["mode"] == "mtp-batch" or e["mtp_recovery"] == "prefix":
                        assert row["target_replayed_inputs"] == 0
                if all("mtp_batch_stats" in r for r in responses):
                    shared = [
                        w["requests"][0]["response"]["mtp_batch_stats"] for w in waves
                    ]
                    for w in waves:
                        assert (
                            len(
                                {
                                    r["response"]["batch_cohort_id"]
                                    for r in w["requests"]
                                }
                            )
                            == 1
                        )
                        assert all(
                            r["response"]["mtp_batch_stats"]
                            == w["requests"][0]["response"]["mtp_batch_stats"]
                            for r in w["requests"]
                        )
                    row["shared_mtp_batch_stats_once_per_wave"] = shared
                    row["native_cohort_after_all_first_outputs_tps"] = (
                        31
                        * count
                        / statistics.mean(
                            s["shared_native_generation_seconds"]
                            - s["shared_prefill_complete_seconds"]
                            for s in shared
                        )
                    )
                rows.append(row)
            result["experiments"].append(
                dict(
                    capsule=str(base.relative_to(ROOT.parent)),
                    label=e["label"],
                    engine=e["engine"],
                    mode=e["mode"],
                    depth=e["mtp_depth"],
                    reused=reused,
                    summary=rows,
                )
            )
    indexed = {
        (e["capsule"], e["label"]): {r["request_count"]: r for r in e["summary"]}
        for e in result["experiments"]
    }
    current = "v100-token16-reuse/v100/staged-campaign-v7"
    result["repeat_path_ratios"] = {}
    for name, promoted in campaign["path_promotions"].items():
        if not promoted:
            continue
        a, b = (
            indexed[current, "repeat-" + arm + "-" + name]
            for arm in ["control", "candidate"]
        )
        result["repeat_path_ratios"][name] = {
            str(n): b[n]["complete_request_output_tps"]
            / a[n]["complete_request_output_tps"]
            for n in b
        }
    if campaign["operator_promoted"]:
        for name in ["mtp2-prefix-c1", "ordinary256-c8", "mtp-batch-c8"]:
            base = ROOT / "v100/staged-final-v8"
            result["profiles"][name] = dict(
                reused=False,
                metadata=read(base / (name + "-metadata.json")),
                attribution=read(base / (name + "-attribution.json")),
            )
    for base, names in [
        (
            ROOT.parent / "v100-native-mtp2/v100/full-profile-v14",
            ["mtp2-prefix-full-c1"],
        ),
        (ROOT.parent / "v100-batched-mtp/v100/final-v4", ["ordinary256-control-c8"]),
        (ROOT.parent / "v100-batched-mtp/v100/sdk-final-v9", ["mtp-batch-c8"]),
    ]:
        for name in names:
            result["profiles"][base.name + "/" + name] = dict(
                reused=True,
                metadata=read(base / (name + "-metadata.json")),
                attribution=read(base / (name + "-attribution.json")),
            )
    result["operator_telemetry"] = read(ROOT / "operator-telemetry.json")
    result["graph_transport_probe"] = read(ROOT / "v100/graph-probe-v9/report.json")
    result["operator_profiles"] = {}
    for folder, label, reused in [
        ("final-v4", "control", True),
        ("final-v4", "candidate", True),
        ("staged-final-v8", "candidate", False),
    ]:
        base = ROOT / "v100" / folder
        result["operator_profiles"][folder + "/" + label] = dict(
            reused=reused,
            metadata=read(base / (label + "-metadata.json")),
            attribution=read(base / (label + "-attribution.json")),
        )
    (ROOT / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                in [
                    "external_speed_goal_complete",
                    "operator_promotion",
                    "path_promotions",
                    "screen_ratios",
                    "repeat_path_ratios",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
