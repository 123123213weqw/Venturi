"""Recompute native head-GDN engine results with explicit fixed timing scopes."""

import hashlib
import json
from pathlib import Path
import runpy
import statistics

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RAW = OUT / "v100"
telemetry = runpy.run_path(str(OUT.parent / "v100-batched-mtp/summarize.py"))[
    "telemetry"
]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    campaign = read(RAW / "campaign-v5/campaign-report.json")
    assert campaign["completed"] and all(campaign["path_promotions"].values())
    assert len(campaign["experiments"]) == 12
    old_path = ROOT / "benchmarks/v100-mtp-draft-residual/metrics.json"
    old = read(old_path)
    result = dict(
        scope=__doc__,
        fixed_work="23 pinned formatted IDs; 32 greedy outputs ignoring EOS; 8 prepared slots of256 context; two V100s logical0/1 mapped physical1/0; row budget32; native post-first31/(generation-first-ready), HTTP32*N/global cohort wall.",
        library_sha256=campaign["library_sha256"],
        screen_ratios=campaign["screen_ratios"],
        matrices={},
        achieved_hardware_counters=None,
        reused_stronger_reference=dict(
            path=str(old_path.relative_to(ROOT)), sha256=sha(old_path)
        ),
        ordinary256_candidate_not_tested=True,
        automatic_strategy_router=False,
        full_goal_complete=False,
    )
    for mode in ("mtp2", "mtp-batch"):
        result["matrices"][mode] = {}
        for arm in ("control", "candidate"):
            label = f"matrix-{mode}-{arm}"
            entry = next(e for e in campaign["experiments"] if e["label"] == label)
            assert (
                entry["completed"]
                and entry["stable_outputs"]
                and entry["projection_rows"] == 32
            )
            data = read(RAW / "campaign-v5" / label / "timed/report.json")
            assert (
                data["completed"]
                and data["all_outputs_equal_first_wave"]
                and data["case"]["max_new_tokens"] == 32
                and len(data["case"]["prompt_token_ids"]) == 23
            )
            rows = []
            for count in (1, 2, 4, 8):
                waves = [w for w in data["waves"] if w["request_count"] == count]
                assert len(waves) == 3
                for wave in waves:
                    assert (
                        wave["emitted_tokens"] == 32 * count
                        and len(wave["requests"]) == count
                    )
                    for request in wave["requests"]:
                        response = request["response"]
                        assert (
                            response["tokens"] == data["case"]["expected_generated_ids"]
                        )
                        stats = response["mtp_stats"]
                        assert (
                            stats["forced_proposals"]
                            == stats["replayed_input_tokens"]
                            == 0
                            and stats["emitted_tokens"] == 32
                        )
                        assert (stats["accepted_tokens"], stats["proposed_tokens"]) == (
                            (17, 26) if mode == "mtp2" else (14, 16)
                        )
                        if arm == "candidate" and mode == "mtp2":
                            assert stats["peer_target_request_device_bytes"] > 0
                value = dict(
                    requests=count,
                    trials=3,
                    http_tps=32
                    * count
                    / statistics.mean(w["wall_seconds"] for w in waves),
                    trial_http_tps=[32 * count / w["wall_seconds"] for w in waves],
                    mean_cohort_seconds=statistics.mean(
                        w["wall_seconds"] for w in waves
                    ),
                    mean_server_cpu_core_seconds=statistics.mean(
                        w["server_cpu_core_seconds"] for w in waves
                    ),
                    mean_client_completion_seconds=statistics.mean(
                        r["client_completion_seconds"]
                        for w in waves
                        for r in w["requests"]
                    ),
                    max_client_completion_seconds=max(
                        r["client_completion_seconds"]
                        for w in waves
                        for r in w["requests"]
                    ),
                    nvml_by_physical_gpu={
                        str(gpu): telemetry(
                            RAW / "campaign-v5" / label,
                            data,
                            count,
                            gpu,
                            campaign["utc_offset_seconds"],
                        )
                        for gpu in (1, 0)
                    },
                    acceptance_per_request=dict(
                        accepted=17 if mode == "mtp2" else 14,
                        proposed=26 if mode == "mtp2" else 16,
                        forced=0,
                        target_replay=0,
                    ),
                )
                summary = next(
                    s for s in entry["summary"] if s["request_count"] == count
                )
                assert (
                    abs(
                        summary["aggregate_complete_request_output_tps"]
                        - value["http_tps"]
                    )
                    < 1e-10
                )
                if count == 1:
                    timings = [
                        w["requests"][0]["response"]["native_timings"] for w in waves
                    ]
                    value.update(
                        native_post_first_tps=31
                        / statistics.mean(
                            t["native_generation_seconds"]
                            - t["native_time_to_first_token_seconds"]
                            for t in timings
                        ),
                        trial_native_post_first_tps=[
                            31
                            / (
                                t["native_generation_seconds"]
                                - t["native_time_to_first_token_seconds"]
                            )
                            for t in timings
                        ],
                        mean_native_first_ready_seconds=statistics.mean(
                            t["native_time_to_first_token_seconds"] for t in timings
                        ),
                        mean_native_generation_seconds=statistics.mean(
                            t["native_generation_seconds"] for t in timings
                        ),
                    )
                assert all(
                    v["samples"] > 0 for v in value["nvml_by_physical_gpu"].values()
                )
                rows.append(value)
            result["matrices"][mode][arm] = rows
        result["matrices"][mode]["native_ratios"] = [
            dict(
                requests=a["requests"],
                http_ratio=b["http_tps"] / a["http_tps"],
                cpu_ratio=b["mean_server_cpu_core_seconds"]
                / a["mean_server_cpu_core_seconds"],
                paired_trial_http_ratios=[
                    y / x for x, y in zip(a["trial_http_tps"], b["trial_http_tps"])
                ],
            )
            for a, b in zip(
                result["matrices"][mode]["control"],
                result["matrices"][mode]["candidate"],
            )
        ]
    single = result["matrices"]["mtp2"]["candidate"][0]
    external = old["single_decode_comparison"]["best_known_pinned_external_tps"]
    floor = external * 1.05
    result["single_decode_comparison"] = dict(
        native_tps=single["native_post_first_tps"],
        trial_tps=single["trial_native_post_first_tps"],
        best_retained_external_tps=external,
        required_floor=floor,
        ratio=single["native_post_first_tps"] / external,
        all_trials_clear_floor=all(
            v >= floor for v in single["trial_native_post_first_tps"]
        ),
    )
    result["selected_http"] = {}
    for count in (1, 2, 4, 8):
        mode = "mtp2" if count == 1 else "mtp-batch"
        row = next(
            r for r in result["matrices"][mode]["candidate"] if r["requests"] == count
        )
        external = old["best_known_pinned_external_http_by_count"][str(count)]
        prior = old["best_known_native_http_by_count"][str(count)]
        result["selected_http"][str(count)] = dict(
            mode=mode,
            mtp_depth=2 if mode == "mtp2" else 1,
            http_tps=row["http_tps"],
            prior_best_native_tps=prior["http_tps"],
            over_prior_best_native_ratio=row["http_tps"] / prior["http_tps"],
            retained_external=external,
            over_retained_external_ratio=row["http_tps"] / external["http_tps"],
            all_three_trials_clear_external_floor=all(
                v >= 1.05 * external["http_tps"] for v in row["trial_http_tps"]
            ),
        )
    result["numeric_goal_against_retained_baselines_met"] = result[
        "single_decode_comparison"
    ]["all_trials_clear_floor"] and all(
        c["all_three_trials_clear_external_floor"]
        for c in result["selected_http"].values()
    )
    result["completion_caveat"] = (
        "All numerical >=5% conditions now pass versus stronger retained pinned baselines. Goal is not marked complete here: finish matched nonspeculative controls for the new profile and final source/correctness/configuration/comparison audit; fresh competitor validation may raise the reference."
    )
    profiles = {}
    for arm in ("control", "candidate"):
        for mode in ("mtp2-c1", "mtp-batch-c8"):
            label = arm + "-" + mode
            data = read(RAW / "profile-v6" / (label + "-attribution.json"))
            cohort = next(x for x in data["cohorts"] if x["name"] == "native_unsplit")
            profiles[label] = dict(
                host_span_ms=cohort["host_cohort_span_nanoseconds"] / 1e6,
                kernels=cohort["kernels"],
                summed_kernel_ms_not_critical_path=cohort["kernel_nanoseconds"] / 1e6,
                process_activity_gap_ms_not_device_idle=cohort[
                    "gaps_between_this_process_activities_nanoseconds"
                ]
                / 1e6,
                per_device_kernel_union_ms={
                    str(k): v / 1e6
                    for k, v in cohort["per_device_kernel_union_nanoseconds"].items()
                },
                simultaneous_two_device_kernel_ms=cohort[
                    "simultaneous_two_device_kernel_nanoseconds"
                ]
                / 1e6,
                transfers=cohort["transfers"],
                kernel_attribution=cohort["kernel_by_name"],
            )
    result["separate_profiles"] = profiles
    result["profile_comparisons"] = {}
    for mode in ("mtp2-c1", "mtp-batch-c8"):
        a, b = profiles["control-" + mode], profiles["candidate-" + mode]
        result["profile_comparisons"][mode] = dict(
            h2d_byte_ratio=b["transfers"]["host_to_device"]["bytes"]
            / a["transfers"]["host_to_device"]["bytes"],
            d2h_byte_ratio=b["transfers"]["device_to_host"]["bytes"]
            / a["transfers"]["device_to_host"]["bytes"],
            summed_kernel_ratio=b["summed_kernel_ms_not_critical_path"]
            / a["summed_kernel_ms_not_critical_path"],
            instrumented_host_ratio=b["host_span_ms"] / a["host_span_ms"],
        )
    resources = read(RAW / "profile-v6/resources.json")
    assert resources["library_sha256"] == campaign["library_sha256"]
    result["resource_scope"] = (
        "Exact current library static registers/shared/local sizes and CUDA-API theoretical residency, joined to actual traced launches. Achieved occupancy/DRAM/compute/cache/stall counters unavailable under unchanged policy1; no repeated denied counter campaign."
    )
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                single=result["single_decode_comparison"],
                selected_http=result["selected_http"],
                profile_comparisons=result["profile_comparisons"],
                full_goal_complete=False,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
