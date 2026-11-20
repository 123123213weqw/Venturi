"""Recompute execution-policy screens/matrices with complete timing scopes."""

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


def read(p):
    return json.loads(p.read_text())


def path_name(e):
    return (
        "mtp2"
        if e["mode"] == "scalar"
        else "mtp-batch"
        if e["mode"] == "mtp-batch"
        else "ordinary256"
    )


def aggregate(entries, count):
    waves = [
        w
        for e in entries
        for w in e["benchmark"]["waves"]
        if w["request_count"] == count
    ]
    responses = [r["response"] for w in waves for r in w["requests"]]
    result = dict(
        requests=count,
        trials=len(waves),
        complete_http_tps=32
        * count
        / statistics.mean(w["wall_seconds"] for w in waves),
        trial_complete_http_tps=[32 * count / w["wall_seconds"] for w in waves],
        mean_complete_http_wall_seconds=statistics.mean(
            w["wall_seconds"] for w in waves
        ),
        mean_server_cpu_core_seconds=statistics.mean(
            w["server_cpu_core_seconds"] for w in waves
        ),
        mean_client_completion_seconds=statistics.mean(
            r["client_completion_seconds"] for w in waves for r in w["requests"]
        ),
        max_client_completion_seconds=max(
            r["client_completion_seconds"] for w in waves for r in w["requests"]
        ),
    )
    if all("native_timings" in r for r in responses):
        times = [r["native_timings"] for r in responses]
        result["mean_native_ttft_seconds"] = statistics.mean(
            t["native_time_to_first_token_seconds"] for t in times
        )
        result["mean_native_generation_seconds"] = statistics.mean(
            t["native_generation_seconds"] for t in times
        )
        if count == 1:
            result["native_single_post_first_tps"] = 31 / statistics.mean(
                t["native_generation_seconds"] - t["native_time_to_first_token_seconds"]
                for t in times
            )
            result["trial_native_single_post_first_tps"] = [
                31
                / (
                    t["native_generation_seconds"]
                    - t["native_time_to_first_token_seconds"]
                )
                for t in times
            ]
    stats = [r["mtp_stats"] for r in responses if "mtp_stats" in r]
    if stats:
        result["mtp_algorithm_counters"] = [
            {
                k: s[k]
                for k in (
                    "proposed_tokens",
                    "accepted_tokens",
                    "verification_calls",
                    "target_forward_calls",
                    "target_input_tokens",
                    "draft_forward_calls",
                    "draft_input_tokens",
                    "replayed_input_tokens",
                    "forced_proposals",
                )
            }
            for s in stats
        ]
    result["telemetry_by_experiment"] = [
        dict(
            experiment=e["label"],
            physical_gpu={
                str(g): telemetry(e["directory"], e["benchmark"], count, g, 28800)
                for g in [1, 0]
            },
        )
        for e in entries
    ]
    return result


def main():
    campaign = read(RAW / "campaign-v4/campaign-report.json")
    assert campaign["completed"] and campaign["model_qualified"]
    grouped = {}
    for e in campaign["experiments"]:
        assert e["completed"] and e["stable_outputs"]
        entry = dict(e)
        entry["directory"] = RAW / "campaign-v4" / e["label"]
        entry["benchmark"] = read(entry["directory"] / "timed/report.json")
        kind = "screen" if e["label"].startswith("screen-") else "repeat"
        grouped.setdefault((kind, path_name(e), e["source_scope_mode"]), []).append(
            entry
        )
    result = dict(
        scope="Complete HTTP32*N/global cohort wall includes reset/admission/queue/reply/client archival/future collection; native single31/(generation minus TTFT). No concurrent time sums and no streaming TTFT claim.",
        profiles=dict(target="q8-residual-column-mlp", draft="q8-residual", kv="fp16"),
        execution_policy_only=True,
        arithmetic_changed=False,
        source_check_cadence_changed_and_declared=True,
        original_default_is_per_weight=True,
        achieved_occupancy=None,
        fresh_external_campaign_run=False,
        screen_ratios=campaign["screen_ratios"],
        path_promotions=campaign["path_promotions"],
        screens={},
        matrices={},
        adoption_checks={},
        profiles_separate={},
        external_speed_goal_complete=False,
    )
    for kind, name, scope in sorted(grouped):
        entries = grouped[(kind, name, scope)]
        counts = sorted(
            {w["request_count"] for e in entries for w in e["benchmark"]["waves"]}
        )
        target = result["screens" if kind == "screen" else "matrices"]
        target.setdefault(name, {})[str(scope)] = {
            str(n): aggregate(entries, n) for n in counts
        }
    for name, promoted in campaign["path_promotions"].items():
        if not promoted:
            result["adoption_checks"][name] = dict(
                promoted=False,
                adopted=False,
                reason="Predeclared paired screen gate failed; no full matrix promoted",
            )
            continue
        arms = result["matrices"][name]
        ratios = {
            n: arms["1"][n]["complete_http_tps"] / arms["0"][n]["complete_http_tps"]
            for n in arms["0"]
        }
        decode = (
            arms["1"]["1"]["native_single_post_first_tps"]
            / arms["0"]["1"]["native_single_post_first_tps"]
        )
        http = ratios["1"]
        improving_decode = sum(
            b > a
            for a, b in zip(
                arms["0"]["1"]["trial_native_single_post_first_tps"],
                arms["1"]["1"]["trial_native_single_post_first_tps"],
            )
        )
        improving_http = sum(
            b > a
            for a, b in zip(
                arms["0"]["1"]["trial_complete_http_tps"],
                arms["1"]["1"]["trial_complete_http_tps"],
            )
        )
        adopted = (
            min(ratios.values()) >= 0.98
            and http >= 0.99
            and (
                (decode > 1.01 and improving_decode >= 2)
                or (http > 1.01 and improving_http >= 2)
            )
        )
        result["adoption_checks"][name] = dict(
            promoted=True,
            adopted=adopted,
            http_ratios=ratios,
            c1_decode_ratio=decode,
            improving_decode_trials=improving_decode,
            improving_http_trials=improving_http,
        )
    profile = RAW / "profile-v5"
    counts = read(profile / "scoped-syscalls.json")
    result["source_check_audit"] = dict(
        prior_control_pairs=6674,
        candidate_pairs=counts["checkpoint_calls"]["descriptor"],
        prior_control_syscalls=13348,
        candidate_syscalls=sum(counts["checkpoint_calls"].values()),
        prior_count_reference="benchmarks/v100-source-validation/v100/probe-v1/syscall-report.json",
        prior_count_reference_sha256=hashlib.sha256(
            (
                ROOT
                / "benchmarks/v100-source-validation/v100/probe-v1/syscall-report.json"
            ).read_bytes()
        ).hexdigest(),
        syscall_reduction_fraction=1 - counts["checkpoint_calls"]["descriptor"] / 6674,
        predicate_preserved=True,
        cadence_changed=True,
        source_trace_instrumented_not_serving_tps=True,
    )
    for scope in [0, 1]:
        for mode in ["mtp2-c1", "mtp-batch-c8"]:
            name = f"scope{scope}-{mode}"
            attribution = read(profile / (name + "-attribution.json"))
            assert not attribution["achieved_hardware_counters"]
            cohort = attribution["cohorts"][0]
            result["profiles_separate"][name] = dict(
                library_sha256=read(profile / (name + "-metadata.json"))[
                    "library_sha256"
                ],
                source_validation_policy=read(profile / (name + "-metadata.json"))[
                    "source_validation_policy"
                ],
                host_span_ms=cohort["host_cohort_span_nanoseconds"] / 1e6,
                kernel_count=cohort["kernels"],
                summed_kernel_ms=cohort["kernel_nanoseconds"] / 1e6,
                process_activity_gaps_ms=cohort[
                    "gaps_between_this_process_activities_nanoseconds"
                ]
                / 1e6,
                per_device_kernel_union_ms={
                    k: v / 1e6
                    for k, v in cohort["per_device_kernel_union_nanoseconds"].items()
                },
                simultaneous_two_device_kernel_ms=cohort[
                    "simultaneous_two_device_kernel_nanoseconds"
                ]
                / 1e6,
                transfers=cohort["transfers"],
                top_kernels=cohort["kernel_by_name"][:12],
            )
    for mode in ["mtp2-c1", "mtp-batch-c8"]:
        a, b = [result["profiles_separate"][f"scope{s}-{mode}"] for s in [0, 1]]
        assert a["kernel_count"] == b["kernel_count"]
        assert all(
            (a["transfers"][d]["count"], a["transfers"][d]["bytes"])
            == (b["transfers"][d]["count"], b["transfers"][d]["bytes"])
            for d in a["transfers"]
        )
    prior = read(OUT.parent / "v100-mtp-draft-residual/metrics.json")
    result["previous_accepted_full_matrix_single_comparison_explicitly_reused"] = prior[
        "single_decode_comparison"
    ]
    result["previous_accepted_best_http_by_count_explicitly_reused"] = prior[
        "best_known_native_http_by_count"
    ]
    result["full_goal_scope_not_redefined"] = True
    restoration = OUT / "production-restoration.json"
    result["production_restoration"] = (
        read(restoration) if restoration.exists() else None
    )
    result["prototype_promoted_to_production"] = any(
        c["adopted"] for c in result["adoption_checks"].values()
    )
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                screen_ratios=result["screen_ratios"],
                adoption_checks=result["adoption_checks"],
                source_check_audit=result["source_check_audit"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
