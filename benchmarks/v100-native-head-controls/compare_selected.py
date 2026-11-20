"""Compare fresh selected native cells to every retained repeated reference."""

import json
from pathlib import Path
import statistics

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def read(path):
    return json.loads(path.read_text())


def enrich_external(row):
    value = dict(row)
    raw = read(ROOT / row["report"])
    waves = [wave for wave in raw["waves"] if wave["request_count"] == row["requests"]]
    value.update(
        mean_server_cpu_core_seconds=statistics.mean(wave["server_cpu_core_seconds"] for wave in waves),
        trial_http_tps=[32 * row["requests"] / wave["wall_seconds"] for wave in waves],
        mean_cohort_seconds=statistics.mean(wave["wall_seconds"] for wave in waves),
        mean_client_completion_seconds=statistics.mean(request["client_completion_seconds"] for wave in waves for request in wave["requests"]),
    )
    return value


def main():
    assert not (OUT / 'evidence-sha256.json').exists(), 'Never overwrite frozen evidence'
    controls = read(OUT / "recomputed-controls.json")
    previous = read(OUT.parent / "v100-native-head-gdn/metrics.json")
    ordinary = read(OUT / "ordinary-metrics.json")
    quality = read(OUT / "ordinary-qualification.json")
    aligned = read(OUT / "v100/aligned-v18/aligned/comparison.json")
    assert aligned["completed"] and aligned["aligned_rows"] == 11
    assert all(row["finite"] and row["input_ids_aligned"] for row in aligned["rows"])
    final = [row for row in controls["repeated_cells"] if row["tag"] == "controls-v17" and row["engine"] == "venturi"]
    assert len(final) == 4 and {row["requests"] for row in final} == {1, 2, 4, 8}
    result = dict(
        fixed_work="Same two V100s; official source revision; declared different Q8 compute/encoding profiles; FP16 KV;23 formatted IDs,32 greedy outputs ignoring EOS;8 prepared256-context slots; common complete-cohort wall. Native post-first31/(generation-first-ready), comparator31/(reported predicted time). Diagnostic-only EOS mask differs and is declared.",
        selected_http={},
        achieved_hardware_counters=None,
        automatic_strategy_router=False,
        original_bf16_corpus_quality=False,
        full_vocabulary_cross_engine_kl=None,
        full_goal_complete=False,
    )
    for row in sorted(final, key=lambda item: item["requests"]):
        count = row["requests"]
        external = enrich_external(controls["strongest_external_http"][str(count)])
        prior_row = ordinary["matrices"]["candidate"][-1] if count == 8 else previous["matrices"]["mtp2" if count == 1 else "mtp-batch"]["candidate"][0 if count == 1 else (1 if count == 2 else 2)]
        assert prior_row["requests"] == count
        earlier_rates = prior_row["trial_http_tps"]
        all_observed_external = [r for r in controls["unpromoted_screens"] if r["engine"] == "llama" and r["requests"] == count]
        highest_screen = max([external, *all_observed_external], key=lambda item: item["http_tps"])
        result["selected_http"][str(count)] = dict(
            native=row,
            strongest_repeated_external=external,
            ratio=row["http_tps"] / external["http_tps"],
            required_floor=1.05 * external["http_tps"],
            all_fresh_trials_clear_floor=all(rate >= 1.05 * external["http_tps"] for rate in row["trial_http_tps"]),
            prior_qualified_trials=earlier_rates,
            all_prior_trials_clear_new_floor=all(rate >= 1.05 * external["http_tps"] for rate in earlier_rates),
            highest_observed_screen_not_silently_promoted=highest_screen,
            all_fresh_trials_also_clear_highest_screen_floor=all(rate >= 1.05 * highest_screen["http_tps"] for rate in row["trial_http_tps"]),
        )
    single = next(row for row in final if row["requests"] == 1)
    external = controls["strongest_external_single"]
    earlier = previous["single_decode_comparison"]["trial_tps"]
    confirmations = [rate for row in controls["repeated_cells"] if row["engine"] == "venturi" and row["depth"] == 2 and row["requests"] == 1 for rate in row["trial_post_first_tps"]]
    result["single_post_first"] = dict(
        native_tps=single["post_first_tps"],
        fresh_trials=single["trial_post_first_tps"],
        stronger_external=external,
        required_floor=external["post_first_tps"] * 1.05,
        ratio=single["post_first_tps"] / external["post_first_tps"],
        earlier_qualified_trials=earlier,
        all_confirmations=confirmations,
        all_trials_clear_floor=all(rate >= external["post_first_tps"] * 1.05 for rate in earlier + confirmations),
    )
    result["native_reference_quality"] = quality
    result["finite_cross_engine_diagnostic"] = dict(
        rows=11,
        top1_equal_count=aligned["top1_equal_count"],
        mismatches=[row for row in aligned["rows"] if not row["top1_equal"]],
        top64_overlap_range=[min(row["top64_overlap"] for row in aligned["rows"]), max(row["top64_overlap"] for row in aligned["rows"])],
        max_absolute_logprob_difference_over_external_top64=max(row["max_abs_logprob_difference_over_external_top64"] for row in aligned["rows"]),
        top64_is_not_full_vocabulary_kl=True,
        exact_arithmetic_or_corpus_parity_claim=False,
    )
    result["separate_ordinary_c8_profiles"] = {}
    for arm in ["control", "candidate"]:
        profile = read(OUT / "v100/profile-v15" / (arm + "-ordinary256-c8-attribution.json"))
        cohort = next(row for row in profile["cohorts"] if row["name"] == "native_unsplit")
        result["separate_ordinary_c8_profiles"][arm] = dict(
            instrumented_host_span_ms=cohort["host_cohort_span_nanoseconds"] / 1e6,
            summed_kernel_ms_not_critical_path=cohort["kernel_nanoseconds"] / 1e6,
            simultaneous_two_device_kernel_ms=cohort["simultaneous_two_device_kernel_nanoseconds"] / 1e6,
            process_activity_gaps_ms_not_device_idle=cohort["gaps_between_this_process_activities_nanoseconds"] / 1e6,
            transfers=cohort["transfers"],
        )
    result["nonspeculative_single_loss_preserved"] = dict(native_post_first_tps=ordinary["matrices"]["candidate"][0]["native_post_first_tps"], scope="Ordinary mode is not the selected single strategy; retained comparator no-speculative decode is faster. No universal-mode win.")
    result["scalar_mtp2_parallel_loss_preserved"] = previous["matrices"]["mtp2"]["candidate"][-1]
    result["numeric_goal_met"] = result["single_post_first"]["all_trials_clear_floor"] and all(row["all_fresh_trials_clear_floor"] and row["all_prior_trials_clear_new_floor"] for row in result["selected_http"].values())
    result["completion_audit_required"] = "Source/payload/library pins, all terminal states, matched-control support/failures, precision disclosures and evidence freeze still require the explicit final audit. Numeric wins alone do not set goal completion."
    (OUT / "selected-comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(single={key: result["single_post_first"][key] for key in ["native_tps", "fresh_trials", "required_floor", "ratio", "all_trials_clear_floor"]}, external_single_tps=external["post_first_tps"], http={key: dict(native=row["native"]["http_tps"], external=row["strongest_repeated_external"]["http_tps"], ratio=row["ratio"], fresh_clear=row["all_fresh_trials_clear_floor"], prior_clear=row["all_prior_trials_clear_new_floor"], highest_screen=row["highest_observed_screen_not_silently_promoted"]["http_tps"], also_clear_screen_floor=row["all_fresh_trials_also_clear_highest_screen_floor"]) for key, row in result["selected_http"].items()}, numeric_goal_met=result["numeric_goal_met"], aligned_top1=aligned["top1_equal_count"], profiles=result["separate_ordinary_c8_profiles"], full_goal_complete=False), indent=2))


if __name__ == "__main__":
    main()
