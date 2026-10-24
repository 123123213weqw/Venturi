"""Recompute native/new-arithmetic/external comparisons with explicit scopes."""

import json
from pathlib import Path
import runpy
import statistics

OUT = Path(__file__).resolve().parent
telemetry = runpy.run_path(str(OUT.parent / "v100-batched-mtp/summarize.py"))[
    "telemetry"
]


def read(p):
    return json.loads(p.read_text())


def main():
    base = OUT / "v100/campaign-v6"
    campaign = read(base / "campaign-report.json")
    assert campaign["completed"]
    result = dict(
        external_speed_goal_complete=False,
        scope="32 emitted outputs/request divided by complete HTTP cohort wall. Native single post-first31/(generation minus TTFT); llama31/predicted_ms. No sums of request times, no streaming TTFT claim.",
        screen_ratios=campaign["screen_ratios"],
        path_promotions=campaign["path_promotions"],
        experiments=[],
        adoption_checks={},
        best_repeated_native_http_by_count={},
        fresh_external_http_by_count={},
        achieved_occupancy=None,
    )
    for e in campaign["experiments"]:
        if not e["completed"]:
            continue
        b = read(base / e["label"] / "timed/report.json")
        rows = []
        for count in sorted({w["request_count"] for w in b["waves"]}):
            waves = [w for w in b["waves"] if w["request_count"] == count]
            requests = [r for w in waves for r in w["requests"]]
            responses = [r["response"] for r in requests]
            assert all(w["emitted_tokens"] == 32 * count for w in waves)
            assert all(
                r["tokens"] == b["case"]["expected_generated_ids"] for r in responses
            )
            row = dict(
                requests=count,
                repeats=len(waves),
                http_complete_cohort_output_tps=32
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
                        base / e["label"], b, count, g, campaign["utc_offset_seconds"]
                    )
                    for g in [1, 0]
                },
            )
            if all("native_timings" in r for r in responses):
                ts = [r["native_timings"] for r in responses]
                row["mean_native_ttft_seconds"] = statistics.mean(
                    t["native_time_to_first_token_seconds"] for t in ts
                )
                row["mean_native_generation_seconds"] = statistics.mean(
                    t["native_generation_seconds"] for t in ts
                )
                if count == 1:
                    row["native_post_first_tps"] = 31 / statistics.mean(
                        t["native_generation_seconds"]
                        - t["native_time_to_first_token_seconds"]
                        for t in ts
                    )
            else:
                row["external_engine_timings"] = [r["timings"] for r in responses]
                if count == 1:
                    assert all(r["timings"]["predicted_n"] == 32 for r in responses)
                    row["external_post_first_tps"] = 31 / statistics.mean(
                        r["timings"]["predicted_ms"] / 1000 for r in responses
                    )
            if all("mtp_stats" in r for r in responses):
                ss = [r["mtp_stats"] for r in responses]
                row["per_request_mtp_counters"] = ss
                row["accepted_over_proposed"] = sum(
                    s["accepted_tokens"] for s in ss
                ) / sum(s["proposed_tokens"] for s in ss)
                assert all(
                    s["replayed_input_tokens"] == s["forced_proposals"] == 0 for s in ss
                )
            if all("mtp_batch_stats" in r for r in responses):
                shared = []
                for w in waves:
                    s = w["requests"][0]["response"]["mtp_batch_stats"]
                    assert all(
                        r["response"]["mtp_batch_stats"] == s for r in w["requests"]
                    )
                    assert (
                        len({r["response"]["batch_cohort_id"] for r in w["requests"]})
                        == 1
                    )
                    shared.append(s)
                row["shared_mtp_stats_once_per_wave"] = shared
                row["native_cohort_post_all_first_output_tps"] = (
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
                label=e["label"],
                engine=e["engine"],
                mode=e["mode"],
                depth=e["mtp_depth"],
                profile=e["projection_profile"],
                native_projection_rows=e["projection_rows"]
                if e["engine"] == "venturi"
                else None,
                external_microbatch_tokens=256 if e["engine"] == "llama" else None,
                rows=rows,
            )
        )
    bcase = b["case"]
    ix = {
        e["label"]: {r["requests"]: r for r in e["rows"]} for e in result["experiments"]
    }
    for name, promoted in campaign["path_promotions"].items():
        if not promoted:
            continue
        a, b = (ix["repeat-" + arm + "-" + name] for arm in ["control", "candidate"])
        assert set(a) == set(b) == {1, 2, 4, 8}
        assert all(a[n]["repeats"] == b[n]["repeats"] == 3 for n in a)
        ratios = {
            str(n): b[n]["http_complete_cohort_output_tps"]
            / a[n]["http_complete_cohort_output_tps"]
            for n in a
        }
        cells = {}
        for n, threshold in [(1, 1.01), (8, 1.02)]:
            improves = sum(
                y > x
                for x, y in zip(a[n]["trial_output_tps"], b[n]["trial_output_tps"])
            )
            cells[str(n)] = dict(
                ratio=ratios[str(n)],
                threshold=threshold,
                improving_trials=improves,
                qualifies=ratios[str(n)] > threshold and improves >= 2,
            )
        stable = all(r >= 0.98 for r in ratios.values())
        result["adoption_checks"][name] = dict(
            ratios=ratios,
            cells=cells,
            no_cell_regresses_over_two_percent=stable,
            qualifies=stable and any(c["qualifies"] for c in cells.values()),
        )
    for n in [1, 2, 4, 8]:
        native = []
        external = []
        for e in result["experiments"]:
            if not e["label"].startswith("repeat"):
                continue
            for r in e["rows"]:
                if r["requests"] != n:
                    continue
                entry = dict(
                    label=e["label"],
                    mode=e["mode"],
                    depth=e["depth"],
                    profile=e["profile"],
                    http_tps=r["http_complete_cohort_output_tps"],
                )
                if e["engine"] == "venturi":
                    name = e["label"].removeprefix("repeat-candidate-")
                    if (
                        e["label"].startswith("repeat-candidate-")
                        and not result["adoption_checks"][name]["qualifies"]
                    ):
                        continue
                    native.append(entry)
                else:
                    external.append(entry)
        if native and external:
            a = max(native, key=lambda r: r["http_tps"])
            b = max(external, key=lambda r: r["http_tps"])
            result["best_repeated_native_http_by_count"][str(n)] = dict(
                **a, over_fresh_external_ratio=a["http_tps"] / b["http_tps"]
            )
            result["fresh_external_http_by_count"][str(n)] = b
    if "repeat-candidate-mtp2" in ix:
        a = ix["repeat-candidate-mtp2"][1]["native_post_first_tps"]
        b = ix["repeat-llama-mtp2-fresh"][1]["external_post_first_tps"]
        result["single_decode_comparison"] = dict(
            native_tps=a,
            external_tps=b,
            ratio=a / b,
            goal_floor=1.05 * b,
            additional_native_gain_needed_ratio=1.05 * b / a,
            goal_met=a >= 1.05 * b,
        )
    historical = []
    for cap, folder in [
        ("v100-q8-wmma", "campaign-v5"),
        ("v100-strided-output", "external-depth-v9"),
    ]:
        base_old = OUT.parent / cap / "v100" / folder
        old_campaign = read(base_old / "campaign-report.json")
        assert (
            OUT.parent / cap / "source/tools/benchmark_completion_api.py"
        ).read_bytes() == (
            OUT.parents[1] / "tools/benchmark_completion_api.py"
        ).read_bytes()
        for e in old_campaign["experiments"]:
            if (
                e["engine"] != "llama"
                or not e["label"].startswith("repeat")
                or not e["completed"]
            ):
                continue
            release = read(OUT.parent / "v100-llama-comparison/v100/screen-report.json")
            assert (
                release["llama_library_hashes"]["llama-server"]
                == "e5b1b5782cd5b4acec5316ac5bd40123e0b0c9826643ae4148b0ec6e359f5fa2"
            )
            assert e["command"][0] == str(
                Path(release["llama_binary_directory"]) / "llama-server"
            )
            assert e["library_sha256"] is None, (
                "Historical field names the native library only; do not invent an old per-item binary hash"
            )
            old = read(base_old / e["label"] / "timed/report.json")
            for n in sorted({w["request_count"] for w in old["waves"]}):
                waves = [w for w in old["waves"] if w["request_count"] == n]
                assert len(waves) == 3 and old["case"] == bcase
                r = dict(
                    capsule=cap + "/v100/" + folder,
                    label=e["label"],
                    requests=n,
                    depth=e["mtp_depth"],
                    http_tps=32 * n / statistics.mean(w["wall_seconds"] for w in waves),
                )
                if n == 1:
                    r["post_first_tps"] = 31 / statistics.mean(
                        x["response"]["timings"]["predicted_ms"] / 1000
                        for w in waves
                        for x in w["requests"]
                    )
                historical.append(r)
    result["retained_stronger_pinned_external_repeats"] = historical
    result["best_known_pinned_external_http_by_count"] = {}
    for n in [1, 2, 4, 8]:
        candidates = [r for r in historical if r["requests"] == n]
        if str(n) in result["fresh_external_http_by_count"]:
            candidates.append(
                dict(
                    capsule="current fresh",
                    **result["fresh_external_http_by_count"][str(n)],
                )
            )
        best = max(candidates, key=lambda r: r["http_tps"])
        result["best_known_pinned_external_http_by_count"][str(n)] = best
        native = result["best_repeated_native_http_by_count"][str(n)]
        native["over_best_known_pinned_external_ratio"] = (
            native["http_tps"] / best["http_tps"]
        )
    if "single_decode_comparison" in result:
        best = max(
            [r["post_first_tps"] for r in historical if r["requests"] == 1]
            + [result["single_decode_comparison"]["external_tps"]]
        )
        r = result["single_decode_comparison"]
        r.update(
            best_known_pinned_external_tps=best,
            best_known_goal_floor=best * 1.05,
            over_best_known_ratio=r["native_tps"] / best,
            additional_gain_to_best_known_goal_ratio=best * 1.05 / r["native_tps"],
            best_known_goal_met=r["native_tps"] >= best * 1.05,
        )
    result["complete_http_matrix_goal_met"] = all(
        r["over_best_known_pinned_external_ratio"] >= 1.05
        for r in result["best_repeated_native_http_by_count"].values()
    ) and set(result["best_repeated_native_http_by_count"]) == {"1", "2", "4", "8"}
    result["external_speed_goal_complete"] = (
        result["complete_http_matrix_goal_met"]
        and result["single_decode_comparison"]["best_known_goal_met"]
    )
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                adoption=result["adoption_checks"],
                single=result.get("single_decode_comparison"),
                best_http=result["best_repeated_native_http_by_count"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
