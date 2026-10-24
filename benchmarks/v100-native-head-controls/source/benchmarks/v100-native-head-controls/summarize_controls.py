"""Retain repeated cells, valid prefixes, failures and all stronger references.

An incomplete campaign can contain a completed three-trial count prefix. A
single-repeat screen is preserved but never silently promoted to a baseline.
Metrics alone do not establish final correctness or goal completion.
"""

import hashlib
import json
from pathlib import Path
import runpy
import statistics

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
telemetry = runpy.run_path(str(OUT.parent / "v100-batched-mtp/summarize.py"))["telemetry"]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def option(command, name):
    return command[command.index(name) + 1] if name in command else None


def main():
    assert not (OUT / 'evidence-sha256.json').exists(), 'Never overwrite frozen evidence'
    reference = read(OUT.parent / "v100-native-head-gdn/v100/campaign-v5/matrix-mtp2-candidate/timed/report.json")["case"]
    historical = read(OUT / "historical-external-audit.json")
    original_pins = read(OUT.parent / "v100-llama-comparison/v100/screen-report.json")["llama_library_hashes"]
    cells, screens, failures, captured = [], [], [], []
    for capture in sorted(OUT.glob("*-capture-manifest.json")):
        tag = capture.name.removesuffix("-capture-manifest.json")
        directory = OUT / "v100" / tag
        manifest = read(capture)
        assert all(sha(directory / rel) == item["sha256"] for rel, item in manifest["manifest"].items())
        captured.append(dict(tag=tag, files=len(manifest["manifest"]), exit_code=manifest["exit_code"], manifest_sha256=sha(capture)))
        campaign_path = directory / "campaign-report.json"
        if not campaign_path.exists():
            continue
        campaign = read(campaign_path)
        for experiment in campaign["experiments"]:
            label = experiment["label"]
            base = directory / label
            if not experiment["completed"]:
                failures.append(dict(tag=tag, label=label, failure=experiment.get("failure"), server_exit_before=experiment.get("server_exit_code_before_cleanup"), server_exit_after=experiment.get("server_exit_code_after_cleanup"), not_counted_as_zero_tps=True))
            path = base / "timed/report.json"
            if not path.exists():
                continue
            data = read(path)
            assert data["case"]["prompt_token_ids"] == reference["prompt_token_ids"]
            assert data["case"]["max_new_tokens"] == 32
            command = read(base / "command.json")
            is_native = experiment["engine"] == "venturi"
            loaded_pins_verified = None
            if not is_native:
                assert option(command, "-ctk") == option(command, "-ctv") == "f16"
                assert option(command, "--parallel") == "8" and option(command, "-c") == "2048"
                env = read(base / "actual-environment.json")
                assert env["GGML_CUDA_DISABLE_GRAPHS"] in (["1"] if experiment.get("compatibility_disable_graphs") else [None, "0"])
                loaded = read(base / "loaded-library-pins.json")
                engine = {Path(p).name: value["sha256"] for p, value in loaded.items() if Path(p).name.startswith(("libllama", "libggml"))}
                assert engine and all(original_pins.get(name) == h for name, h in engine.items())
                loaded_pins_verified = engine
            else:
                assert option(command, "--slots") == "8" and option(command, "--capacity") == "256"
            for count in sorted({wave["request_count"] for wave in data["waves"]}):
                waves = [wave for wave in data["waves"] if wave["request_count"] == count]
                ids = []
                for wave in waves:
                    assert wave["emitted_tokens"] == 32 * count and len(wave["requests"]) == count
                    for request in wave["requests"]:
                        response = request["response"]
                        assert response["tokens_evaluated"] == 23
                        assert response["tokens_predicted"] == len(response["tokens"]) == 32
                        if is_native:
                            assert response["tokens"] == reference["expected_generated_ids"]
                            if "mtp_stats" in response:
                                stats = response["mtp_stats"]
                                assert stats["forced_proposals"] == stats["replayed_input_tokens"] == 0
                        else:
                            assert response["timings"]["prompt_n"] == 23
                            assert response["timings"]["cache_n"] == 0
                        ids.append(response["tokens"])
                assert all(value == ids[0] for value in ids), "Unstable greedy trajectory in a measured cell"
                row = dict(
                    tag=tag, label=label, engine=experiment["engine"],
                    mode=experiment["mode"], depth=experiment["mtp_depth"],
                    requests=count, trials=len(waves),
                    cpu_sampling_fallback=experiment.get("cpu_sampling_fallback", False),
                    sampling_scope="Flags record the requested sampler route; retained server logs disclose automatic backend-offload fallback. Do not infer executed GPU sampling solely from a flag.",
                    compatibility_disable_graphs=experiment.get("compatibility_disable_graphs", False),
                    whole_report_completed=data["completed"],
                    valid_completed_count_prefix=not data["completed"],
                    report=str(path.relative_to(ROOT)), report_sha256=sha(path),
                    command_sha256=sha(base / "command.json"),
                    loaded_engine_pins_verified=loaded_pins_verified,
                    greedy_ids_match_native_story=ids[0] == reference["expected_generated_ids"],
                    http_tps=count * 32 / statistics.mean(wave["wall_seconds"] for wave in waves),
                    trial_http_tps=[count * 32 / wave["wall_seconds"] for wave in waves],
                    mean_cohort_seconds=statistics.mean(wave["wall_seconds"] for wave in waves),
                    mean_server_cpu_core_seconds=statistics.mean(wave["server_cpu_core_seconds"] for wave in waves),
                    request_completion_seconds={
                        "mean": statistics.mean(q["client_completion_seconds"] for wave in waves for q in wave["requests"]),
                        "min": min(q["client_completion_seconds"] for wave in waves for q in wave["requests"]),
                        "max": max(q["client_completion_seconds"] for wave in waves for q in wave["requests"]),
                    },
                    nvml_by_physical_gpu={str(gpu): telemetry(base, data, count, gpu, campaign["utc_offset_seconds"]) for gpu in (1, 0)},
                    streaming_client_ttft_seconds=None,
                    speculative_acceptance_per_completed_request=[
                        dict(
                            accepted=q["response"]["mtp_stats"]["accepted_tokens"],
                            proposed=q["response"]["mtp_stats"]["proposed_tokens"],
                            forced=q["response"]["mtp_stats"]["forced_proposals"],
                            target_replayed_input_tokens=q["response"]["mtp_stats"]["replayed_input_tokens"],
                        ) if "mtp_stats" in q["response"] else dict(
                            accepted=q["response"]["timings"]["draft_n_accepted"],
                            proposed=q["response"]["timings"]["draft_n"],
                        )
                        for wave in waves for q in wave["requests"]
                        if "mtp_stats" in q["response"] or "draft_n" in q["response"].get("timings", {})
                    ],
                )
                assert all(v["samples"] > 0 for v in row["nvml_by_physical_gpu"].values())
                if count == 1:
                    responses = [wave["requests"][0]["response"] for wave in waves]
                    if is_native:
                        times = [r["native_timings"] for r in responses]
                        elapsed = [t["native_generation_seconds"] - t["native_time_to_first_token_seconds"] for t in times]
                        row["native_first_ready_seconds"] = [t["native_time_to_first_token_seconds"] for t in times]
                    else:
                        times = [r["timings"] for r in responses]
                        elapsed = [t["predicted_ms"] / 1000 for t in times]
                        row["external_prompt_eval_seconds_not_client_ttft"] = [t["prompt_ms"] / 1000 for t in times]
                    row.update(post_first_tps=31 / statistics.mean(elapsed), trial_post_first_tps=[31 / seconds for seconds in elapsed])
                if len(waves) >= 3:
                    assert {wave["repeat"] for wave in waves} == set(range(len(waves)))
                    cells.append(row)
                else:
                    screens.append(row)
    external = historical["comparable_repeated_cells"] + [row for row in cells if row["engine"] == "llama"]
    strongest_http = {str(count): max((row for row in external if row["requests"] == count), key=lambda row: row["http_tps"]) for count in (1, 2, 4, 8)}
    strongest_single = max((row for row in external if "post_first_tps" in row), key=lambda row: row["post_first_tps"])
    result = dict(scope=__doc__, captured=captured, repeated_cells=cells, unpromoted_screens=screens, failures=failures, strongest_external_http=strongest_http, strongest_external_single=strongest_single, achieved_hardware_counters=None, full_goal_complete=False)
    (OUT / "recomputed-controls.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(captured=len(captured), repeated_cells=len(cells), screens=len(screens), failed_arms=len(failures), strongest_single=strongest_single["post_first_tps"], strongest_http={k: row["http_tps"] for k, row in strongest_http.items()}, full_goal_complete=False), indent=2))


if __name__ == "__main__":
    main()
