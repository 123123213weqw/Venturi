"""Recompute completed ordinary controls; never infer final goal completion."""

import hashlib
import json
from pathlib import Path
import runpy
import statistics

OUT = Path(__file__).resolve().parent
RAW = OUT / "v100/controls-v1"
telemetry = runpy.run_path(str(OUT.parent / "v100-batched-mtp/summarize.py"))["telemetry"]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (OUT / 'evidence-sha256.json').exists(), 'Never overwrite frozen evidence'
    campaign = read(RAW / "campaign-report.json")
    result = dict(
        scope="Ordinary Q8, no speculative decoding: 32*N/global complete-cohort wall; single post-first=31/(generation-first-ready). Admission, prefill and HTTP overhead stay in the complete-cohort denominator.",
        library_sha256=campaign["library_sha256"],
        projection_rows=256,
        physical_gpus=[1, 0],
        matrices={},
        native_ratios=[],
        full_goal_complete=False,
        achieved_hardware_counters=None,
        raw_manifest={},
    )
    for arm in ("control", "candidate"):
        label = f"matrix-ordinary256-{arm}"
        entry = next(e for e in campaign["experiments"] if e["label"] == label)
        path = RAW / label / "timed/report.json"
        data = read(path)
        assert entry["completed"] and entry["stable_outputs"]
        assert data["completed"] and data["all_outputs_equal_first_wave"]
        assert len(data["case"]["prompt_token_ids"]) == 23
        expected = data["case"]["expected_generated_ids"]
        assert len(expected) == data["case"]["max_new_tokens"] == 32
        rows = []
        for count in (1, 2, 4, 8):
            waves = [w for w in data["waves"] if w["request_count"] == count]
            assert len(waves) == 3 and {w["repeat"] for w in waves} == {0, 1, 2}
            for wave in waves:
                assert wave["emitted_tokens"] == 32 * count
                assert len(wave["requests"]) == count
                assert len({r["response"]["batch_cohort_id"] for r in wave["requests"]}) == 1
                for request in wave["requests"]:
                    response = request["response"]
                    assert response["tokens"] == expected
                    assert response["tokens_evaluated"] == 23
                    assert response["tokens_predicted"] == 32
                    assert response["tokens_cached"] == 0
                    assert response["batch_request_count"] == count
                    assert "mtp_stats" not in response
                    assert response["batch_stats"]["group_forward_calls"] == 32
            row = dict(
                requests=count,
                trials=3,
                http_tps=32 * count / statistics.mean(w["wall_seconds"] for w in waves),
                trial_http_tps=[32 * count / w["wall_seconds"] for w in waves],
                mean_cohort_seconds=statistics.mean(w["wall_seconds"] for w in waves),
                mean_server_cpu_core_seconds=statistics.mean(w["server_cpu_core_seconds"] for w in waves),
                mean_client_completion_seconds=statistics.mean(r["client_completion_seconds"] for w in waves for r in w["requests"]),
                max_client_completion_seconds=max(r["client_completion_seconds"] for w in waves for r in w["requests"]),
                mean_admission_seconds=statistics.mean(r["response"]["batch_admission_delay_seconds"] for w in waves for r in w["requests"]),
                nvml_by_physical_gpu={str(gpu): telemetry(RAW / label, data, count, gpu, campaign["utc_offset_seconds"]) for gpu in (1, 0)},
            )
            assert all(v["samples"] > 0 for v in row["nvml_by_physical_gpu"].values())
            summary = next(s for s in entry["summary"] if s["request_count"] == count)
            assert abs(row["http_tps"] - summary["aggregate_complete_request_output_tps"]) < 1e-10
            if count == 1:
                times = [w["requests"][0]["response"]["native_timings"] for w in waves]
                elapsed = [t["native_generation_seconds"] - t["native_time_to_first_token_seconds"] for t in times]
                row.update(
                    native_post_first_tps=31 / statistics.mean(elapsed),
                    trial_native_post_first_tps=[31 / value for value in elapsed],
                    mean_native_first_ready_seconds=statistics.mean(t["native_time_to_first_token_seconds"] for t in times),
                )
            rows.append(row)
        result["matrices"][arm] = rows
        for p in [path, RAW / label / "command.json", *sorted((RAW / label).glob("gpu-*-samples.csv"))]:
            result["raw_manifest"][str(p.relative_to(OUT))] = sha(p)
    for old, new in zip(result["matrices"]["control"], result["matrices"]["candidate"]):
        result["native_ratios"].append(dict(
            requests=old["requests"],
            http_ratio=new["http_tps"] / old["http_tps"],
            paired_trial_http_ratios=[y / x for x, y in zip(old["trial_http_tps"], new["trial_http_tps"])],
            cpu_ratio=new["mean_server_cpu_core_seconds"] / old["mean_server_cpu_core_seconds"],
        ))
    (OUT / "ordinary-metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"ratios": result["native_ratios"], "candidate_tps": [r["http_tps"] for r in result["matrices"]["candidate"]], "full_goal_complete": False}, indent=2))


if __name__ == "__main__":
    main()
