"""Recompute experimental projection ratios without promoting them to serving TPS."""

import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent


def read(p):
    return json.loads(p.read_text())


def main():
    raw = ROOT / "v100"
    result = dict(
        scope=__doc__,
        new_compute_profiles_not_full_model_qualified=True,
        default_engine_arithmetic_unchanged=True,
        external_speed_goal_complete=False,
        operator_experiments={},
    )
    for name in ["scalar-projection-v2", "vector-v3"]:
        report = read(raw / name / "projection/report.json")
        assert report["completed"] and len(report["cases"]) == 32
        summaries = []
        for tokens in [1, 2, 8, 32]:
            cases = [c for c in report["cases"] if c["tokens"] == tokens]
            for i, profile in enumerate(report["profiles"][1:], 1):
                speed = [
                    c["median_gpu_event_ms"]["original-fp16"]
                    / c["median_gpu_event_ms"][profile]
                    for c in cases
                ]
                errors = [
                    c["original_profile_comparison"][i]["relative_l2"] for c in cases
                ]
                assert all(
                    c["sampled_reference"][i]["relative_l2"] < 0.001 for c in cases
                )
                summaries.append(
                    dict(
                        tokens=tokens,
                        profile=profile,
                        shapes=len(cases),
                        min_projection_speedup=min(speed),
                        median_projection_speedup=statistics.median(speed),
                        max_projection_speedup=max(speed),
                        relative_l2_vs_original_range=[min(errors), max(errors)],
                        original_top1_agrees_all_rows_all_shapes=all(
                            c["same_original_top1"][i] for c in cases
                        ),
                        gate_scope="Independent sampled-row FP64 reconstruction of this explicit activation profile, not exact original FP16 arithmetic or full-model quality",
                    )
                )
        result["operator_experiments"][name] = dict(
            library_sha256=report["library_sha256"], summary=summaries
        )
    for name, labels in [
        ("t8-screen-v1", ["baseline-batch", "adaptive-batch"]),
        ("worker-screen-v1", ["baseline-batch", "worker-batch"]),
    ]:
        result[name] = {}
        for label in labels:
            report = read(raw / name / label / "timed/report.json")
            assert report["completed"] and len(report["waves"]) == 2
            result[name][label] = report["summary"]
    result["actual_competitor_arithmetic"] = read(
        raw / "llama-arithmetic-v2/campaign-report.json"
    )
    result["separate_projection_attribution"] = read(
        raw / "final-v4/projection-attribution.json"
    )
    result["hardware_counter_policy"] = (
        (raw / "final-v4/counter-policy.txt").read_text().strip()
    )
    result["theoretical_residency"] = read(raw / "vector-v3/resources.json")
    result["failed_first_competitor_profile"] = dict(
        health_observer_main_thread_children_only=True,
        salvage_intersected_observer_watchdog=True,
        c1_response_preserved=True,
        c8_incomplete=True,
        no_inference_failure_or_zero_tps_inferred=True,
    )
    (ROOT / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    for row in result["operator_experiments"]["vector-v3"]["summary"]:
        print(row["tokens"], row["profile"], round(row["median_projection_speedup"], 3))


if __name__ == "__main__":
    main()
