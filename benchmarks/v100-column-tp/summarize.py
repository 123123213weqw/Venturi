"""Recompute measured block speedups, telemetry and precision/engine limits."""

import csv
import datetime as dt
import json
from pathlib import Path
import statistics

OUT = Path(__file__).resolve().parent


def read(p):
    return json.loads(p.read_text())


def telemetry(base, waves, gpu):
    fields = [
        "gpu_busy_percent",
        "memory_controller_busy_percent",
        "device_used_mib",
        "power_watts",
        "temperature_c",
        "sm_clock_mhz",
        "memory_clock_mhz",
    ]
    rows = []
    for r in csv.reader((base / "gpu-samples.csv").open()):
        if len(r) != 9 or int(r[1]) != gpu:
            continue
        stamp = (
            dt.datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
            .replace(tzinfo=dt.timezone(dt.timedelta(hours=8)))
            .timestamp()
        )
        if any(w["start_unix"] <= stamp <= w["end_unix"] for w in waves):
            rows.append([float(x) for x in r[2:]])
    return dict(
        samples=len(rows),
        values={
            name: dict(
                mean=statistics.mean(r[i] for r in rows),
                min=min(r[i] for r in rows),
                max=max(r[i] for r in rows),
            )
            for i, name in enumerate(fields)
        }
        if rows
        else {},
    )


def main():
    base = OUT / "v100/mlp-v4/unprofiled"
    report = read(base / "report.json")
    assert report["completed"] and not report["profiled"]
    result = dict(
        scope="Matched primary-only normalized input/output; isolated MLP block, not serving TPS or complete model quality.",
        library_sha256=report["library_sha256"],
        source_artifact_id=report["artifact_id"],
        full_model_implemented=False,
        model_quality_qualified=False,
        external_speed_goal_complete=False,
        precision=report["precision"],
        achieved_hardware_counters=None,
        cases=[],
    )
    for c in report["cases"]:
        rows = []
        assert (
            c["matched_primary_only_input_output"]
            and c["local_gate_up_swiglu_bytes_equal"]
        )
        for mode in ["row", "column"]:
            waves = [w for w in c["waves"] if w["arm"] == mode]
            assert len(waves) == 3 and all(w["calls"] == 400 for w in waves)
            rows.append(
                dict(
                    arm=mode,
                    repeats=3,
                    calls_per_trial=400,
                    trial_event_ms_per_call=[w["gpu_event_ms_per_call"] for w in waves],
                    median_event_ms_per_call=statistics.median(
                        w["gpu_event_ms_per_call"] for w in waves
                    ),
                    median_wall_us_per_call=statistics.median(
                        w["wall_seconds"] * 1e6 / w["calls"] for w in waves
                    ),
                    median_cpu_core_us_per_call=statistics.median(
                        w["cpu_core_seconds"] * 1e6 / w["calls"] for w in waves
                    ),
                    logical_copy_bytes_each_direction_per_call=c[
                        "logical_copy_bytes_per_direction_per_call"
                    ][mode],
                    telemetry={str(g): telemetry(base, waves, g) for g in [1, 0]},
                )
            )
        assert all(
            a > b
            for a, b in zip(
                rows[0]["trial_event_ms_per_call"], rows[1]["trial_event_ms_per_call"]
            )
        )
        result["cases"].append(
            dict(
                tokens=c["tokens"],
                median_speed_ratio=c["speed_ratio"],
                all_three_trials_faster=True,
                different_fp16_outputs=c["different_fp16_outputs"],
                prior_profile_output_relative_l2=c["difference_l2"],
                sampled_fp64=c["fp64_checks"],
                arms=rows,
            )
        )
    result["separate_profile_attribution"] = read(OUT / "v100/mlp-v4/attribution.json")
    result["resources"] = read(OUT / "v100/mlp-v4/resources.json")
    prior_end = read(OUT / "v100/mlp-v3/completed.json")["ended_unix"]
    assert prior_end < min(w["start_unix"] for c in report["cases"] for w in c["waves"])
    result["prior_profile_ended_before_matched_unprofiled_trials"] = True
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    for c in result["cases"]:
        print(
            c["tokens"],
            c["median_speed_ratio"],
            [
                (
                    a["arm"],
                    a["median_event_ms_per_call"],
                    a["median_cpu_core_us_per_call"],
                    {
                        g: (
                            t["samples"],
                            t["values"].get("gpu_busy_percent"),
                            t["values"].get("device_used_mib"),
                        )
                        for g, t in a["telemetry"].items()
                    },
                )
                for a in c["arms"]
            ],
        )


if __name__ == "__main__":
    main()
