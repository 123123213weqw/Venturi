"""Recompute the bounded operator gate; never convert it into model TPS."""

import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import statistics

OUT = Path(__file__).resolve().parent


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def telemetry(samples, waves, arm, gpu, offset):
    windows = [(w["start_unix"], w["end_unix"]) for w in waves if w["arm"] == arm]
    values = []
    tz = timezone(timedelta(seconds=offset))
    for row in samples:
        if len(row) != 9 or int(row[1]) != gpu:
            continue
        stamp = (
            datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
            .replace(tzinfo=tz)
            .timestamp()
        )
        if any(a <= stamp <= b for a, b in windows):
            values.append([float(v) for v in row[2:]])
    names = [
        "gpu_busy_percent",
        "memory_controller_busy_percent",
        "device_used_mib",
        "power_watts",
        "temperature_c",
        "sm_clock_mhz",
        "memory_clock_mhz",
    ]
    return dict(
        samples=len(values),
        values={
            name: dict(
                mean=statistics.mean(r[i] for r in values),
                min=min(r[i] for r in values),
                max=max(r[i] for r in values),
            )
            for i, name in enumerate(names)
        }
        if values
        else {},
    )


def main():
    base = OUT / "v100/campaign-v2"
    report = read(base / "timed/report.json")
    assert report["completed"] and not report["profiled"]
    assert len(report["cases"]) == 18
    samples = list(
        csv.reader((base / "timed/gpu-samples.csv").read_text().splitlines())
    )
    offset = read(OUT / "telemetry-timezone.json")["utc_offset_seconds"]
    result = dict(
        scope=report["scope"],
        model_dispatch_changed=False,
        new_model_tps_measured=False,
        full_goal_complete=False,
        achieved_occupancy=None,
        cases=[],
    )
    for c in report["cases"]:
        arms = []
        for arm in [0, 1]:
            ws = [w for w in c["waves"] if w["arm"] == arm]
            assert len(ws) == 3 and all(w["calls"] == 1024 for w in ws)
            checks = [v for v in c["reference_checks"] if v["arm"] == arm]
            arms.append(
                dict(
                    arm=arm,
                    median_gpu_event_us_per_call=statistics.median(
                        w["gpu_event_ms_per_call"] * 1000 for w in ws
                    ),
                    trial_gpu_event_us_per_call=[
                        w["gpu_event_ms_per_call"] * 1000 for w in ws
                    ],
                    mean_wall_us_per_call=statistics.mean(
                        w["wall_seconds"] / w["calls"] * 1e6 for w in ws
                    ),
                    mean_cpu_core_us_per_call=statistics.mean(
                        w["cpu_core_seconds"] / w["calls"] * 1e6 for w in ws
                    ),
                    max_sampled_reference_relative_l2=max(
                        v["relative_l2"] for v in checks
                    ),
                    max_sampled_unquantized_activation_relative_l2=max(
                        v["versus_unquantized_fp16_activation_fp64"]["relative_l2"]
                        for v in checks
                    ),
                    telemetry={
                        str(g): telemetry(samples, c["waves"], arm, g, offset)
                        for g in [1, 0]
                    },
                )
            )
        ratio = (
            arms[0]["median_gpu_event_us_per_call"]
            / arms[1]["median_gpu_event_us_per_call"]
        )
        assert ratio == c["operator_throughput_ratio"]
        result["cases"].append(
            dict(
                label=c["label"],
                tokens=c["tokens"],
                shapes=c["shapes"],
                output_dtype=c["output_dtype"],
                operator_throughput_ratio=ratio,
                arms=arms,
            )
        )
    fused = [
        c["operator_throughput_ratio"]
        for c in result["cases"]
        if c["label"].startswith("fused")
    ]
    median = statistics.median(fused)
    minimum = min(c["operator_throughput_ratio"] for c in result["cases"])
    promote = median >= 1.05 and minimum >= 0.98
    assert (
        median == report["median_fused_operator_ratio"]
        and promote == report["promote_to_model"]
    )
    result.update(
        median_fused_operator_ratio=median,
        minimum_operator_ratio=minimum,
        median_fused_gate=1.05,
        minimum_cell_gate=0.98,
        promote_to_model=promote,
        decision="Do not integrate: dominant fused median misses predeclared operator gate"
        if not promote
        else "Operator gate passes; model quality and speed remain unproven",
    )
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
