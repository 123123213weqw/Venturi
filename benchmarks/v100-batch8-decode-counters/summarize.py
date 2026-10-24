"""Recompute counter summaries without CUDA or model dependencies."""

import argparse
import csv
import io
import json
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    a = p.parse_args()
    root = a.root
    measurements = json.loads((root / "measurements.json").read_text())
    restored = json.loads((root / "production-restored.json").read_text())
    assert restored["healthy"] and restored["configuration_unchanged"]
    assert json.loads((root / "benchmark-complete.json").read_text())["passed"]
    timed, reports = [], []
    for arm in measurements:
        items = [json.loads(p.read_text()) for p in (root / arm["name"]).glob("run-*.json")]
        assert items and all(r["passed"] for r in items)
        reports.extend(items)
        for r in items:
            assert r["artifact_id"] == "3d635b87db004f42aefd6730e372e530"
            assert r["batch_size"] == 8 and r["measured_output_tokens"] == 248
            assert r["prompt_lengths"] == [23] * 8
            assert r["reference_positions"] == [54] * 8
            for trial in r["runs"]:
                assert trial["outputs"] == r["expected_outputs"]
                assert all(len(ids) == 32 for ids in trial["outputs"])
                assert trial["info"]["input_rows"] == 248
                if r["profiled"]:
                    assert trial["aggregate_decode_tps"] is None
                elif trial["phase"] == "measured":
                    timed.append(trial)
    assert len(timed) == 6
    devices = reports[0]["devices"]
    samples = [json.loads(line) for line in (root / "gpu-telemetry.jsonl").read_text().splitlines()]
    uuid_to_physical = {}
    for sample in samples:
        if sample.get("returncode") == 0:
            for values in csv.reader(io.StringIO(sample["csv"])):
                uuid_to_physical[values[1].strip().removeprefix("GPU-")] = int(values[0])
    assert len(uuid_to_physical) == 2
    result = dict(
        scope="Qwen3.5-27B Q8/group32, FP16 activations/KV, native two-V100 host TP, no MTP; 8 requests, 23 prompt tokens and 32 outputs each. Decode scope excludes prompt and first output; 31 steps/248 emitted tokens, with one continuation API re-entry.",
        counters="Each GPU profiled separately, three ranges, two application-replay passes per invocation. Per-GPU bandwidths are not a simultaneous aggregate. Profiler measurements are not unprofiled TPS.",
        precision="Existing qualified q8-residual-column-mlp-head-gdn arithmetic unchanged; this is an attribution experiment, not a kernel optimization.",
        production_restored=restored,
        correctness=dict(passed=True, full_model_executions=len(reports),
                         split_cohorts=sum(len(r["runs"]) for r in reports),
                         full_model_state_scope="Greedy IDs and cursor positions checked; bitwise canonical caches checked on the aligned tiny fixture only."),
        per_gpu=[],
    )
    mean_wall = statistics.mean(t["wall_seconds"] for t in timed)
    mean_cpu = statistics.mean(t["process_cpu_seconds"] for t in timed)
    result["unprofiled_decode"] = dict(
        repeats=len(timed), emitted_tokens_per_repeat=248,
        mean_wall_seconds=mean_wall,
        min_wall_seconds=min(t["wall_seconds"] for t in timed),
        max_wall_seconds=max(t["wall_seconds"] for t in timed),
        aggregate_tps=248 / mean_wall, per_request_tps=31 / mean_wall,
        mean_process_cpu_seconds=mean_cpu, mean_cpu_cores=mean_cpu / mean_wall,
    )
    for arm in measurements:
        if "ranges" not in arm:
            continue
        rows = arm["ranges"]
        selected = int(rows[0]["device"])
        ns = sum(r["gpu__time_duration.sum"] for r in rows)
        read = sum(r["dram__bytes_read.sum"] for r in rows)
        write = sum(r["dram__bytes_write.sum"] for r in rows)
        occupancy = [r["sm__warps_active.avg.pct_of_peak_sustained_active"] for r in rows]
        raw = (root / (arm["name"] + ".csv")).read_text()
        parsed = [r for r in csv.DictReader(io.StringIO(raw[raw.find('"ID",'):])) if r.get("ID")]
        assert all(int(r["profiler__replayer_passes"]) == 2 for r in parsed)
        result["per_gpu"].append(dict(
            logical_device=selected, physical_device=uuid_to_physical[devices[selected]["uuid"]],
            uuid=devices[selected]["uuid"], ranges=len(rows),
            mean_read_GB_per_31_steps=read / len(rows) / 1e9,
            mean_write_GB_per_31_steps=write / len(rows) / 1e9,
            mean_profiled_range_seconds=ns / len(rows) / 1e9,
            profiled_read_plus_write_GBs= (read + write) / ns,
            mean_dram_pct=statistics.mean(r["dram__throughput.avg.pct_of_peak_sustained_elapsed"] for r in rows),
            mean_achieved_occupancy_pct=statistics.mean(occupancy),
            min_achieved_occupancy_pct=min(occupancy), max_achieved_occupancy_pct=max(occupancy),
            profiled_duration_vs_unprofiled_wall_ratio=ns / len(rows) / 1e9 / mean_wall,
        ))
    gpu = {}
    for sample in samples:
        if sample.get("returncode") != 0 or not any(t["start_monotonic"] <= sample["monotonic"] <= t["end_monotonic"] for t in timed):
            continue
        for line in csv.reader(io.StringIO(sample["csv"])):
            values = [v.strip() for v in line]
            index = int(values[0])
            gpu.setdefault(index, []).append(list(map(float, values[2:])))
    result["unprofiled_gpu_telemetry"] = [{
        "physical_index": index, "samples": len(rows),
        "mean_busy_pct": statistics.mean(r[1] for r in rows),
        "max_memory_MiB": max(r[0] for r in rows),
        "mean_power_W": statistics.mean(r[2] for r in rows),
        "mean_SM_clock_MHz": statistics.mean(r[3] for r in rows),
        "mean_memory_clock_MHz": statistics.mean(r[4] for r in rows),
        "temperature_min_C": min(r[5] for r in rows),
        "temperature_max_C": max(r[5] for r in rows),
    } for index, rows in sorted(gpu.items())]
    (root / "recomputed.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
