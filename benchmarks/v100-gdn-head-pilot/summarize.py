"""Recompute corrected native GDN operator metrics, separate from model TPS."""

import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import statistics

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RAW = OUT / "v100"


def read(path):
    return json.loads(path.read_text())


def telemetry(path, start, end):
    rows = []
    for line in csv.reader(path.open()):
        if len(line) != 9:
            raise ValueError("Unexpected NVML telemetry row: " + str(line))
        timestamp = (
            datetime.strptime(line[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
            .replace(tzinfo=timezone(timedelta(hours=8)))
            .timestamp()
        )
        if not start <= timestamp <= end:
            continue
        values = [
            None if v.strip() in ("N/A", "[N/A]") else float(v.strip())
            for v in line[2:]
        ]
        rows.append((int(line[1]), values))
    fields = (
        "sampled_gpu_busy_percent",
        "sampled_memory_controller_busy_percent",
        "used_mib_both_operator_arms_resident",
        "power_watts",
        "temperature_celsius",
        "sm_clock_mhz",
        "memory_clock_mhz",
    )
    result = {}
    for device in (0, 1):
        selected = [v for d, v in rows if d == device]
        assert selected, ("Missing timed NVML samples", start, end, device)
        result[str(device)] = dict(
            samples=len(selected),
            **{
                field: dict(min=min(v), mean=statistics.mean(v), max=max(v))
                if (v := [r[i] for r in selected if r[i] is not None])
                else None
                for i, field in enumerate(fields)
            },
        )
    return result


def main():
    report = read(RAW / "trained-v8/measurement/report.json")
    assert (
        report["completed"]
        and report["integration_gate"]["passed"]
        and len(report["cells"]) == 6
    )
    profile = read(RAW / "profile-v9/attribution.json")
    profmeta = read(RAW / "profile-v9/measurement/report.json")
    cells = []
    for cell in report["cells"]:
        trials = cell["trials"]
        assert len(trials) == 6
        med = {
            mode: statistics.median(
                t["native_wall_seconds"] / t["iterations"]
                for t in trials
                if t["mode"] == mode
            )
            for mode in ("row", "head")
        }
        assert med == cell["median_block_seconds"]
        speedup = med["row"] / med["head"]
        assert speedup == cell["speedup"]
        cpu = {
            mode: statistics.median(
                t["process_cpu_core_seconds"] / t["iterations"]
                for t in trials
                if t["mode"] == mode
            )
            for mode in ("row", "head")
        }
        work = cell["total_rows"]
        assert work == cell["sequences"] * cell["tokens_per_sequence"]
        assert all(t["iterations"] == 2000 for t in trials)
        transport = {}
        for mode in ("row", "head"):
            times = [t for t in trials if t["mode"] == mode]
            transport[mode] = {
                k: times[0][k] / 2000
                for k in (
                    "h2d_bytes",
                    "d2h_bytes",
                    "h2d_calls",
                    "d2h_calls",
                    "d2d_calls",
                )
            }
            assert all(
                {k: t[k] / 2000 for k in transport[mode]} == transport[mode]
                for t in times
            )
        cells.append(
            dict(
                layer=cell["layer"],
                sequences=cell["sequences"],
                steps=cell["tokens_per_sequence"],
                blocks_per_arm_per_trial=2000,
                trials_per_arm=3,
                median_block_microseconds={k: v * 1e6 for k, v in med.items()},
                median_process_cpu_microseconds_per_block={
                    k: v * 1e6 for k, v in cpu.items()
                },
                speedup=speedup,
                latency_reduction_fraction=1 - 1 / speedup,
                paired_trial_speedups=[
                    next(
                        t["native_wall_seconds"]
                        for t in trials
                        if t["trial"] == i and t["mode"] == "row"
                    )
                    / next(
                        t["native_wall_seconds"]
                        for t in trials
                        if t["trial"] == i and t["mode"] == "head"
                    )
                    for i in (1, 2, 3)
                ],
                projected_output_relative_l2_vs_row=cell["output_relative_l2_vs_row"],
                source_reference_max_relative_l2=max(
                    c["relative_l2"] for c in cell["independent_reference"]
                ),
                source_reference_min_cosine=min(
                    c["cosine"] for c in cell["independent_reference"]
                ),
                transport_per_block=transport,
                telemetry_by_trial=[
                    dict(
                        mode=t["mode"],
                        trial=t["trial"],
                        devices=telemetry(
                            RAW / "trained-v8/measurement/gpu-samples.csv",
                            t["start_unix"],
                            t["end_unix"],
                        ),
                    )
                    for t in trials
                ],
            )
        )
    trace = []
    for op in profile["operators"]:
        gates = sum(
            k["count"] for k in op["kernel_by_name"] if "gdn_gates<" in k["name"]
        )
        assert gates == (64 if op["name"].endswith("_head") else 32)
        name = op["name"]
        meta = next(
            t
            for c in profmeta["cells"]
            for t in c["trials"]
            if name
            == f"gdn_{c['layer']}_{c['sequences']}x{c['tokens_per_sequence']}_{t['mode']}"
        )
        for direction, prefix in [("host_to_device", "h2d"), ("device_to_host", "d2h")]:
            assert (
                op["transfers"][direction]["bytes"] == meta[prefix + "_bytes"]
                and op["transfers"][direction]["count"] == meta[prefix + "_calls"]
            )
        trace.append(
            dict(
                name=name,
                measured_blocks=32,
                host_span_ms=op["host_span_nanoseconds"] / 1e6,
                kernels=op["kernels"],
                summed_kernel_ms_not_critical_path=op["kernel_nanoseconds"] / 1e6,
                process_activity_gap_ms_not_device_idle=op[
                    "gaps_between_this_process_activities_nanoseconds"
                ]
                / 1e6,
                per_device_kernel_union_ms={
                    k: v / 1e6
                    for k, v in op["per_device_kernel_union_nanoseconds"].items()
                },
                simultaneous_two_device_kernel_ms=op[
                    "simultaneous_two_device_kernel_nanoseconds"
                ]
                / 1e6,
                transfers=op["transfers"],
                gates_launches=gates,
                kernel_attribution=op["kernel_by_name"],
            )
        )
    resource = (
        ROOT / "benchmarks/v100-mtp-draft-residual/v100/profile-v4/resources.json"
    )
    r = read(resource)
    assert (
        r["library_sha256"]
        == report["blocks_sha256"]
        == "6d43160ce4e993b4be5ed600d9be21aad070634729d8e5f7b3f74b734da85de2"
    )
    result = dict(
        scope=__doc__,
        corrected_schedule=True,
        model_integrated=False,
        new_model_tps=False,
        integration_gate=report["integration_gate"],
        cells=cells,
        separate_profile=trace,
        achieved_hardware_counters=None,
        hardware_counter_policy="RmProfilingAdminOnly: 1, unchanged; no repeated counter campaign",
        reused_exact_cuda_library_resources=dict(
            path=str(resource.relative_to(ROOT)),
            sha256=hashlib.sha256(resource.read_bytes()).hexdigest(),
            scope="Compiled/static resources only, not achieved occupancy; actual grids/registers/shared sizes in current attribution.json",
        ),
        initial_positive_pilot_superseded="trained-v5 used per-sequence gates in both arms; corrected trained-v8 matches production batched gates; both preserved",
        raw_report_signature_label_note="Raw final_state_signature is the state after the initial one-block correctness run, not the timed 2000-block final state. Equal timed-final canonical state/history bytes are asserted separately by validate() after the last trial.",
        correctness_scope="9 focused two-GPU tests, trained layer0/62 blocks with 20 independent sequence references and exact row/head pre-output states; not a new 98-row model teacher or end-to-end MTP qualification",
    )
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    for c in cells:
        print(
            c["layer"],
            c["sequences"],
            c["steps"],
            c["median_block_microseconds"],
            round(c["speedup"], 5),
            "CPU",
            c["median_process_cpu_microseconds_per_block"],
        )
    for t in trace:
        print(
            t["name"],
            "kernels",
            t["kernels"],
            "kernelms",
            round(t["summed_kernel_ms_not_critical_path"], 3),
            "gapms",
            round(t["process_activity_gap_ms_not_device_idle"], 3),
            "H2D",
            t["transfers"]["host_to_device"],
        )


if __name__ == "__main__":
    main()
