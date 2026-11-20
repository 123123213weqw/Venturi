"""Recompute syscall counts and a CPU opportunity estimate, never predicted TPS."""

from datetime import datetime, timedelta, timezone
import csv
import hashlib
import json
from pathlib import Path
import statistics

OUT = Path(__file__).resolve().parent
RAW = OUT / "v100/probe-v1"
ROOT = OUT.parents[1]


def read(path):
    return json.loads(path.read_text())


def telemetry(calls):
    rows = {0: [], 1: []}
    with (RAW / "gpu-telemetry.csv").open() as f:
        for r in csv.DictReader(f, skipinitialspace=True):
            epoch = (
                datetime.strptime(r["timestamp"], "%Y/%m/%d %H:%M:%S.%f")
                .replace(tzinfo=timezone(timedelta(hours=8)))
                .timestamp()
            )
            if any(c["start_unix"] <= epoch <= c["end_unix"] for c in calls):
                rows[int(r["index"])].append(r)
    result = {}
    for device, samples in rows.items():
        assert samples, "No samples inside marked intervals"
        columns = {}
        for key in samples[0]:
            if key in ("timestamp", "index"):
                continue
            values = [float(r[key].split()[0]) for r in samples]
            columns[key] = dict(
                mean=statistics.mean(values), min=min(values), max=max(values)
            )
        result[str(device)] = dict(samples=len(samples), columns=columns)
    return result


def main():
    untraced = read(RAW / "untraced.json")
    traced = read(RAW / "traced.json")
    counts = read(RAW / "syscall-report.json")
    estimates = {}
    total = 0
    for name in ("target", "draft"):
        micro = read(RAW / (name + "-microbench.json"))
        pair = (
            statistics.median(micro["seconds"]) / micro["verification_pairs_per_repeat"]
        )
        path = next(
            p
            for p in counts["checkpoint_calls_by_path"]
            if ("mtp-q8" in p) == (name == "draft")
        )
        c = counts["checkpoint_calls_by_path"][path]
        assert c["descriptor"] == c["pathname"]
        value = pair * c["descriptor"]
        total += value
        estimates[name] = dict(
            path=path,
            verification_pairs=c["descriptor"],
            microbench_pair_seconds=pair,
            cpu_opportunity_estimate_seconds=value,
            microbench_repeat_seconds=micro["seconds"],
        )
    calls = untraced["calls"]
    wall = statistics.mean(c["native_call_wall_seconds"] for c in calls)
    references = {}
    for rel in (
        "benchmarks/v100-mtp-draft-residual/metrics.json",
        "benchmarks/v100-mtp-draft-residual/v100/profile-v4/resources.json",
        "benchmarks/v100-mtp-draft-residual/v100/profile-v4/residual-draft-mtp2-c1-attribution.json",
        "benchmarks/v100-mtp-draft-residual/v100/profile-v4/residual-draft-mtp2-c1-metadata.json",
    ):
        references[rel] = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
    result = dict(
        scope="Unchanged inference: raw native C1 MTP2, not complete HTTP cohort throughput or a new optimization comparison",
        library_sha256=untraced["library_sha256"],
        source_guards_unchanged=True,
        raw_native_call_repeats=len(calls),
        mean_raw_native_wall_seconds=wall,
        trial_raw_native_wall_seconds=[c["native_call_wall_seconds"] for c in calls],
        mean_process_cpu_seconds=statistics.mean(
            c["process_cpu_seconds"] for c in calls
        ),
        mean_calling_thread_cpu_seconds=statistics.mean(
            c["thread_cpu_seconds"] for c in calls
        ),
        acceptance=[
            dict(
                accepted=c["stats"]["accepted_tokens"],
                proposed=c["stats"]["proposed_tokens"],
            )
            for c in calls
        ],
        total_checkpoint_metadata_syscalls=sum(counts["checkpoint_calls"].values()),
        verification_pairs=counts["checkpoint_calls"]["descriptor"],
        cpu_microbench_opportunities=estimates,
        summed_cpu_opportunity_estimate_seconds=total,
        opportunity_estimate_over_raw_call_wall=total / wall,
        opportunity_is_not_critical_path_or_tps_prediction=True,
        instrumented_raw_call_wall_seconds=traced["calls"][0][
            "native_call_wall_seconds"
        ],
        instrumented_metadata_syscall_seconds=sum(
            counts["instrumented_syscall_seconds"].values()
        ),
        untraced_telemetry=telemetry(calls),
        instrumented_telemetry=telemetry(traced["calls"]),
        telemetry_utc_offset_seconds=28800,
        new_cuda_profile_or_hardware_counters=False,
        achieved_occupancy=None,
        unchanged_library_prior_timeline_and_goal_references=references,
        eligibility_gate=dict(
            greater_than_one_percent=total > wall * 0.01,
            greater_than_five_ms=total > 0.005,
            eligible_for_followup_test=total > wall * 0.01 and total > 0.005,
            implementation_or_adoption_gate_passed=False,
        ),
        next_action="Bounded resident-only lexical per-forward source-check deduplication prototype: preserve forced start/end checks and streaming guards; deterministic mid-call mutation/failure/reset/concurrency tests before any C1/C8 speed comparison. Not implemented here.",
        external_speed_goal_complete=False,
        inference_optimization_implemented_in_this_capsule=False,
    )
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "verification_pairs",
                    "summed_cpu_opportunity_estimate_seconds",
                    "opportunity_estimate_over_raw_call_wall",
                    "eligibility_gate",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
