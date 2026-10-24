"""Remote-only public-Op trace; timing is separate from unprofiled measurements."""

import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess

ROOT = Path("/home/test/wangyue/venturi-work/qwen")
OUT = ROOT / "prefill-full"


def main():
    commands = []
    resources = {}
    for name in ("prefill_baseline", "prefill_r128_w32", "prefill_r128_w64"):
        trace = OUT / ("public-" + name)
        command = [
            "nsys",
            "profile",
            "--trace=cuda",
            "--cuda-graph-trace=node",
            "--sample=none",
            "--cpuctxsw=none",
            "--force-overwrite=false",
            "-o",
            str(trace),
            str(OUT / name / "bench"),
            "512",
        ]
        commands.append(command)
        with trace.with_suffix(".stdout").open("w") as stdout:
            with trace.with_suffix(".stderr").open("w") as stderr:
                subprocess.run(command, stdout=stdout, stderr=stderr, check=True)
        command = [
            "nsys",
            "export",
            "--type",
            "sqlite",
            "--output",
            str(trace.with_suffix(".sqlite")),
            str(trace.with_suffix(".nsys-rep")),
        ]
        commands.append(command)
        with trace.with_suffix(".export.log").open("w") as sink:
            subprocess.run(command, stdout=sink, stderr=subprocess.STDOUT, check=True)
        with sqlite3.connect(trace.with_suffix(".sqlite")) as db:
            rows = db.execute("""SELECT blockX*blockY*blockZ, registersPerThread,
                staticSharedMemory+dynamicSharedMemory, gridX*gridY*gridZ,
                COUNT(*), AVG(end-start)/1000.0
                FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id
                WHERE s.value LIKE '%q4_rowsplit_gemm_mma_kernel%'
                GROUP BY 1,2,3,4""").fetchall()
        resources[name] = []
        for threads, registers, shared, grid, count, duration in rows:
            blocks = min(
                24, 1536 // threads, 65536 // (registers * threads), 102400 // shared
            )
            resources[name].append(
                dict(
                    threads=threads,
                    registers_per_thread=registers,
                    shared_bytes_per_block=shared,
                    grid_blocks=grid,
                    calls=count,
                    resource_only_blocks_per_sm_upper_bound=blocks,
                    resource_only_thread_occupancy_upper_bound=blocks * threads / 1536,
                    instrumented_mean_kernel_us=duration,
                )
            )
        if not rows:
            raise AssertionError("Missing target kernel in public-Op trace")
    baseline = json.loads((ROOT / "native/venturi-qwen-runner.build.json").read_text())
    archives = {
        p: hashlib.sha256(Path(p).read_bytes()).hexdigest() == expected
        for p, expected in baseline["linked_archive_sha256"].items()
    }
    policy = [
        s
        for s in Path("/proc/driver/nvidia/params").read_text().splitlines()
        if s.startswith("RmProfilingAdminOnly:")
    ]
    result = dict(
        public_trace_resources=resources,
        commands=commands,
        archive_matches_production_build=archives,
        production_runner_sha256=hashlib.sha256(
            (ROOT / "native/venturi-qwen-runner").read_bytes()
        ).hexdigest(),
        current_counter_policy=policy,
        limitations=[
            "Resource bounds are not achieved occupancy.",
            "Hardware counter access remains restricted; no new counter probe or privilege change.",
            "Instrumented public-Op timing is not model latency or unprofiled throughput.",
        ],
    )
    (OUT / "diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
