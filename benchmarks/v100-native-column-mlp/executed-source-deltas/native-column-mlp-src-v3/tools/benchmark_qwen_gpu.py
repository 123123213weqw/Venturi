"""Unprofiled production-API GPU telemetry and CPU/wall accounting.

NVML GPU busy and memory-controller busy are time-based samples, not achieved
SM occupancy, FLOPS utilization or bandwidth percentage. Samples are whole-device
and coarse; exclude the first second of each wave to reduce boundary averaging.
"""

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
from time import time

from benchmark_qwen_cpu import measure
from run_qwen_validation import idle_gpu
from venturi.backends.qwen import QwenNativeSession


FIELDS = [
    "timestamp",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "power.draw",
    "temperature.gpu",
    "clocks.sm",
    "clocks.mem",
]


def telemetry(path, waves):
    rows = []
    for row in csv.reader(path.read_text().splitlines()):
        if len(row) != len(FIELDS):
            continue
        try:
            stamp = datetime.strptime(
                row[0].strip(), "%Y/%m/%d %H:%M:%S.%f"
            ).timestamp()
        except ValueError:
            continue
        values = {
            name: float(value) if value.strip() not in ("N/A", "[N/A]") else None
            for name, value in zip(FIELDS[1:], row[1:])
        }
        rows.append(dict(timestamp=stamp, **values))
    summaries = {}
    for mode in ("single", "parallel"):
        chosen = [
            row
            for row in rows
            if any(
                w["mode"] == mode
                and w["start_unix"] + 1 <= row["timestamp"] <= w["end_unix"]
                for w in waves
            )
        ]
        summaries[mode] = dict(samples=len(chosen))
        for name in FIELDS[1:]:
            values = [r[name] for r in chosen if r[name] is not None]
            summaries[mode][name] = (
                dict(
                    mean=statistics.mean(values),
                    minimum=min(values),
                    maximum=max(values),
                )
                if values
                else None
            )
    return summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("--runner", required=True)
    p.add_argument("--cpu-wait", choices=("auto", "blocking"), default="blocking")
    p.add_argument("--output-directory", required=True)
    args = p.parse_args()
    out = Path(args.output_directory)
    out.mkdir(parents=True, exist_ok=False)
    background = idle_gpu()
    prompts = [
        f"Document {i}. "
        + "The quick brown fox jumps over the lazy dog. " * 50
        + "\nContinue the explanation:"
        for i in range(8)
    ]
    generation = dict(raw_prompt=True, max_new_tokens=64, ignore_eos=True)
    waves = []
    with QwenNativeSession(
        args.checkpoint,
        runner=args.runner,
        max_context=2048,
        max_concurrency=4,
        prefill_chunk=512,
        cpu_wait=args.cpu_wait,
        timeout=600,
    ) as model:
        ready = model.ready
        for mode in ("single", "parallel"):
            model.generate_many(prompts, mode=mode, **generation)
        with (out / "telemetry.csv").open("w") as sink:
            sampler = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-gpu=" + ",".join(FIELDS),
                    "--format=csv,noheader,nounits",
                    "-lms",
                    "200",
                ],
                stdout=sink,
                stderr=subprocess.DEVNULL,
            )
            try:
                for mode in ("single", "parallel", "parallel", "single"):
                    start = time()
                    measured = measure(
                        model,
                        lambda: model.generate_many(prompts, mode=mode, **generation),
                    )
                    wave = dict(
                        mode=mode, start_unix=start, end_unix=time(), **measured
                    )
                    waves.append(wave)
                    print(
                        mode, measured["result"]["output_tokens_per_second"], flush=True
                    )
            finally:
                sampler.terminate()
                try:
                    sampler.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    sampler.kill()
                    sampler.wait()
    if sampler.returncode not in (0, -15):
        raise RuntimeError(f"GPU telemetry sampler failed: {sampler.returncode}")
    summaries = {}
    for mode in ("single", "parallel"):
        chosen = [w for w in waves if w["mode"] == mode]
        seconds = sum(w["wall_seconds"] for w in chosen)
        tokens = sum(
            len(r["generated_ids"]) for w in chosen for r in w["result"]["results"]
        )
        summaries[mode] = dict(
            output_tokens=tokens,
            client_wall_seconds=seconds,
            output_tokens_per_second=tokens / seconds,
            native_cpu_percent=100
            * sum(w["native_cpu_seconds"] for w in chosen)
            / seconds,
            mean_decode_batch=sum(
                w["result"]["runtime"]["decode_row_rounds"] for w in chosen
            )
            / sum(w["result"]["runtime"]["decode_rounds"] for w in chosen),
        )
    report = dict(
        method=__doc__,
        ready=ready,
        background=background,
        prompts=prompts,
        generation=generation,
        waves=waves,
        summaries=summaries,
        telemetry=telemetry(out / "telemetry.csv", waves),
        process_exit=model._process.returncode,
        runner_sha256=hashlib.sha256(Path(args.runner).read_bytes()).hexdigest(),
    )
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(dict(summaries=summaries, telemetry=report["telemetry"]), indent=2)
    )


if __name__ == "__main__":
    main()
