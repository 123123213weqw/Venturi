"""Explicit process CPU/wall accounting and aggregation of recorded measurements.

No inference engine, process launcher, implicit GPU query or model dependency.
Summary consumes the existing measurement-record schema; it does not establish
that a caller's result came from a particular implementation or model.
"""

import os
from pathlib import Path
import resource
import statistics
from time import perf_counter


def process_cpu(pid):
    # comm may contain spaces or ')' characters; the final ')' closes field 2.
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


def python_cpu():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def measure_process(pid, call):
    before = process_cpu(pid)
    py_before = python_cpu()
    started = perf_counter()
    result = call()
    seconds = perf_counter() - started
    cpu = process_cpu(pid) - before
    py_cpu = python_cpu() - py_before
    return dict(
        wall_seconds=seconds,
        native_cpu_seconds=cpu,
        python_cpu_seconds=py_cpu,
        native_cpu_percent=100 * cpu / seconds,
        result=result,
    )


def summary(runs):
    seconds = sum(r["wall_seconds"] for r in runs)
    cpu = sum(r["native_cpu_seconds"] for r in runs)
    py_cpu = sum(r["python_cpu_seconds"] for r in runs)
    tokens = sum(
        len(x["generated_ids"])
        for r in runs
        for x in (r["result"].get("results", [r["result"]]))
    )
    return dict(
        wall_seconds=seconds,
        native_cpu_seconds=cpu,
        python_cpu_seconds=py_cpu,
        native_cpu_percent=100 * cpu / seconds,
        total_cpu_seconds_per_output_token=(cpu + py_cpu) / tokens,
        output_tokens=tokens,
        output_tokens_per_second=tokens / seconds,
        median_first_token_seconds=statistics.median(
            x["timings"]["first_token_seconds"]
            for r in runs
            for x in r["result"].get("results", [r["result"]])
        ),
    )
