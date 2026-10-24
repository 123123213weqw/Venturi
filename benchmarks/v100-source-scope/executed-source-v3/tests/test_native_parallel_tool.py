"""Concurrent throughput uses global wall time, not summed request latency."""

from concurrent.futures import ThreadPoolExecutor
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import time

import pytest


def tool():
    path = (
        Path(__file__).resolve().parents[1] / "tools/benchmark_qwen_native_parallel.py"
    )
    spec = spec_from_file_location("native_parallel_tool", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_parallel_global_wall_and_separate_thread_cpu():
    module = tool()

    def call():
        time.sleep(0.02)
        return [2, 3]

    with ThreadPoolExecutor(max_workers=2) as pool:
        r = module.run_wave(pool, [call, call], 2)
    assert r["emitted_tokens"] == 4
    assert r["aggregate_output_tokens_per_wall_second"] == 4 / r["wall_seconds"]
    assert r["process_cpu_seconds"] >= 0
    assert len(r["requests"]) == 2
    assert all(
        x["ids"] == [2, 3] and x["thread_cpu_seconds"] >= 0 for x in r["requests"]
    )
    assert all(
        0.02 <= x["completion_latency_seconds"] <= r["wall_seconds"]
        for x in r["requests"]
    )


def test_native_parallel_rejects_unequal_output_work():
    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(AssertionError, match="unequal"):
            tool().run_wave(pool, [lambda: [1]], 2)
