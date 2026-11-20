"""Process CPU accounting must not be confused with internal host-work counters."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from benchmark_qwen_cpu import process_cpu, summary  # noqa: E402


def test_proc_stat_handles_spaces_and_parentheses_in_comm(monkeypatch):
    # Field 14 is utime; field 15 is stime. Count all process threads, not wall time.
    fields = ["S"] + ["0"] * 10 + ["250", "50"] + ["0"] * 10
    monkeypatch.setattr(
        Path, "read_text", lambda self: "123 (worker ) name) " + " ".join(fields)
    )
    monkeypatch.setattr("benchmark_qwen_cpu.os.sysconf", lambda key: 100)
    assert process_cpu(123) == 3.0


def test_cpu_percent_can_exceed_one_core_and_parallel_wall_is_not_summed():
    item = dict(generated_ids=[1, 2], timings=dict(first_token_seconds=0.2))
    runs = [
        dict(
            wall_seconds=2.0,
            native_cpu_seconds=3.0,
            python_cpu_seconds=0.1,
            result=dict(results=[item] * 4),
        )
    ]
    r = summary(runs)
    assert r["native_cpu_percent"] == 150
    assert r["output_tokens"] == 8
    assert r["output_tokens_per_second"] == 4
    assert r["total_cpu_seconds_per_output_token"] == pytest.approx(3.1 / 8)
    assert r["median_first_token_seconds"] == 0.2
