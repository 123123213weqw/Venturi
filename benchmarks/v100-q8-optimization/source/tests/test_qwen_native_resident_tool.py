"""Native timing distinguishes a first token from continuation throughput."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


def tool():
    path = (
        Path(__file__).resolve().parents[1] / "tools/benchmark_qwen_native_resident.py"
    )
    spec = spec_from_file_location("resident_timing_tool", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_timing_aggregate_scopes():
    rows = [
        dict(
            ttft=dict(wall_seconds=2.0, process_cpu_seconds=1.0),
            decode=dict(wall_seconds=4.0, process_cpu_seconds=2.0, output_tokens=8),
            unsplit=dict(wall_seconds=6.0, process_cpu_seconds=3.0, output_tokens=9),
        ),
        dict(
            ttft=dict(wall_seconds=4.0, process_cpu_seconds=2.0),
            decode=dict(wall_seconds=8.0, process_cpu_seconds=4.0, output_tokens=8),
            unsplit=dict(wall_seconds=12.0, process_cpu_seconds=6.0, output_tokens=9),
        ),
    ]
    result = tool().aggregate(rows)
    assert result["decode"]["output_tokens_per_wall_second"] == 8 / 6
    assert result["unsplit"]["output_tokens_per_wall_second"] == 9 / 9
    assert "output_tokens_per_wall_second" not in result["ttft"]
    rows[1]["decode"]["output_tokens"] = 7
    with pytest.raises(ValueError, match="unequal"):
        tool().aggregate(rows)


def test_native_timing_measured_result():
    value, timing = tool().measured(lambda: [1, 2])
    assert value == [1, 2]
    assert timing["wall_seconds"] >= 0
    assert timing["process_cpu_seconds"] >= 0
    assert timing["end_unix"] >= timing["start_unix"]
