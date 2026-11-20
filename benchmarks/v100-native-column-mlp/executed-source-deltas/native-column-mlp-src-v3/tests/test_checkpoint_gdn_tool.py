"""Qualification tooling preserves failures and reports primitive, not model TPS."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch


def load_tool():
    spec = spec_from_file_location(
        "checkpoint_gdn_tool",
        Path(__file__).resolve().parents[1] / "tools/qualify_checkpoint_gdn.py",
    )
    module = module_from_spec(spec)
    import sys

    tools = str(Path(spec.origin).parent)
    sys.path.insert(0, tools)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(tools)
    return module


def test_bf16_oracle_gate_preserves_counterexample():
    check = load_tool().bf16_check
    ideal = torch.tensor([[1e-7, -0.2, 4.0]], dtype=torch.float64)
    actual = ideal.bfloat16()
    assert check(actual, ideal)["passed"]
    actual[0, 0] = 0.1
    failure = check(actual, ideal)
    assert not failure["passed"] and failure["failed_elements"] == 1
    assert failure["first_failure_indices"] == [[0, 0]]


def test_comparison_separates_cpu_wall_and_event_times():
    module = load_tool()
    phases = [
        dict(
            condition=name,
            wall_us_per_call=10.0 + i,
            event_us_per_call=2.0 + i,
            process_cpu_percent=90.0 + i,
            calls_per_second=1000.0 + i,
        )
        for i, name in enumerate(
            ("torch-reference", "native", "native", "torch-reference")
        )
    ]
    summary = module.comparison(phases)
    assert summary["native"]["wall_us_per_call"] == 11.5
    assert summary["native"]["event_us_per_call"] == 3.5
    assert summary["native"]["process_cpu_percent"] == 91.5
