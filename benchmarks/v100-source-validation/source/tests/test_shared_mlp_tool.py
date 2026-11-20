"""Composite BF16 diagnostics must not be mistaken for per-Op qualification."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from benchmark_shared_mlp import qualify  # noqa: E402


def test_composite_metric_is_explicitly_diagnostic_and_retains_error():
    expected = torch.tensor([[1.0, 0.0]], dtype=torch.bfloat16)
    actual = torch.tensor([[1.0, 0.125]], dtype=torch.bfloat16)
    result = qualify(actual, expected)
    assert result["diagnostic_budget_met"] is False
    assert result["max_abs_error"] == 0.125
    assert result["relative_rms_error"] > 0
    assert "passed" not in result
    assert "diagnostic" in result["scope"]
