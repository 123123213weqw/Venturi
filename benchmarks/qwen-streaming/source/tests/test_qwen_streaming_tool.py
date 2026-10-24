"""Full-model diagnostic gates must reject disagreements and nonfinite values."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import pytest
import torch


def load():
    spec = spec_from_file_location(
        "qwen_stream_tool",
        Path(__file__).resolve().parents[1] / "tools/qualify_qwen_streaming.py",
    )
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_logit_gate_rejects_counterexamples():
    tool = load()
    reference = torch.tensor([1.0, 2.0, 4.0, -1.0])
    assert tool.logits_comparison(reference, reference)["passed"]
    assert tool.logits_comparison(torch.zeros(4), torch.zeros(4))["passed"]
    assert tool.metrics(torch.empty(0, 3), torch.empty(0, 3))["max_abs_error"] == 0
    changed = reference.clone()
    changed[0] = 9
    assert not tool.logits_comparison(changed, reference)["passed"]
    with pytest.raises(ValueError, match="finite"):
        tool.logits_comparison(torch.tensor([float("nan")]), torch.ones(1))
    with pytest.raises(ValueError, match="nonfinite"):
        tool.cpu_greedy(torch.tensor([float("inf")]))
    assert tool.cpu_greedy(torch.tensor([2.0, 2.0, 1.0])) == 0
