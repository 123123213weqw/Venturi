"""Full-model diagnostic gates must reject disagreements and nonfinite values."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import pytest
import torch

from venturi.ops import TorchBackend, prepare_greedy


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


def test_planned_selection_status_checked_before_token_use():
    tool = load()
    # Explicit reference backend only for testing the control-flow contract.
    plan = prepare_greedy(TorchBackend(), vocab_size=4, device="cpu")
    out = torch.empty(1, 2, dtype=torch.int64)
    scratch = torch.empty(plan.workspace_bytes, dtype=torch.uint8)
    assert (
        tool.planned_greedy(torch.tensor([1.0, 3.0, 3.0, -5.0]), plan, out, scratch)
        == 1
    )
    with pytest.raises(ValueError, match="nonfinite"):
        tool.planned_greedy(
            torch.tensor([float("nan"), 3.0, 2.0, 1.0]), plan, out, scratch
        )
