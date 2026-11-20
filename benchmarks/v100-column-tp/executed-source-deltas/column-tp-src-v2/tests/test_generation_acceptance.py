import pytest
import torch

from tools.run_generation_acceptance import compare


def test_large_fp16_logit_rounding_is_relative():
    result = compare(
        torch.tensor([1000.5], dtype=torch.float16), torch.tensor([1000.25])
    )
    assert result["max_abs_error"] == 0.25
    assert result["max_error_over_bound"] < 1


@pytest.mark.parametrize(
    "actual,expected", [([float("nan")], [1.0]), ([1.0], [float("inf")])]
)
def test_nonfinite_logits_never_pass(actual, expected):
    with pytest.raises(RuntimeError, match="Nonfinite"):
        compare(torch.tensor(actual), torch.tensor(expected))


def test_wrong_fp32_logits_fail():
    with pytest.raises(RuntimeError, match="numerical tolerance"):
        compare(torch.tensor([0.1, 0.2]), torch.tensor([0.1, 0.5]))


def test_diagnostic_drift_is_reported_not_marked_close():
    result = compare(
        torch.tensor([0.1, 0.2]), torch.tensor([0.1, 0.5]), require_close=False
    )
    assert not result["within_tolerance"]
    assert result["max_error_over_bound"] > 1
