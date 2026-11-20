"""Benchmark storage accounting counts physical tensors, not logical references."""

from pathlib import Path
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from benchmark_residency_attention import check, storage_report  # noqa: E402


def test_storage_accounting_deduplicates_shared_planes():
    a, b = torch.empty(12), torch.empty(7, dtype=torch.float16)
    shared = [SimpleNamespace(storage_tensors=(a, b)) for _ in range(3)]
    result = storage_report(shared)
    assert result["unique_weight_bytes"] == 62
    assert result["logical_plan_weight_bytes"] == 186
    assert result["unique_weight_allocations"] == 2
    assert result["all_shape_pointers_identical"]
    separate = [
        SimpleNamespace(storage_tensors=(a.clone(), b.clone())) for _ in range(3)
    ]
    result = storage_report(separate)
    assert result["unique_weight_bytes"] == result["logical_plan_weight_bytes"] == 186
    assert result["unique_weight_allocations"] == 6
    assert not result["all_shape_pointers_identical"]


def test_operator_qualification_never_hides_nonfinite_results():
    assert check(torch.ones(2), torch.ones(2))["passed"]
    invalid = check(torch.tensor([float("nan")]), torch.zeros(1))
    assert not invalid["passed"]
    assert invalid["failed_elements"] == 1
    assert not check(torch.tensor([float("inf")]), torch.zeros(1))["passed"]
