from types import SimpleNamespace

import pytest
import torch

from venturi.benchmark import benchmark_tokens
from venturi.generation import GenerationResult


def install_fake(monkeypatch, *, counts=(3,), device="cpu"):
    import venturi.benchmark

    calls = []

    def generate(model, prompt_ids, **options):
        count = counts[len(calls) % len(counts)]
        calls.append(options)
        return GenerationResult(
            tuple(prompt_ids), (2,) * count, "length", 0.2, 0.4, 0.8, 128,
            time_to_first_token_seconds=0.3,
        )

    monkeypatch.setattr(venturi.benchmark, "generate_tokens", generate)
    return SimpleNamespace(device=torch.device(device)), calls


def test_counts_weighted_rates_and_warmup(monkeypatch):
    model, calls = install_fake(monkeypatch, counts=(3, 1))
    result = benchmark_tokens(model, [4, 5], warmup=1, repeats=2, ignore_eos=True)
    assert len(calls) == 3
    assert all(call["eos_token_ids"] == () for call in calls)
    assert result["decode_tokens_per_second"] == 2 / 0.8
    assert result["request_tokens_per_second"] == 4 / 1.6
    assert result["time_to_first_token_seconds"] == {"min": 0.3, "median": 0.3, "max": 0.3}
    assert not result["identical_generated_ids"]
    assert result["runs"][0]["decode_tokens_per_second"] is None
    assert result["peak_cuda_allocated_bytes"] is None


def test_one_token_is_not_a_decode_step(monkeypatch):
    model, calls = install_fake(monkeypatch, counts=(1,))
    result = benchmark_tokens(model, [4], warmup=0, repeats=2)
    assert all("eos_token_ids" not in call for call in calls)
    assert result["decode_tokens_per_second"] is None
    assert result["identical_generated_ids"]
    assert result["eos_policy"] == "model_config"


@pytest.mark.parametrize("kwargs", [
    {"warmup": -1}, {"warmup": True}, {"repeats": 0}, {"repeats": 1.2},
    {"max_new_tokens": 0}, {"max_new_tokens": True}, {"ignore_eos": 1},
])
def test_bad_parameters_never_execute(monkeypatch, kwargs):
    model, calls = install_fake(monkeypatch)
    with pytest.raises(ValueError):
        benchmark_tokens(model, [4], **kwargs)
    assert calls == []


def test_cuda_peak_scope_and_reset(monkeypatch):
    import venturi.benchmark

    model, _ = install_fake(monkeypatch, device="cuda:0")
    events = []
    monkeypatch.setattr(venturi.benchmark, "_synchronize", lambda d: events.append("sync"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda d: events.append("reset"))
    for name, value in {
        "memory_allocated": 100, "memory_reserved": 200,
        "max_memory_allocated": 150, "max_memory_reserved": 250,
    }.items():
        monkeypatch.setattr(torch.cuda, name, lambda d, value=value: value)
    result = benchmark_tokens(model, [4], warmup=0, repeats=2)
    assert events == ["sync", "reset", "sync", "reset"]
    assert result["peak_cuda_allocated_bytes"] == 150
    assert result["runs"][0]["cuda_allocator"]["extra_peak_allocated_bytes"] == 50
