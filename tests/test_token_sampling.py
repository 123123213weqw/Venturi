"""Universal token sampling correctness: validation, determinism, greedy
collapse, and bit-level agreement between the torch implementation and the
FP64 scalar reference."""

import math

import pytest
import torch

from venturi.engine.sampling import (
    SamplingParams,
    reference_sample_token,
    sample_token,
)


def test_params_validation():
    for bad in (
        dict(temperature=0),
        dict(temperature=-1.0),
        dict(temperature=float("nan")),
        dict(temperature=True),
        dict(top_k=-1),
        dict(top_k=1.5),
        dict(top_p=0),
        dict(top_p=1.5),
        dict(top_p=float("inf")),
        dict(seed=-1),
        dict(seed=1.2),
    ):
        with pytest.raises(ValueError):
            SamplingParams(**bad)
    assert SamplingParams(seed=0).seed == 0  # zero seed is explicit and valid
    with pytest.raises(TypeError):
        sample_token(torch.zeros(4), params="loose")


def test_same_seed_is_deterministic_and_seeds_differ():
    logits = torch.tensor([1.0, 4.0, 2.0, 0.5, 3.0])
    params = SamplingParams(temperature=1.0, seed=1234)
    first = [sample_token(logits, params) for _ in range(32)]
    second = [sample_token(logits, params) for _ in range(32)]
    assert first == second
    other = [
        sample_token(logits, SamplingParams(temperature=1.0, seed=99)) for _ in range(32)
    ]
    assert other != first or len(set(first)) == 1  # degenerate support may collide


def test_top_k_one_and_cold_temperature_collapse_to_argmax():
    logits = torch.tensor([0.1, 3.0, 2.9, -1.0, 0.0])
    for params in (
        SamplingParams(temperature=1e9, top_k=1, seed=7),
        SamplingParams(temperature=1e-9, seed=7),
    ):
        assert all(sample_token(logits, params) == 1 for _ in range(16))


def test_reference_agrees_bit_for_bit():
    torch.manual_seed(11)
    logits = torch.randn(512)
    for params in (
        SamplingParams(temperature=1.0),
        SamplingParams(temperature=0.37),
        SamplingParams(temperature=2.5, top_k=32),
        SamplingParams(temperature=0.8, top_p=0.9),
        SamplingParams(temperature=1.4, top_k=16, top_p=0.6),
    ):
        for draw in (0.0, 1e-9, 0.25, 0.5, 0.749999, 0.999999):
            assert sample_token(logits, params, uniform=draw) == reference_sample_token(
                logits, params, draw
            ), (params, draw)


def test_distribution_shape():
    torch.manual_seed(3)
    logits = torch.randn(64) * 3
    params = SamplingParams(temperature=0.8, top_k=10, top_p=0.85, seed=5)
    generator = torch.Generator().manual_seed(5)
    counts = {}
    for _ in range(4000):
        token = sample_token(logits, params, generator)
        counts[token] = counts.get(token, 0) + 1
    assert set(counts) <= set(range(64))
    assert sum(counts.values()) == 4000
    assert len(counts) <= 10  # top-k bound respected
    head = int(torch.topk(logits, 3).indices[0])
    tail = int(torch.topk(logits, 3).indices[-1])
    assert counts[head] >= counts[tail]  # mass concentrates on the head


def test_top_p_boundary_semantics():
    logits = torch.tensor([math.log(0.7), math.log(0.3)])
    for draw in (0.0, 0.5, 0.999):
        assert reference_sample_token(logits, SamplingParams(top_p=0.7), draw) == 0
        assert sample_token(logits, SamplingParams(top_p=0.7), uniform=draw) == 0
    # top_p=1.0 keeps both; the second token is reachable.
    assert reference_sample_token(logits, SamplingParams(top_p=1.0), 0.99) == 1
    assert sample_token(logits, SamplingParams(top_p=1.0), uniform=0.99) == 1


def test_nonfinite_rejected():
    for bad in (torch.tensor([1.0, float("nan")]), torch.tensor([float("inf"), 0.5])):
        with pytest.raises(ValueError):
            sample_token(bad, SamplingParams())
        with pytest.raises(ValueError):
            reference_sample_token(bad, SamplingParams(), 0.5)


def test_draw_bounds_validated():
    logits = torch.tensor([1.0, 2.0])
    with pytest.raises(ValueError):
        reference_sample_token(logits, SamplingParams(), 1.0)
    with pytest.raises(ValueError):
        sample_token(logits, SamplingParams(), uniform=1.5)
