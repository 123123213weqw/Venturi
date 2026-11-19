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
        dict(seed=2**64),
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

    def draw_sequence(seed):
        rng = torch.Generator(device="cpu").manual_seed(seed)
        return [sample_token(logits, params, rng) for _ in range(32)]

    first = draw_sequence(1234)
    assert first == draw_sequence(1234) and len(set(first)) > 1
    assert first != draw_sequence(99)


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


@pytest.mark.parametrize(
    "top_k,top_p,draw,expected",
    [
        (1, 1.0, 0.0, 0),
        (2, 1.0, 0.75, 1),
        (0, 0.25, 0.99, 1),
    ],
)
def test_tied_scores_keep_ascending_token_ids(top_k, top_p, draw, expected):
    logits = torch.zeros(8, dtype=torch.float64)
    params = SamplingParams(top_k=top_k, top_p=top_p)
    assert sample_token(logits, params, uniform=draw) == expected
    assert reference_sample_token(logits, params, draw) == expected


def test_inverse_cdf_rounding_cannot_return_zero_mass_tail():
    logits = torch.tensor([0.0, -5 / 17, -5 / 19, -1000.0], dtype=torch.float64)
    draw = math.nextafter(1.0, 0.0)
    assert sample_token(logits, SamplingParams(), uniform=draw) == 1
    assert reference_sample_token(logits, SamplingParams(), draw) == 1


def test_reference_and_sampler_reject_empty_row():
    with pytest.raises(ValueError, match="nonempty"):
        sample_token(torch.empty(0), SamplingParams())
    with pytest.raises(ValueError, match="nonempty"):
        reference_sample_token([], SamplingParams(), 0.0)


def test_finite_extreme_logits_can_be_rescaled_before_difference():
    logits = torch.tensor([1e308, -1e308], dtype=torch.float64)
    params = SamplingParams(temperature=1e308)
    # Centered logits [0, -2] give the second ID about 0.119 probability.
    assert sample_token(logits, params, uniform=0.95) == 1
    assert reference_sample_token(logits, params, 0.95) == 1
