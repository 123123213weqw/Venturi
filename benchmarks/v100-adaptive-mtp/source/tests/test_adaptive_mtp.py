"""Policy learning is bounded, transparent and independent of model kernels."""

import math

import pytest

from venturi.adaptive_mtp import AdaptiveMTPPolicy, MTPWorkload


def workload(count=1, prompt=23, budget=32, stopping=False):
    return MTPWorkload((prompt,) * count, (budget,) * count, stopping)


def observe(policy, work, depth, cost, **kw):
    policy.observe(
        work,
        depth,
        wall_seconds=cost * sum(work.output_budgets),
        emitted_tokens=sum(work.output_budgets),
        **kw,
    )


@pytest.mark.parametrize(
    "count,expected", [(1, [0, 1, 2]), (2, [0, 1]), (4, [0, 1]), (8, [0, 1])]
)
def test_cold_probes_then_measured_strategy(count, expected):
    policy, work = AdaptiveMTPPolicy(), workload(count)
    for depth in expected:
        result = policy.choose(work)
        assert result.depth == depth and result.reason == "cold-start-probe"
        observe(policy, work, depth, 0.03 - depth * 0.005)
    assert policy.choose(work).depth == expected[-1]
    assert policy.choose(work).reason == "measured-cost"


def test_feedback_can_disable_and_reenable_mtp():
    p, w = AdaptiveMTPPolicy(alpha=1, probe_interval=4), workload()
    for d, cost in [(0, 0.03), (1, 0.02), (2, 0.025)]:
        observe(p, w, d, cost)
    assert p.choose(w).depth == 1
    observe(p, w, 1, 0.04, proposed_tokens=20, accepted_tokens=0)
    assert p.choose(w).depth == 2
    observe(p, w, 2, 0.05, proposed_tokens=30, accepted_tokens=0)
    assert p.choose(w).depth == 0
    observe(p, w, 0, 0.03)
    observe(p, w, 0, 0.03)
    observe(p, w, 0, 0.03)
    assert p.choose(w).reason == "periodic-recheck"
    assert p.choose(w).depth == 1
    observe(p, w, 1, 0.015, proposed_tokens=20, accepted_tokens=18)
    assert p.choose(w).depth == 2  # The other stale alternative is checked too.
    observe(p, w, 2, 0.05)
    assert p.choose(w).depth == 1
    row = p.info()["buckets"][0]["samples"][1]
    assert row["acceptance_rate"] == 18 / 40


def test_probe_free_manual_learning_and_margin():
    p, w = AdaptiveMTPPolicy(probe_interval=0), workload()
    assert p.choose(w).depth == 0
    assert p.choose(w, override=2).reason == "manual-override"
    observe(p, w, 2, 0.02)
    assert p.choose(w).reason == "unmeasured-baseline"
    observe(p, w, 0, 0.0201)
    assert p.choose(w).depth == 0  # Less than the 3% safety margin.
    observe(p, w, 1, 0.015)
    assert p.choose(w).depth == 1


def test_workload_buckets_do_not_share_unrelated_costs():
    p = AdaptiveMTPPolicy(probe_interval=0, max_buckets=2)
    w = workload()
    observe(p, w, 0, 0.03)
    observe(p, w, 2, 0.01)
    assert p.choose(workload(prompt=24)).depth == 2
    assert p.choose(workload(prompt=33)).depth == 0
    assert p.choose(workload(stopping=True)).depth == 0
    assert len(p.info()["buckets"]) == 2
    assert p.choose(w).depth == 0  # Evicted, not stale cross-bucket reuse.


@pytest.mark.parametrize("count,budget", [(1, 1), (2, 1), (8, 1)])
def test_single_output_does_not_probe_useless_drafts(count, budget):
    p, w = AdaptiveMTPPolicy(), workload(count, budget=budget)
    assert p.choose(w).depth == 0
    with pytest.raises(ValueError):
        p.choose(w, override=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(alpha=0),
        dict(alpha=math.nan),
        dict(minimum_gain=1),
        dict(probe_interval=True),
        dict(max_buckets=0),
    ],
)
def test_invalid_policy(kwargs):
    with pytest.raises(ValueError):
        AdaptiveMTPPolicy(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(wall_seconds=math.nan),
        dict(wall_seconds=-1),
        dict(emitted_tokens=True),
        dict(emitted_tokens=33),
        dict(accepted_tokens=2, proposed_tokens=1),
    ],
)
def test_invalid_observation_is_not_learned(kwargs):
    p = AdaptiveMTPPolicy()
    args = dict(wall_seconds=1, emitted_tokens=32)
    args.update(kwargs)
    with pytest.raises(ValueError):
        p.observe(workload(), 1, **args)
    assert not p.info()["buckets"]


@pytest.mark.parametrize(
    "supported,override",
    [((0, 1), 2), ((0, 1, 2), True), ((1, 2), None), ((0, 0), None)],
)
def test_unsupported_selection_fails_closed(supported, override):
    with pytest.raises(ValueError):
        AdaptiveMTPPolicy().choose(workload(2), supported=supported, override=override)
