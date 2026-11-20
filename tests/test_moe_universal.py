"""Universal MoE layer tests: router oracle equality, reference execution,
profiler coverage semantics. CPU-only."""

import pytest
import torch
from torch import nn

from venturi.models.moe.geometry import ExpertGeometry
from venturi.models.moe.layer import RoutedMoE, routed_experts
from venturi.models.moe.profiler import RouteProfiler
from venturi.models.moe.routing import TopKRouter
from venturi.models.moe_routing import RoutingSpec, reference_route_experts


class Bank:
    """Minimal store contract: in-RAM experts, ascending read."""

    def __init__(self, geometry, dtype=torch.float32):
        self.geometry = geometry
        g = torch.Generator().manual_seed(11)
        self.dtype = dtype
        self.experts = {
            e: (torch.randn(2 * geometry.intermediate_size, geometry.hidden_size, generator=g),
                torch.randn(geometry.hidden_size, geometry.intermediate_size, generator=g))
            for e in range(geometry.num_experts)
        }

    def read(self, expert, *, device, dtype):
        assert 0 <= expert < self.geometry.num_experts
        return tuple(t.to(device=device, dtype=dtype) for t in self.experts[expert])


GEOMETRY = ExpertGeometry(hidden_size=8, num_experts=6, top_k=2, intermediate_size=4)


def make_router(spec):
    router = TopKRouter(GEOMETRY, spec)
    with torch.no_grad():
        router.weight.copy_(torch.randn(GEOMETRY.num_experts, GEOMETRY.hidden_size, generator=torch.Generator().manual_seed(3)))
        router.correction_bias.copy_(torch.linspace(-0.5, 0.5, GEOMETRY.num_experts))
    return router


@pytest.mark.parametrize("scoring,normalize", [
    ("softmax", True), ("softmax", False), ("sigmoid", True), ("sqrtsoftplus", False),
])
def test_router_matches_oracle_exactly(scoring, normalize):
    router = make_router(RoutingSpec(top_k=GEOMETRY.top_k, scoring=scoring, normalize=normalize, scale=1.7))
    hidden = torch.randn(5, 3, GEOMETRY.hidden_size, generator=torch.Generator().manual_seed(5))
    weights, indices = router(hidden)
    logits = nn.functional.linear(hidden.reshape(-1, GEOMETRY.hidden_size), router.weight).float()
    bias = router.correction_bias.float().expand(logits.shape[0], -1)
    expected_weights, expected_indices = reference_route_experts(
        logits, bias, RoutingSpec(top_k=GEOMETRY.top_k, scoring=scoring, normalize=normalize, scale=1.7))
    assert torch.equal(indices.reshape(-1, GEOMETRY.top_k), expected_indices)
    assert torch.equal(weights.reshape(-1, GEOMETRY.top_k), expected_weights)


def test_routed_execution_matches_tokenwise_reference():
    router = make_router(RoutingSpec(top_k=GEOMETRY.top_k, scoring="sigmoid", normalize=True))
    store = Bank(GEOMETRY)
    hidden = torch.randn(2, 5, GEOMETRY.hidden_size, generator=torch.Generator().manual_seed(9))
    weights, indices = router(hidden)
    output = routed_experts(hidden, indices, weights, store)
    reference = torch.zeros_like(hidden)
    flat = hidden.reshape(-1, GEOMETRY.hidden_size)
    w = weights.reshape(-1, GEOMETRY.top_k)
    i = indices.reshape(-1, GEOMETRY.top_k)
    for expert in sorted(set(i.reshape(-1).tolist())):  # expert-major, matching index_add_
        gate_up, down = store.experts[expert]
        token_rows, slots = (i == expert).nonzero(as_tuple=True)
        gate, up = nn.functional.linear(flat[token_rows], gate_up).chunk(2, dim=-1)
        values = nn.functional.linear(nn.functional.silu(gate) * up, down)
        reference.reshape(-1, GEOMETRY.hidden_size).index_add_(
            0, token_rows, values * w[token_rows, slots, None])
    torch.testing.assert_close(output, reference, rtol=0, atol=0)


def test_clamped_swiglu_limits_gate_and_up():
    router = make_router(RoutingSpec(top_k=GEOMETRY.top_k, scoring="sigmoid", normalize=True))
    store = Bank(GEOMETRY)
    model = RoutedMoE(GEOMETRY, router, store, swiglu_limit=0.05)
    hidden = torch.randn(4, GEOMETRY.hidden_size, generator=torch.Generator().manual_seed(2))
    output = model(hidden)
    weights, indices = router(hidden)
    manual = routed_experts(hidden, indices, weights, store, swiglu_limit=0.05)
    torch.testing.assert_close(output, manual, rtol=0, atol=0)
    unlimited = routed_experts(hidden, indices, weights, store, swiglu_limit=0.0)
    assert not torch.equal(output, unlimited)  # the clamp actually bites


def test_shared_expert_and_scale_compose():
    router = make_router(RoutingSpec(top_k=GEOMETRY.top_k, scoring="softmax", normalize=True))
    store = Bank(GEOMETRY)
    shared = nn.Linear(GEOMETRY.hidden_size, GEOMETRY.hidden_size)
    gate = nn.Linear(GEOMETRY.hidden_size, 1, bias=False)
    model = RoutedMoE(GEOMETRY, router, store, routed_scale=2.0, shared_expert=shared, shared_gate=gate)
    hidden = torch.randn(3, GEOMETRY.hidden_size)
    output = model(hidden)
    weights, indices = router(hidden)
    expected = routed_experts(hidden, indices, weights, store) * 2.0
    expected = expected + torch.sigmoid(gate(hidden)) * shared(hidden)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_profiler_hot_set_and_coverage():
    profiler = RouteProfiler(num_layers=2, num_experts=8)
    router = make_router(RoutingSpec(top_k=2, scoring="softmax", normalize=True))
    router.attach_profiler(profiler)
    profiler.enter_layer(0)
    router(torch.randn(6, GEOMETRY.hidden_size))
    report = profiler.coverage_report(layer=0)
    assert report[0]["tokens"] == 12  # 6 tokens x top-2
    ranked = report[0]["experts"]
    assert ranked[0]["coverage"] == ranked[0]["tokens"] / 12
    assert abs(ranked[-1]["coverage"] - 1.0) < 1e-9
    hot = profiler.hot_set(coverage=0.99, layer=0)[0]
    assert 1 <= len(hot) <= 8
    with pytest.raises(ValueError):
        profiler.hot_set(coverage=1.5)
    with pytest.raises(ValueError):
        profiler.enter_layer(9)


def test_geometry_and_router_validation():
    with pytest.raises(ValueError):
        ExpertGeometry(hidden_size=0, num_experts=4, top_k=2, intermediate_size=3)
    with pytest.raises(ValueError):
        ExpertGeometry(hidden_size=8, num_experts=4, top_k=5, intermediate_size=3)
    with pytest.raises(ValueError):
        TopKRouter(GEOMETRY, RoutingSpec(top_k=3, scoring="softmax", normalize=True))
    router = make_router(RoutingSpec(top_k=GEOMETRY.top_k, scoring="softmax", normalize=True))
    with pytest.raises(ValueError):
        router(torch.zeros(2, 7))  # hidden mismatch
