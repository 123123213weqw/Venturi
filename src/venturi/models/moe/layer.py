"""Universal routed MoE block: router + selected-expert execution reference.

Correctness-first torch execution over any store satisfying the expert-store
contract (`.geometry` + `read(expert, *, device, dtype) -> (gate_up, down)`).
Expert math is oracle-exact clamped SwiGLU; iteration order matches the
pinned flash-next reference (ascending expert ids, index_add_ accumulation).
"""

import torch
from torch import nn

from venturi.models.moe.geometry import ExpertGeometry
from venturi.models.moe.routing import TopKRouter


def _clamped_swiglu(gate, up, limit):
    """Training-compatible SwiGLU clamps on any device/dtype.

    Mirrors venturi.models.moe_routing.reference_expert_activation (the
    CPU-FP32 pinned oracle) without its CPU-only admission checks:
    upper-only gate clamp, two-sided up clamp.
    """
    if limit:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return nn.functional.silu(gate) * up


def routed_experts(hidden_states, indices, scores, store, *, swiglu_limit: float = 0.0):
    """Execute selected experts in ascending id order (pinned reference)."""
    geometry = store.geometry
    if not isinstance(geometry, ExpertGeometry):
        raise TypeError("Store must expose an ExpertGeometry")
    if indices.dtype != torch.int64 or scores.dtype != hidden_states.dtype:
        raise ValueError("Indices must be int64 and scores match the stream dtype")
    if indices.device != hidden_states.device or scores.device != hidden_states.device:
        raise ValueError("Routing tensors must share the stream device")
    if indices.numel() and (indices.min() < 0 or indices.max() >= geometry.num_experts):
        raise ValueError("Router selected expert ids outside the bank")
    output = torch.zeros_like(hidden_states)
    flat = hidden_states.reshape(-1, geometry.hidden_size)
    flat_indices = indices.reshape(-1, geometry.top_k)
    flat_scores = scores.reshape(-1, geometry.top_k)
    for expert in torch.unique(flat_indices).tolist():
        token_rows, slots = torch.where(flat_indices == expert)
        gate_up, down = store.read(expert, device=hidden_states.device, dtype=hidden_states.dtype)
        gate, up = nn.functional.linear(flat[token_rows], gate_up).chunk(2, dim=-1)
        activated = _clamped_swiglu(gate, up, swiglu_limit)
        current = nn.functional.linear(activated, down)
        weighted = current.to(output.dtype) * flat_scores[token_rows, slots, None].to(output.dtype)
        output.reshape(-1, geometry.hidden_size).index_add_(0, token_rows, weighted.to(output.dtype))
    return output


class RoutedMoE(nn.Module):
    """Router + routed experts + optional shared expert and output scaling."""

    def __init__(self, geometry: ExpertGeometry, router: TopKRouter, store,
                 *, swiglu_limit: float = 0.0, routed_scale: float = 1.0,
                 shared_expert: nn.Module = None, shared_gate: nn.Module = None):
        super().__init__()
        if router.geometry is not geometry:
            raise ValueError("Router geometry must be the expert geometry")
        self.geometry = geometry
        self.router = router
        self.store = store
        self.expert_dispatch = None  # optional seam, mirrors flash-next
        self.swiglu_limit = swiglu_limit
        self.routed_scale = routed_scale
        self.shared_expert = shared_expert
        self.shared_gate = shared_gate

    def forward(self, hidden_states):
        weights, indices = self.router(hidden_states)
        if self.expert_dispatch is None:
            routed = routed_experts(hidden_states, indices, weights, self.store,
                                    swiglu_limit=self.swiglu_limit)
        else:
            routed = self.expert_dispatch(hidden_states, indices, weights)
        result = routed * self.routed_scale
        if self.shared_expert is not None:
            shared = self.shared_expert(hidden_states)
            if self.shared_gate is not None:
                shared = torch.sigmoid(self.shared_gate(hidden_states)) * shared
            result = result + shared
        return result
