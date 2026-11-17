"""Universal top-k expert routing: a torch module wrapper over the pinned
semantic oracle (`venturi.models.moe_routing.reference_route_experts`).

The oracle is the single source of routing truth (scoring family, correction
bias affecting selection but not weights, top_k==1 non-normalization,
scaling); this module only adds weight ownership, device/dtype discipline and
an optional routing-profiler hook. flash-next's softmax router and the
DeepSeek/GLM sigmoid-noaux family are both configurations of this module.
"""

import torch
from torch import nn

from venturi.models.moe_routing import RoutingSpec

# Device-independent re-implementation of the pinned oracle semantics
# (venturi.models.moe_routing.reference_route_experts). Byte-exact equality
# with the oracle is locked by tests/test_moe_universal.py on CPU; the oracle
# itself stays CPU-only by design.
def _route(flat_logits, bias, spec):
    scores = flat_logits / spec.temperature
    if spec.scoring == "softmax":
        scores = scores.softmax(dim=-1)
    elif spec.scoring == "sigmoid":
        scores = scores.sigmoid()
    else:
        scores = torch.nn.functional.softplus(scores).sqrt()
    indices = (scores + bias).topk(spec.top_k, dim=-1)[1]
    weights = scores.gather(1, indices)
    if spec.normalize and spec.top_k > 1:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + spec.normalization_epsilon)
    return weights * spec.scale, indices


class TopKRouter(nn.Module):
    """Linear router logits + oracle-exact selection and weighting."""

    def __init__(self, geometry, spec: RoutingSpec):
        super().__init__()
        if spec.top_k != geometry.top_k:
            raise ValueError("Routing spec top_k must match the expert geometry")
        self.geometry = geometry
        self.spec = spec
        self.weight = nn.Parameter(torch.zeros(geometry.num_experts, geometry.hidden_size))
        self.correction_bias = nn.Parameter(torch.zeros(geometry.num_experts), requires_grad=False)
        self._profiler = None

    def attach_profiler(self, profiler):
        """Optional side-channel histogram; routing math is unchanged."""
        self._profiler = profiler

    def forward(self, hidden_states):
        if hidden_states.dtype != self.weight.dtype or hidden_states.device != self.weight.device:
            raise ValueError("Router input dtype/device must match the router weight")
        if hidden_states.shape[-1] != self.geometry.hidden_size:
            raise ValueError("Router input hidden size differs from geometry")
        logits = nn.functional.linear(hidden_states, self.weight)
        flat = logits.reshape(-1, self.geometry.num_experts).to(torch.float32)
        bias = self.correction_bias.to(torch.float32).expand(flat.shape[0], -1)
        weights, indices = _route(flat, bias, self.spec)
        if self._profiler is not None:
            self._profiler.record(indices)
        return weights.to(logits.dtype), indices.reshape(*logits.shape[:-1], self.geometry.top_k)
