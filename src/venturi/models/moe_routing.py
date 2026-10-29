"""CPU MoE semantic oracles, not performance kernels or a complete model.

Shared routing/activation specifications are independent of a checkpoint and
storage format. In particular, correction bias affects selection, NOT weights.
DeepSeek V4.1 semantics are checked against its pinned reference Gate.forward;
this is not an inference fallback and is not wired into existing native models.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RoutingSpec:
    top_k: int
    scoring: str
    normalize: bool
    scale: float = 1.0
    temperature: float = 1.0
    normalization_epsilon: float = 1e-20

    def __post_init__(self):
        if type(self.top_k) is not int or self.top_k < 1:
            raise ValueError("Positive integer top_k required")
        if self.scoring not in ("softmax", "sigmoid", "sqrtsoftplus"):
            raise ValueError("Explicit supported router score function required")
        if type(self.normalize) is not bool:
            raise TypeError("Normalization must be explicit bool")
        for name in ("scale", "temperature", "normalization_epsilon"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Finite nonnegative {name} required")
        if self.temperature == 0 or self.normalization_epsilon == 0:
            raise ValueError("Positive temperature and normalization epsilon required")


def _cpu_float32(tensor):
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or tensor.dtype != torch.float32:
        raise ValueError("Explicit float32 CPU oracle tensors required")
    if not torch.isfinite(tensor).all():
        raise ValueError("Finite oracle inputs required")


def reference_route_experts(logits, correction_bias, spec):
    """Return (FP32 weights, int64 indices), using unbiased selected scores.

    Bias can be shared [experts] or per token [tokens, experts], so callers can
    select a separate image-span bias without mixing it into routing weights.
    A top_k of one intentionally does NOT normalize, matching DS41 training.
    """
    if not isinstance(spec, RoutingSpec):
        raise TypeError("Explicit RoutingSpec required")
    _cpu_float32(logits)
    _cpu_float32(correction_bias)
    if logits.ndim != 2 or not logits.shape[0] or spec.top_k > logits.shape[1]:
        raise ValueError("Nonempty [tokens, experts] logits and valid top_k required")
    if correction_bias.shape not in (logits.shape[1:], logits.shape):
        raise ValueError("Correction bias must be shared or token-shaped")
    scores = logits / spec.temperature
    if spec.scoring == "softmax":
        scores = scores.softmax(dim=-1)
    elif spec.scoring == "sigmoid":
        scores = scores.sigmoid()
    else:
        scores = F.softplus(scores).sqrt()
    indices = (scores + correction_bias).topk(spec.top_k, dim=-1)[1]
    weights = scores.gather(1, indices)
    if spec.normalize and spec.top_k > 1:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + spec.normalization_epsilon)
    return weights * spec.scale, indices


def reference_expert_activation(gate, up, *, limit):
    """Training-compatible SwiGLU clamps: upper-only gate, two-sided up.

    The returned FP32 activation must still be cast to the model's expert dtype
    before the down projection. No projection/quantization is implemented here.
    """
    _cpu_float32(gate)
    _cpu_float32(up)
    if gate.shape != up.shape or gate.ndim != 2:
        raise ValueError("Matching [tokens, channels] activation branches required")
    if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not math.isfinite(limit) or limit < 0:
        raise ValueError("Finite nonnegative clamp limit required")
    if limit:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return F.silu(gate) * up
