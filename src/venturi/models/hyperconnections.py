"""CPU hyper-connection semantic oracles, not a GPU or full-model backend.

The residual stream is [batch, tokens, streams, hidden]. A sublayer computes
new pre/post/residual coefficients BEFORE changing that stream, but consumes
the PREVIOUS sublayer's pre coefficients. This one-sublayer lag is intentional.
No existing native generation path is redirected to these reference functions.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class HyperConnectionSpec:
    streams: int = 4
    sinkhorn_iterations: int = 20
    epsilon: float = 1e-6
    norm_epsilon: float = 1e-6

    def __post_init__(self):
        for field in ("streams", "sinkhorn_iterations"):
            value = getattr(self, field)
            if type(value) is not int or value < 1:
                raise ValueError(f"Positive integer {field} required")
        for field in ("epsilon", "norm_epsilon"):
            value = getattr(self, field)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"Finite positive {field} required")

    @property
    def coefficient_count(self):
        return self.streams * (self.streams + 2)


def _tensor(value, *, fp32=False):
    dtypes = (torch.float32,) if fp32 else (torch.float32, torch.float16, torch.bfloat16)
    if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
            or value.dtype not in dtypes or not torch.isfinite(value).all()):
        raise ValueError("Finite CPU oracle tensors with explicit supported dtype required")


def _stream(value):
    _tensor(value)
    if value.ndim != 4 or any(n == 0 for n in value.shape):
        raise ValueError("Nonempty [batch, tokens, streams, hidden] residual required")


def reference_hc_coefficients(projected, scales, bases, spec):
    """FP32 pre/post/comb; comb's axes are [source stream, destination stream].

    The first iteration is row softmax PLUS epsilon then column normalization.
    Subsequent iterations alternate row/column division, each denominator plus
    epsilon. Finite iteration counts need NOT produce exact unit row sums.
    This is a CPU reading of the official kernel equations, NOT GPU validation.
    """
    if not isinstance(spec, HyperConnectionSpec):
        raise TypeError("Explicit HyperConnectionSpec required")
    for value in (projected, scales, bases):
        _tensor(value, fp32=True)
    if (projected.ndim != 3 or any(n == 0 for n in projected.shape)
            or projected.shape[-1] != spec.coefficient_count
            or scales.shape != (3,) or bases.shape != (spec.coefficient_count,)):
        raise ValueError("Matching projected coefficients, three scales and bases required")
    h, eps = spec.streams, spec.epsilon
    pre = (projected[..., :h] * scales[0] + bases[:h]).sigmoid() + eps
    post = 2 * (projected[..., h:2*h] * scales[1] + bases[h:2*h]).sigmoid()
    matrix = (projected[..., 2*h:] * scales[2] + bases[2*h:]).unflatten(-1, (h, h))
    matrix = matrix.softmax(-1) + eps
    matrix = matrix / (matrix.sum(-2, keepdim=True) + eps)
    for _ in range(spec.sinkhorn_iterations - 1):
        matrix = matrix / (matrix.sum(-1, keepdim=True) + eps)
        matrix = matrix / (matrix.sum(-2, keepdim=True) + eps)
    for value in (pre, post, matrix):
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite hyper-connection arithmetic")
    return pre, post, matrix


def reference_hc_mixes(stream, projection, scales, bases, spec):
    """One RMS statistic over ALL streams/hidden channels, not per stream."""
    _stream(stream)
    _tensor(projection, fp32=True)
    if not isinstance(spec, HyperConnectionSpec):
        raise TypeError("Explicit HyperConnectionSpec required")
    if (stream.shape[-2] != spec.streams
            or projection.shape != (spec.coefficient_count, spec.streams * stream.shape[-1])):
        raise ValueError("Projection must match the flattened multi-stream residual")
    flat = stream.flatten(2).float()
    inv_rms = torch.rsqrt(flat.square().mean(-1, keepdim=True) + spec.norm_epsilon)
    projected = F.linear(flat, projection) * inv_rms
    return reference_hc_coefficients(projected, scales, bases, spec)


def reference_hc_pre(stream, previous_pre):
    """Collapse streams with PREVIOUS sublayer coefficients, FP32 accumulation."""
    _stream(stream)
    _tensor(previous_pre, fp32=True)
    if previous_pre.shape != stream.shape[:-1]:
        raise ValueError("Pre coefficients must match residual batch/tokens/streams")
    return (stream.float() * previous_pre[..., None]).sum(2).to(stream.dtype)


def reference_hc_post(output, residual, post, combination):
    """Expand output and combine residual with a source->destination matrix.

    In particular this is NOT combination @ residual: the matrix is transposed
    relative to that familiar expression. Output dtype follows the sublayer.
    """
    _stream(residual)
    _tensor(output)
    _tensor(post, fp32=True)
    _tensor(combination, fp32=True)
    b, t, h, d = residual.shape
    if output.shape != (b, t, d) or post.shape != (b, t, h) or combination.shape != (b, t, h, h):
        raise ValueError("Output and coefficients must match the residual geometry")
    expanded = output.unsqueeze(-2) * post.unsqueeze(-1)
    mixed = (residual.unsqueeze(-2) * combination.unsqueeze(-1)).sum(2)
    return (expanded + mixed).to(output.dtype)


def reference_identity_pre(stream):
    """Initial collapse selects stream zero; it is NOT the average of streams."""
    _stream(stream)
    result = torch.zeros(stream.shape[:-1], dtype=torch.float32)
    result[..., 0] = 1
    return result
