"""Universal token sampling over logits: temperature, top-k, top-p, seed.

Model-agnostic by construction: the only input is one row of logits. The
correctness-first implementation runs in float64 on CPU tensors with a single
inverse-CDF draw, so it is bit-comparable against the scalar reference
(`reference_sample_token`) and fully reproducible for a fixed seed. GPU-native
sampling kernels and in-loop native sampling are future optimizations, not
part of this contract.
"""

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class SamplingParams:
    """Explicit sampling controls; invalid values fail closed.

    temperature: logit divisor, must be finite and positive
    top_k:       keep only the k highest logits (0 keeps all)
    top_p:       keep the smallest descending-probability prefix whose mass
                 reaches top_p (1.0 keeps all)
    seed:        deterministic sampling; None draws from global torch RNG
    """

    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    seed: int | None = None

    def __post_init__(self):
        if type(self.temperature) not in (int, float) or not (
            math.isfinite(self.temperature) and self.temperature > 0
        ):
            raise ValueError("temperature must be a finite positive number")
        if type(self.top_k) is not int or self.top_k < 0:
            raise ValueError("top_k must be a nonnegative int (0 keeps all)")
        if type(self.top_p) not in (int, float) or not (
            math.isfinite(self.top_p) and 0 < self.top_p <= 1
        ):
            raise ValueError("top_p must be in (0, 1]")
        if self.seed is not None and (type(self.seed) is not int or self.seed < 0):
            raise ValueError("seed must be a nonnegative int or None")


def _candidate_probs(logits: torch.Tensor, params: SamplingParams):
    """Shared truncation math: descending (probabilities, token ids)."""
    if logits.ndim != 1 or logits.numel() == 0:
        raise ValueError("sampling expects one nonempty logits row")
    scores = logits.double()
    if not torch.isfinite(scores).all():
        raise ValueError("sampling rejected nonfinite logits")
    keep = min(params.top_k, scores.numel()) if params.top_k else scores.numel()
    values, indices = torch.topk(scores, keep)
    weights = torch.exp((values - values[0]) / params.temperature)
    probabilities = weights / weights.sum()
    if params.top_p < 1:
        cumulative = torch.cumsum(probabilities, 0)
        # Keep tokens whose cumulative mass BEFORE them is still below the
        # budget; the first token always survives, so the prefix is nonempty.
        probabilities = torch.where(cumulative - probabilities < params.top_p, probabilities, torch.zeros(()))
        probabilities = probabilities / probabilities.sum()
    return probabilities, indices


def sample_token(
    logits: torch.Tensor,
    params: SamplingParams,
    generator: torch.Generator | None = None,
    *,
    uniform: float | None = None,
) -> int:
    """Draw one token id from a logits row with the given controls.

    `uniform` injects the [0,1) draw explicitly (tests, replay); otherwise the
    seeded `generator` (or the params seed, or global torch RNG) supplies it.
    """
    if not isinstance(params, SamplingParams):
        raise TypeError("sampling requires SamplingParams")
    if generator is None and params.seed is not None:
        generator = torch.Generator()
        generator.manual_seed(params.seed)
    probabilities, indices = _candidate_probs(logits, params)
    if uniform is None:
        draw = float(torch.rand((), generator=generator)) if generator is not None else float(torch.rand(()))
    else:
        draw = uniform
    return int(indices[_inverse_cdf(probabilities, draw)])


def _inverse_cdf(probabilities: torch.Tensor, draw: float) -> int:
    if not 0 <= draw < 1:
        raise ValueError("uniform draw must lie in [0, 1)")
    total = 0.0
    for index, probability in enumerate(probabilities.tolist()):
        total += probability
        if draw < total:
            return index
    return len(probabilities) - 1  # float rounding: the last candidate absorbs


def reference_sample_token(logits, params: SamplingParams, draw: float) -> int:
    """Scalar float64 golden reference: same semantics, no torch primitives."""
    if not 0 <= draw < 1:
        raise ValueError("uniform draw must lie in [0, 1)")
    scores = [float(value) for value in logits]
    for value in scores:
        if not math.isfinite(value):
            raise ValueError("sampling rejected nonfinite logits")
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    if params.top_k:
        order = order[: params.top_k]
    peak = scores[order[0]]
    weights = [math.exp((scores[i] - peak) / params.temperature) for i in order]
    total = sum(weights)
    probabilities = [weight / total for weight in weights]
    if params.top_p < 1:
        kept, cumulative = [], 0.0
        for index, probability in zip(order, probabilities):
            kept.append(index)
            cumulative += probability
            if cumulative >= params.top_p:
                break
        order, total = kept, sum(probabilities[: len(kept)])
        probabilities = [probabilities[: len(kept)][i] / total for i in range(len(kept))]
    running = 0.0
    for index, probability in zip(order, probabilities):
        running += probability
        if draw < running:
            return index
    return order[-1]
