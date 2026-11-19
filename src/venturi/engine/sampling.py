"""Universal token sampling over logits: temperature, top-k, top-p, seed.

Model-agnostic by construction: the only input is one row of logits. The
correctness-first implementation runs in float64 on CPU tensors with a single
inverse-CDF draw. Exact ties use ascending token IDs, matching the scalar
reference. CDF boundaries can differ by floating-point rounding. A fixed seed
replays a draw sequence for the same logits and PyTorch environment. GPU-native
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
    seed:        unsigned 64-bit deterministic seed; None uses the global CPU RNG
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
        if self.seed is not None and (
            type(self.seed) is not int or not 0 <= self.seed < 2**64
        ):
            raise ValueError("seed must be an unsigned 64-bit int or None")


def _candidate_probs(logits: torch.Tensor, params: SamplingParams):
    """Shared truncation math: descending (probabilities, token ids)."""
    if logits.ndim != 1 or logits.numel() == 0:
        raise ValueError("sampling expects one nonempty logits row")
    scores = logits.to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(scores).all():
        raise ValueError("sampling rejected nonfinite logits")
    keep = min(params.top_k, scores.numel()) if params.top_k else scores.numel()
    # Stable ordering breaks equal scores by the original (lowest) token ID.
    values, indices = torch.sort(scores, descending=True, stable=True)
    values, indices = values[:keep], indices[:keep]
    temperature = params.temperature
    centered = values - values[0]
    scaled = centered / temperature
    # Opposite-signed finite FP64 extremes can overflow during subtraction
    # even when the temperature-scaled difference is representable.
    if torch.isinf(centered).any():
        scaled = torch.where(
            torch.isinf(centered),
            values / temperature - values[0] / temperature,
            scaled,
        )
    weights = torch.exp(scaled)
    probabilities = weights / weights.sum()
    if params.top_p < 1:
        cumulative = torch.cumsum(probabilities, 0)
        # First crossing is inclusive; a rounded CDF short of the budget
        # keeps the full support. Avoid subtracting a probability from its CDF.
        threshold = cumulative.new_tensor(params.top_p)
        prefix = int(torch.searchsorted(cumulative, threshold)) + 1
        probabilities, indices = probabilities[:prefix], indices[:prefix]
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
    seeded CPU `generator` (or the params seed, or global CPU RNG) supplies
    it. A params seed alone builds a fresh generator for this draw; pass a
    persistent generator to consume a sequence across calls.
    """
    if not isinstance(params, SamplingParams):
        raise TypeError("sampling requires SamplingParams")
    if generator is None and params.seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(params.seed)
    probabilities, indices = _candidate_probs(logits, params)
    if uniform is None:
        draw = float(
            torch.rand((), generator=generator, device="cpu", dtype=torch.float64)
        )
    else:
        draw = uniform
    return int(indices[_inverse_cdf(probabilities, draw)])


def _inverse_cdf(probabilities: torch.Tensor, draw: float) -> int:
    if not 0 <= draw < 1:
        raise ValueError("uniform draw must lie in [0, 1)")
    last_positive, total = 0, 0.0
    for index, probability in enumerate(probabilities.tolist()):
        if probability > 0.0:
            last_positive = index
        total += probability
        if draw < total:
            return index
    return last_positive  # A rounding residual must not select zero mass.


def reference_sample_token(logits, params: SamplingParams, draw: float) -> int:
    """Scalar float64 golden reference: same semantics, no torch primitives."""
    if not 0 <= draw < 1:
        raise ValueError("uniform draw must lie in [0, 1)")
    scores = [float(value) for value in logits]
    if not scores:
        raise ValueError("sampling expects one nonempty logits row")
    for value in scores:
        if not math.isfinite(value):
            raise ValueError("sampling rejected nonfinite logits")
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    if params.top_k:
        order = order[: params.top_k]
    peak = scores[order[0]]
    temperature = params.temperature
    weights = []
    for index in order:
        centered = scores[index] - peak
        scaled = (
            scores[index] / temperature - peak / temperature
            if math.isinf(centered)
            else centered / temperature
        )
        weights.append(math.exp(scaled))
    total = sum(weights)
    probabilities = [weight / total for weight in weights]
    if params.top_p < 1:
        kept, cumulative = [], 0.0
        for index, probability in zip(order, probabilities):
            kept.append(index)
            cumulative += probability
            if cumulative >= params.top_p:
                break
        order, probabilities = kept, probabilities[: len(kept)]
        total = sum(probabilities)
        probabilities = [probability / total for probability in probabilities]
    last_positive, running = order[0], 0.0
    for index, probability in zip(order, probabilities):
        if probability > 0.0:
            last_positive = index
        running += probability
        if draw < running:
            return index
    return last_positive
