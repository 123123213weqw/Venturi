"""Sampling strategy interface (Protocol, no implementation yet).

Defines the contract for token sampling. Current engine is greedy-only;
future implementations include temperature, top-p, top-k, etc.
"""
from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class Sampler(Protocol):
    """Contract for token samplers."""

    def configure(self, **params: Any) -> None:
        """Set sampling parameters (temperature, top_p, top_k, etc.)."""
        ...

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample token IDs from logits [batch, vocab]."""
        ...
