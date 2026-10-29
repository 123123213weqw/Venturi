"""Structural requirements of the existing greedy loop, not a serving API.

The implementation still uses PyTorch tensors. These protocols only remove the
unrelated dependency on a particular model family; they do not supply scheduling,
MTP, sampling, graph capture, or an alternative tensor representation.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence


class GenerationConfig(Protocol):
    vocab_size: int
    max_position_embeddings: int
    bos_token_id: int | None
    eos_token_ids: tuple[int, ...]


class ExecutionDevice(Protocol):
    type: str


class GenerationSession(Protocol):
    model: GenerationModel
    capacity: int
    cache_bytes: int

    def forward(self, token_ids: Sequence[int]) -> Any: ...

    def reset(self) -> None: ...


class GenerationModel(Protocol):
    config: GenerationConfig
    device: ExecutionDevice

    def create_session(self, capacity: int) -> GenerationSession: ...
