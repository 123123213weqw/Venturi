"""Abstract base class for quantization methods.

Mirrors vLLM's QuantizationConfig pattern: each quantization method
implements this interface and registers itself in the registry. Model
adapters call create_linear() without knowing which method is active.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QuantizationConfig(ABC):
    """Base class for all quantization methods.

    A quantization method describes how weights are stored, how to
    compute their byte layout, and whether it is qualified for execution
    on a given GPU architecture. Subclasses register themselves in
    QUANTIZATION_METHODS via the registry in __init__.py.
    """

    name: str

    @abstractmethod
    def encoded_bytes(self, n: int, k: int) -> int:
        """Bytes of the stored representation for an n × k matrix."""

    @abstractmethod
    def upload_bytes(self, n: int, k: int) -> int:
        """Resident VRAM bytes (may differ from stored, e.g., no alignment gaps)."""

    @abstractmethod
    def execution_qualified(self, capability: tuple[int, int]) -> bool:
        """Whether this method may create a native model on this architecture."""

    @abstractmethod
    def create_linear(self, n: int, k: int, **kwargs: Any) -> Any:
        """Create a linear layer with this quantization applied.

        The returned object is backend-specific (CUDA kernel handle,
        torch module, etc.) and depends on the execution context.
        """
