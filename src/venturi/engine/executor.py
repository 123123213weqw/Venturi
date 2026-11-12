"""GPU executor interface (Protocol, no implementation yet).

Defines the contract for GPU resource management: allocation,
synchronization, and memory introspection. The current CudaBackend
in ops/blocks.py satisfies this interface implicitly.
"""
from typing import Protocol, runtime_checkable


@runtime_checkable
class GPUExecutor(Protocol):
    """Contract for GPU execution backends."""

    def allocate(self, bytes: int) -> int:
        """Allocate device memory. Returns pointer."""
        ...

    def free(self, ptr: int) -> None:
        """Free device memory."""
        ...

    def sync(self) -> None:
        """Synchronize all pending work on this executor."""
        ...

    def mem_info(self) -> tuple[int, int]:
        """Returns (free_bytes, total_bytes)."""
        ...
