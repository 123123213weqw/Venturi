"""Request scheduler interface (Protocol, no implementation yet).

Defines the contract for request scheduling. Future implementations
include continuous batching (multiple requests share GPU cycles).
"""
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Scheduler(Protocol):
    """Contract for request schedulers."""

    def submit(self, request: Any) -> str:
        """Submit a generation request. Returns request ID."""
        ...

    def next_batch(self) -> list[Any]:
        """Get the next batch of requests to execute."""
        ...

    def complete(self, request_id: str) -> Any:
        """Mark a request as complete. Returns the result."""
        ...

    def pending(self) -> int:
        """Number of pending requests."""
        ...
