"""Execution engine core (vLLM-style).

Currently contains interface definitions (Protocols) only.
Implementations will be added as the engine evolves:
- executor: CudaBackend (existing, in ops/blocks.py)
- scheduler: sequential (current) → continuous batching (future)
- sampler: greedy (current) → temperature/top-p (future)
"""
from .executor import GPUExecutor
from .sampler import Sampler
from .scheduler import Scheduler

__all__ = ["GPUExecutor", "Scheduler", "Sampler"]
