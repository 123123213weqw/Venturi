"""Owned, architecture-independent checkpoint tensors loaded through Siphon or a reference."""

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import Any, Iterator, Mapping

import torch

from .checkpoint import CheckpointError, Checkpoint, inspect_checkpoint
from .dtypes import torch_dtype


@dataclass(frozen=True)
class LoadStats:
    loader: str
    device: str
    inspect_seconds: float
    load_seconds: float
    total_seconds: float
    tensor_count: int
    tensor_bytes: int
    cuda_allocated_bytes_delta: int | None


@dataclass(frozen=True)
class LoadedCheckpoint:
    """Raw storage only; tensor names, dtypes, shapes, and packing are unchanged.

    The mapping is read-only; callers can still mutate tensor contents. Weights
    remain valid after the loader closes. No forward pass or state is allocated.
    """

    checkpoint: Checkpoint
    weights: Mapping[str, torch.Tensor]
    stats: LoadStats

    @property
    def config(self) -> Mapping[str, Any]:
        return self.checkpoint.config

    def summary(self) -> dict[str, Any]:
        return self.checkpoint.summary() | {"load": asdict(self.stats)}


def _siphon_tensors(
    checkpoint: Checkpoint,
    device: torch.device,
    backend: str | None,
    buffer_size: int | None,
) -> Iterator[tuple[str, torch.Tensor]]:
    try:
        from siphon import Backend, BackendPolicy, safe_open
    except ImportError as error:
        raise RuntimeError(
            "Install the user's Siphon native loader from https://github.com/123123213weqw/siphon; see README.md"
        ) from error
    options: dict[str, Any] = {"copy": True}
    if backend is not None:
        available = Backend.__members__ | BackendPolicy.__members__
        names = [name.strip().upper() for name in backend.split(",")]
        if any(name not in available for name in names):
            raise ValueError(
                f"Unknown Siphon backend {backend!r}; choose from {sorted(available)}"
            )
        candidates = [available[name] for name in names]
        options["backend"] = candidates[0] if len(candidates) == 1 else candidates
    if buffer_size is not None:
        options["buffer_size"] = buffer_size
    # One multi-file open lets Siphon schedule all shards together. copy=True is
    # essential: copy=False tensors alias a rotating buffer that closes here.
    with (
        torch.cuda.device(device),
        safe_open(
            [str(path) for path in checkpoint.files],
            framework="pt",
            device=str(device),
            **options,
        ) as reader,
    ):
        yield from reader.tensors()


def _reference_tensors(
    checkpoint: Checkpoint, device: torch.device
) -> Iterator[tuple[str, torch.Tensor]]:
    from safetensors import safe_open

    for path in checkpoint.files:
        with safe_open(str(path), framework="pt", device=str(device)) as reader:
            for name in reader.keys():
                yield name, reader.get_tensor(name)


def load_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cuda:0",
    loader: str = "siphon",
    siphon_backend: str | None = None,
    buffer_size: int | None = None,
    require_float_dtype: str | None = None,
) -> LoadedCheckpoint:
    """Load a storage-validated checkpoint without interpreting model semantics.

    Siphon is the production CUDA path. The explicit safetensors backend allows
    CPU tests and reference comparisons; it is never an automatic fallback.
    Siphon environment variables retain their normal meaning unless overridden.
    """
    started = perf_counter()
    if loader not in ("siphon", "safetensors"):
        raise ValueError(f"Unknown loader: {loader!r}")
    if buffer_size is not None and (type(buffer_size) is not int or buffer_size <= 0):
        raise ValueError("buffer_size must be a positive integer")
    if loader != "siphon" and (siphon_backend is not None or buffer_size is not None):
        raise ValueError("Siphon options cannot be used with the safetensors loader")
    target = torch.device(device)
    if target.type not in ("cuda", "cpu"):
        raise ValueError("Only CUDA devices and the CPU reference path are supported")
    if loader == "siphon" and target.type != "cuda":
        raise ValueError(
            "Siphon requires CUDA; use loader='safetensors' explicitly for CPU"
        )
    if target.type == "cpu":
        target = torch.device("cpu")
    checkpoint = inspect_checkpoint(path)
    if require_float_dtype is not None:
        if require_float_dtype not in ("F16", "BF16", "F32", "F64"):
            raise ValueError("require_float_dtype must be F16, BF16, F32, or F64")
        mismatched = [
            name
            for name, spec in checkpoint.tensors.items()
            if not spec.dtype.startswith(("I", "U"))
            and spec.dtype != "BOOL"
            and spec.dtype != require_float_dtype
        ]
        if mismatched:
            raise CheckpointError(
                f"Expected floating-point storage {require_float_dtype}; mismatched tensors: {mismatched[:5]}. No dtype conversion is performed."
            )
    inspected = perf_counter()
    dtypes = {
        name: torch_dtype(spec.dtype) for name, spec in checkpoint.tensors.items()
    }
    allocated_before: int | None = None
    if target.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable; no automatic CPU fallback is performed"
            )
        if target.index is None:
            target = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.synchronize(target)
        allocated_before = torch.cuda.memory_allocated(target)

    weights: dict[str, torch.Tensor] = {}
    iterator = (
        _siphon_tensors(checkpoint, target, siphon_backend, buffer_size)
        if loader == "siphon"
        else _reference_tensors(checkpoint, target)
    )
    # Explicitly close the generator on failure to release the native loader.
    try:
        for name, tensor in iterator:
            spec = checkpoint.tensors.get(name)
            if spec is None or name in weights:
                raise CheckpointError(
                    f"Loader returned an unexpected or duplicate tensor: {name}"
                )
            if tuple(tensor.shape) != spec.shape or tensor.dtype != dtypes[name]:
                raise CheckpointError(
                    f"Loaded tensor disagrees with checkpoint header: {name}"
                )
            if tensor.device != target:
                raise CheckpointError(
                    f"Tensor {name} was loaded onto {tensor.device}, expected {target}"
                )
            weights[name] = tensor
    finally:
        iterator.close()
    if weights.keys() != checkpoint.tensors.keys():
        raise CheckpointError("Loader did not return every checkpoint tensor")
    allocated_delta: int | None = None
    if target.type == "cuda":
        torch.cuda.synchronize(target)
        allocated_delta = torch.cuda.memory_allocated(target) - allocated_before
    finished = perf_counter()
    stats = LoadStats(
        loader=loader,
        device=str(target),
        inspect_seconds=inspected - started,
        load_seconds=finished - inspected,
        total_seconds=finished - started,
        tensor_count=len(weights),
        tensor_bytes=checkpoint.tensor_bytes,
        cuda_allocated_bytes_delta=allocated_delta,
    )
    return LoadedCheckpoint(checkpoint, MappingProxyType(weights), stats)
