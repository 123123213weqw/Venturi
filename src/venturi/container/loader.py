"""Owned, architecture-independent checkpoint tensors loaded through a staged transfer or a reference."""

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import Any, Iterator, Mapping

import torch

from venturi.container.checkpoint import CheckpointError, Checkpoint, inspect_checkpoint
from venturi.dtypes import torch_dtype


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


def _staged_tensors(
    checkpoint: Checkpoint,
    device: torch.device,
    buffer_size: int | None,
    slots: int | None,
    readers: int | None,
) -> Iterator[tuple[str, torch.Tensor]]:
    from venturi.container.safetensors import SafetensorsContainer
    from venturi.container.staged_transfer import (
        DEFAULT_READERS,
        DEFAULT_SLOTS,
        DEFAULT_SLOT_BYTES,
        StagedTransfer,
        pread_fill,
    )

    transfer = StagedTransfer(
        device,
        slot_bytes=DEFAULT_SLOT_BYTES if buffer_size is None else buffer_size,
        slots=DEFAULT_SLOTS if slots is None else slots,
        readers=DEFAULT_READERS if readers is None else readers,
    )
    for path in checkpoint.files:
        container = SafetensorsContainer(path)
        names = sorted(
            (name for name, spec in checkpoint.tensors.items() if spec.file == path),
            key=lambda name: container.tensor_info(name).byte_offset,
        )
        fd = os.open(str(path), os.O_RDONLY)
        try:
            identity = container.stamp
            status = os.fstat(fd)
            if (
                status.st_dev,
                status.st_ino,
                status.st_size,
                status.st_mtime_ns,
                status.st_ctime_ns,
            ) != identity:
                raise CheckpointError(f"Shard changed while loading: {path}")
            for name in names:
                info = container.tensor_info(name)
                spec = checkpoint.tensors[name]
                tensor = torch.empty(
                    spec.shape, dtype=torch_dtype(spec.dtype), device=device
                )

                def read_chunk(offset, count, dest, fd=fd, base=info.byte_offset):
                    return pread_fill(fd, base + offset, count, dest)

                transfer.upload(tensor, read_chunk)
                yield name, tensor
            status = os.fstat(fd)
            if (
                status.st_dev,
                status.st_ino,
                status.st_size,
                status.st_mtime_ns,
                status.st_ctime_ns,
            ) != identity:
                raise CheckpointError(f"Shard changed while loading: {path}")
        finally:
            transfer.close()
            os.close(fd)


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
    loader: str = "staged",
    buffer_size: int | None = None,
    slots: int | None = None,
    readers: int | None = None,
    require_float_dtype: str | None = None,
) -> LoadedCheckpoint:
    """Load a storage-validated checkpoint without interpreting model semantics.

    The staged loader is the production CUDA path: concurrent shard reads into
    a pinned slot ring with non-blocking device copies. The explicit
    safetensors backend allows CPU tests and reference comparisons; it is
    never an automatic fallback.
    """
    started = perf_counter()
    if loader not in ("staged", "safetensors"):
        raise ValueError(f"Unknown loader: {loader!r}")
    if buffer_size is not None and (type(buffer_size) is not int or buffer_size <= 0):
        raise ValueError("buffer_size must be a positive integer")
    if slots is not None and (type(slots) is not int or slots <= 0):
        raise ValueError("slots must be a positive integer")
    if readers is not None and (type(readers) is not int or readers <= 0):
        raise ValueError("readers must be a positive integer")
    if loader != "staged" and (buffer_size is not None or slots is not None or readers is not None):
        raise ValueError("Staging options cannot be used with the safetensors loader")
    target = torch.device(device)
    if target.type not in ("cuda", "cpu"):
        raise ValueError("Only CUDA devices and the CPU reference path are supported")
    if loader == "staged" and target.type != "cuda":
        raise ValueError(
            "The staged loader requires CUDA; use loader='safetensors' explicitly for CPU"
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
        _staged_tensors(checkpoint, target, buffer_size, slots, readers)
        if loader == "staged"
        else _reference_tensors(checkpoint, target)
    )
    # Explicitly close the generator on failure to release shard descriptors.
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
