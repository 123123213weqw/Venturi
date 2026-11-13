"""Pluggable weight container interface.

A container knows how to store and retrieve tensor bytes on disk; it does
not interpret quantization semantics or model architecture. Implementations
exist for safetensors (primary, HuggingFace standard), GGUF (secondary,
llama.cpp ecosystem), and .ninfer (legacy compatibility).
"""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class TensorInfo:
    """Metadata for one tensor within a container."""
    name: str
    dtype: str          # "uint8", "float16", "bfloat16", "float32"
    shape: tuple[int, ...]
    byte_offset: int    # absolute offset within the container file(s)
    byte_size: int      # total bytes of this tensor's data


@dataclass(frozen=True)
class ByteSpan:
    """A contiguous byte region in a file, for C++ mmap (VRSpan-compatible)."""
    path: str
    logical_start: int  # offset in the logical concatenated payload
    file_offset: int    # byte offset within the file
    bytes: int          # number of bytes
    file_bytes: int     # total file size (for C++ fstat validation)


@runtime_checkable
class WeightContainer(Protocol):
    """Uniform interface for any weight storage format.

    Implementations must be read-only and deterministic: calling any method
    twice must return identical results (unless the underlying file changed,
    which should raise). No implicit conversion between formats.
    """

    def list_tensors(self) -> list[str]:
        """All tensor names in the container."""
        ...

    def tensor_info(self, name: str) -> TensorInfo:
        """Shape/dtype/offset/size for one tensor."""
        ...

    def metadata(self) -> dict[str, Any]:
        """File-level metadata (quantization config, model info, provenance)."""
        ...

    def byte_spans(self) -> list[ByteSpan]:
        """Contiguous byte regions for C++ WeightSource mmap."""
        ...

    def read_bytes(self, tensor_name: str, offset: int, size: int) -> bytes:
        """Read a byte range from one tensor's data region."""
        ...

    def total_payload_bytes(self) -> int:
        """Total bytes of tensor data (for memory budgeting)."""
        ...

    def model_config(self) -> dict[str, Any]:
        """The model's architecture configuration (hidden_size, layers, etc.)."""
        ...

    def binding_map(self) -> dict[str, str]:
        """Logical role name → tensor name in this container.

        E.g. {"text/token_embedding": "token_embd.weight", ...}
        For containers without explicit bindings, this maps directly.
        """
        ...
