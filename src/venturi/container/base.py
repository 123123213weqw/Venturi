"""Storage tensor metadata and byte-span records; no model/execution policy."""

from dataclasses import dataclass


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
