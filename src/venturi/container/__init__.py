"""Pluggable weight container layer.

The engine consumes only the WeightContainer interface; it never knows
which file format is on disk. Adding a format = adding a Container class
+ one line in open_container(). Removing a format = deleting that class.

Usage:
    from venturi.container import open_container
    container = open_container("model.safetensors")  # or .ninfer, .gguf
    weights = QwenQ8Weights.from_container(container)
"""

from pathlib import Path

from .base import ByteSpan, TensorInfo, WeightContainer
from .safetensors import SafetensorsContainer

# Legacy re-exports (backward compatibility; new code should not use these).


def open_container(path: str | Path) -> WeightContainer:
    """Open any weight file, auto-detecting the format by magic bytes.

    Supported formats:
        .safetensors  — HuggingFace standard (primary)
                directory/    — Sharded safetensors (model_dir/model-*.safetensors)
        .gguf         — llama.cpp format (future)

    Returns a WeightContainer regardless of format. The engine code that
    consumes the returned object is format-agnostic.
    """
    path = Path(path)

    # Directory → sharded safetensors
    if path.is_dir():
        shards = sorted(path.glob("*.safetensors"))
        if not shards:
            raise ValueError(f"No .safetensors files in directory: {path}")
        if len(shards) == 1:
            return SafetensorsContainer(shards[0])
        raise NotImplementedError(
            f"Multi-shard directory loading not yet implemented ({len(shards)} shards)"
        )

    # Single file → detect by magic bytes (not extension, which may lie)
    if not path.is_file():
        raise FileNotFoundError(f"Weight file not found: {path}")

    with open(path, "rb") as f:
        magic = f.read(16)


    # safetensors: 8-byte little-endian header length followed by JSON
    # The first 8 bytes are a valid u64 that's reasonable in size (< 1GB)
    # and the next bytes look like JSON ('{' or whitespace)
    import struct
    if len(magic) >= 9:
        header_len = struct.unpack("<Q", magic[:8])[0]
        if 0 < header_len < 1024**3 and magic[8:9] in (b"{", b" ", b"\t", b"\n"):
            return SafetensorsContainer(path)

    if magic.startswith(b"GGUF"):
        raise NotImplementedError("GGUF container not yet implemented")

    raise ValueError(
        f"Unrecognized weight format: {path} (magic={magic[:8]!r}). "
        f"Supported: .safetensors"
    )


__all__ = [
    "WeightContainer", "TensorInfo", "ByteSpan",
    "SafetensorsContainer",     "open_container",
    # Legacy
    ]
