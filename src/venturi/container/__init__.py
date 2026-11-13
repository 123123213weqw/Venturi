"""Pluggable weight container layer."""
from .base import ByteSpan, TensorInfo, WeightContainer
from .safetensors import SafetensorsContainer
from pathlib import Path as _Path
import struct as _struct

def open_container(path):
    """Open any weight file, auto-detecting the format."""
    path = _Path(path)
    if path.is_dir():
        shards = sorted(path.glob("*.safetensors"))
        if len(shards) == 1: return SafetensorsContainer(shards[0])
        raise NotImplementedError("Multi-shard not yet implemented")
    with open(path, "rb") as f: magic = f.read(16)
    if len(magic) >= 9:
        header_len = _struct.unpack("<Q", magic[:8])[0]
        if 0 < header_len < 1024**3 and magic[8:9] in (b"{", b" ", b"\t", b"\n"):
            return SafetensorsContainer(path)
    raise ValueError(f"Unrecognized format: {path}")

__all__ = ["WeightContainer", "TensorInfo", "ByteSpan",
           "SafetensorsContainer", "open_container"]
