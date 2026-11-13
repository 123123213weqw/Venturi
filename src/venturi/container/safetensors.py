"""Safetensors-based weight container (primary, HuggingFace standard).

Quantized weights are stored as regular safetensors tensors with a naming
convention (`.codes` / `.scales` suffixes) and file-level metadata carrying
the quantization config. Any safetensors-compatible tool can open and
inspect these files.

File layout (safetensors standard):
    [8B header_length][JSON header][tensor data blocks]

The JSON header maps tensor names to {dtype, shape, data_offsets: [begin, end]}.
data_offsets are relative to the start of the data section. This container
computes absolute file offsets for each tensor and exposes them via
byte_spans() for the C++ WeightSource mmap path.
"""

import json
import struct
from pathlib import Path
from typing import Any

from .base import ByteSpan, TensorInfo

_DTYPE_SIZES = {
    "BOOL": 1, "U8": 1, "I8": 1,
    "F16": 2, "BF16": 2,
    "I16": 2, "U16": 2,
    "F32": 4, "I32": 4, "U32": 4,
    "F64": 8, "I64": 8, "U64": 8,
}

# Map safetensors dtype strings to our dtype names
_DTYPE_MAP = {
    "U8": "uint8", "I8": "int8",
    "F16": "float16", "BF16": "bfloat16", "F32": "float32",
}


class SafetensorsContainer:
    """Read quantized weights from a safetensors file."""

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve(strict=True)
        self._tensors: dict[str, TensorInfo] = {}
        self._metadata: dict[str, Any] = {}
        self._data_start: int = 0
        self._logical_offsets: dict[str, int] = {}
        self._total_payload: int = 0
        self._parse()

    def _parse(self):
        """Parse the safetensors header and compute tensor offsets."""
        with open(self.path, "rb") as f:
            header_len_bytes = f.read(8)
            if len(header_len_bytes) != 8:
                raise ValueError("Truncated safetensors header length")
            header_len = struct.unpack("<Q", header_len_bytes)[0]
            if header_len > 512 * 1024 * 1024:
                raise ValueError("Safetensors header exceeds 512MB bound")
            header_json = f.read(header_len)
            if len(header_json) != header_len:
                raise ValueError("Truncated safetensors header")
            self._data_start = 8 + header_len

        header = json.loads(header_json)
        self._metadata = header.pop("__metadata__", {})

        # Build TensorInfo for each tensor, computing absolute file offsets.
        # Also compute logical offsets (concatenated payload coordinates).
        logical = 0
        for name, spec in sorted(header.items()):
            dtype = spec["dtype"]
            shape = tuple(spec["shape"])
            begin, end = spec["data_offsets"]
            nbytes = end - begin
            self._tensors[name] = TensorInfo(
                name=name,
                dtype=_DTYPE_MAP.get(dtype, dtype.lower()),
                shape=shape,
                byte_offset=self._data_start + begin,
                byte_size=nbytes,
            )
            self._logical_offsets[name] = logical
            logical += nbytes
        self._total_payload = logical

    def list_tensors(self) -> list[str]:
        return list(self._tensors.keys())

    def tensor_info(self, name: str) -> TensorInfo:
        if name not in self._tensors:
            raise KeyError(f"Tensor {name!r} not in {self.path.name}")
        return self._tensors[name]

    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def logical_offset(self, name: str) -> int:
        """Offset of this tensor in the logical concatenated payload."""
        return self._logical_offsets[name]

    def byte_spans(self) -> list[ByteSpan]:
        """One span per tensor, sorted by file offset (for C++ mmap).

        Note: safetensors stores tensors sorted by name, but their data
        offsets may not be contiguous. Each tensor gets its own span.
        """
        total = self.path.stat().st_size
        spans = []
        sorted_by_offset = sorted(self._tensors.values(), key=lambda t: t.byte_offset)
        for info in sorted_by_offset:
            spans.append(ByteSpan(
                path=str(self.path),
                logical_start=self._logical_offsets[info.name],
                file_offset=info.byte_offset,
                bytes=info.byte_size,
                file_bytes=total,
            ))
        return spans

    def read_bytes(self, tensor_name: str, offset: int, size: int) -> bytes:
        info = self.tensor_info(tensor_name)
        if offset < 0 or size < 0 or offset + size > info.byte_size:
            raise ValueError(
                f"Read [{offset}:{offset+size}) exceeds {tensor_name!r} "
                f"size {info.byte_size}"
            )
        with open(self.path, "rb") as f:
            f.seek(info.byte_offset + offset)
            data = f.read(size)
        if len(data) != size:
            raise ValueError(f"Truncated read from {tensor_name!r}")
        return data

    def total_payload_bytes(self) -> int:
        return self._total_payload

    def model_config(self) -> dict:
        """Read config.json from the same directory (HuggingFace convention)."""
        config_path = self.path.parent / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"config.json not found alongside {self.path.name}"
            )
        return json.loads(config_path.read_text())

    def binding_map(self) -> dict[str, str]:
        """Identity mapping: safetensors tensor names ARE the binding names.

        For quantized matrices, each logical weight is stored as two tensors:
        "{source}.codes" and "{source}.scales". The admission layer knows
        to look for both when the plan declares a quantized format.
        """
        return {name: name for name in self._tensors}
