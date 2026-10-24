"""Byte-aligned safetensors storage types. Packed INT4 uses integer containers."""

import torch

# Native sub-byte float formats need a different storage/shape contract; reject
# them rather than treating one logical value as one byte. No dtype is cast.
DTYPE_INFO = {
    "BOOL": ("bool", 1),
    "I8": ("int8", 1),
    "U8": ("uint8", 1),
    "I16": ("int16", 2),
    "U16": ("uint16", 2),
    "I32": ("int32", 4),
    "U32": ("uint32", 4),
    "I64": ("int64", 8),
    "U64": ("uint64", 8),
    "F16": ("float16", 2),
    "BF16": ("bfloat16", 2),
    "F32": ("float32", 4),
    "F64": ("float64", 8),
    "C64": ("complex64", 8),
    "F8_E4M3": ("float8_e4m3fn", 1),
    "F8_E4M3FNUZ": ("float8_e4m3fnuz", 1),
    "F8_E5M2": ("float8_e5m2", 1),
    "F8_E5M2FNUZ": ("float8_e5m2fnuz", 1),
    "F8_E8M0": ("float8_e8m0fnu", 1),
}


def torch_dtype(code: str) -> torch.dtype:
    name = DTYPE_INFO[code][0]
    dtype = getattr(torch, name, None)
    if dtype is None:
        raise ValueError(
            f"Installed PyTorch does not support safetensors dtype {code} ({name})"
        )
    return dtype
