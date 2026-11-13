"""Encoded matrix geometry, independent of file containers and model adapters."""

ROW_FORMATS = {
    "q4_g64_fp16": (4, 64),
    "q5_g64_fp16": (5, 64),
    "q6_g64_fp16": (6, 64),
    "q8_g32_fp16": (8, 32),
}


def _align(n, alignment=256):
    return (n + alignment - 1) // alignment * alignment


def row_geometry(shape, format):
    """GPU-local packed scratch geometry, not checkpoint file offsets."""
    n, k = shape
    bits, group = ROW_FORMATS[format]
    groups = _align(k, 128) // group
    base_bytes = n * groups * 32
    high_bytes = n * groups * (8 * (bits - 4) if bits in (5, 6) else 0)
    high_offset = _align(base_bytes)
    scales_offset = high_offset + _align(high_bytes)
    return (
        bits,
        group,
        groups,
        high_offset,
        scales_offset,
        scales_offset + n * groups * 2,
    )


def row_payload_bytes(shape, format):
    """Physical code/high/scale bytes, without GPU-local alignment padding."""
    n, k = shape
    bits, group = ROW_FORMATS[format]
    padded = _align(k, 128)
    return n * (padded * bits // 8 + padded // group * 2)
