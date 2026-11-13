"""Weight codec registry: the engine-wide, architecture-neutral source of truth.

A codec describes how one matrix role is stored as explicit safetensors planes:
quantized row-split layouts (packed codes [+ high plane] + FP16 scales) and
dense floating layouts. Everything format-related consults this module:
admission geometry, byte accounting, storage verification, reference
dequantization, and the per-architecture execution-qualification table.

No codec implies conversion: every format loads in its own encoding, and a
codec becomes executable on an architecture only after its qualification
campaign lands with evidence.
"""

from dataclasses import dataclass

import numpy as np
import torch

from venturi.quantization.layout import ROW_FORMATS, row_payload_bytes

DENSE_ITEMSIZE = {"fp16": 2, "bf16": 2, "fp32": 4}


@dataclass(frozen=True)
class Codec:
    fmt: str
    layout: str
    bits: int = 0
    group: int = 0
    scale_dtype: str = ""
    high_plane: bool = False
    dense: bool = False
    forbidden_codes: tuple = ()

    @property
    def quantized(self) -> bool:
        return self.layout == "row_split_k128_v1"

    def encoded_bytes(self, n: int, k: int) -> int:
        """Bytes of the stored representation for an n x k matrix."""
        if self.dense:
            return n * k * DENSE_ITEMSIZE[self.fmt]
        return row_payload_bytes((n, k), self.fmt)

    def upload_bytes(self, n: int, k: int) -> int:
        """Resident VRAM upload size: plane payloads without inter-plane padding.

        For q8_g32_fp16 this equals the historical budget formula
        n * (padded + padded // 32 * 2).
        """
        if self.dense:
            return n * k * DENSE_ITEMSIZE[self.fmt]
        padded = (k + 127) // 128 * 128
        codes = padded * self.bits // 8
        scales = (padded // self.group) * 2
        return n * (codes + scales)


def _row_codec(fmt):
    bits, group = ROW_FORMATS[fmt]
    return Codec(
        fmt=fmt,
        layout="row_split_k128_v1",
        bits=bits,
        group=group,
        scale_dtype="fp16",
        high_plane=bits in (5, 6),
        forbidden_codes=(-128,) if bits == 8 else (),
    )


CODECS = {
    **{fmt: _row_codec(fmt) for fmt in ROW_FORMATS},
    **{
        fmt: Codec(fmt=fmt, layout="contiguous_le_v1", dense=True)
        for fmt in ("fp16", "bf16", "fp32")
    },
}

# Per-architecture execution qualification. A codec enters a set only after
# its qualification campaign on that architecture passes; loading tiers are
# independent of this table. q4_g64_fp16 qualified on SM89 (RTX 4080) on
# 2026-11-11: kernel bit-tests, synthetic native-vs-reference token equality,
# and the real 14.38GB artifact resident smoke.
EXECUTION_QUALIFIED = {
    (7, 0): {"q8_g32_fp16"},
    (8, 9): {"q8_g32_fp16", "q4_g64_fp16", "q5_g64_fp16", "q6_g64_fp16"},
    (8, 6): {"q8_g32_fp16"},
}


def codec_for(fmt: str) -> Codec:
    try:
        return CODECS[fmt]
    except KeyError:
        raise ValueError(
            f"Unknown weight format {fmt!r}; registered: {sorted(CODECS)}"
        ) from None


def dequantize_planes(codes, scales, high, shape, fmt):
    """Explicit CPU reference decode of admitted code/scales, without storage IO."""
    codec = codec_for(fmt)
    n, k = shape
    if not codec.quantized:
        raise ValueError("Expected a quantized codec")
    groups = ((k + 127) // 128 * 128) // codec.group
    c = codes.numpy().reshape(n, groups, 32)
    scale = scales.numpy().reshape(n, groups).astype(np.float32)
    if not np.isfinite(scale).all() or (scale < 0).any():
        raise ValueError("Invalid stored scale")
    if codec.bits == 8:
        if (c == -128).any():
            raise ValueError("Forbidden Q8 code")
        signed = c.astype(np.float32)
    else:
        lanes = np.empty((n, groups, 64), dtype=np.int32)
        lanes[:, :, 0::2] = c & 15
        lanes[:, :, 1::2] = c >> 4
        if codec.high_plane:
            hb = codec.bits - 4
            if high is None:
                raise ValueError("Missing high-bit plane")
            hi = high.numpy().reshape(n, groups, 8 * hb)
            for lane in range(64):
                lanes[:, :, lane] |= (
                    (hi[:, :, lane * hb // 8] >> (lane * hb % 8)) & ((1 << hb) - 1)
                ).astype(np.int32) << 4
        sign = 1 << (codec.bits - 1)
        signed = np.where(lanes >= sign, lanes - (1 << codec.bits), lanes).astype(
            np.float32
        )
    values = (signed * scale[:, :, None]).reshape(n, -1)[:, :k]
    return torch.from_numpy(np.ascontiguousarray(values)).to(torch.float16)
