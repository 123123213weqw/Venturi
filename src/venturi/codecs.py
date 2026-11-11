"""Weight codec registry: the engine-wide, architecture-neutral source of truth.

A codec describes how one matrix role is stored in the NInfer container:
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

from .ninfer import ROW_FORMATS, row_geometry

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
        return row_geometry((n, k), self.fmt)[-1]

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
    (8, 9): {"q8_g32_fp16", "q4_g64_fp16"},
    (8, 6): {"q8_g32_fp16"},
}


def codec_for(fmt: str) -> Codec:
    try:
        return CODECS[fmt]
    except KeyError:
        raise ValueError(
            f"Unknown weight format {fmt!r}; registered: {sorted(CODECS)}"
        ) from None


def dequant_q4_reference(artifact, obj):
    """Vectorized decode_row: nibble unpack, two's complement, g64 FP16 scales.

    Returns FP16 [n, k]; small-int times FP16-scale products are exact in
    FP16, so this equals the native dequantization kernel bit for bit.
    """
    n, k = obj["shape"]
    raw = artifact.read_object(obj["id"], max_bytes=obj["bytes"])
    *_, high_offset, scales_offset, _ = row_geometry((n, k), "q4_g64_fp16")
    groups = ((k + 127) // 128 * 128) // 64
    codes = (
        np.frombuffer(raw, dtype=np.uint8, count=n * groups * 32)
        .reshape(n, groups, 32)
    )
    lanes = np.empty((n, groups, 64), dtype=np.int8)
    lanes[:, :, 0::2] = (codes & 0x0F).astype(np.int8)
    lanes[:, :, 1::2] = (codes >> 4).astype(np.int8)
    signed = np.where(lanes >= 8, lanes - 16, lanes).astype(np.float32)
    scales = (
        np.frombuffer(
            raw,
            dtype="<f2",
            count=n * groups,
            offset=scales_offset,
        )
        .reshape(n, groups)
        .astype(np.float32)
    )
    values = (signed * scales[:, :, None]).reshape(n, groups * 64)[:, :k]
    return torch.from_numpy(np.ascontiguousarray(values)).to(torch.float16)
