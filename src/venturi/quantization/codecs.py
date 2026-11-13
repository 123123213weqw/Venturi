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

from venturi.ninfer import ROW_FORMATS, row_geometry

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


def dequant_row_reference(artifact, obj):
    """Vectorized decode_row for any row codec (q4/q5/q6_g64).

    Returns FP16 [n, k]; small-int times FP16-scale products are exact in
    FP16, so this equals the native dequantization kernel bit for bit.
    """
    fmt = obj.get("format", "q4_g64_fp16")
    codec = codec_for(fmt)
    bits = codec.bits
    n, k = obj["shape"]
    raw = artifact.read_object(obj["id"], max_bytes=obj["bytes"])
    _, _, _, high_off, scales_off, _ = row_geometry((n, k), fmt)
    groups = ((k + 127) // 128 * 128) // codec.group
    codes = (
        np.frombuffer(raw, dtype=np.uint8, count=n * groups * 32)
        .reshape(n, groups, 32)
    )
    lanes = np.empty((n, groups, 64), dtype=np.int32)
    lanes[:, :, 0::2] = codes & 0x0F
    lanes[:, :, 1::2] = codes >> 4
    if bits > 4:
        hb = bits - 4
        hb_stride = groups * hb * 8  # bytes per row
        high = np.frombuffer(raw, dtype=np.uint8, count=n * hb_stride, offset=high_off).reshape(n, groups, hb * 8)
        extra = np.zeros((n, groups, 64), dtype=np.int32)
        for lane in range(64):
            v = np.zeros((n, groups), dtype=np.int32)
            for b in range(hb):
                byte_i = (lane * hb + b) >> 3
                bit_i = (lane * hb + b) & 7
                v |= ((high[:, :, byte_i].astype(np.int32) >> bit_i) & 1) << b
            extra[:, :, lane] = v
        lanes |= extra << 4
    sign = 1 << (bits - 1)
    span = 1 << bits
    signed = np.where(lanes >= sign, lanes - span, lanes).astype(np.float32)
    scales = (
        np.frombuffer(raw, dtype="<f2", count=n * groups, offset=scales_off)
        .reshape(n, groups)
        .astype(np.float32)
    )
    values = (signed * scales[:, :, None]).reshape(n, groups * 64)[:, :k]
    return torch.from_numpy(np.ascontiguousarray(values)).to(torch.float16)


def dequant_q4_reference(artifact, obj):
    return dequant_row_reference(artifact, obj)
