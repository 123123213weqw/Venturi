"""Weight codec registry: the single source of truth for storage formats.

A codec describes how one matrix role is encoded in the NInfer container:
quantized row-split layouts (packed codes [+ high plane] + FP16 scales) and
dense floating layouts. Loading is codec-driven; execution is separately
qualified (EXECUTION_QUALIFIED) so a format can load and byte-verify before
any native kernel path exists for it. No codec implies conversion: every
format loads in its own encoding.
"""

from dataclasses import dataclass

from ...ninfer import ROW_FORMATS, row_geometry

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

    def plane_bytes(self, n: int, k: int):
        """(codes_bytes, high_bytes, scale_bytes) for row-split codecs."""
        bits, group, groups, high_offset, scales_offset, total = row_geometry(
            (n, k), self.fmt
        )
        return high_offset, scales_offset - high_offset, total - scales_offset


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

# Formats that may create a native model today. Everything else loads,
# byte-verifies and accounts, and fails model creation explicitly.
EXECUTION_QUALIFIED = {"q8_g32_fp16"}


def codec_for(fmt: str) -> Codec:
    try:
        return CODECS[fmt]
    except KeyError:
        raise ValueError(
            f"Unknown weight format {fmt!r}; registered: {sorted(CODECS)}"
        ) from None
